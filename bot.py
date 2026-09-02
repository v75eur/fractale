import time, requests, io, pytz, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

# ===== CONFIGURATION NTFY =====
NTFY_XAU = os.getenv("NTFY_XAU", "https://ntfy.sh/rick-xau-fractale-secret-2026")
NTFY_EUR = os.getenv("NTFY_EUR", "https://ntfy.sh/rick-eur-fractale-secret-2026")
NTFY_GBP = os.getenv("NTFY_GBP", "https://ntfy.sh/rick-gbp-fractale-secret-2026")
NTFY_V75 = os.getenv("NTFY_V75", "https://ntfy.sh/rick-v75-fractale-secret-2026")

PAIRS = {
    "XAUUSD": {"symbol": "GC=F", "ntfy": NTFY_XAU, "dec": 2, "name": "XAUUSD (Or)"},
    "EURUSD": {"symbol": "EURUSD=X", "ntfy": NTFY_EUR, "dec": 5, "name": "EURUSD"},
    "GBPUSD": {"symbol": "GBPUSD=X", "ntfy": NTFY_GBP, "dec": 5, "name": "GBPUSD"},
    "V75": {"symbol": "R_75", "ntfy": NTFY_V75, "dec": 2, "name": "Volatility 75", "source": "deriv"},
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def send(url, title, msg, img=None):
    for i in range(5):
        try:
            h = {"Title": title}
            if img:
                h["Filename"] = "chart.png"
                h["X-Message"] = msg
                r = requests.post(url, data=img, headers={k: v.encode('utf-8') for k, v in h.items()}, timeout=30)
            else:
                r = requests.post(url, data=msg.encode('utf-8'), headers=h, timeout=15)
            if r.status_code == 200:
                log(f"✅ {title}")
                return True
            log(f"⚠️ Tentative {i+1}: status {r.status_code}")
            time.sleep(2**i)
        except Exception as e:
            log(f"❌ Erreur: {e}")
            time.sleep(2**i)
    return False

def get_candles_deriv(sym, granularity=60):
    try:
        import websocket as ws_client
        log(f"🔌 Connexion Deriv pour {sym}...")
        ws = ws_client.create_connection('wss://ws.binaryws.com/websockets/v3?app_id=1089', timeout=10)
        ws.send(json.dumps({"ticks_history": sym, "count": 100, "end": "latest", "start": 1, "style": "candles", "granularity": granularity}))
        r = json.loads(ws.recv())
        ws.close()
        if "candles" in r:
            candles = [{"t": c["epoch"], "o": float(c["open"]), "h": float(c["high"]), "l": float(c["low"]), "c": float(c["close"])} for c in r["candles"]]
            log(f"✅ Deriv: {len(candles)} bougies pour {sym}")
            return candles
        else:
            log(f"⚠️ Deriv: pas de bougies pour {sym}")
            return []
    except Exception as e:
        log(f"❌ Deriv {sym}: {e}")
        return []

def get_candles(sym, interval="1h", range_days="60d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={range_days}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json()["chart"]["result"][0]
        closes = data["indicators"]["quote"][0]["close"]
        opens = data["indicators"]["quote"][0]["open"]
        highs = data["indicators"]["quote"][0]["high"]
        lows = data["indicators"]["quote"][0]["low"]
        timestamps = data["timestamp"]
        candles = []
        for i in range(len(closes)):
            if closes[i] and opens[i] and highs[i] and lows[i]:
                candles.append({"t": timestamps[i], "o": opens[i], "h": highs[i], "l": lows[i], "c": closes[i]})
        candles = candles[-100:] if len(candles) > 100 else candles
        log(f"✅ Yahoo: {len(candles)} bougies pour {sym} en {interval}")
        return candles
    except Exception as e:
        log(f"❌ Yahoo {sym}: {e}")
        return []

def calc_sma(candles, period=3):
    closes = [c["c"] for c in candles]
    if len(closes) < period:
        return [None] * len(closes)
    sma = []
    for i in range(len(closes)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(sum(closes[i-period+1:i+1]) / period)
    return sma

def detect_cross_sma(sma3, sma10):
    if len(sma3) < 2 or len(sma10) < 2:
        return None, False
    if sma3[-1] is None or sma10[-1] is None or sma3[-2] is None or sma10[-2] is None:
        return None, False
    if sma3[-1] > sma10[-1] and sma3[-2] <= sma10[-2]:
        return "HAUT", True
    elif sma3[-1] < sma10[-1] and sma3[-2] >= sma10[-2]:
        return "BAS", True
    return None, False

def detect_trend(candles, period=20):
    if len(candles) < period:
        return "INCONNUE", 0
    closes = np.array([c["c"] for c in candles[-period:]])
    slope = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
    if slope > 0.001:
        return "HAUSSIERE", slope
    elif slope < -0.001:
        return "BAISSIERE", slope
    return "LATERALE", slope

def detect_pattern(candle, prev_candle):
    body = abs(candle["c"] - candle["o"])
    upper_wick = candle["h"] - max(candle["c"], candle["o"])
    lower_wick = min(candle["c"], candle["o"]) - candle["l"]
    pattern = None
    
    if lower_wick > body * 2 and lower_wick > upper_wick * 2:
        pattern = "PIN_BAR_HAUT"
    elif upper_wick > body * 2 and upper_wick > lower_wick * 2:
        pattern = "PIN_BAR_BAS"
    
    if prev_candle:
        if candle["c"] > prev_candle["o"] and candle["o"] < prev_candle["c"] and body > prev_candle["h"] - prev_candle["l"]:
            pattern = "ENGULFING_HAUT"
        elif candle["c"] < prev_candle["o"] and candle["o"] > prev_candle["c"] and body > prev_candle["h"] - prev_candle["l"]:
            pattern = "ENGULFING_BAS"
    
    if body < (candle["h"] - candle["l"]) * 0.1:
        pattern = "DOJI"
    
    return pattern

def chart_fractale_sma(candles, sma3, sma10, title, price, dec=2, cross_type=None, trend=None):
    if len(candles) < 10:
        return None
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0d1117')
    
    for i, c in enumerate(candles):
        col = '#26a69a' if c["c"] >= c["o"] else '#ef5350'
        ax.plot([i, i], [c["l"], c["h"]], color=col, lw=1.2)
        ax.add_patch(plt.Rectangle((i-0.35, min(c["o"], c["c"])), 0.7, abs(c["c"]-c["o"]) or 0.0001, facecolor=col, edgecolor=col))
    
    ax.plot(range(len(sma3)), sma3, color='#00b4d8', lw=2, label='SMA 3')
    ax.plot(range(len(sma10)), sma10, color='#ff6b6b', lw=2, label='SMA 10')
    
    ax.axhline(price, color='#f59e0b', ls='-.', lw=2, alpha=0.7)
    ax.text(len(candles)-1, price, f'  {price:.{dec}f}', color='#f59e0b', fontsize=12, fontweight='bold')
    
    if cross_type:
        ax.set_title(f"{title} - Croisement SMA {cross_type} | Tendance: {trend}", color='white', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"{title} - {trend}", color='white', fontsize=14, fontweight='bold')
    
    ax.set_ylabel("Prix", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='gray')
    ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', framealpha=0.9)
    
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0a0a0a')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def analyze_fractale(key, info):
    log(f"🔍 Analyse Fractale {key}...")
    src = info.get("source", "yahoo")
    
    if src == "deriv":
        candles_h8 = get_candles_deriv(info["symbol"], granularity=28800)
        candles_h4 = get_candles_deriv(info["symbol"], granularity=14400)
        candles_h2 = get_candles_deriv(info["symbol"], granularity=7200)
        candles_h1 = get_candles_deriv(info["symbol"], granularity=3600)
        candles_m15 = get_candles_deriv(info["symbol"], granularity=900)
    else:
        candles_h8 = get_candles(info["symbol"], interval="1h", range_days="60d")
        if len(candles_h8) >= 8:
            df = pd.DataFrame(candles_h8)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            resampled = df.resample('8H').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
            candles_h8 = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
            candles_h8 = candles_h8[-50:]
        candles_h4 = get_candles(info["symbol"], interval="1h", range_days="60d")
        if len(candles_h4) >= 4:
            df = pd.DataFrame(candles_h4)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            resampled = df.resample('4H').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
            candles_h4 = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
            candles_h4 = candles_h4[-50:]
        candles_h2 = get_candles(info["symbol"], interval="1h", range_days="7d")
        if len(candles_h2) >= 2:
            df = pd.DataFrame(candles_h2)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            resampled = df.resample('2H').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
            candles_h2 = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
            candles_h2 = candles_h2[-50:]
        candles_h1 = get_candles(info["symbol"], interval="1h", range_days="3d")
        candles_m15 = get_candles(info["symbol"], interval="15m", range_days="2d")
    
    if not candles_h8 or not candles_h4 or not candles_h2 or not candles_h1 or not candles_m15:
        log(f"⚠️ Pas de données pour {key}")
        return
    
    # Analyses
    trend_h8, _ = detect_trend(candles_h8)
    trend_h4, _ = detect_trend(candles_h4)
    
    # SMA H1
    sma3_h1 = calc_sma(candles_h1, 3)
    sma10_h1 = calc_sma(candles_h1, 10)
    cross_h1, cross_h1_bool = detect_cross_sma(sma3_h1, sma10_h1)
    
    # SMA M15
    sma3_m15 = calc_sma(candles_m15, 3)
    sma10_m15 = calc_sma(candles_m15, 10)
    cross_m15, cross_m15_bool = detect_cross_sma(sma3_m15, sma10_m15)
    
    # Pattern H2
    pattern_h2 = None
    if len(candles_h2) >= 2:
        pattern_h2 = detect_pattern(candles_h2[-1], candles_h2[-2])
    
    cp = candles_m15[-1]["c"]
    dec = info["dec"]
    
    msg = ""
    conseil = ""
    condition_remplie = False
    
    # Conditions
    if trend_h8 == trend_h4 and trend_h8 != "LATERALE" and trend_h8 != "INCONNUE":
        
        if trend_h8 == "HAUSSIERE":
            if cross_h1_bool and cross_h1 == "HAUT" and cross_m15_bool and cross_m15 == "HAUT":
                if pattern_h2 in ["PIN_BAR_HAUT", "ENGULFING_HAUT", "DOJI"]:
                    condition_remplie = True
                    conseil = f"ACHAT (H1 SMA HAUT + M15 SMA HAUT | H2: {pattern_h2})"
                    msg = f"SIGNAL ACHAT FRACTALE\nTendance H8: {trend_h8} | H4: {trend_h4}\nH1: Croisement SMA 3/10 HAUT\nM15: Croisement SMA 3/10 HAUT\nPattern H2: {pattern_h2}"
        
        elif trend_h8 == "BAISSIERE":
            if cross_h1_bool and cross_h1 == "BAS" and cross_m15_bool and cross_m15 == "BAS":
                if pattern_h2 in ["PIN_BAR_BAS", "ENGULFING_BAS", "DOJI"]:
                    condition_remplie = True
                    conseil = f"VENTE (H1 SMA BAS + M15 SMA BAS | H2: {pattern_h2})"
                    msg = f"SIGNAL VENTE FRACTALE\nTendance H8: {trend_h8} | H4: {trend_h4}\nH1: Croisement SMA 3/10 BAS\nM15: Croisement SMA 3/10 BAS\nPattern H2: {pattern_h2}"
    
    if condition_remplie:
        full_msg = f"{msg}\n\nPrix: {cp:.{dec}f}\n{datetime.now(pytz.timezone('Africa/Porto-Novo')).strftime('%H:%M')}H Benin\nFractale Bot"
        
        log(f"📤 SIGNAL {key} - {conseil}")
        send(info["ntfy"], f"ALERTE {key} - {conseil}", full_msg)
        
        # 5 graphiques
        img_h8 = chart_fractale_sma(candles_h8, [], [], f"{info['name']} H8", cp, dec, None, trend_h8)
        if img_h8:
            time.sleep(1)
            send(info["ntfy"], f"{key} H8 - Tendance", f"Tendance {trend_h8}", img_h8)
        
        img_h4 = chart_fractale_sma(candles_h4, [], [], f"{info['name']} H4", cp, dec, None, trend_h4)
        if img_h4:
            time.sleep(1)
            send(info["ntfy"], f"{key} H4 - Confirmation", f"Confirmation {trend_h4}", img_h4)
        
        img_h2 = chart_fractale_sma(candles_h2, [], [], f"{info['name']} H2", cp, dec, None, pattern_h2)
        if img_h2:
            time.sleep(1)
            send(info["ntfy"], f"{key} H2 - Pattern", f"Pattern {pattern_h2}", img_h2)
        
        img_h1 = chart_fractale_sma(candles_h1, sma3_h1, sma10_h1, f"{info['name']} H1", cp, dec, cross_h1, trend_h4)
        if img_h1:
            time.sleep(1)
            send(info["ntfy"], f"{key} H1 - SMA", f"Croisement SMA 3/10 {cross_h1}", img_h1)
        
        img_m15 = chart_fractale_sma(candles_m15, sma3_m15, sma10_m15, f"{info['name']} M15", cp, dec, cross_m15, trend_h4)
        if img_m15:
            time.sleep(1)
            send(info["ntfy"], f"{key} M15 - SMA", f"Croisement SMA 3/10 {cross_m15}", img_m15)
    else:
        log(f"⏭️ SILENCE {key} - Pas de signal Fractale")

if __name__ == "__main__":
    log("🚀 FRACTALE BOT - SMA 3/10 (H8+H4+H2+H1+M15)")
    now = datetime.now(pytz.timezone('Africa/Porto-Novo'))
    h, j = now.hour, now.weekday()
    
    log("→ V75 (7j/7)")
    analyze_fractale("V75", PAIRS["V75"])
    
    if j < 5:
        log(f"📊 Analyse Forex {h}H")
        for key in ["XAUUSD", "EURUSD", "GBPUSD"]:
            log(f"→ {key}")
            analyze_fractale(key, PAIRS[key])
    else:
        log(f"💤 Forex ferme week-end")
    
    log("✅ Termine")
