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

def channel(cd, lb=40):
    if len(cd) < 20:
        return None
    rc = cd[-lb:] if len(cd) >= lb else cd
    nr, off = len(rc), len(cd)-len(rc)
    ph, pl = [], []
    for i in range(2, nr-2):
        if all(rc[j]["h"] <= rc[i]["h"] for j in range(max(0,i-2), min(nr,i+3)) if j != i):
            ph.append((i, rc[i]["h"]))
        if all(rc[j]["l"] >= rc[i]["l"] for j in range(max(0,i-2), min(nr,i+3)) if j != i):
            pl.append((i, rc[i]["l"]))
    cls = [c["c"] for c in rc]
    sg = np.polyfit(np.arange(nr), cls, 1)[0]
    if sg >= 0:
        if len(pl) < 2: return None
        xs, ys = np.array([p[0] for p in pl]), np.array([p[1] for p in pl])
    else:
        if len(ph) < 2: return None
        xs, ys = np.array([p[0] for p in ph]), np.array([p[1] for p in ph])
    sl, ic = np.polyfit(xs, ys, 1)
    xa = np.arange(nr)
    base = sl*xa + ic
    hi = np.array([c["h"] for c in rc])
    lo = np.array([c["l"] for c in rc])
    return {"x": np.arange(off, len(cd)), "upper": base+np.max(hi-base), "lower": base+np.min(lo-base), "slope": sl}

def detect_long_candle(candle, ratio=2.0):
    """Détecte une bougie longue (corps > ratio * moyenne)"""
    body = abs(candle["c"] - candle["o"])
    if body == 0:
        return False
    return body > (candle["h"] - candle["l"]) * 0.5

def chart_fractale(candles, sma3, sma10, title, price, dec=2, signal=None, trend=None):
    if len(candles) < 10:
        return None
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0d1117')
    
    for i, c in enumerate(candles):
        col = '#26a69a' if c["c"] >= c["o"] else '#ef5350'
        ax.plot([i, i], [c["l"], c["h"]], color=col, lw=1.2)
        ax.add_patch(plt.Rectangle((i-0.35, min(c["o"], c["c"])), 0.7, abs(c["c"]-c["o"]) or 0.0001, facecolor=col, edgecolor=col))
    
    if sma3 and sma10:
        ax.plot(range(len(sma3)), sma3, color='#00b4d8', lw=2, label='SMA 3')
        ax.plot(range(len(sma10)), sma10, color='#ff6b6b', lw=2, label='SMA 10')
    
    ax.axhline(price, color='#f59e0b', ls='-.', lw=2, alpha=0.7)
    ax.text(len(candles)-1, price, f'  {price:.{dec}f}', color='#f59e0b', fontsize=12, fontweight='bold')
    
    if signal:
        ax.set_title(f"{title} - {signal} | Tendance: {trend}", color='white', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"{title} - Tendance: {trend}", color='white', fontsize=14, fontweight='bold')
    
    ax.set_ylabel("Prix", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='gray')
    if sma3 and sma10:
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
    
    # Récupérer H2 et H1
    if src == "deriv":
        candles_h2 = get_candles_deriv(info["symbol"], granularity=7200)
        candles_h1 = get_candles_deriv(info["symbol"], granularity=3600)
    else:
        candles_h2 = get_candles(info["symbol"], interval="1h", range_days="60d")
        if len(candles_h2) >= 2:
            df = pd.DataFrame(candles_h2)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            resampled = df.resample('2H').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
            candles_h2 = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
            candles_h2 = candles_h2[-50:]
        candles_h1 = get_candles(info["symbol"], interval="1h", range_days="7d")
    
    if not candles_h2 or not candles_h1:
        log(f"⚠️ Pas de données pour {key}")
        return
    
    # Canal H2 (tendance)
    ch_h2 = channel(candles_h2)
    if ch_h2:
        if ch_h2["slope"] > 0.0001:
            tendance = "HAUSSIERE"
        elif ch_h2["slope"] < -0.0001:
            tendance = "BAISSIERE"
        else:
            tendance = "LATERALE"
    else:
        tendance = "INCONNUE"
    
    # SMA H1
    sma3_h1 = calc_sma(candles_h1, 3)
    sma10_h1 = calc_sma(candles_h1, 10)
    cross_type, cross_bool = detect_cross_sma(sma3_h1, sma10_h1)
    
    # Longue bougie H1
    last_candle = candles_h1[-1]
    is_long = detect_long_candle(last_candle)
    is_green = last_candle["c"] >= last_candle["o"]
    is_red = last_candle["c"] < last_candle["o"]
    
    cp = candles_h1[-1]["c"]
    dec = info["dec"]
    
    msg = ""
    conseil = ""
    signal_type = ""
    condition_remplie = False
    
    # ===== LOGIQUE =====
    if tendance == "HAUSSIERE":
        # Signal 1 : Croisement SMA 3/10 HAUT
        if cross_bool and cross_type == "HAUT":
            condition_remplie = True
            signal_type = "CROISEMENT SMA 3/10 HAUT"
            conseil = "ACHAT (Canal H2 HAUSSIER + Croisement SMA 3/10 HAUT)"
            msg = f"SIGNAL ACHAT FRACTALE\nTendance H2: HAUSSIERE\nH1: SMA 3 croise au-dessus SMA 10"
        
        # Signal 2 : Longue bougie verte
        elif is_long and is_green:
            condition_remplie = True
            signal_type = "LONGUE BOUGIE VERTE"
            conseil = "ACHAT (Canal H2 HAUSSIER + Longue bougie verte H1)"
            msg = f"SIGNAL ACHAT FRACTALE\nTendance H2: HAUSSIERE\nH1: Longue bougie verte detectee"
    
    elif tendance == "BAISSIERE":
        # Signal 1 : Croisement SMA 3/10 BAS
        if cross_bool and cross_type == "BAS":
            condition_remplie = True
            signal_type = "CROISEMENT SMA 3/10 BAS"
            conseil = "VENTE (Canal H2 BAISSIER + Croisement SMA 3/10 BAS)"
            msg = f"SIGNAL VENTE FRACTALE\nTendance H2: BAISSIERE\nH1: SMA 3 croise en dessous SMA 10"
        
        # Signal 2 : Longue bougie rouge
        elif is_long and is_red:
            condition_remplie = True
            signal_type = "LONGUE BOUGIE ROUGE"
            conseil = "VENTE (Canal H2 BAISSIER + Longue bougie rouge H1)"
            msg = f"SIGNAL VENTE FRACTALE\nTendance H2: BAISSIERE\nH1: Longue bougie rouge detectee"
    
    if condition_remplie:
        full_msg = f"{msg}\n\nPrix: {cp:.{dec}f}\n{datetime.now(pytz.timezone('Africa/Porto-Novo')).strftime('%H:%M')}H Benin\nFractale Bot"
        
        log(f"📤 SIGNAL {key} - {conseil}")
        send(info["ntfy"], f"ALERTE {key} - {conseil}", full_msg)
        
        # Graphique H2
        img_h2 = chart_fractale(candles_h2, None, None, f"{info['name']} H2 - Canal", cp, dec, None, tendance)
        if img_h2:
            time.sleep(1)
            send(info["ntfy"], f"{key} H2 - Canal", f"Canal {tendance}", img_h2)
        
        # Graphique H1 avec SMA
        img_h1 = chart_fractale(candles_h1, sma3_h1, sma10_h1, f"{info['name']} H1 - {signal_type}", cp, dec, signal_type, tendance)
        if img_h1:
            time.sleep(1)
            send(info["ntfy"], f"{key} H1 - Signal", signal_type, img_h1)
    else:
        log(f"⏭️ SILENCE {key} - Pas de signal Fractale")

if __name__ == "__main__":
    log("🚀 FRACTALE BOT - H2 (canal) + H1 (SMA 3/10 ou longue bougie)")
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
