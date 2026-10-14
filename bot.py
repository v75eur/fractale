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
        candles = candles[-120:] if len(candles) > 120 else candles
        log(f"✅ Yahoo: {len(candles)} bougies pour {sym} en {interval} (sur {range_days})")
        return candles
    except Exception as e:
        log(f"❌ Yahoo {sym}: {e}")
        return []

def resample_candles(candles, rule):
    if not candles:
        return []
    df = pd.DataFrame(candles)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df.set_index('t', inplace=True)
    resampled = df.resample(rule).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
    out = [{"t": int(ts.timestamp()), "o": row['o'], "h": row['h'], "l": row['l'], "c": row['c']} for ts, row in resampled.iterrows()]
    return out[-100:]

def calc_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(candles)):
        h, l, c_prev = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    return sum(tr_list[-period:]) / period

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

def chart_fractale(cd, cp, info, title_suffix="", sl_price=None):
    if len(cd) < 5:
        return None
    ch = channel(cd)
    name, dec = info["name"], info["dec"]
    fig, ax = plt.subplots(figsize=(18,10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0d1117')
    for i, c in enumerate(cd):
        col = '#26a69a' if c["c"]>=c["o"] else '#ef5350'
        ax.plot([i,i], [c["l"],c["h"]], color=col, lw=1.2)
        ax.add_patch(plt.Rectangle((i-0.35, min(c["o"],c["c"])), 0.7, abs(c["c"]-c["o"]) or 0.0001, facecolor=col, edgecolor=col))
    if ch:
        cc = '#a371f7' if ch["slope"]>0 else '#f0883e'
        ax.plot(ch["x"], ch["upper"], color=cc, lw=2.5, ls='--', alpha=0.9)
        ax.plot(ch["x"], ch["lower"], color=cc, lw=2.5, ls='--', alpha=0.9)
        ax.fill_between(ch["x"], ch["lower"], ch["upper"], color=cc, alpha=0.06)
    
    ax.axhline(cp, color='#f59e0b', ls='-.', lw=3)
    ax.text(2, cp, f' Prix: {cp:.{dec}f}', color='#f59e0b', fontsize=14, va='bottom', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1400', alpha=0.85))

    if sl_price:
        ax.axhline(sl_price, color='#ef5350', ls='--', lw=2.5)
        ax.text(2, sl_price, f' Stop-Loss: {sl_price:.{dec}f}', color='#ef5350', fontsize=14, va='top', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#0a0a0a', alpha=0.85))

    n = len(cd)
    step = max(1, n//8)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([datetime.fromtimestamp(cd[i]["t"]).strftime('%d/%m\n%Hh') for i in range(0,n,step)], color='white', fontsize=11)
    ax.set_title(f"  {name} — {title_suffix} | Prix: {cp:.{dec}f}", color='white', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Prix", color='white', fontsize=13)
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(True, alpha=0.12, color='gray')
    for sp in ax.spines.values():
        sp.set_color('#333333')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0a0a0a')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def analyze_fractale(key, info):
    log(f"🔍 Analyse Fractale Multi-TF (H8+H4+H2+H1) avec historique élargi pour {key}...")
    # On charge un historique large de 60 jours en H1 pour nourrir proprement toutes les unités supérieures
    candles_h1 = get_candles(info["symbol"], interval="1h", range_days="60d")
    if not candles_h1 or len(candles_h1) < 50:
        log(f"⚠️ Données insuffisantes pour {key}")
        return

    candles_h2 = resample_candles(candles_h1, '2H')
    candles_h4 = resample_candles(candles_h1, '4H')
    candles_h8 = resample_candles(candles_h1, '8H')

    ch_h8 = channel(candles_h8)
    ch_h4 = channel(candles_h4)
    ch_h2 = channel(candles_h2)
    ch_h1 = channel(candles_h1)

    t8 = "HAUSSIERE" if ch_h8 and ch_h8["slope"] > 0.0001 else ("BAISSIERE" if ch_h8 and ch_h8["slope"] < -0.0001 else "LATERALE")
    t4 = "HAUSSIERE" if ch_h4 and ch_h4["slope"] > 0.0001 else ("BAISSIERE" if ch_h4 and ch_h4["slope"] < -0.0001 else "LATERALE")
    t2 = "HAUSSIERE" if ch_h2 and ch_h2["slope"] > 0.0001 else ("BAISSIERE" if ch_h2 and ch_h2["slope"] < -0.0001 else "LATERALE")

    cp = candles_h1[-1]["c"]
    dec = info["dec"]
    atr_h1 = calc_atr(candles_h1, 14)
    seuil_atr = 0.00008 if key in ["EURUSD", "GBPUSD"] else (0.4 if key == "XAUUSD" else 4.0)
    tolerance = 0.0004 if key in ["EURUSD", "GBPUSD"] else (1.0 if key == "XAUUSD" else 10.0)

    condition_remplie = False
    conseil = ""
    msg = ""
    sl_price = 0.0

    # Alignement macro H8 + H4 + H2 requis, avec cassure des bornes en H1
    if t8 == "HAUSSIERE" and t4 == "HAUSSIERE" and t2 == "HAUSSIERE" and ch_h1 and cp > ch_h1["upper"][-1]:
        condition_remplie = True
        recent_low = min([c["l"] for c in candles_h1[-5:]])
        sl_price = recent_low - tolerance
        conseil = "ACHAT (Multi-TF H8-H1 Haussier + Breakout H1)"
        msg = f"SIGNAL ACHAT FRACTALE PRO\nAlignement H8+H4+H2 Haussier\nCassure Haussière Canal H1\nEntrée: {cp:.{dec}f}\n🛑 Stop-Loss: {sl_price:.{dec}f}"

    elif t8 == "BAISSIERE" and t4 == "BAISSIERE" and t2 == "BAISSIERE" and ch_h1 and cp < ch_h1["lower"][-1]:
        condition_remplie = True
        recent_high = max([c["h"] for c in candles_h1[-5:]])
        sl_price = recent_high + tolerance
        conseil = "VENTE (Multi-TF H8-H1 Baissier + Breakdown H1)"
        msg = f"SIGNAL VENTE FRACTALE PRO\nAlignement H8+H4+H2 Baissier\nCassure Baissière Canal H1\nEntrée: {cp:.{dec}f}\n🛑 Stop-Loss: {sl_price:.{dec}f}"

    if condition_remplie:
        if atr_h1 < seuil_atr:
            log(f"⏭️ SILENCE {key} - ATR H1 trop faible ({atr_h1:.5f}) [Bruit de marché]")
            return

        full_msg = f"{msg}\n\nATR H1: {atr_h1:.{dec}f}\n{datetime.now(pytz.timezone('Africa/Porto-Novo')).strftime('%H:%M')}H Benin\nFractale Pro Multi-TF"

        log(f"📤 SIGNAL {key} - {conseil}")
        send(info["ntfy"], f"ALERTE {key} - {conseil}", full_msg)

        img_h8 = chart_fractale(candles_h8, cp, info, "Canal Macro H8 (60j)", sl_price)
        if img_h8:
            time.sleep(1)
            send(info["ntfy"], f"{key} H8 - Macro", "Canal H8", img_h8)

        img_h1 = chart_fractale(candles_h1, cp, info, "Canal & Trigger H1", sl_price)
        if img_h1:
            time.sleep(1)
            send(info["ntfy"], f"{key} H1 - Gâchette", "Canal H1 Breakout", img_h1)
    else:
        log(f"⏭️ SILENCE {key} - Pas d'alignement complet H8->H2 ou pas de cassure H1")

if __name__ == "__main__":
    log("🚀 FRACTALE BOT PRO (60J) - Cascades H8 + H4 + H2 + H1 + ATR + SL")
    now = datetime.now(pytz.timezone('Africa/Porto-Novo'))
    h, j = now.hour, now.weekday()

    if j < 5:
        log(f"📊 Analyse Forex Multi-TF {h}H")
        for key in ["XAUUSD", "EURUSD", "GBPUSD"]:
            log(f"→ {key}")
            analyze_fractale(key, PAIRS[key])
    else:
        log(f"💤 Forex fermé week-end")

    log("✅ Terminé")
