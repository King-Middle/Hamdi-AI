"""
=============================================================
  TEST_FOREX_ADVANCED_COMPARISON.PY
  Compares Original and Advanced Forex strategies month-by-month.
  Checks the impact of the "Look-Ahead Bias" (Future Leak) by
  running both strategies in Biased and Corrected (Real-life) modes.
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Forex.keras"
SYMBOL           = "eurusd=x"
INTERVAL         = "1h"
WINDOW_SIZE      = 48
PREDICT_SIZE     = 4
STARTING_CAPITAL = 1_000.0
RISK_PCT         = 0.02
SPREAD_PIPS      = 0.8  # 0.8 pips spread cost for real-life testing
SPREAD_PRICE     = SPREAD_PIPS * 0.0001

DATA_START = "2025-12-01"
DATA_END   = "2026-06-13"

REPORT_PERIODS = [
    ("Dec 2025", "2025-12-01", "2026-01-01"),
    ("Jan 2026", "2026-01-01", "2026-02-01"),
    ("Feb 2026", "2026-02-01", "2026-03-01"),
    ("Mar 2026", "2026-03-01", "2026-04-01"),
    ("Apr 2026", "2026-04-01", "2026-05-01"),
    ("May 2026", "2026-05-01", "2026-06-01"),
    ("Jun 2026", "2026-06-01", "2026-06-15"),
]

MACRO_SYMBOLS = {
    'VIX'        : '^VIX',   'US10Y': '^TNX',
    'DXY'        : 'UUP',    'XAUUSD': 'GC=F',
    'SPX'        : '^GSPC',  'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F'
}

# ─── STRATEGY DEFINITIONS ──────────────────────────────────────────────────────
STRATEGIES = {
    "Original": {
        "conv_thresh": 1.0,
        "min_move": 0.05,
        "sl_floor": 0.20,
        "tp_mode": "MODEL",
        "fixed_rr": 2.0,
        "holding_bars": 4
    },
    "Advanced": {
        "conv_thresh": 1.0,
        "min_move": 0.15,
        "sl_floor": 0.05,
        "tp_mode": "FIXED",
        "fixed_rr": 1.5,
        "holding_bars": 4
    }
}

# ─── DATA PIPELINE ─────────────────────────────────────────────────────────────

def download_data(start, end):
    data = yf.download(SYMBOL, start=start, end=end, interval=INTERVAL,
                       auto_adjust=True, progress=False)
    for name, ticker in MACRO_SYMBOLS.items():
        temp = yf.download(ticker, start=start, end=end, interval=INTERVAL,
                           auto_adjust=True, progress=False)[['Close']]
        temp.rename(columns={'Close': name}, inplace=True)
        data = data.join(temp, how='outer')
    data.ffill(inplace=True); data.bfill(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[data['Close'].notna()]; data.dropna(inplace=True)
    return data

def build_windows(data):
    scaler = MinMaxScaler()
    sd     = scaler.fit_transform(data)
    cv, hv, lv = data['Close'].values, data['High'].values, data['Low'].values
    X, y = [], []
    for i in range(WINDOW_SIZE, len(sd) - PREDICT_SIZE):
        X.append(sd[i - WINDOW_SIZE : i])
        cc = cv[i]
        y.append([
            ((np.max(hv[i:i+PREDICT_SIZE]) - cc) / cc) * 100,
            ((np.min(lv[i:i+PREDICT_SIZE]) - cc) / cc) * 100
        ])
    s = WINDOW_SIZE
    e = len(sd) - PREDICT_SIZE
    return (np.array(X), np.array(y),
            cv[s:e], hv[s:e], lv[s:e], data.index[s:e],
            hv, lv, cv)

# ─── TRADE ENGINE ──────────────────────────────────────────────────────────────

def resolve_trade(direction, entry, tp_price, sl_price, future_h, future_l, future_c):
    for idx, (bh, bl, bc) in enumerate(zip(future_h, future_l, future_c)):
        bars_elapsed = idx + 1
        if direction == "LONG":
            if bh >= tp_price and bl <= sl_price: return "LOSS", sl_price, bars_elapsed
            if bh >= tp_price:                    return "WIN",  tp_price, bars_elapsed
            if bl <= sl_price:                    return "LOSS", sl_price, bars_elapsed
        else:
            if bl <= tp_price and bh >= sl_price: return "LOSS", sl_price, bars_elapsed
            if bl <= tp_price:                    return "WIN",  tp_price, bars_elapsed
            if bh >= sl_price:                    return "LOSS", sl_price, bars_elapsed
    return "NEUTRAL", future_c[-1], len(future_h)

def pnl_for_trade(outcome, direction, entry, exit_p, tp_p, sl_p, risk, sl_d_price, use_spread):
    if sl_d_price < 1e-10: return 0.0
    
    if outcome == "WIN":
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        raw_pnl = risk * rr
    elif outcome == "LOSS":
        raw_pnl = -risk
    else:
        move = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
        frac = move / sl_d_price
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        frac = max(-1.0, min(rr, frac))
        raw_pnl = risk * frac
        
    if use_spread:
        cost = risk * (SPREAD_PRICE / sl_d_price)
        return raw_pnl - cost
    return raw_pnl

# ─── BACKTEST RUNNER ───────────────────────────────────────────────────────────

def run_backtest(y_pred, closes, all_h, all_lows, all_c, timestamps, params, mode, use_spread, start_idx, end_idx):
    """
    mode: "biased" (evaluates same bar) or "corrected" (evaluates next bar onwards)
    """
    ct = params["conv_thresh"]
    mm = params["min_move"]
    sf = params["sl_floor"]
    tm = params["tp_mode"]
    rr = params["fixed_rr"]
    hb = params["holding_bars"]

    account = STARTING_CAPITAL
    eq      = [account]
    trades  = []

    i = start_idx
    while i < end_idx:
        ph = y_pred[i, 0]
        pl = y_pred[i, 1]

        dom   = max(ph, abs(pl))
        ratio = ph / abs(pl) if abs(pl) > 1e-8 else 99.0

        if dom < mm:
            i += 1; continue

        if ratio >= ct:
            direction = "LONG"
            model_tp  = ph
            model_sl  = pl
        elif (1.0 / ratio) >= ct:
            direction = "SHORT"
            model_tp  = pl
            model_sl  = ph
        else:
            i += 1; continue

        entry = closes[i]

        if direction == "LONG":
            sl_pct   = min(model_sl, -sf)
            tp_pct   = model_tp if tm == "MODEL" else sf * rr
            tp_pct   = max(tp_pct, sf * 0.5)
        else:
            sl_pct   = max(model_sl, sf)
            tp_pct   = model_tp if tm == "MODEL" else -(sf * rr)
            tp_pct   = min(tp_pct, -(sf * 0.5))

        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)

        sl_d_price = abs(sl_price - entry)
        if sl_d_price < 1e-10:
            i += 1; continue

        raw = WINDOW_SIZE + i
        
        # Biased vs. Corrected Future Slice
        if mode == "biased":
            # Includes current bar (look-ahead leak)
            if raw + hb >= len(all_h): break
            fh = all_h [raw : raw + hb]
            fl = all_lows[raw : raw + hb]
            fc = all_c  [raw : raw + hb]
        else:
            # Starts from next bar (real-life)
            if raw + 1 + hb >= len(all_h): break
            fh = all_h [raw + 1 : raw + 1 + hb]
            fl = all_lows[raw + 1 : raw + 1 + hb]
            fc = all_c  [raw + 1 : raw + 1 + hb]

        outcome, exit_p, bars = resolve_trade(direction, entry, tp_price, sl_price, fh, fl, fc)

        risk = account * RISK_PCT
        pnl  = pnl_for_trade(outcome, direction, entry, exit_p, tp_price, sl_price, risk, sl_d_price, use_spread)

        account = max(account + pnl, 0.01)
        eq.append(account)

        trades.append({
            "outcome": outcome,
            "pnl": pnl,
            "balance": account
        })

        i += bars

    df_trades = pd.DataFrame(trades)
    n = len(df_trades)
    if n == 0:
        return account, df_trades, eq, {"n": 0, "wr": 0.0, "pf": 0.0, "ret": 0.0}
        
    nw = (df_trades.outcome == "WIN").sum()
    wr = (nw / n) * 100
    gross_win = df_trades.loc[df_trades.pnl > 0, "pnl"].sum()
    gross_loss = abs(df_trades.loc[df_trades.pnl < 0, "pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    ret = (account - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    metrics = {"n": n, "wr": wr, "pf": pf, "ret": ret}
    return account, df_trades, eq, metrics

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    print("Downloading data...")
    data = download_data(DATA_START, DATA_END)
    (X_full, y_true_full, closes_full, highs_full, lows_full,
     ts_full, all_h, all_l, all_c) = build_windows(data)
    
    print("Running predictions...")
    y_pred_full = model.predict(X_full, batch_size=64, verbose=0)

    # We will test 4 configurations:
    # 1. Original (Biased, No Spread) - This was the benchmark from Test_Forex_Optimizer
    # 2. Original (Corrected, With Spread) - True Original performance
    # 3. Advanced (Biased, With Spread) - What we ran previously
    # 4. Advanced (Corrected, With Spread) - True Advanced performance
    
    modes = [
        ("Original (Biased, No Spread)", "Original", "biased", False),
        ("Original (Corrected, With Spread)", "Original", "corrected", True),
        ("Advanced (Biased, With Spread)", "Advanced", "biased", True),
        ("Advanced (Corrected, With Spread)", "Advanced", "corrected", True),
    ]

    all_results = {}
    
    for run_name, strat_name, mode, use_spread in modes:
        all_results[run_name] = {}
        for p_label, p_start, p_end in REPORT_PERIODS:
            ts_start = pd.Timestamp(p_start).tz_localize("UTC") if ts_full.tz is not None else pd.Timestamp(p_start)
            ts_end   = pd.Timestamp(p_end).tz_localize("UTC")   if ts_full.tz is not None else pd.Timestamp(p_end)
            period_mask = (ts_full >= ts_start) & (ts_full < ts_end)
            
            if period_mask.sum() < 10:
                continue
                
            pm = np.array(period_mask)
            idx_start = int(np.argmax(pm))
            idx_end   = int(len(pm) - np.argmax(pm[::-1]))
            
            final, df, eq, m = run_backtest(
                y_pred_full, closes_full, all_h, all_l, all_c, ts_full,
                STRATEGIES[strat_name], mode, use_spread, idx_start, idx_end
            )
            all_results[run_name][p_label] = (final, m, eq)

    # --- Print Text Summary & Write to File ---
    txt_lines = []
    txt_lines.append("="*85)
    txt_lines.append("           FOREX STRATEGY BIAS ANALYSIS & COMPARISON SUMMARY")
    txt_lines.append("="*85)
    txt_lines.append("\nThis document compares the Original and Advanced strategies to see")
    txt_lines.append("how much of the performance is 'real' vs how much was a result of")
    txt_lines.append("Look-Ahead Bias (evaluating trades on the entry bar instead of the next bar).")
    txt_lines.append("\nSpread Simulated: 0.8 pips (0.008%) per trade where specified.")
    
    for p_label, _, _ in REPORT_PERIODS:
        txt_lines.append("\n" + "-"*85)
        txt_lines.append(f" PERIOD: {p_label}")
        txt_lines.append("-"*85)
        txt_lines.append(f"  {'Strategy Run Profile':<40} | {'Trades':>6} | {'Win%':>6} | {'PF':>5} | {'Return':>9}")
        txt_lines.append(f"  {'-'*40} + {'-'*6} + {'-'*6} + {'-'*5} + {'-'*9}")
        for run_name, _, _, _ in modes:
            if p_label in all_results[run_name]:
                final, m, _ = all_results[run_name][p_label]
                txt_lines.append(f"  {run_name:<40} | {m['n']:>6} | {m['wr']:>5.1f}% | {m['pf']:>5.2f} | {m['ret']:>+8.2f}%")
                
    summary_text = "\n".join(txt_lines)
    print(summary_text)
    
    with open("/Users/ardaunal/Desktop/Hamdi/Strategy_Comparison.txt", "w") as f:
        f.write(summary_text)
    print("\n✅ Saved detailed report to /Users/ardaunal/Desktop/Hamdi/Strategy_Comparison.txt")

    # --- Generate Chart ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 2, figsize=(18, 22), sharex=False)
    fig.patch.set_facecolor('#0d0d1a')
    
    periods = [p[0] for p in REPORT_PERIODS]
    
    for idx, p_label in enumerate(periods):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        ax.set_facecolor('#14142b')
        ax.grid(color='#2a2a4e', linestyle='--', linewidth=0.5)
        
        for run_name, _, _, _ in modes:
            if p_label in all_results[run_name]:
                final, m, eq = all_results[run_name][p_label]
                ax.plot(eq, label=f"{run_name[:20]}... ({m['ret']:+.1f}%)", linewidth=1.5)
        
        ax.axhline(STARTING_CAPITAL, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(f"Period: {p_label}", color='#e0e0ff', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', facecolor='#14142b', edgecolor='#2a2a4e')
        ax.tick_params(colors='#9999cc', labelsize=8)
        
    # Hide the 8th subplot (since we only have 7 periods)
    axes[3, 1].axis('off')
    
    plt.suptitle("Strategy Comparison: Biased vs. Corrected (Real-life) Results", color='white', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("/Users/ardaunal/Desktop/Hamdi/Forex_Strategy_Comparison.png", dpi=150, facecolor=fig.get_facecolor())
    print("✅ Saved comparison chart to /Users/ardaunal/Desktop/Hamdi/Forex_Strategy_Comparison.png")

if __name__ == "__main__":
    main()
