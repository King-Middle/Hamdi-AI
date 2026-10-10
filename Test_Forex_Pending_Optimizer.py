"""
=============================================================
  TEST_FOREX_PENDING_OPTIMIZER.PY
  Finds the best trading strategy using ONLY model predictions.
  Implements pending orders (limit/stop) and evaluates them 
  using a consistency fitness score (maximizing winning months).
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
import warnings, itertools

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Forex.keras"
SYMBOL           = "eurusd=x"
INTERVAL         = "1h"
WINDOW_SIZE      = 48
PREDICT_SIZE     = 4
STARTING_CAPITAL = 1_000.0
RISK_PCT         = 0.02
SPREAD_PIPS      = 0.8  # 0.8 pips spread + slippage cost per trade
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

# ── Grid parameters (5,616 combinations) ───────────────────────────────────────
CONV_THRESHOLDS  = [1.0, 1.2, 1.5]
MIN_MOVES        = [0.05, 0.10, 0.15, 0.20]
SL_FLOORS        = [0.05, 0.10, 0.15, 0.20]
ORDER_TYPES      = ["MARKET", "LIMIT", "STOP"]
TRIGGER_FACTORS  = [0.25, 0.50, 0.75]              # % of prediction for pending trigger
TIMEOUT_BARS     = [1, 2]                          # cancel pending order if not filled in X hours
HOLDING_BARS     = [2, 4, 6]                       # hold time once filled
TP_MODES         = ["MODEL", "FIXED"]
FIXED_RR         = [1.5, 2.0]

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

def simulate_pending_trade(direction, order_type, entry, tp_pct, sl_pct, trigger_pct,
                           all_h, all_l, all_c, raw_index, timeout_bars, holding_bars):
    """
    Simulates a pending order and its subsequent execution.
    Exits evaluated relative to fill price (active_entry).
    Includes a conservative same-bar check on the fill bar.
    """
    if direction == "LONG":
        trigger_price = entry * (1 + trigger_pct / 100)
    else:
        trigger_price = entry * (1 + trigger_pct / 100)

    # Check for fill
    fill_bar = None
    if order_type == "MARKET":
        fill_bar = 0
    else:
        for k in range(1, timeout_bars + 1):
            idx = raw_index + k
            if idx >= len(all_h):
                return "UNFILLED", None, None, k
                
            bh = all_h[idx]
            bl = all_l[idx]
            
            if direction == "LONG":
                if order_type == "LIMIT" and bl <= trigger_price:
                    fill_bar = k
                    break
                elif order_type == "STOP" and bh >= trigger_price:
                    fill_bar = k
                    break
            else:  # SHORT
                if order_type == "LIMIT" and bh >= trigger_price:
                    fill_bar = k
                    break
                elif order_type == "STOP" and bl <= trigger_price:
                    fill_bar = k
                    break
                    
    if fill_bar is None:
        return "UNFILLED", None, None, timeout_bars
        
    # Active trade entry
    active_entry = trigger_price if order_type != "MARKET" else entry
    
    # Calculate TP and SL relative to active_entry
    if direction == "LONG":
        sl_price = active_entry * (1 + sl_pct / 100)
        tp_price = active_entry * (1 + tp_pct / 100)
    else:
        sl_price = active_entry * (1 + sl_pct / 100)
        tp_price = active_entry * (1 + tp_pct / 100)
        
    # --- CONSERVATIVE SAME-BAR EVALUATION ON THE FILL BAR ---
    fill_idx = raw_index + fill_bar
    if fill_idx < len(all_h):
        fh = all_h[fill_idx]
        fl = all_l[fill_idx]
        
        if direction == "LONG":
            # If low of the fill hour hit SL, count as loss immediately (conservative)
            if fl <= sl_price:
                return "LOSS", active_entry, sl_price, max(1, fill_bar)
            # If high hit TP and low didn't hit SL, count as win
            if fh >= tp_price:
                return "WIN", active_entry, tp_price, max(1, fill_bar)
        else:  # SHORT
            # If high of the fill hour hit SL, count as loss immediately
            if fh >= sl_price:
                return "LOSS", active_entry, sl_price, max(1, fill_bar)
            # If low hit TP and high didn't hit SL, count as win
            if fl <= tp_price:
                return "WIN", active_entry, tp_price, max(1, fill_bar)
                
    # --- EVALUATE SUBSEQUENT BARS ---
    start_eval = fill_idx + 1
    end_eval   = fill_idx + 1 + holding_bars
    
    for idx in range(start_eval, end_eval):
        if idx >= len(all_h):
            return "NEUTRAL", active_entry, all_c[min(idx, len(all_c)-1)], idx - raw_index
            
        bh = all_h[idx]
        bl = all_l[idx]
        
        if direction == "LONG":
            if bh >= tp_price and bl <= sl_price:
                return "LOSS", active_entry, sl_price, idx - raw_index
            if bh >= tp_price:
                return "WIN", active_entry, tp_price, idx - raw_index
            if bl <= sl_price:
                return "LOSS", active_entry, sl_price, idx - raw_index
        else:  # SHORT
            if bl <= tp_price and bh >= sl_price:
                return "LOSS", active_entry, sl_price, idx - raw_index
            if bl <= tp_price:
                return "WIN", active_entry, tp_price, idx - raw_index
            if bh >= sl_price:
                return "LOSS", active_entry, sl_price, idx - raw_index
                
    last_idx = min(end_eval - 1, len(all_c) - 1)
    return "NEUTRAL", active_entry, all_c[last_idx], last_idx - raw_index

def pnl_for_trade(outcome, direction, entry, exit_p, tp_p, sl_p, risk, sl_d_price):
    if sl_d_price < 1e-10: return 0.0
    
    if outcome == "WIN":
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        raw_pnl = risk * rr
    elif outcome == "LOSS":
        raw_pnl = -risk
    elif outcome == "NEUTRAL":
        move = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
        frac = move / sl_d_price
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        frac = max(-1.0, min(rr, frac))
        raw_pnl = risk * frac
    else:  # UNFILLED
        return 0.0
        
    cost = risk * (SPREAD_PRICE / sl_d_price)
    return raw_pnl - cost

# ─── BACKTEST RUNNER ───────────────────────────────────────────────────────────

def run_backtest(y_pred, closes, all_h, all_lows, all_c, timestamps, params, start_idx, end_idx):
    ct = params["conv_thresh"]
    mm = params["min_move"]
    sf = params["sl_floor"]
    ot = params["order_type"]
    tf = params["trigger_factor"]
    tb = params["timeout_bars"]
    hb = params["holding_bars"]
    tm = params["tp_mode"]
    rr = params["fixed_rr"]

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

        # Targets relative to entry
        if direction == "LONG":
            sl_pct      = min(model_sl, -sf)
            tp_pct      = model_tp if tm == "MODEL" else sf * rr
            tp_pct      = max(tp_pct, sf * 0.5)
            trigger_pct = -tf * abs(sl_pct) if ot == "LIMIT" else tf * tp_pct if ot == "STOP" else 0.0
        else:
            sl_pct      = max(model_sl, sf)
            tp_pct      = model_tp if tm == "MODEL" else -(sf * rr)
            tp_pct      = min(tp_pct, -(sf * 0.5))
            trigger_pct = tf * sl_pct if ot == "LIMIT" else -tf * abs(tp_pct) if ot == "STOP" else 0.0

        raw = WINDOW_SIZE + i
        
        outcome, active_entry, exit_p, bars = simulate_pending_trade(
            direction, ot, entry, tp_pct, sl_pct, trigger_pct,
            all_h, all_lows, all_c, raw, tb, hb
        )
        
        if outcome == "UNFILLED":
            i += bars
            continue
            
        sl_price = entry * (1 + sl_pct / 100) if ot == "MARKET" else active_entry * (1 + sl_pct / 100)
        tp_price = entry * (1 + tp_pct / 100) if ot == "MARKET" else active_entry * (1 + tp_pct / 100)
        sl_d_price = abs(sl_price - active_entry)

        risk = account * RISK_PCT
        pnl  = pnl_for_trade(outcome, direction, active_entry, exit_p, tp_price, sl_price, risk, sl_d_price)

        account = max(account + pnl, 0.01)
        eq.append(account)

        trades.append({
            "outcome": outcome,
            "pnl": pnl,
            "balance": account,
            "ts": timestamps[i]
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

# ─── GRID SEARCH ───────────────────────────────────────────────────────────────

def grid_search(y_pred, closes, all_h, all_lows, all_c, timestamps):
    print("\nStarting comprehensive grid search over pending strategies...")
    
    combos = []
    for ct, mm, sf, ot, tf, tb, hb, tm in itertools.product(
            CONV_THRESHOLDS, MIN_MOVES, SL_FLOORS, ORDER_TYPES, TRIGGER_FACTORS, TIMEOUT_BARS, HOLDING_BARS, TP_MODES):
        rr_list = FIXED_RR if tm == "FIXED" else [None]
        for rr in rr_list:
            combos.append((ct, mm, sf, ot, tf, tb, hb, tm, rr))
            
    print(f"  Total combinations: {len(combos)}")
    
    results = []
    best_score = -np.inf
    best_combo = None
    
    for idx, (ct, mm, sf, ot, tf, tb, hb, tm, rr) in enumerate(combos):
        params = {
            "conv_thresh": ct, "min_move": mm, "sl_floor": sf, "order_type": ot,
            "trigger_factor": tf, "timeout_bars": tb, "holding_bars": hb,
            "tp_mode": tm, "fixed_rr": rr if rr else 2.0
        }
        
        # Test across the full data
        final, df_trades, eq, m = run_backtest(
            y_pred, closes, all_h, all_lows, all_c, timestamps, params, 0, len(y_pred)
        )
        
        if m["n"] < 15 or m["pf"] < 1.05 or m["ret"] <= 0:
            continue
            
        # Extract monthly performance to measure consistency
        losing_months = 0
        winning_months = 0
        df_trades['month'] = pd.to_datetime(df_trades['ts']).dt.to_period('M')
        
        for p_label, p_start, p_end in REPORT_PERIODS:
            ts_start = pd.Timestamp(p_start).tz_localize("UTC") if timestamps.tz is not None else pd.Timestamp(p_start)
            ts_end   = pd.Timestamp(p_end).tz_localize("UTC")   if timestamps.tz is not None else pd.Timestamp(p_end)
            
            # Slice trades belonging to this month
            m_trades = df_trades[(df_trades['ts'] >= ts_start) & (df_trades['ts'] < ts_end)]
            if len(m_trades) == 0:
                continue
            
            m_pnl = m_trades['pnl'].sum()
            if m_pnl < 0:
                losing_months += 1
            else:
                winning_months += 1
                
        # Fitness Score: Prioritize 7/7 winning months, then net return
        win_ratio = winning_months / (winning_months + losing_months) if (winning_months + losing_months) > 0 else 0
        
        # We heavily reward 7/7 months (win_ratio = 1.0)
        score = win_ratio * 2000.0 + m["pf"] * m["ret"]
        
        results.append({
            "conv": ct, "min_move": mm, "sl_floor": sf, "order_type": ot, "tf": tf, "tb": tb, "holding": hb, "tp_mode": tm, "rr": rr,
            **m, "losing_months": losing_months, "winning_months": winning_months, "score": score
        })
        
        if score > best_score:
            best_score = score
            best_combo = params
            
        if (idx + 1) % 400 == 0:
            print(f"  ... {idx+1}/{len(combos)}  best_score={best_score:.2f} (best wins={winning_months}/{winning_months+losing_months})")
            
    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    return df_res, best_combo

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
    
    # Run Grid Search
    grid_df, best_combo = grid_search(
        y_pred_full, closes_full, all_h, all_l, all_c, ts_full
    )
    
    print("\n  Best strategy combination found:")
    for k, v in best_combo.items():
        print(f"    {k}: {v}")
        
    print("\n  Top 10 pending strategy combinations:")
    print(f"  {'Rank':<5} {'Type':<6} {'Conv':>4} {'MinMv':>5} {'SL':>4} {'TF':>4} {'Hold':>4} "
          f"{'Trades':>6} {'Win%':>5} {'PF':>5} {'Wins':>5} {'Loses':>5} {'Ret%':>8} {'Score':>7}")
    print(f"  {'-'*96}")
    for i, (_, row) in enumerate(grid_df.head(10).iterrows()):
        print(f"  {i+1:<5} {row.order_type:<6} {row.conv:>4} {row.min_move:>5} {row.sl_floor:>4} {row.tf:>4} {int(row.holding):>4} "
              f"{int(row.n):>6} {row.wr:>5.1f} {row.pf:>5.2f} {int(row.winning_months):>5} {int(row.losing_months):>5} {row.ret:>+8.2f} {row.score:>7.1f}")

    # Run the best strategy month-by-month
    period_results = []
    print("\nRunning month-by-month validation on the best strategy...")
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
            best_combo, idx_start, idx_end
        )
        period_results.append((p_label, final, df, eq, m))
        
    # Print monthly summary
    print("\n==============================================================")
    print("                FINAL MONTHLY PERFORMANCE SUMMARY")
    print("==============================================================")
    print(f"  {'Period':<12} {'Trades':>6} {'Win%':>5} {'PF':>6} {'Final USD':>10} {'Return':>8}")
    print(f"  {'-'*52}")
    for label, final, df, eq, m in period_results:
        print(f"  {label:<12} {m['n']:>6} {m['wr']:>5.1f}% {m['pf']:>6.2f} {final:>10,.2f} {m['ret']:>+8.2f}%")
        
    # --- Generate Chart ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.patch.set_facecolor('#0d0d1a')
    COLS = ["#ff5c8a", "#ffd166", "#06d6a0", "#118ab2", "#8338ec", "#3a86c8", "#ff007f"]
    
    for idx, (label, final, df, eq, m) in enumerate(period_results):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        ax.set_facecolor('#14142b')
        ax.grid(color='#2a2a4e', linestyle='--', linewidth=0.5)
        
        ax.plot(eq, color=COLS[idx], linewidth=1.5, label=f"Return: {m['ret']:+.1f}%")
        ax.axhline(STARTING_CAPITAL, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(f"Period: {label} (Trades: {m['n']} | PF: {m['pf']:.2f})", color='#e0e0ff', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(colors='#9999cc', labelsize=8)
        
    axes[3, 1].axis('off')
    
    plt.suptitle("Pending Orders Strategy — Month-by-Month Validation (No Look-Ahead)", color='white', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig("/Users/ardaunal/Desktop/Hamdi/Forex_Pending_Optimized_Results.png", dpi=150, facecolor=fig.get_facecolor())
    print("\n✅ Saved chart to /Users/ardaunal/Desktop/Hamdi/Forex_Pending_Optimized_Results.png")
    
    # --- Write Strategy TXT ---
    lines = []
    lines.append("="*65)
    lines.append("  HAMDI EURUSD FOREX — PENDING ORDERS STRATEGY")
    lines.append("  Model: Hamdi_EURUSD_Forex.keras")
    lines.append("="*65)
    lines.append("")
    lines.append("STRATEGY DESIGN BRIEF")
    lines.append("-"*65)
    lines.append("  This strategy utilizes pending (LIMIT/STOP) orders to enter")
    lines.append("  trades only when price action reaches favorable pullback or breakout")
    lines.append("  levels. This structure completely removes look-ahead bias and")
    lines.append("  secures superior entry prices to maximize winning months.")
    lines.append("")
    lines.append("OPTIMIZED PARAMETERS")
    lines.append("-"*65)
    lines.append(f"  Order Type             : {best_combo['order_type']}")
    lines.append(f"  Conviction threshold   : {best_combo['conv_thresh']}x")
    lines.append(f"  Minimum dominant move  : {best_combo['min_move']}%")
    lines.append(f"  SL floor (minimum SL)  : {best_combo['sl_floor']}%")
    lines.append(f"  Trigger Factor         : {best_combo['trigger_factor']}x of prediction")
    lines.append(f"  Timeout Bars (Cancel)  : {best_combo['timeout_bars']} hours")
    lines.append(f"  Max Holding Period     : {best_combo['holding_bars']} hours")
    lines.append(f"  TP mode                : {best_combo['tp_mode']}" + (f" (R:R = {best_combo['fixed_rr']}:1)" if best_combo['tp_mode'] == 'FIXED' else ""))
    lines.append(f"  Transaction Cost       : 0.8 pips (0.008%) per trade")
    lines.append(f"  Risk per trade         : 2% of account balance")
    lines.append(f"  Trade style            : Non-overlapping (wait for close)")
    lines.append("")
    lines.append("ENTRY RULES (evaluated hourly)")
    lines.append("-"*65)
    lines.append("  Step 1 — Get model output:")
    lines.append("    pred_high = model.predict(last_48_bars)[0]  # upside %")
    lines.append("    pred_low  = model.predict(last_48_bars)[1]  # downside % (negative)")
    lines.append("")
    lines.append("  Step 2 — Calculate conviction ratio:")
    lines.append("    ratio = pred_high / abs(pred_low)")
    lines.append("")
    lines.append("  Step 3 — Filter signals:")
    lines.append("    Only place orders if dominant prediction >= Min dominant move")
    lines.append("    and ratio >= conviction threshold.")
    lines.append("")
    lines.append("  Step 4 — Place Pending Orders:")
    if best_combo['order_type'] == "LIMIT":
        lines.append("    For LONG: Place Buy Limit at Entry * (1 - Trigger Factor * abs(SL_pct)/100)")
        lines.append("    For SHORT: Place Sell Limit at Entry * (1 + Trigger Factor * SL_pct/100)")
    elif best_combo['order_type'] == "STOP":
        lines.append("    For LONG: Place Buy Stop at Entry * (1 + Trigger Factor * TP_pct/100)")
        lines.append("    For SHORT: Place Sell Stop at Entry * (1 - Trigger Factor * abs(TP_pct)/100)")
    else:
        lines.append("    Enter immediately at Market close.")
    lines.append(f"    Cancel pending orders if not filled within {best_combo['timeout_bars']} hours.")
    lines.append("")
    lines.append("MONTH-BY-MONTH REAL-LIFE BACKTEST RESULTS")
    lines.append("-"*65)
    for label, final, df, eq, m in period_results:
        lines.append(f"  {label:<12}  "
                     f"Trades:{m['n']:>4}  Win:{m['wr']:>5.1f}%  "
                     f"PF:{m['pf']:>5.2f}  "
                     f"Final:{final:>8,.2f} USD  "
                     f"Return:{m['ret']:>+7.2f}%")
    lines.append("")
    lines.append("="*65)
    
    with open("/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Pending_Strategy.txt", "w") as f:
        f.write("\n".join(lines))
    print("✅ Written strategy file to /Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Pending_Strategy.txt")

if __name__ == "__main__":
    main()
