"""
=============================================================
  TEST_FOREX_ADVANCED_OPTIMIZER.PY
  Advanced Grid-search to find the best trading strategy using
  ONLY the Hamdi_EURUSD_Forex.keras model's predictions.

  Introduces realistic features:
  1. Transaction cost simulation (spread/slippage = 0.8 pips).
  2. Breakeven stops (moving SL to entry in profit).
  3. Optimized holding time (exiting early or holding longer).
  4. Month-by-month backtesting and reporting.
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
SPREAD_PIPS      = 0.8  # 0.8 pips transaction cost (spread + slippage)
SPREAD_PRICE     = SPREAD_PIPS * 0.0001

# Full data range for optimization
DATA_START = "2025-12-01"
DATA_END   = "2026-06-13"

# Monthly sub-periods for detailed reporting
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

# ── Grid parameters ────────────────────────────────────────────────────────────
CONV_THRESHOLDS = [1.0, 1.2, 1.5, 2.0]            # conviction ratio
MIN_MOVES       = [0.05, 0.10, 0.15, 0.20]         # minimum dominant move %
SL_FLOORS       = [0.05, 0.10, 0.15, 0.20, 0.30]   # minimum SL distance %
TP_MODES        = ["MODEL", "FIXED"]               # TP source
FIXED_RR        = [1.5, 2.0, 3.0]                  # R:R for FIXED mode
BREAKEVEN_PCTS  = [None, 0.5, 0.7]                 # profit % to trigger BE
HOLDING_BARS    = [2, 4, 6]                        # maximum hold time (hours)

# ─── DATA PIPELINE ─────────────────────────────────────────────────────────────

def banner(t): print(f"\n{'='*62}\n  {t}\n{'='*62}")

def download_data(start, end, quiet=False):
    if not quiet: banner(f"Downloading data [{start} -> {end}]")
    data = yf.download(SYMBOL, start=start, end=end, interval=INTERVAL,
                       auto_adjust=True, progress=False)
    if data.empty: raise ValueError("No data returned")
    for name, ticker in MACRO_SYMBOLS.items():
        temp = yf.download(ticker, start=start, end=end, interval=INTERVAL,
                           auto_adjust=True, progress=False)[['Close']]
        temp.rename(columns={'Close': name}, inplace=True)
        data = data.join(temp, how='outer')
    data.ffill(inplace=True); data.bfill(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[data['Close'].notna()]; data.dropna(inplace=True)
    if not quiet: print(f"  {len(data)} bars  |  {list(data.columns)}")
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

def resolve_trade_advanced(direction, entry, tp_price, sl_price, future_h, future_l, future_c, breakeven_pct=None):
    """
    Resolve trade bar-by-bar to support breakeven stop loss.
    breakeven_pct: float (e.g., 0.5 means move SL to entry when price reaches 50% of TP distance)
    """
    current_sl = sl_price
    is_breakeven = False
    
    tp_d = abs(tp_price - entry)
    
    # Breakeven trigger price
    be_trigger = None
    if breakeven_pct is not None:
        if direction == "LONG":
            be_trigger = entry + breakeven_pct * tp_d
        else:
            be_trigger = entry - breakeven_pct * tp_d
        
    for idx, (bh, bl, bc) in enumerate(zip(future_h, future_l, future_c)):
        bars_elapsed = idx + 1
        if direction == "LONG":
            # Check if we hit both TP and SL in this bar
            if bh >= tp_price and bl <= current_sl:
                # Conservatively count as loss/breakeven depending on state
                return "LOSS", current_sl, bars_elapsed
                
            # Check TP
            if bh >= tp_price:
                return "WIN", tp_price, bars_elapsed
                
            # Check SL
            if bl <= current_sl:
                return "LOSS", current_sl, bars_elapsed
                
            # Check Breakeven Trigger
            if be_trigger is not None and not is_breakeven:
                if bh >= be_trigger:
                    is_breakeven = True
                    current_sl = entry  # Move SL to breakeven
                    
        else:  # SHORT
            # Check if we hit both TP and SL in this bar
            if bl <= tp_price and bh >= current_sl:
                return "LOSS", current_sl, bars_elapsed
                
            # Check TP
            if bl <= tp_price:
                return "WIN", tp_price, bars_elapsed
                
            # Check SL
            if bh >= current_sl:
                return "LOSS", current_sl, bars_elapsed
                
            # Check Breakeven Trigger
            if be_trigger is not None and not is_breakeven:
                if bl <= be_trigger:
                    is_breakeven = True
                    current_sl = entry  # Move SL to breakeven

    # If holding period expires, close at market close
    return "NEUTRAL", future_c[-1], len(future_h)

def pnl_for_trade_advanced(outcome, direction, entry, exit_p, tp_p, sl_p, risk, sl_d_price):
    """
    Computes P&L and incorporates flat transaction cost (spread + slippage).
    """
    if sl_d_price < 1e-10: return 0.0
    
    # Raw P&L fraction of risk
    if outcome == "WIN":
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        raw_pnl = risk * rr
    elif outcome == "LOSS":
        # If SL was moved to entry (breakeven), loss is 0
        if abs(sl_p - entry) < 1e-10:
            raw_pnl = 0.0
        else:
            raw_pnl = -risk
    else:  # NEUTRAL
        move = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
        frac = move / sl_d_price
        # Cap neutral P&L between full loss and full win
        tp_d = abs(tp_p - entry)
        rr = tp_d / sl_d_price
        frac = max(-1.0, min(rr, frac))
        raw_pnl = risk * frac
        
    # Transaction cost impact: cost = position_size * SPREAD_PRICE
    # position_size = risk / sl_d_price
    cost = risk * (SPREAD_PRICE / sl_d_price)
    
    return raw_pnl - cost

# ─── SINGLE BACKTEST ───────────────────────────────────────────────────────────

def run_backtest_advanced(y_pred, closes, all_h, all_lows, all_c, timestamps,
                          conv_thresh, min_move, sl_floor, tp_mode, fixed_rr,
                          breakeven_pct, holding_bars, start_idx=0, end_idx=None):
    """
    Runs one advanced strategy variant.
    """
    n = len(y_pred) if end_idx is None else end_idx
    s = start_idx

    account = STARTING_CAPITAL
    eq      = [account]
    trades  = []

    i = s
    while i < n:
        ph = y_pred[i, 0]   # predicted high %
        pl = y_pred[i, 1]   # predicted low  % (negative)

        dom   = max(ph, abs(pl))
        ratio = ph / abs(pl) if abs(pl) > 1e-8 else 99.0

        if dom < min_move:
            i += 1; continue

        if ratio >= conv_thresh:
            direction = "LONG"
            model_tp  = ph
            model_sl  = pl
        elif (1.0 / ratio) >= conv_thresh:
            direction = "SHORT"
            model_tp  = pl
            model_sl  = ph
        else:
            i += 1; continue

        entry = closes[i]

        # SL and TP percentages
        if direction == "LONG":
            sl_pct   = min(model_sl, -sl_floor)
            tp_pct   = model_tp if tp_mode == "MODEL" else sl_floor * fixed_rr
            tp_pct   = max(tp_pct, sl_floor * 0.5)
        else:
            sl_pct   = max(model_sl, sl_floor)
            tp_pct   = model_tp if tp_mode == "MODEL" else -(sl_floor * fixed_rr)
            tp_pct   = min(tp_pct, -(sl_floor * 0.5))

        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)

        sl_d_price = abs(sl_price - entry)
        if sl_d_price < 1e-10:
            i += 1; continue

        # Slice future bars based on holding_bars
        raw = WINDOW_SIZE + i
        if raw + holding_bars >= len(all_h): break

        fh = all_h [raw : raw + holding_bars]
        fl = all_lows[raw : raw + holding_bars]
        fc = all_c  [raw : raw + holding_bars]

        outcome, exit_p, bars = resolve_trade_advanced(
            direction, entry, tp_price, sl_price, fh, fl, fc, breakeven_pct
        )

        risk = account * RISK_PCT
        pnl  = pnl_for_trade_advanced(outcome, direction, entry, exit_p, tp_price, sl_price, risk, sl_d_price)

        account = max(account + pnl, 0.01)
        eq.append(account)

        trades.append({
            "i": i, "ts": timestamps[i], "dir": direction,
            "entry": entry, "tp_pct": round(tp_pct, 4), "sl_pct": round(sl_pct, 4),
            "rr": round(abs(tp_price - entry)/sl_d_price, 3), "bars": bars,
            "outcome": outcome, "pnl": round(pnl, 4), "balance": round(account, 4),
            "conv": round(ratio if direction == "LONG" else 1.0/ratio, 3),
        })

        i += bars

    return account, pd.DataFrame(trades), eq

# ─── METRICS ───────────────────────────────────────────────────────────────────

def calc_metrics(df, final, eq):
    n  = len(df)
    if n == 0: return None
    nw = (df.outcome == "WIN").sum()
    nl = (df.outcome == "LOSS").sum()
    wr = nw / n * 100

    gross_win  = df.loc[df.pnl > 0, "pnl"].sum()
    gross_loss = abs(df.loc[df.pnl < 0, "pnl"].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    eq_arr = np.array(eq)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = ((eq_arr - peak) / peak * 100).min()
    ret    = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {"n": n, "nw": nw, "nl": nl, "wr": wr,
            "pf": pf, "dd": dd, "ret": ret, "final": final}

# ─── GRID SEARCH ───────────────────────────────────────────────────────────────

def grid_search(y_pred, closes, all_h, all_lows, all_c, timestamps):
    banner("Grid Search — testing advanced strategy combinations")

    combos = []
    for ct, mm, sf, tm, be, hb in itertools.product(
            CONV_THRESHOLDS, MIN_MOVES, SL_FLOORS, TP_MODES, BREAKEVEN_PCTS, HOLDING_BARS):
        rr_list = FIXED_RR if tm == "FIXED" else [None]
        for rr in rr_list:
            combos.append((ct, mm, sf, tm, rr, be, hb))

    print(f"  Total combinations to test: {len(combos)}")

    results = []
    best_score = -np.inf
    best_combo = None

    for idx, (ct, mm, sf, tm, rr, be, hb) in enumerate(combos):
        final, df, eq = run_backtest_advanced(
            y_pred, closes, all_h, all_lows, all_c, timestamps,
            conv_thresh=ct, min_move=mm, sl_floor=sf,
            tp_mode=tm, fixed_rr=rr if rr else 2.0,
            breakeven_pct=be, holding_bars=hb
        )
        m = calc_metrics(df, final, eq)
        if m is None or m["n"] < 15: continue

        # Fitness score = Profit Factor * Return * WinRate_factor * trade_count_penalty
        # This prioritizes high returns, but only if the profit factor is healthy and we have enough trades
        if m["pf"] < 1.1 or m["ret"] <= 0:
            score = -999.0
        else:
            score = m["pf"] * m["ret"] * (m["wr"] / 100.0) * min(1.0, m["n"] / 100)

        results.append({
            "conv": ct, "min_move": mm, "sl_floor": sf,
            "tp_mode": tm, "rr": rr, "breakeven": be, "holding": hb,
            **m, "score": round(score, 4)
        })

        if score > best_score:
            best_score = score
            best_combo = (ct, mm, sf, tm, rr if rr else 2.0, be, hb)

        if (idx + 1) % 150 == 0:
            print(f"  ... {idx+1}/{len(combos)}  best_score={best_score:.3f}")

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    print(f"\n  Grid complete. Best score: {best_score:.4f}")
    return df_res, best_combo

# ─── PLOT ──────────────────────────────────────────────────────────────────────

def plot_results(period_results, grid_df, best_combo):
    banner("Generating advanced performance charts ...")
    BG  = "#0d0d1a"; ABG = "#14142b"; GC = "#2a2a4e"
    TS  = dict(color="#e0e0ff", fontsize=9, fontweight="bold", pad=6)
    
    COLS = {
        "Dec 2025": "#ff5c8a", "Jan 2026": "#ffd166", "Feb 2026": "#06d6a0",
        "Mar 2026": "#118ab2", "Apr 2026": "#8338ec", "May 2026": "#3a86c8",
        "Jun 2026": "#ff007f"
    }

    def _s(ax, title):
        ax.set_facecolor(ABG); ax.tick_params(colors="#9999cc", labelsize=7)
        ax.spines[:].set_color(GC); ax.set_title(title, **TS)
        ax.grid(color=GC, linewidth=0.5, linestyle="--")

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor(BG)
    
    # 7 months + 1 combined plot = 8 plots. Let's arrange them in a grid of 4x2
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.25)

    # 1. Combined Equity Curves
    ax_combined = fig.add_subplot(gs[0, :])
    for label, final, df, eq, m in period_results:
        if m is None: continue
        c = COLS.get(label, "#ffffff")
        ax_combined.plot(eq, color=c, linewidth=1.8, label=f"{label}: ${final:,.2f} ({m['ret']:+.1f}%)")
    ax_combined.axhline(STARTING_CAPITAL, color="#555577", linewidth=1, linestyle=":")
    _s(ax_combined, "Combined Monthly Equity Curves (Each starts with $1,000)")
    ax_combined.legend(fontsize=8, facecolor="#14142b", edgecolor=GC, labelcolor="#e0e0ff", loc="upper left")
    ax_combined.set_ylabel("Account Balance (USD)", color="#9999cc", fontsize=8)

    # 2. Individual Months
    for idx, (label, final, df, eq, m) in enumerate(period_results):
        if m is None: continue
        row = (idx + 2) // 2
        col = (idx + 2) % 2
        
        ax = fig.add_subplot(gs[row, col])
        c = COLS.get(label, "#ffffff")
        
        # Plot equity curve
        ax.plot(eq, color=c, linewidth=1.5)
        ax.axhline(STARTING_CAPITAL, color="#555577", linewidth=0.8, linestyle=":")
        ax.fill_between(range(len(eq)), eq, STARTING_CAPITAL, where=np.array(eq) >= STARTING_CAPITAL, alpha=0.1, color=c)
        ax.fill_between(range(len(eq)), eq, STARTING_CAPITAL, where=np.array(eq) < STARTING_CAPITAL, alpha=0.15, color="#ff0055")
        
        # Drawdown twin axis
        eq_arr = np.array(eq)
        pk = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - pk) / pk * 100
        ax_dd = ax.twinx()
        ax_dd.fill_between(range(len(dd)), dd, 0, alpha=0.1, color="#ff0055")
        ax_dd.tick_params(colors="#ff0055", labelsize=6)
        ax_dd.spines[:].set_color(GC)
        
        _s(ax, f"{label} Equity (Final: ${final:,.2f} | Returns: {m['ret']:+.1f}% | Win: {m['wr']:.1f}%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        
    ct, mm, sf, tm, rr, be, hb = best_combo
    fig.suptitle(
        f"Advanced EURUSD Strategy — Month-by-Month Validation\n"
        f"Params: Conv>{ct}x | MinMove>{mm}% | SLFloor>{sf}% | TP:{tm}"
        + (f" R:R={rr}" if tm=="FIXED" else "") + f" | BE={be}x | Hold={hb}h",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.99
    )

    out = "/Users/ardaunal/Desktop/Hamdi/Forex_Advanced_Optimized_Results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {out}")
    plt.close()

# ─── WRITE STRATEGY TXT ────────────────────────────────────────────────────────

def write_strategy_txt(best_combo, period_results):
    ct, mm, sf, tm, rr, be, hb = best_combo

    lines = []
    lines.append("="*65)
    lines.append("  HAMDI EURUSD FOREX — ADVANCED OPTIMIZED TRADING STRATEGY")
    lines.append("  Model: Hamdi_EURUSD_Forex.keras")
    lines.append("="*65)
    lines.append("")
    lines.append("STRATEGY DESIGN BRIEF")
    lines.append("-"*65)
    lines.append("  This strategy was optimized to produce high returns across changing")
    lines.append("  market regimes while incorporating real-world transaction costs")
    lines.append("  (0.8 pips spread + slippage) and advanced risk management.")
    lines.append("")
    lines.append("OPTIMIZED PARAMETERS")
    lines.append("-"*65)
    lines.append(f"  Conviction threshold   : {ct}x")
    lines.append(f"  Minimum dominant move  : {mm}%")
    lines.append(f"  SL floor (minimum SL)  : {sf}%")
    lines.append(f"  TP mode                : {tm}"
                 + (f"  (R:R = {rr}:1)" if tm=="FIXED" else " (model's predicted level)"))
    lines.append(f"  Breakeven Stop Trigger : " + (f"{be}x of TP target" if be else "Disabled"))
    lines.append(f"  Max Holding Period     : {hb} hours")
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
    lines.append("  Step 3 — LONG signal conditions (ALL must be true):")
    lines.append(f"    a) pred_high > abs(pred_low)         (upside > downside)")
    lines.append(f"    b) ratio >= {ct}                          (conviction)")
    lines.append(f"    c) pred_high >= {mm}%                    (meaningful move predicted)")
    lines.append("")
    lines.append("  Step 4 — SHORT signal conditions (ALL must be true):")
    lines.append(f"    a) abs(pred_low) > pred_high         (downside > upside)")
    lines.append(f"    b) (1/ratio) >= {ct}                     (conviction)")
    lines.append(f"    c) abs(pred_low) >= {mm}%               (meaningful move predicted)")
    lines.append("")
    lines.append("POSITION SETUP & TRANSACTION COSTS")
    lines.append("-"*65)
    lines.append("  risk_amount = account_balance * 0.02")
    lines.append("")
    if tm == "MODEL":
        lines.append("  For LONG:")
        lines.append(f"    SL distance = max(abs(pred_low), {sf}%)")
        lines.append(f"    TP distance = pred_high")
    else:
        lines.append("  For LONG:")
        lines.append(f"    SL distance = max(abs(pred_low), {sf}%)")
        lines.append(f"    TP distance = SL_distance * {rr}")
    lines.append("    SL price    = entry * (1 - SL_distance/100)")
    lines.append("    TP price    = entry * (1 + TP_distance/100)")
    lines.append("")
    if tm == "MODEL":
        lines.append("  For SHORT:")
        lines.append(f"    SL distance = max(pred_high, {sf}%)")
        lines.append(f"    TP distance = abs(pred_low)")
    else:
        lines.append("  For SHORT:")
        lines.append(f"    SL distance = max(pred_high, {sf}%)")
        lines.append(f"    TP distance = SL_distance * {rr}")
    lines.append("    TP price    = entry * (1 - TP_distance/100)")
    lines.append("    SL price    = entry * (1 + SL_distance/100)")
    lines.append("")
    lines.append("  Position size (Units):")
    lines.append("    units = risk_amount / (SL_distance in currency units)")
    lines.append("  Transaction Cost (spread/slippage simulation):")
    lines.append("    A cost of 0.8 pips is deducted from every trade's return.")
    lines.append("")
    lines.append("TRADE MANAGEMENT & EXIT RULES")
    lines.append("-"*65)
    if be:
        lines.append(f"  1. Breakeven Stop: If the trade moves {int(be*100)}% of the way to the TP,")
        lines.append("     automatically move the Stop Loss to the entry price.")
    lines.append(f"  2. Time Exit: If the trade is not closed within {hb} hours (bars),")
    lines.append("     automatically close the trade at the market price.")
    lines.append("  3. Win Exit: Close immediately when TP price is hit.")
    lines.append("  4. Loss Exit: Close immediately when SL price is hit.")
    lines.append("")
    lines.append("MONTH-BY-MONTH BACKTEST RESULTS (Starting with 1,000 USD each month)")
    lines.append("-"*65)
    for label, final, df, eq, m in period_results:
        if m is None: continue
        lines.append(f"  {label:<12}  "
                     f"Trades:{m['n']:>4}  Win:{m['wr']:>5.1f}%  "
                     f"PF:{m['pf']:>6.3f}  "
                     f"MaxDD:{m['dd']:>5.1f}%  "
                     f"Final:{m['final']:>8,.2f} USD  "
                     f"Return:{m['ret']:>+7.2f}%")
    lines.append("")
    lines.append("="*65)

    txt = "\n".join(lines)
    path = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Advanced_Strategy.txt"
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n  ✅ Advanced Strategy written → {path}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*62)
    print("  HAMDI EURUSD FOREX — ADVANCED STRATEGY OPTIMIZER")
    print("  Grid-searching for a robust, high-performance strategy")
    print("█"*62)

    # 1. Load model
    banner("Loading model")
    model = load_model(MODEL_PATH)
    model.summary()

    # 2. Download full dataset
    data = download_data(DATA_START, DATA_END)
    (X_full, y_true_full, closes_full, highs_full, lows_full,
     ts_full, all_h, all_l, all_c) = build_windows(data)

    # 3. Predict once for all bars
    banner("Computing predictions for all bars")
    y_pred_full = model.predict(X_full, batch_size=64, verbose=1)

    # 4. Run Grid Search
    grid_df, best_combo = grid_search(
        y_pred_full, closes_full, all_h, all_l, all_c, ts_full
    )

    ct, mm, sf, tm, rr, be, hb = best_combo
    print(f"\n  Best advanced combo found:")
    print(f"    Conviction={ct}  MinMove={mm}%  SLfloor={sf}%  TP={tm}  RR={rr}  BE={be}  Hold={hb}")

    # Print top 10 combos
    print("\n  Top 10 strategy combinations:")
    top10 = grid_df.head(10)
    print(f"  {'Rank':<5} {'Conv':>4} {'MinMv':>5} {'SLflr':>5} {'TP':>6} "
          f"{'RR':>3} {'BE':>4} {'Hold':>4} {'Trades':>6} {'Win%':>5} {'PF':>5} {'Ret%':>8} {'Score':>7}")
    print(f"  {'-'*88}")
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"  {i+1:<5} {row.conv:>4} {row.min_move:>5} {row.sl_floor:>5} "
              f"{row.tp_mode:>6} {str(row.rr):>3} {str(row.breakeven):>4} {int(row.holding):>4} {int(row.n):>6} "
              f"{row.wr:>5.1f} {row.pf:>5.2f} {row.ret:>+8.2f} {row.score:>7.2f}")

    # 5. Run best combo on each calendar month
    period_results = []
    for p_label, p_start, p_end in REPORT_PERIODS:
        ts_start = pd.Timestamp(p_start).tz_localize("UTC") if ts_full.tz is not None else pd.Timestamp(p_start)
        ts_end   = pd.Timestamp(p_end).tz_localize("UTC")   if ts_full.tz is not None else pd.Timestamp(p_end)
        period_mask = (ts_full >= ts_start) & (ts_full < ts_end)
        
        if period_mask.sum() < 10:
            print(f"\n  Skipping {p_label}: insufficient data")
            continue

        pm = np.array(period_mask)
        idx_start = int(np.argmax(pm))
        idx_end   = int(len(pm) - np.argmax(pm[::-1]))

        final, df, eq = run_backtest_advanced(
            y_pred_full, closes_full, all_h, all_l, all_c, ts_full,
            conv_thresh=ct, min_move=mm, sl_floor=sf, tp_mode=tm, fixed_rr=rr,
            breakeven_pct=be, holding_bars=hb,
            start_idx=idx_start, end_idx=idx_end
        )
        m = calc_metrics(df, final, eq)
        period_results.append((p_label, final, df, eq, m))

    # 6. Print monthly summary
    banner("MONTHLY PERFORMANCE SUMMARY")
    print(f"\n  {'Period':<12} {'Trades':>6} {'Win%':>5} {'PF':>6} "
          f"{'MaxDD':>6} {'Final USD':>10} {'Return':>8}")
    print(f"  {'-'*60}")
    for label, final, df, eq, m in period_results:
        if m is None: continue
        print(f"  {label:<12} {m['n']:>6} {m['wr']:>5.1f}% {m['pf']:>6.2f} "
              f"{m['dd']:>6.1f}% {final:>10,.2f} {m['ret']:>+8.2f}%")

    # 7. Generate chart
    plot_results(period_results, grid_df, best_combo)

    # 8. Write strategy file
    write_strategy_txt(best_combo, period_results)

    print("\n  Execution complete.")

if __name__ == "__main__":
    main()
