"""
=============================================================
  TEST_FOREX_OPTIMIZER.PY
  Grid-search to find the best trading strategy using
  ONLY the Hamdi_EURUSD_Forex.keras model's predictions.

  Model outputs: [pred_high%, pred_low%]
    pred_high = expected max upside over next 4 bars (%)
    pred_low  = expected max downside over next 4 bars (%)

  Strategy Framework (all variants use model predictions):
  ─────────────────────────────────────────────────────────
  DIRECTION  : LONG  if pred_high > abs(pred_low)
               SHORT if abs(pred_low) > pred_high
  
  CONVICTION : ratio = pred_high / abs(pred_low)  (or inverse)
               Only trade when ratio >= CONV_THRESH
  
  MIN MOVE   : Only trade when dominant prediction >= MIN_MOVE %
               (filters out near-zero predictions = model uncertain)
  
  SL         : max(abs(model_downside), SL_FLOOR)
               Never tighter than SL_FLOOR — model often underestimates noise
  
  TP (two modes):
    MODE A "Model TP" : Use model's predicted upside as TP target
    MODE B "Fixed RR" : TP = SL × R_RATIO  (fixed reward:risk)

  Grid tested: 5 conv × 4 min_move × 4 sl_floor × 2 tp_mode × 3 rr
             = up to 480 combinations
  Best selected by: highest Profit Factor + positive return
  ─────────────────────────────────────────────────────────
  Results: best variant printed + chart saved
  If strategy is profitable → strategy written to .txt file
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
import warnings, itertools, textwrap
warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Forex.keras"
SYMBOL           = "eurusd=x"
INTERVAL         = "1h"
WINDOW_SIZE      = 48
PREDICT_SIZE     = 4
STARTING_CAPITAL = 1_000.0
RISK_PCT         = 0.02

# Backtest period (full data for robust optimization)
DATA_START = "2025-12-01"
DATA_END   = "2026-06-13"

# Sub-periods for final reporting
REPORT_PERIODS = [
    ("1 Month",  "2026-05-01", "2026-06-13"),
    ("2 Months", "2026-04-01", "2026-06-13"),
    ("6 Months", "2025-12-01", "2026-06-13"),
]

MACRO_SYMBOLS = {
    'VIX'        : '^VIX',   'US10Y': '^TNX',
    'DXY'        : 'UUP',    'XAUUSD': 'GC=F',
    'SPX'        : '^GSPC',  'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F'
}

# ── Grid parameters ────────────────────────────────────────────────────────────
CONV_THRESHOLDS = [1.0, 1.2, 1.5, 2.0, 3.0]       # conviction ratio
MIN_MOVES       = [0.05, 0.10, 0.15, 0.20]          # minimum dominant move %
SL_FLOORS       = [0.05, 0.10, 0.15, 0.20]          # minimum SL distance %
TP_MODES        = ["MODEL", "FIXED"]                 # TP source
FIXED_RR        = [1.5, 2.0, 3.0]                   # R:R for FIXED mode

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

def resolve_trade(direction, tp_price, sl_price, future_h, future_l, future_c):
    for bh, bl, bc in zip(future_h, future_l, future_c):
        if direction == "LONG":
            if bh >= tp_price and bl <= sl_price: return "LOSS", sl_price, 1
            if bh >= tp_price:                    return "WIN",  tp_price, 1
            if bl <= sl_price:                    return "LOSS", sl_price, 1
        else:
            if bl <= tp_price and bh >= sl_price: return "LOSS", sl_price, 1
            if bl <= tp_price:                    return "WIN",  tp_price, 1
            if bh >= sl_price:                    return "LOSS", sl_price, 1
    return "NEUTRAL", future_c[-1], PREDICT_SIZE

def pnl_for_trade(outcome, direction, entry, exit_p, tp_p, sl_p, risk):
    sl_d = abs(sl_p - entry)
    tp_d = abs(tp_p - entry)
    if sl_d < 1e-10: return 0.0
    rr = tp_d / sl_d
    if outcome == "WIN":     return +risk * rr
    if outcome == "LOSS":    return -risk
    move = (exit_p - entry) if direction == "LONG" else (entry - exit_p)
    frac = max(-1.0, min(rr, move / sl_d))
    return risk * frac

# ─── SINGLE BACKTEST ───────────────────────────────────────────────────────────

def run_backtest(y_pred, closes, all_h, all_lows, all_c, timestamps,
                 conv_thresh, min_move, sl_floor, tp_mode, fixed_rr,
                 start_idx=0, end_idx=None):
    """
    Run one strategy variant on precomputed predictions.
    Returns: final_balance, trades_df, equity_curve
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

        dom   = max(ph, abs(pl))      # dominant prediction
        ratio = ph / abs(pl) if abs(pl) > 1e-8 else 99.0

        if dom < min_move:
            i += 1; continue

        if ratio >= conv_thresh:
            direction = "LONG"
            model_tp  = ph
            model_sl  = pl          # negative
        elif (1.0 / ratio) >= conv_thresh:
            direction = "SHORT"
            model_tp  = pl          # negative (down = profit for short)
            model_sl  = ph          # positive (up = loss for short)
        else:
            i += 1; continue        # no clear conviction

        entry = closes[i]

        # SL price — floor ensures minimum distance
        if direction == "LONG":
            sl_pct   = min(model_sl, -sl_floor)     # at least sl_floor below
            tp_pct   = model_tp if tp_mode == "MODEL" else sl_floor * fixed_rr
            tp_pct   = max(tp_pct, sl_floor * 0.5)  # minimum TP
        else:  # SHORT
            sl_pct   = max(model_sl, sl_floor)      # at least sl_floor above
            tp_pct   = model_tp if tp_mode == "MODEL" else -(sl_floor * fixed_rr)
            tp_pct   = min(tp_pct, -(sl_floor * 0.5))

        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)

        sl_d = abs(sl_price - entry)
        tp_d = abs(tp_price - entry)
        if sl_d < 1e-10 or tp_d < 1e-10:
            i += 1; continue

        # Future bars
        raw = WINDOW_SIZE + i
        if raw + PREDICT_SIZE >= len(all_h): break

        fh = all_h [raw : raw + PREDICT_SIZE]
        fl = all_lows[raw : raw + PREDICT_SIZE]
        fc = all_c  [raw : raw + PREDICT_SIZE]

        outcome, exit_p, bars = resolve_trade(direction, tp_price, sl_price, fh, fl, fc)

        risk = account * RISK_PCT
        pnl  = pnl_for_trade(outcome, direction, entry, exit_p, tp_price, sl_price, risk)

        account = max(account + pnl, 0.01)
        eq.append(account)

        trades.append({
            "i": i, "ts": timestamps[i], "dir": direction,
            "entry": entry, "tp_pct": round(tp_pct, 4), "sl_pct": round(sl_pct, 4),
            "rr": round(tp_d/sl_d, 3), "bars": bars,
            "outcome": outcome, "pnl": round(pnl, 4), "balance": round(account, 4),
            "conv": round(ratio if direction == "LONG" else 1.0/ratio, 3),
        })

        i += bars   # NON-OVERLAPPING

    return account, pd.DataFrame(trades), eq

# ─── METRICS ───────────────────────────────────────────────────────────────────

def calc_metrics(df, final, eq):
    n  = len(df)
    if n == 0: return None
    nw = (df.outcome == "WIN").sum()
    nl = (df.outcome == "LOSS").sum()
    wr = nw / n * 100

    gross_win  = df.loc[df.outcome=="WIN",  "pnl"].sum()
    gross_loss = abs(df.loc[df.outcome=="LOSS", "pnl"].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else 99.0

    eq_arr = np.array(eq)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = ((eq_arr - peak) / peak * 100).min()
    ret    = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {"n": n, "nw": nw, "nl": nl, "wr": wr,
            "pf": pf, "dd": dd, "ret": ret, "final": final}

# ─── GRID SEARCH ───────────────────────────────────────────────────────────────

def grid_search(y_pred, closes, all_h, all_lows, all_c, timestamps):
    banner("Grid Search — testing all strategy variants")

    combos = []
    for ct, mm, sf, tm in itertools.product(
            CONV_THRESHOLDS, MIN_MOVES, SL_FLOORS, TP_MODES):
        rr_list = FIXED_RR if tm == "FIXED" else [None]
        for rr in rr_list:
            combos.append((ct, mm, sf, tm, rr))

    print(f"  Total combinations: {len(combos)}")

    results = []
    best_score = -np.inf
    best_combo = None

    for idx, (ct, mm, sf, tm, rr) in enumerate(combos):
        final, df, eq = run_backtest(
            y_pred, closes, all_h, all_lows, all_c, timestamps,
            conv_thresh=ct, min_move=mm, sl_floor=sf,
            tp_mode=tm, fixed_rr=rr if rr else 2.0
        )
        m = calc_metrics(df, final, eq)
        if m is None or m["n"] < 10: continue

        # Score: profit factor × return sign (must be positive)
        # Also penalise extremely few trades (unreliable)
        score = m["pf"] * np.sign(m["ret"]) * min(1.0, m["n"] / 50)

        results.append({
            "conv": ct, "min_move": mm, "sl_floor": sf,
            "tp_mode": tm, "rr": rr,
            **m, "score": round(score, 4)
        })

        if score > best_score:
            best_score = score
            best_combo = (ct, mm, sf, tm, rr if rr else 2.0)

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx+1}/{len(combos)}  best_score={best_score:.3f}")

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    print(f"\n  Grid complete.  Best score: {best_score:.4f}")
    return df_res, best_combo

# ─── DETAILED ANALYSIS ─────────────────────────────────────────────────────────

def full_analysis(y_pred, closes, all_h, all_lows, all_c, timestamps,
                  best_combo, period_label="Full"):
    ct, mm, sf, tm, rr = best_combo
    final, df, eq = run_backtest(
        y_pred, closes, all_h, all_lows, all_c, timestamps,
        conv_thresh=ct, min_move=mm, sl_floor=sf, tp_mode=tm, fixed_rr=rr
    )
    m = calc_metrics(df, final, eq)
    return final, df, eq, m

def print_detail(m, df, label, combo):
    ct, mm, sf, tm, rr = combo
    banner(f"Best Strategy — {label}")
    print(f"  Parameters:")
    print(f"    Conviction threshold : {ct}x")
    print(f"    Min dominant move    : {mm}%")
    print(f"    SL floor             : {sf}%")
    print(f"    TP mode              : {tm}" + (f"  (R:R {rr}:1)" if tm=="FIXED" else " (model prediction)"))
    print()
    print(f"  Performance:")
    print(f"    Trades               : {m['n']}")
    print(f"    Win rate             : {m['wr']:.1f}%")
    print(f"    Profit factor        : {m['pf']:.3f}")
    print(f"    Max drawdown         : {m['dd']:.1f}%")
    print(f"    Starting capital     : USD 1,000.00")
    print(f"    Final balance        : USD {m['final']:,.2f}")
    print(f"    Return               : {m['ret']:+.2f}%")
    print()
    avg_w = df.loc[df.outcome=="WIN","pnl"].mean()  if m['nw']>0 else 0
    avg_l = df.loc[df.outcome=="LOSS","pnl"].mean() if m['nl']>0 else 0
    avg_rr = df.rr.mean()
    print(f"    Avg R:R              : {avg_rr:.2f}")
    print(f"    Avg win              : USD {avg_w:+.4f}")
    print(f"    Avg loss             : USD {avg_l:+.4f}")

# ─── PLOT ──────────────────────────────────────────────────────────────────────

def plot_results(period_results, grid_df, best_combo):
    banner("Generating charts …")
    BG  = "#0a0a1a"; ABG = "#111128"; GC = "#232348"
    TS  = dict(color="#e0e0ff", fontsize=10, fontweight="bold", pad=8)
    COLS = {"1 Month":"#00e5ff", "2 Months":"#69ff47", "6 Months":"#ffd166"}
    OC   = {"WIN":"#00e5ff", "LOSS":"#f72585", "NEUTRAL":"#888899"}

    def _s(ax, title):
        ax.set_facecolor(ABG); ax.tick_params(colors="#9999cc", labelsize=8)
        ax.spines[:].set_color(GC); ax.set_title(title, **TS)
        ax.grid(color=GC, linewidth=0.5, linestyle="--")

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.38)

    # Row 0: Grid search scores (top 30 combos)
    ax0 = fig.add_subplot(gs[0, :])
    top = grid_df.head(30)
    bar_c = ["#ffd166" if r > 0 else "#f72585" for r in top.ret]
    ax0.bar(range(len(top)), top.score.values, color=bar_c, width=0.8, alpha=0.85)
    ax0.axhline(0, color="#555577", linewidth=1, linestyle="--")
    _s(ax0, "Grid Search — Top 30 Strategy Scores  (gold=profitable, red=losing)")
    ax0.set_xlabel("Strategy rank", color="#9999cc", fontsize=8)
    ax0.set_ylabel("Score (PF x sign(ret))", color="#9999cc", fontsize=8)

    # Rows 1–3: Per-period results
    for row_i, (label, final, df, eq, m) in enumerate(period_results):
        if m is None: continue
        c   = COLS.get(label, "#ffffff")
        row = row_i + 1

        # Per-trade P&L
        ax1 = fig.add_subplot(gs[row, 0])
        bc  = [OC.get(o, "#888") for o in df.outcome]
        ax1.bar(range(len(df)), df.pnl.values, color=bc, width=1.0, alpha=0.85)
        ax1.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
        _s(ax1, f"{label} — Per-Trade P&L (USD)")
        ax1.set_xlabel("Trade #", color="#9999cc", fontsize=7)

        # Outcome pie
        ax2 = fig.add_subplot(gs[row, 1])
        nw = (df.outcome=="WIN").sum(); nl = (df.outcome=="LOSS").sum()
        nn = (df.outcome=="NEUTRAL").sum()
        vs = [v for v in [nw,nl,nn] if v>0]
        ls = [l for l,v in [("WIN",nw),("LOSS",nl),("NEUTRAL",nn)] if v>0]
        cs = [OC[l] for l in ls]
        wg, _, at = ax2.pie(vs, labels=ls, colors=cs, autopct="%1.1f%%",
                            startangle=90,
                            textprops={"color":"#e0e0ff","fontsize":8},
                            wedgeprops={"edgecolor":ABG,"linewidth":1.5})
        for a in at: a.set_color("#0a0a1a")
        ax2.set_facecolor(ABG)
        ax2.set_title(f"{label} Outcomes\nFinal: {final:,.2f} USD ({m['ret']:+.1f}%)", **TS)

        # Equity + drawdown
        ax3 = fig.add_subplot(gs[row, 2])
        ea  = np.array(eq)
        ax3.plot(ea, color=c, linewidth=1.4)
        ax3.axhline(STARTING_CAPITAL, color="#555577", linewidth=0.8, linestyle=":")
        ax3.fill_between(range(len(ea)), ea, STARTING_CAPITAL,
                         where=ea >= STARTING_CAPITAL, alpha=0.15, color=c)
        ax3.fill_between(range(len(ea)), ea, STARTING_CAPITAL,
                         where=ea < STARTING_CAPITAL, alpha=0.20, color="#f72585")
        pk = np.maximum.accumulate(ea); dd = (ea - pk) / pk * 100
        ax4 = ax3.twinx()
        ax4.fill_between(range(len(dd)), dd, 0, alpha=0.18, color="#f72585")
        ax4.set_ylabel("Drawdown %", color="#f72585", fontsize=7)
        ax4.tick_params(colors="#f72585", labelsize=7); ax4.spines[:].set_color(GC)
        _s(ax3, f"{label} Equity  |  MaxDD {m['dd']:.1f}%")
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax3.set_xlabel("Trade #", color="#9999cc", fontsize=7)

    # Row 4: Combined equity
    ax_eq = fig.add_subplot(gs[4, :])
    for label, final, df, eq, m in period_results:
        if m is None: continue
        c   = COLS.get(label, "#ffffff")
        pct = m["ret"]
        ax_eq.plot(eq, color=c, linewidth=1.8,
                   label=f"{label}  ->  {final:,.2f} USD  ({pct:+.1f}%)")
    ax_eq.axhline(STARTING_CAPITAL, color="#555577", linewidth=1, linestyle=":",
                  label=f"Start {STARTING_CAPITAL:,.0f} USD")
    _s(ax_eq, "Combined Equity Curves — All Periods")
    ax_eq.legend(fontsize=8, facecolor="#18183a", edgecolor=GC, labelcolor="#e0e0ff")
    ax_eq.set_xlabel("Trade #", color="#9999cc", fontsize=8)
    ax_eq.set_ylabel("Balance (USD)", color="#9999cc", fontsize=8)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))

    ct, mm, sf, tm, rr = best_combo
    fig.suptitle(
        f"Hamdi EURUSD Forex — Optimized Strategy Backtest\n"
        f"Best params: Conv>{ct}x | MinMove>{mm}% | SL>={sf}% | TP:{tm}"
        + (f" R:R={rr}" if tm=="FIXED" else " (model)"),
        color="#ffffff", fontsize=12, fontweight="bold", y=0.998
    )

    out = "/Users/ardaunal/Desktop/Hamdi/Forex_Optimized_Results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {out}")
    plt.close()

# ─── WRITE STRATEGY TXT ────────────────────────────────────────────────────────

def write_strategy_txt(best_combo, period_results, grid_df):
    ct, mm, sf, tm, rr = best_combo

    # Check if strategy is profitable overall
    profitable_count = sum(1 for _, final, _, _, m in period_results
                           if m and m["ret"] > 0)
    is_profitable    = profitable_count >= 2  # profitable in at least 2 of 3 periods

    if not is_profitable:
        print("\n  ⚠️  Strategy was NOT consistently profitable across periods.")
        print("  Strategy.txt NOT written.")
        return False

    lines = []
    lines.append("="*65)
    lines.append("  HAMDI EURUSD FOREX — OPTIMIZED TRADING STRATEGY")
    lines.append("  Model: Hamdi_EURUSD_Forex.keras")
    lines.append("="*65)
    lines.append("")
    lines.append("MODEL OVERVIEW")
    lines.append("-"*65)
    lines.append("  Architecture : LSTM(128) -> LSTM(64) -> LSTM(32) -> Dense(2)")
    lines.append("  Input        : 48 hourly bars of EURUSD OHLCV +")
    lines.append("                 7 macro indicators (VIX, US10Y, DXY,")
    lines.append("                 XAUUSD, SPX, EuroStoxx50, Brent)")
    lines.append("  Output       : [pred_high%, pred_low%]")
    lines.append("                 pred_high = expected max upside in next 4 hrs (%)")
    lines.append("                 pred_low  = expected max downside in next 4 hrs (%)")
    lines.append("")
    lines.append("OPTIMIZED STRATEGY PARAMETERS")
    lines.append("-"*65)
    lines.append(f"  Conviction threshold   : {ct}x")
    lines.append(f"  Minimum dominant move  : {mm}%")
    lines.append(f"  SL floor (minimum SL)  : {sf}%")
    lines.append(f"  TP mode                : {tm}"
                 + (f"  (R:R = {rr}:1)" if tm=="FIXED" else " (model's predicted level)"))
    lines.append(f"  Risk per trade         : 2% of account balance")
    lines.append(f"  Trade style            : Non-overlapping (wait for close)")
    lines.append(f"  Interval               : 1 hour bars")
    lines.append("")
    lines.append("ENTRY RULES (evaluated every closed hourly bar)")
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
    lines.append("  Step 5 — If no signal: DO NOT trade. Skip to next bar.")
    lines.append("")
    lines.append("POSITION SETUP")
    lines.append("-"*65)
    lines.append("  risk_amount = account_balance * 0.02    (2% of balance)")
    lines.append("")
    if tm == "MODEL":
        lines.append("  For LONG:")
        lines.append(f"    SL distance = max(abs(pred_low), {sf}%)   [floor at {sf}%]")
        lines.append(f"    TP distance = pred_high")
        lines.append(f"    SL price    = entry * (1 - SL_distance/100)")
        lines.append(f"    TP price    = entry * (1 + TP_distance/100)")
        lines.append("")
        lines.append("  For SHORT:")
        lines.append(f"    SL distance = max(pred_high, {sf}%)        [floor at {sf}%]")
        lines.append(f"    TP distance = abs(pred_low)")
        lines.append(f"    TP price    = entry * (1 - TP_distance/100)")
        lines.append(f"    SL price    = entry * (1 + SL_distance/100)")
    else:
        lines.append("  For LONG:")
        lines.append(f"    SL distance = max(abs(pred_low), {sf}%)   [floor at {sf}%]")
        lines.append(f"    TP distance = SL_distance * {rr}            [fixed {rr}:1 R:R]")
        lines.append(f"    SL price    = entry * (1 - SL_distance/100)")
        lines.append(f"    TP price    = entry * (1 + TP_distance/100)")
        lines.append("")
        lines.append("  For SHORT:")
        lines.append(f"    SL distance = max(pred_high, {sf}%)        [floor at {sf}%]")
        lines.append(f"    TP distance = SL_distance * {rr}            [fixed {rr}:1 R:R]")
        lines.append(f"    TP price    = entry * (1 - TP_distance/100)")
        lines.append(f"    SL price    = entry * (1 + SL_distance/100)")
    lines.append("")
    lines.append("  Position size:")
    lines.append("    units = risk_amount / SL_price_distance_in_currency")
    lines.append("    (sized so that if SL is hit, exactly 2% of account is lost)")
    lines.append("")
    lines.append("EXIT RULES")
    lines.append("-"*65)
    lines.append("  Priority 1 — TP hit first   → CLOSE at TP price (WIN)")
    lines.append("  Priority 2 — SL hit first   → CLOSE at SL price (LOSS)")
    lines.append("  Priority 3 — 4 bars elapsed → CLOSE at market close (NEUTRAL)")
    lines.append("  If both TP & SL are in same bar → treat as SL (conservative)")
    lines.append("")
    lines.append("  After closing: wait for next hourly bar before evaluating again.")
    lines.append("  NEVER hold overlapping positions on this model.")
    lines.append("")
    lines.append("BACKTEST RESULTS")
    lines.append("-"*65)
    for label, final, df, eq, m in period_results:
        if m is None: continue
        lines.append(f"  {label:<12}  "
                     f"Trades:{m['n']:>5}  Win:{m['wr']:>5.1f}%  "
                     f"PF:{m['pf']:>6.3f}  "
                     f"Start:1,000 USD  Final:{m['final']:>8,.2f} USD  "
                     f"Return:{m['ret']:>+7.2f}%")
    lines.append("")
    lines.append("RISK WARNINGS")
    lines.append("-"*65)
    lines.append("  * Past performance does not guarantee future results.")
    lines.append("  * This strategy was optimized on historical data.")
    lines.append("  * Always test on a demo account before live trading.")
    lines.append("  * Never risk money you cannot afford to lose.")
    lines.append("  * Forex carries significant risk of loss.")
    lines.append("")
    lines.append("="*65)

    txt = "\n".join(lines)
    path = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Strategy.txt"
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n  ✅ Strategy written → {path}")
    return True

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*62)
    print("  HAMDI EURUSD FOREX — STRATEGY OPTIMIZER")
    print("  Grid-searching for the best model-based strategy")
    print("█"*62)

    # 1. Load model
    banner("Loading model")
    model = load_model(MODEL_PATH)
    model.summary()

    # 2. Download full 6-month dataset once
    data = download_data(DATA_START, DATA_END)
    (X_full, y_true_full, closes_full, highs_full, lows_full,
     ts_full, all_h, all_l, all_c) = build_windows(data)

    # 3. Predict once for all bars
    banner("Computing predictions for all bars")
    y_pred_full = model.predict(X_full, batch_size=64, verbose=1)
    print(f"  Predictions: {y_pred_full.shape}")

    # 4. Grid search on full dataset
    grid_df, best_combo = grid_search(
        y_pred_full, closes_full, all_h, all_l, all_c, ts_full
    )

    ct, mm, sf, tm, rr = best_combo
    print(f"\n  Best combo found:")
    print(f"    Conviction={ct}  MinMove={mm}%  SLfloor={sf}%  TP={tm}  RR={rr}")

    # Print top 10 combos
    print("\n  Top 10 strategy combinations:")
    top10 = grid_df.head(10)
    print(f"  {'Rank':<5} {'Conv':>5} {'MinMv':>6} {'SLflr':>6} {'TP':>6} "
          f"{'RR':>4} {'Trades':>7} {'Win%':>6} {'PF':>6} {'Ret%':>7} {'Score':>7}")
    print(f"  {'-'*72}")
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"  {i+1:<5} {row.conv:>5} {row.min_move:>6} {row.sl_floor:>6} "
              f"{row.tp_mode:>6} {str(row.rr):>4} {int(row.n):>7} "
              f"{row.wr:>6.1f} {row.pf:>6.3f} {row.ret:>+7.2f} {row.score:>7.4f}")

    # 5. Run best combo on each reporting period
    period_results = []
    for p_label, p_start, p_end in REPORT_PERIODS:
        # Build index slice for this sub-period
        ts_start = pd.Timestamp(p_start).tz_localize("UTC") if ts_full.tz is not None else pd.Timestamp(p_start)
        ts_end   = pd.Timestamp(p_end).tz_localize("UTC")   if ts_full.tz is not None else pd.Timestamp(p_end)
        period_mask = (ts_full >= ts_start) & (ts_full < ts_end)
        if period_mask.sum() < 20:
            print(f"\n  Skipping {p_label}: insufficient data")
            period_results.append((p_label, STARTING_CAPITAL, pd.DataFrame(), [STARTING_CAPITAL], None))
            continue

        pm = np.array(period_mask)
        idx_start = int(np.argmax(pm))
        idx_end   = int(len(pm) - np.argmax(pm[::-1]))

        final, df, eq = run_backtest(
            y_pred_full, closes_full, all_h, all_l, all_c, ts_full,
            conv_thresh=ct, min_move=mm, sl_floor=sf, tp_mode=tm, fixed_rr=rr,
            start_idx=int(idx_start), end_idx=int(idx_end)
        )
        m = calc_metrics(df, final, eq)
        period_results.append((p_label, final, df, eq, m))
        print_detail(m, df, p_label, best_combo)

        csv = f"/Users/ardaunal/Desktop/Hamdi/Optimized_Log_{p_label.replace(' ','_')}.csv"
        df.to_csv(csv, index=False)
        print(f"  Log: {csv}")

    # 6. Final summary
    banner("FINAL SUMMARY")
    print(f"\n  {'Period':<12} {'Trades':>7} {'Win%':>6} {'PF':>7} "
          f"{'MaxDD':>7} {'Final USD':>12} {'Return':>8}")
    print(f"  {'-'*65}")
    for label, final, df, eq, m in period_results:
        if m is None:
            print(f"  {label:<12}   —  insufficient data")
            continue
        print(f"  {label:<12} {m['n']:>7} {m['wr']:>6.1f}% {m['pf']:>7.3f} "
              f"{m['dd']:>7.1f}% {final:>12,.2f} {m['ret']:>+8.2f}%")

    # 7. Plot
    plot_results(period_results, grid_df, best_combo)

    # 8. Write strategy txt if profitable
    wrote = write_strategy_txt(best_combo, period_results, grid_df)
    if not wrote:
        print("\n  Model strategy did not achieve consistent profitability.")
        print("  This suggests the model needs retraining or different features.")

    print("\n  Files:")
    print("  • Forex_Optimized_Results.png")
    if wrote:
        print("  • Hamdi_EURUSD_Strategy.txt")
    for label, *_ in REPORT_PERIODS:
        print(f"  • Optimized_Log_{label.replace(' ','_')}.csv")
    print()

if __name__ == "__main__":
    main()
