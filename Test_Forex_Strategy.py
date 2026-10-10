"""
=============================================================
  TEST_FOREX_STRATEGY.PY  v2
  Strategy Backtest — Hamdi_EURUSD_Forex.keras

  Strategy Rules:
  ─────────────────────────────────────────────────────────
  LONG  when pred_high > abs(pred_low)  (upside > downside)
    TP = entry × (1 + pred_high / 100)
    SL = entry × (1 + pred_low  / 100)   ← negative → below entry

  SHORT when abs(pred_low) > pred_high  (downside > upside)
    TP = entry × (1 + pred_low  / 100)   ← below entry
    SL = entry × (1 + pred_high / 100)   ← above entry

  TP/SL resolution (bar-by-bar, next PREDICT_SIZE=4 candles):
    LONG:  bar.high >= TP_price → WIN first
           bar.low  <= SL_price → LOSS first
           If both in same bar  → LOSS (conservative / worst-case)
    SHORT: bar.low  <= TP_price → WIN first
           bar.high >= SL_price → LOSS first
           If both in same bar  → LOSS
    If 4 bars expire untouched  → NEUTRAL, exit at 4th bar close

  NON-OVERLAPPING trades:
    After opening a trade, the next trade is only considered
    after the current one closes (step forward by resolved bars).

  Position Sizing:
    risk_amount = account × RISK_PCT  (default 2 %)
    Units sized so SL hit = lose exactly risk_amount
    TP hit → win = risk_amount × (TP_dist / SL_dist)

  Starting capital : USD 1,000
  Periods          : 1-month, 2-month, 6-month
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
RISK_PCT         = 0.02          # 2 % of account per trade risk

MACRO_SYMBOLS = {
    'VIX'        : '^VIX',
    'US10Y'      : '^TNX',
    'DXY'        : 'UUP',
    'XAUUSD'     : 'GC=F',
    'SPX'        : '^GSPC',
    'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F'
}

PERIODS = [
    ("1 Month",  "2026-05-01", "2026-06-13"),
    ("2 Months", "2026-04-01", "2026-06-13"),
    ("6 Months", "2025-12-01", "2026-06-13"),
]

# ─── DATA ──────────────────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{'='*62}\n  {text}\n{'='*62}")

def download_data(start, end):
    """Same pipeline as Education Forex.py"""
    data = yf.download(SYMBOL, start=start, end=end, interval=INTERVAL,
                       auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No EURUSD data for {start} -> {end}")

    for name, ticker in MACRO_SYMBOLS.items():
        temp = yf.download(ticker, start=start, end=end, interval=INTERVAL,
                           auto_adjust=True, progress=False)[['Close']]
        temp.rename(columns={'Close': name}, inplace=True)
        data = data.join(temp, how='outer')

    data.ffill(inplace=True)
    data.bfill(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[data['Close'].notna()]
    data.dropna(inplace=True)
    return data

def build_windows(data):
    """Exact Education Forex.py sliding-window logic."""
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y_raw = [], []
    close_v = data['Close'].values
    high_v  = data['High'].values
    low_v   = data['Low'].values

    for i in range(WINDOW_SIZE, len(scaled_data) - PREDICT_SIZE):
        X.append(scaled_data[i - WINDOW_SIZE : i])
        cc = close_v[i]
        fh = high_v[i : i + PREDICT_SIZE]
        fl = low_v [i : i + PREDICT_SIZE]
        pct_h = ((np.max(fh) - cc) / cc) * 100
        pct_l = ((np.min(fl) - cc) / cc) * 100
        y_raw.append([pct_h, pct_l])

    return (np.array(X),
            np.array(y_raw),
            close_v[WINDOW_SIZE : len(scaled_data) - PREDICT_SIZE],
            high_v [WINDOW_SIZE : len(scaled_data) - PREDICT_SIZE],
            low_v  [WINDOW_SIZE : len(scaled_data) - PREDICT_SIZE],
            data.index[WINDOW_SIZE : len(scaled_data) - PREDICT_SIZE],
            data['High'].values,
            data['Low'].values,
            data['Close'].values)

# ─── TRADE RESOLUTION ──────────────────────────────────────────────────────────

def resolve_trade(direction, entry, tp_price, sl_price,
                  future_highs, future_lows, future_closes):
    """
    Walk next PREDICT_SIZE bars one-by-one.
    Returns (outcome, exit_price, bars_held)
      outcome: 'WIN' | 'LOSS' | 'NEUTRAL'
    """
    for bar_idx, (bh, bl, bc) in enumerate(zip(future_highs, future_lows, future_closes)):
        if direction == "LONG":
            tp_hit = bh >= tp_price
            sl_hit = bl <= sl_price
            if tp_hit and sl_hit:   return "LOSS",    sl_price,  bar_idx + 1
            if tp_hit:              return "WIN",     tp_price,  bar_idx + 1
            if sl_hit:              return "LOSS",    sl_price,  bar_idx + 1
        else:  # SHORT
            tp_hit = bl <= tp_price
            sl_hit = bh >= sl_price
            if tp_hit and sl_hit:   return "LOSS",    sl_price,  bar_idx + 1
            if tp_hit:              return "WIN",     tp_price,  bar_idx + 1
            if sl_hit:              return "LOSS",    sl_price,  bar_idx + 1

    # Expired → close at last candle's close price
    return "NEUTRAL", future_closes[-1], PREDICT_SIZE

def calc_pnl(outcome, direction, entry, exit_price,
             tp_price, sl_price, risk_amount):
    """
    Calculate dollar P&L for a resolved trade.

    WIN  → gain = risk_amount × R:R
    LOSS → lose = risk_amount
    NEUTRAL → proportional gain/loss based on actual move vs SL distance
    """
    sl_dist = abs(sl_price - entry)   # dollar distance to SL
    tp_dist = abs(tp_price - entry)   # dollar distance to TP

    if sl_dist < 1e-10:
        return 0.0

    rr = tp_dist / sl_dist

    if outcome == "WIN":
        return +risk_amount * rr

    if outcome == "LOSS":
        return -risk_amount

    # NEUTRAL: actual move expressed relative to SL distance
    # Positive = moved toward TP, negative = moved toward SL
    if direction == "LONG":
        actual_move = exit_price - entry       # + = favourable
    else:
        actual_move = entry - exit_price       # + = favourable (price fell)

    fraction = actual_move / sl_dist           # +1 = moved full SL toward TP
    fraction = max(-1.0, min(rr, fraction))    # cap at ±limits
    return risk_amount * fraction

# ─── STRATEGY RUNNER ───────────────────────────────────────────────────────────

def run_strategy(model, data, period_label):
    banner(f"Running: {period_label}")

    (X, y_true, closes, highs, lows, timestamps,
     all_highs, all_lows, all_closes) = build_windows(data)

    n = len(X)
    y_pred = model.predict(X, batch_size=64, verbose=0)

    print(f"  Bars        : {len(data)}")
    print(f"  Windows     : {n}")

    account      = STARTING_CAPITAL
    equity_curve = [account]
    trades       = []

    i = 0   # NON-OVERLAPPING: advance i by bars_held after each trade
    while i < n:
        pred_h = y_pred[i, 0]
        pred_l = y_pred[i, 1]
        entry  = closes[i]

        # Direction: long if upside > downside, short otherwise
        if pred_h >= abs(pred_l):
            direction = "LONG"
            tp_pct, sl_pct = pred_h, pred_l
        else:
            direction = "SHORT"
            tp_pct, sl_pct = pred_l, pred_h    # tp is negative, sl is positive

        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)

        tp_dist = abs(tp_price - entry)
        sl_dist = abs(sl_price - entry)

        # Skip degenerate predictions (no meaningful move predicted)
        if sl_dist < 1e-8 or tp_dist < 1e-8:
            i += 1
            continue

        # Check we have future bars
        raw_idx = WINDOW_SIZE + i
        if raw_idx + PREDICT_SIZE >= len(all_highs):
            break

        future_highs  = all_highs [raw_idx : raw_idx + PREDICT_SIZE]
        future_lows   = all_lows  [raw_idx : raw_idx + PREDICT_SIZE]
        future_closes = all_closes[raw_idx : raw_idx + PREDICT_SIZE]

        outcome, exit_price, bars_held = resolve_trade(
            direction, entry, tp_price, sl_price,
            future_highs, future_lows, future_closes
        )

        risk_amount = account * RISK_PCT
        pnl = calc_pnl(outcome, direction, entry, exit_price,
                       tp_price, sl_price, risk_amount)

        account += pnl
        account  = max(account, 0.01)
        equity_curve.append(account)

        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        trades.append({
            "trade_n"   : len(trades),
            "bar_i"     : i,
            "timestamp" : timestamps[i],
            "direction" : direction,
            "entry"     : round(entry, 5),
            "tp_price"  : round(tp_price, 5),
            "sl_price"  : round(sl_price, 5),
            "tp_pct"    : round(tp_pct, 4),
            "sl_pct"    : round(sl_pct, 4),
            "rr"        : round(rr, 3),
            "risk_usd"  : round(risk_amount, 4),
            "bars_held" : bars_held,
            "outcome"   : outcome,
            "exit"      : round(exit_price, 5),
            "pnl_usd"   : round(pnl, 4),
            "balance"   : round(account, 4),
        })

        # Move to next non-overlapping window
        i += bars_held

    return equity_curve, pd.DataFrame(trades), account

# ─── STATS & OUTPUT ────────────────────────────────────────────────────────────

def print_stats(df, final, label):
    banner(f"Results: {label}")
    start = STARTING_CAPITAL
    profit = final - start
    ret    = profit / start * 100

    n  = len(df)
    nw = (df.outcome == "WIN").sum()
    nl = (df.outcome == "LOSS").sum()
    nn = (df.outcome == "NEUTRAL").sum()
    wr = nw / n * 100 if n > 0 else 0

    rr_mean  = df.rr.mean()
    avg_w    = df.loc[df.outcome=="WIN",     "pnl_usd"].mean() if nw > 0 else 0
    avg_l    = df.loc[df.outcome=="LOSS",    "pnl_usd"].mean() if nl > 0 else 0
    best     = df.pnl_usd.max()
    worst    = df.pnl_usd.min()
    peak     = np.maximum.accumulate(df.balance.values)
    dd       = ((df.balance.values - peak) / peak * 100).min()

    print(f"  Starting capital  : USD {start:>10,.2f}")
    print(f"  Final balance     : USD {final:>10,.2f}")
    print(f"  Total P&L         : USD {profit:>+10,.2f}  ({ret:+.2f}%)")
    print(f"  Max Drawdown      : {dd:.2f}%")
    print()
    print(f"  Total trades      : {n}")
    print(f"  LONG              : {(df.direction=='LONG').sum()}")
    print(f"  SHORT             : {(df.direction=='SHORT').sum()}")
    print(f"  Wins (TP)         : {nw}  ({wr:.1f}%)")
    print(f"  Losses (SL)       : {nl}  ({nl/n*100:.1f}%)")
    print(f"  Neutral (expired) : {nn}  ({nn/n*100:.1f}%)")
    print()
    print(f"  Avg R:R           : {rr_mean:.2f}")
    print(f"  Avg win           : USD {avg_w:+.4f}")
    print(f"  Avg loss          : USD {avg_l:+.4f}")
    print(f"  Best trade        : USD {best:+.4f}")
    print(f"  Worst trade       : USD {worst:+.4f}")

def print_sample_trades(df, n=20, label=""):
    print(f"\n  Sample Trades — {label}  (first {n})")
    hdr = f"  {'#':<5} {'Dir':<6} {'Entry':>9} {'TP%':>8} {'SL%':>8} {'R:R':>5} {'Bars':>5} {'Out':<8} {'PnL $':>9} {'Balance':>10}"
    print(f"  {'-'*len(hdr.rstrip())}")
    print(hdr)
    print(f"  {'-'*len(hdr.rstrip())}")
    for _, r in df.head(n).iterrows():
        print(f"  {int(r.trade_n):<5} {r.direction:<6} {r.entry:>9.5f} "
              f"{r.tp_pct:>+8.4f} {r.sl_pct:>+8.4f} {r.rr:>5.2f} "
              f"{int(r.bars_held):>5} {r.outcome:<8} "
              f"{r.pnl_usd:>+9.4f} {r.balance:>10.2f}")

# ─── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_all(results):
    banner("Generating charts …")

    COLORS = {
        "1 Month" : "#00e5ff",
        "2 Months": "#69ff47",
        "6 Months": "#ffd166",
    }
    OC = {"WIN": "#00e5ff", "LOSS": "#f72585", "NEUTRAL": "#888899"}

    BG  = "#0a0a1a"
    ABG = "#111128"
    GC  = "#232348"
    TS  = dict(color="#e0e0ff", fontsize=10, fontweight="bold", pad=8)

    def _s(ax, title):
        ax.set_facecolor(ABG)
        ax.tick_params(colors="#9999cc", labelsize=8)
        ax.spines[:].set_color(GC)
        ax.set_title(title, **TS)
        ax.grid(color=GC, linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, facecolor="#18183a", edgecolor=GC, labelcolor="#e0e0ff")

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: Combined equity curves ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    max_len = max(len(eq) for _, eq, _, _ in results)
    for label, equity, df, final in results:
        c   = COLORS[label]
        pct = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        ax0.plot(equity, color=c, linewidth=1.8,
                 label=f"{label}  ->  {final:,.2f} USD  ({pct:+.1f}%)")
    ax0.axhline(STARTING_CAPITAL, color="#555577", linewidth=1.0, linestyle=":",
                label=f"Start {STARTING_CAPITAL:,.0f} USD")
    _s(ax0, f"Equity Curve  |  Start {STARTING_CAPITAL:,.0f} USD  |  Risk {RISK_PCT*100:.0f}% per trade")
    ax0.set_xlabel("Trade number", color="#9999cc", fontsize=8)
    ax0.set_ylabel("Balance (USD)", color="#9999cc", fontsize=8)
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── Rows 1-3: Per-period breakdown ───────────────────────────────────────
    for row, (label, equity, df, final) in enumerate(results):
        c      = COLORS[label]
        r      = row + 1
        profit = final - STARTING_CAPITAL
        pct    = profit / STARTING_CAPITAL * 100

        # Per-trade P&L bars
        ax1 = fig.add_subplot(gs[r, 0])
        bar_c = [OC.get(o, "#888899") for o in df.outcome]
        ax1.bar(range(len(df)), df.pnl_usd.values, color=bar_c, width=1.0, alpha=0.85)
        ax1.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
        _s(ax1, f"{label} — Per-Trade P&L (USD)")
        ax1.set_xlabel("Trade #", color="#9999cc", fontsize=7)
        ax1.set_ylabel("P&L (USD)", color="#9999cc", fontsize=7)

        # Outcome pie
        ax2 = fig.add_subplot(gs[r, 1])
        nw = (df.outcome == "WIN").sum()
        nl = (df.outcome == "LOSS").sum()
        nn = (df.outcome == "NEUTRAL").sum()
        vals  = [v for v in [nw, nl, nn] if v > 0]
        lbls  = [l for l, v in [("WIN",nw),("LOSS",nl),("NEUTRAL",nn)] if v > 0]
        clrs  = [OC[l] for l in lbls]
        wedges, texts, atexts = ax2.pie(
            vals, labels=lbls, colors=clrs, autopct="%1.1f%%", startangle=90,
            textprops={"color": "#e0e0ff", "fontsize": 8},
            wedgeprops={"edgecolor": ABG, "linewidth": 1.5}
        )
        for at in atexts: at.set_color("#0a0a1a")
        ax2.set_facecolor(ABG)
        ax2.set_title(f"{label} Outcome\nFinal: {final:,.2f} USD  ({pct:+.1f}%)", **TS)

        # Equity + drawdown
        ax3 = fig.add_subplot(gs[r, 2])
        eq_arr = np.array(equity)
        ax3.plot(eq_arr, color=c, linewidth=1.4)
        ax3.axhline(STARTING_CAPITAL, color="#555577", linewidth=0.8, linestyle=":")
        ax3.fill_between(range(len(eq_arr)), eq_arr, STARTING_CAPITAL,
                         where=eq_arr >= STARTING_CAPITAL, alpha=0.15, color=c)
        ax3.fill_between(range(len(eq_arr)), eq_arr, STARTING_CAPITAL,
                         where=eq_arr < STARTING_CAPITAL, alpha=0.20, color="#f72585")
        peak = np.maximum.accumulate(eq_arr)
        dd   = (eq_arr - peak) / peak * 100
        ax3_dd = ax3.twinx()
        ax3_dd.fill_between(range(len(dd)), dd, 0, alpha=0.18, color="#f72585")
        ax3_dd.set_ylabel("Drawdown %", color="#f72585", fontsize=7)
        ax3_dd.tick_params(colors="#f72585", labelsize=7)
        ax3_dd.spines[:].set_color(GC)
        max_dd = dd.min()
        _s(ax3, f"{label} Equity  |  Max DD: {max_dd:.1f}%")
        ax3.set_xlabel("Trade #", color="#9999cc", fontsize=7)
        ax3.set_ylabel("Balance (USD)", color="#9999cc", fontsize=7)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Super title — NO $ sign (breaks mathtext)
    summaries = "  |  ".join(
        f"{lbl}: {fin:,.0f} USD ({(fin-STARTING_CAPITAL)/STARTING_CAPITAL*100:+.1f}%)"
        for lbl, _, _, fin in results
    )
    fig.suptitle(
        f"Hamdi EURUSD Forex — TP/SL Strategy Backtest\n"
        f"Start: {STARTING_CAPITAL:,.0f} USD  |  Risk/Trade: {RISK_PCT*100:.0f}%  |  {summaries}",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.995
    )

    out = "/Users/ardaunal/Desktop/Hamdi/Forex_Strategy_Results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {out}")
    plt.close()

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  HAMDI EURUSD FOREX — TP/SL STRATEGY BACKTEST  v2")
    print(f"  Start: {STARTING_CAPITAL:,.0f} USD  |  Risk: {RISK_PCT*100:.0f}% per trade")
    print("  NON-OVERLAPPING trades (wait for close before next entry)")
    print("="*62)

    banner("Loading model")
    model = load_model(MODEL_PATH)
    model.summary()

    results = []

    for label, start, end in PERIODS:
        print(f"\n{'▓'*62}")
        print(f"  Period: {label}  [{start} -> {end}]")
        print(f"{'▓'*62}")

        data = download_data(start, end)
        print(f"  Downloaded {len(data)} hourly bars")

        if len(data) < WINDOW_SIZE + PREDICT_SIZE + 20:
            print(f"  Not enough data, skipping.")
            continue

        equity, df, final = run_strategy(model, data, label)
        print_stats(df, final, label)
        print_sample_trades(df, n=20, label=label)

        csv = f"/Users/ardaunal/Desktop/Hamdi/Strategy_Log_{label.replace(' ','_')}.csv"
        df.to_csv(csv, index=False)
        print(f"\n  Trade log saved: {csv}")

        results.append((label, equity, df, final))

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("FINAL SUMMARY — All Periods")
    print(f"\n  {'Period':<12} {'Trades':>7} {'Win%':>7} {'Start USD':>12} "
          f"{'Final USD':>12} {'Profit USD':>12} {'Return':>8}")
    print(f"  {'-'*72}")
    for label, equity, df, final in results:
        nw  = (df.outcome == "WIN").sum()
        n   = len(df)
        wr  = nw / n * 100 if n > 0 else 0
        pr  = final - STARTING_CAPITAL
        ret = pr / STARTING_CAPITAL * 100
        print(f"  {label:<12} {n:>7} {wr:>6.1f}% "
              f"{STARTING_CAPITAL:>12,.2f} {final:>12,.2f} {pr:>+12,.2f} {ret:>+7.2f}%")

    plot_all(results)
    print("\nAll done. Check Forex_Strategy_Results.png and the CSV logs.")

if __name__ == "__main__":
    main()
