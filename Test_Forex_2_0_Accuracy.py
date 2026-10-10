"""
=============================================================
  TEST_FOREX_2_0_ACCURACY.PY
  Directional Accuracy & TP/SL Account Simulation
  for: Hamdi_EURUSD_Forex_2.0.keras

  Model trained in: Education Forex 2.0.py
  Output:  single float  →  predicted % change of Close
           positive  →  price goes UP   (open LONG)
           negative  →  price goes DOWN (open SHORT)

  Trading rules:
    Take Profit :  +0.1% from entry
    Stop Loss   :  -0.1% from entry
    Both LONG and SHORT positions are allowed.

  Test period : last 1 calendar month of available data
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save PNGs, never opens windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model   # type: ignore
from datetime import datetime, timedelta
import pickle

# ── Custom loss used in V5 (must be registered before load_model) ─────────────
def directional_huber_loss(y_true, y_pred, delta=0.15):
    huber   = tf.keras.losses.Huber(delta=delta)
    base    = huber(y_true, y_pred)
    wrong   = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
    penalty = tf.reduce_mean(wrong * tf.abs(y_true - y_pred)) * 0.5
    return base + penalty

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH   = "/Users/ardaunal/Desktop/Hamdi/Forex/Hamdi_DNM_Forex_2.0.keras"
SYMBOL       = "EURUSD=X"
INTERVAL     = "1h"

# Test window: last 1 calendar month (approx 30 days back from today)
TEST_END   = datetime.today().strftime("%Y-%m-%d")
TEST_START = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

# Download an extra leading buffer so we have enough history to fill the 48-bar window
BUFFER_DAYS  = 10          # extra calendar days before TEST_START for context
DATA_START   = (datetime.today() - timedelta(days=30 + 48 + BUFFER_DAYS)).strftime("%Y-%m-%d")

# Must match Education Forex 2.0.py exactly
WINDOW_SIZE  = 48
PREDICT_SIZE = 4
THRESHOLD    = 0.05         # % — same noise filter used during training label creation

# Trading account parameters
TP_PCT = 0.1     # +0.10 % → take profit
SL_PCT = -0.1    # -0.10 % → stop loss

# Macro features — EXACTLY as in Education Forex 2.0.py
MACRO_SYMBOLS = {
    'VIX'        : '^VIX',
    'US10Y'      : '^TNX',
    'DXY'        : 'UUP',
    'XAUUSD'     : 'GC=F',
    'SPX'        : '^GSPC',
    'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F',
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def banner(text):
    line = "=" * 64
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def download_data(start, end):
    """Download EURUSD + macro features, merge & clean (mirrors Education Forex 2.0.py)."""
    banner(f"Downloading  [{start}  →  {end}]")

    data = yf.download(SYMBOL, start=start, end=end,
                       interval=INTERVAL, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError("❌ EURUSD data is empty. Check internet / date range.")

    for name, ticker in MACRO_SYMBOLS.items():
        print(f"  Fetching {name:12s} ({ticker}) ...", end=" ", flush=True)
        temp = yf.download(ticker, start=start, end=end,
                           interval=INTERVAL, auto_adjust=True, progress=False)[['Close']]
        temp.rename(columns={'Close': name}, inplace=True)
        data = data.join(temp, how='outer')
        print("✓")

    # ── Exact same cleaning as training ──────────────────────────────────────
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[data['Close'].notna()]
    data.dropna(inplace=True)

    print(f"\n  ✅ {len(data)} hourly bars  |  columns: {list(data.columns)}")
    return data


def add_technical_indicators(df):
    """
    Compute and append 10+ technical indicators to the dataframe.
    All indicators are based on EURUSD OHLCV columns only.
    NaN rows from indicator warmup are forward-filled / back-filled.
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_l  = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs     = avg_g / (avg_l + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD']        = macd
    df['MACD_signal'] = signal
    df['MACD_hist']   = macd - signal

    # ── ATR(14) ──────────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.ewm(com=13, min_periods=14, adjust=False).mean()

    # ── Bollinger %B (20, 2) ─────────────────────────────────────────────────
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    df['BB_pctB'] = (close - lower) / (upper - lower + 1e-9)

    # ── Trend EMAs ───────────────────────────────────────────────────────────
    df['EMA_9']  = close.ewm(span=9,  adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # Normalised distance from EMA (relative to close price)
    df['dist_EMA9']  = (close - df['EMA_9'])  / (close + 1e-9) * 100
    df['dist_EMA21'] = (close - df['EMA_21']) / (close + 1e-9) * 100
    df['dist_EMA50'] = (close - df['EMA_50']) / (close + 1e-9) * 100

    # ── Volume change % ──────────────────────────────────────────────────────
    # EURUSD volume from Yahoo is often 0; clip to avoid division issues
    vol_safe = vol.replace(0, np.nan)
    df['Vol_change'] = vol_safe.pct_change() * 100
    df['Vol_change'].fillna(0, inplace=True)

    # ── Rolling std of returns (24h) ─────────────────────────────────────────
    returns            = close.pct_change() * 100
    df['Return_std24'] = returns.rolling(24).std()

    # ── Price Rate-of-Change (4 bars) ────────────────────────────────────────
    df['ROC_4'] = close.pct_change(periods=4) * 100

    # Clean NaNs introduced by indicator warmup
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df



def build_windows(data):
    """
    Reproduce the EXACT sliding-window + label logic from Education Forex 2.0.py.
    Label = pct_change of Close after PREDICT_SIZE bars (float, regression).
    Windows where |pct_change| < THRESHOLD are skipped (same as training).
    Returns X, y_pct (ground truth % changes), timestamps, close_prices.
    """
    banner("Building sliding windows  (identical to training logic)")

    # Load the training scaler if available to prevent normalization/distribution shift
    SCALER_PATH = MODEL_PATH.replace(".keras", "_scaler.pkl")
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"  ✅ Loaded training scaler from: {SCALER_PATH}")
        scaled_data = scaler.transform(data)
    except Exception as e:
        print(f"  ⚠️ Warning: Could not load scaler, falling back to local fit_transform. Error: {e}")
        scaler      = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

    close_vals  = data['Close'].values
    timestamps  = data.index

    X, y_list, ts_list, close_list = [], [], [], []

    for i in range(WINDOW_SIZE, len(scaled_data) - PREDICT_SIZE):
        current_close = close_vals[i]
        next_close    = close_vals[i + PREDICT_SIZE]

        if current_close == 0:
            continue

        pct_change = ((next_close - current_close) / current_close) * 100

        # Mirror the noise filter from training
        if abs(pct_change) < THRESHOLD:
            continue

        X.append(scaled_data[i - WINDOW_SIZE : i])
        y_list.append(pct_change)
        ts_list.append(timestamps[i])
        close_list.append(current_close)

    X    = np.array(X)
    y    = np.array(y_list)
    ts   = np.array(ts_list)
    clp  = np.array(close_list)

    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    return X, y, ts, clp


def restrict_to_test_window(X, y, ts, clp, test_start_str):
    """Keep only rows whose timestamp falls inside the 1-month test window."""
    test_start = pd.Timestamp(test_start_str, tz='UTC')

    if ts[0].tzinfo is None:          # make timezone-aware if needed
        ts_aware = pd.DatetimeIndex(ts).tz_localize('UTC')
    else:
        ts_aware = pd.DatetimeIndex(ts)

    mask = ts_aware >= test_start
    return X[mask], y[mask], ts[mask], clp[mask]


def simulate_account(y_true_pct, y_pred_pct, close_prices, timestamps,
                     tp_pct=TP_PCT, sl_pct=SL_PCT):
    """
    For every model signal:
      • LONG  if prediction > 0  (model thinks price will rise)
      • SHORT if prediction < 0  (model thinks price will fall)

    P&L of each trade is clipped to [sl_pct, tp_pct] to replicate
    a fixed TP / SL account.

    actual_move = pct_change of Close over next PREDICT_SIZE bars
    For LONG  : pnl = actual_move  → positive = profit
    For SHORT : pnl = –actual_move → negative actual move = profit

    Both are then clipped to [sl_pct, tp_pct].
    """
    records = []
    for i in range(len(y_pred_pct)):
        pred   = y_pred_pct[i]
        actual = y_true_pct[i]

        if pred > 0:
            direction = "LONG"
            raw_pnl   =  actual          # long profits when price rises
        else:
            direction = "SHORT"
            raw_pnl   = -actual          # short profits when price falls

        # Apply TP / SL cap
        pnl = max(sl_pct, min(tp_pct, raw_pnl))

        # Directional hit: did the sign of the prediction match reality?
        dir_correct = (np.sign(pred) == np.sign(actual))

        records.append({
            "timestamp"    : timestamps[i],
            "close"        : close_prices[i],
            "direction"    : direction,
            "pred_pct"     : pred,
            "actual_pct"   : actual,
            "raw_pnl"      : raw_pnl,
            "clipped_pnl"  : pnl,
            "dir_correct"  : dir_correct,
            "outcome"      : "WIN" if pnl >= tp_pct else ("LOSS" if pnl <= sl_pct else "PARTIAL"),
        })

    return pd.DataFrame(records)


def print_results(df, y_true, y_pred):
    banner("📊 DIRECTIONAL ACCURACY")

    n_total   = len(df)
    n_correct = df['dir_correct'].sum()
    dir_acc   = n_correct / n_total * 100 if n_total > 0 else 0

    n_long    = (df['direction'] == 'LONG').sum()
    n_short   = (df['direction'] == 'SHORT').sum()

    long_acc  = df[df['direction'] == 'LONG' ]['dir_correct'].mean() * 100 if n_long  > 0 else 0
    short_acc = df[df['direction'] == 'SHORT']['dir_correct'].mean() * 100 if n_short > 0 else 0

    print(f"\n  Total signals         : {n_total}")
    print(f"  LONG  signals         : {n_long}  |  Directional acc: {long_acc:.2f}%")
    print(f"  SHORT signals         : {n_short}  |  Directional acc: {short_acc:.2f}%")
    print(f"\n  ★ Overall Directional Accuracy : {dir_acc:.2f}%  ({n_correct}/{n_total})")

    banner(f"💰 ACCOUNT SIMULATION  (TP={TP_PCT:+.1f}%  /  SL={SL_PCT:+.1f}%)")

    wins     = (df['outcome'] == 'WIN').sum()
    losses   = (df['outcome'] == 'LOSS').sum()
    partial  = (df['outcome'] == 'PARTIAL').sum()
    win_rate = wins / n_total * 100 if n_total > 0 else 0
    cum_pnl  = df['clipped_pnl'].sum()
    avg_pnl  = df['clipped_pnl'].mean()

    print(f"\n  TP target       : {TP_PCT:+.1f}%")
    print(f"  SL target       : {SL_PCT:+.1f}%")
    print(f"\n  Total trades    : {n_total}")
    print(f"  Wins  (hit TP)  : {wins}  ({win_rate:.1f}%)")
    print(f"  Losses (hit SL) : {losses}  ({losses/n_total*100 if n_total>0 else 0:.1f}%)")
    print(f"  Partial (no TP/SL reached) : {partial}")
    print(f"\n  Avg P&L per trade    : {avg_pnl:+.4f}%")
    print(f"  Cumulative P&L (sum) : {cum_pnl:+.4f}%")

    # Per-direction breakdown
    for d in ['LONG', 'SHORT']:
        sub  = df[df['direction'] == d]
        if len(sub) == 0:
            continue
        w    = (sub['outcome'] == 'WIN').sum()
        wr   = w / len(sub) * 100
        cpnl = sub['clipped_pnl'].sum()
        print(f"\n  [{d}]  trades={len(sub)}  wins={w}  win_rate={wr:.1f}%  cum_pnl={cpnl:+.4f}%")

    return dir_acc, win_rate, cum_pnl


def plot_results(df, y_true, y_pred, dir_acc, win_rate, cum_pnl):
    banner("📈 Plotting …")

    N   = min(400, len(df))
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0d0d1a")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    title_kw = dict(color="#e0e0ff", fontsize=11, fontweight="bold", pad=10)
    ax_bg    = "#141428"
    grid_clr = "#2a2a4a"

    def style(ax, title):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.spines[:].set_color(grid_clr)
        ax.set_title(title, **title_kw)
        ax.grid(color=grid_clr, linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, facecolor="#1a1a30",
                  edgecolor=grid_clr, labelcolor="#e0e0ff")

    # ── 1. Predicted vs Actual % change ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y_true[:N], color="#00e5ff", linewidth=1.0, label="Actual % change", alpha=0.8)
    ax1.plot(y_pred[:N], color="#ff6b35", linewidth=1.0,
             linestyle="--", label="Predicted % change", alpha=0.8)
    ax1.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
    style(ax1, "Predicted vs Actual % Change of Close  (4-bar horizon)")

    # ── 2. Prediction signs ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    colors_correct = ['#00e5ff' if c else '#f72585' for c in df['dir_correct'][:N]]
    ax2.bar(range(N), df['clipped_pnl'].values[:N],
            color=colors_correct, width=1.0, alpha=0.8)
    ax2.axhline(0, color="#555577", linewidth=0.8, linestyle=":")

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#00e5ff', label='Direction Correct'),
                  Patch(facecolor='#f72585', label='Direction Wrong')]
    ax2.legend(handles=legend_els, fontsize=8,
               facecolor="#1a1a30", edgecolor=grid_clr, labelcolor="#e0e0ff")
    style(ax2, "Trade P&L per Signal  (clipped to TP / SL)  — blue=correct dir, pink=wrong")

    # ── 3. LONG vs SHORT P&L distribution ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    long_pnl  = df[df['direction'] == 'LONG' ]['clipped_pnl']
    short_pnl = df[df['direction'] == 'SHORT']['clipped_pnl']
    ax3.hist(long_pnl,  bins=25, color="#00e5ff", alpha=0.7, label="LONG")
    ax3.hist(short_pnl, bins=25, color="#f72585", alpha=0.7, label="SHORT")
    style(ax3, "P&L Distribution  (LONG vs SHORT)")
    ax3.set_xlabel("P&L %", color="#aaaacc", fontsize=8)

    # ── 4. Cumulative P&L curve ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    cum = df['clipped_pnl'].cumsum().values
    pos = np.where(cum >= 0, cum, 0)
    neg = np.where(cum <  0, cum, 0)
    ax4.fill_between(range(len(cum)), pos, alpha=0.25, color="#00e5ff")
    ax4.fill_between(range(len(cum)), neg, alpha=0.25, color="#f72585")
    ax4.plot(cum, color="#e0e0ff", linewidth=1.2)
    ax4.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
    style(ax4, "Cumulative P&L  (%)")

    # ── 5. Scatter: predicted vs actual ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(y_true, y_pred, alpha=0.15, s=6, color="#7b61ff")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax5.plot([mn, mx], [mn, mx], color="#ff6b35",
             linewidth=1.0, linestyle="--", label="Perfect fit")
    style(ax5, "Scatter: Actual vs Predicted")
    ax5.set_xlabel("Actual %", color="#aaaacc", fontsize=8)
    ax5.set_ylabel("Predicted %", color="#aaaacc", fontsize=8)

    # ── 6. Pie chart: outcomes ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    wins    = (df['outcome'] == 'WIN').sum()
    losses  = (df['outcome'] == 'LOSS').sum()
    partial = (df['outcome'] == 'PARTIAL').sum()
    sizes   = [wins, losses, partial]
    labels  = [f"WIN (TP)\n{wins}", f"LOSS (SL)\n{losses}", f"PARTIAL\n{partial}"]
    colors  = ["#00e5ff", "#f72585", "#ffd166"]
    explode = [0.05, 0.05, 0.02]
    wedges, texts, autotexts = ax6.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=140,
        textprops={'color': '#e0e0ff', 'fontsize': 9}
    )
    for at in autotexts:
        at.set_color('#ffffff')
        at.set_fontweight('bold')
    ax6.set_facecolor(ax_bg)
    ax6.set_title("Trade Outcome Breakdown", **title_kw)

    fig.suptitle(
        f"Hamdi EURUSD Forex 2.0 — 1-Month Accuracy Test\n"
        f"Directional Acc: {dir_acc:.1f}%  |  "
        f"Win Rate (TP/SL): {win_rate:.1f}%  |  "
        f"Cum P&L: {cum_pnl:+.3f}%",
        color="#ffffff", fontsize=13, fontweight="bold", y=0.998
    )

    out_path = "/Users/ardaunal/Desktop/Hamdi/Forex_2_0_Accuracy_Results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✅ Saved: {out_path}")


def print_sample_trades(df, n=30):
    banner(f"🔍 Sample Trades (first {min(n, len(df))})")
    header = f"  {'#':<5} {'Timestamp':<20} {'Close':>10} {'Dir':>6} {'Pred%':>8} {'Actual%':>9} {'P&L%':>7} {'OK?':>5}"
    print(header)
    print("  " + "-" * 80)
    for i, row in df.head(n).iterrows():
        ok = "✓" if row['dir_correct'] else "✗"
        ts = str(row['timestamp'])[:19]
        print(f"  {i:<5} {ts:<20} {row['close']:>10.5f} "
              f"{row['direction']:>6} {row['pred_pct']:>+8.4f} "
              f"{row['actual_pct']:>+9.4f} {row['clipped_pnl']:>+7.4f} {ok:>5}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 64)
    print("  HAMDI EURUSD FOREX 2.0 — DIRECTIONAL ACCURACY & TP/SL TEST")
    print(f"  Model   : Hamdi_EURUSD_Forex_2.0.keras")
    print(f"  Test    : {TEST_START}  →  {TEST_END}  (1 month)")
    print(f"  TP      : {TP_PCT:+.1f}%    SL : {SL_PCT:+.1f}%")
    print("█" * 64)

    # 1. Load model
    banner("Loading Keras model")
    try:
        model = load_model(MODEL_PATH)   # V6+ uses standard Huber
    except Exception:
        model = load_model(MODEL_PATH,
                           custom_objects={'directional_huber_loss': directional_huber_loss})
    model.summary()

    # 2. Download full range (buffer + test window) for context
    data = download_data(DATA_START, TEST_END)

    # Add technical indicators to match the 24-feature model shape
    data = add_technical_indicators(data)
    data.drop(columns=['EMA_9', 'EMA_21', 'EMA_50'], inplace=True, errors='ignore')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    if len(data) < WINDOW_SIZE + PREDICT_SIZE + 10:
        raise ValueError(f"Not enough data: {len(data)} bars. Try a wider date range.")

    # 3. Build all windows
    X_all, y_all, ts_all, clp_all = build_windows(data)

    # 4. Keep only windows that START inside the 1-month test period
    X_test, y_test, ts_test, clp_test = restrict_to_test_window(
        X_all, y_all, ts_all, clp_all, TEST_START
    )

    if len(X_test) == 0:
        raise ValueError(
            "No windows fall inside the test window. "
            f"Try extending the test period (current: {TEST_START} → {TEST_END})."
        )

    print(f"\n  Test-window windows : {len(X_test)}")
    print(f"  Date range          : {ts_test[0]} → {ts_test[-1]}")

    # 5. Run predictions
    banner("Running model predictions")
    y_pred = model.predict(X_test, batch_size=64, verbose=1).flatten()

    print(f"\n  Pred range  : [{y_pred.min():.4f}%,  {y_pred.max():.4f}%]")
    print(f"  Actual range: [{y_test.min():.4f}%,  {y_test.max():.4f}%]")

    # 6. Simulate account
    df = simulate_account(y_test, y_pred, clp_test, ts_test)

    # 7. Print trade samples
    print_sample_trades(df, n=30)

    # 8. Print & capture metrics
    dir_acc, win_rate, cum_pnl = print_results(df, y_test, y_pred)

    # 9. Save trade log
    csv_path = "/Users/ardaunal/Desktop/Hamdi/Forex_2_0_Trade_Log.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Full trade log saved → {csv_path}")

    # 10. Plots
    plot_results(df, y_test, y_pred, dir_acc, win_rate, cum_pnl)

    # 11. Final summary
    banner("✅ TEST COMPLETE")
    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  Model     : Hamdi_EURUSD_Forex_2.0.keras              │")
    print(f"  │  Period    : {TEST_START}  →  {TEST_END}          │")
    print(f"  │  Signals   : {len(df):<6}                                   │")
    print(f"  │  Dir. Acc  : {dir_acc:>6.2f}%                               │")
    print(f"  │  Win Rate  : {win_rate:>6.2f}%  (TP={TP_PCT:+.1f}% / SL={SL_PCT:+.1f}%)        │")
    print(f"  │  Cum P&L   : {cum_pnl:>+7.3f}%                              │")
    print(f"  └────────────────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    main()
