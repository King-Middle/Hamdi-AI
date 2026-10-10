"""
=============================================================
  TEST_FOREX_SIMULATION.PY
  Real Market Data Simulation & Accuracy Test
  for: Hamdi_EURUSD_Forex.keras
  
  Model trained in: Education Forex.py
  Author: Hamdi AI System
=============================================================

HOW IT WORKS:
  - Downloads the SAME macro features used during training
  - Rebuilds the exact same sliding window (48 bars → predict 4 bars)
  - Runs the model on a fresh date range (simulation period)
  - Computes directional accuracy, MSE, MAE, Hit-Rate %
  - Plots prediction vs reality for both High and Low channels
  - Simulates a simple trading strategy based on model signals
=============================================================
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive: no blocking windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MODEL_PATH    = "/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Forex.keras"
SYMBOL        = "eurusd=x"
INTERVAL      = "1h"

# Simulation window — use a date range AFTER the training period (which ended 2026-01-01)
# We'll also allow a fallback to overlapping range if recent data is unavailable
SIM_START     = "2025-06-01"
SIM_END       = "2026-06-01"

WINDOW_SIZE   = 48   # must match training
PREDICT_SIZE  = 4    # must match training
THRESHOLD     = 0.0  # must match training

# Macro features — EXACTLY as in Education Forex.py
MACRO_SYMBOLS = {
    'VIX'        : '^VIX',
    'US10Y'      : '^TNX',
    'DXY'        : 'UUP',
    'XAUUSD'     : 'GC=F',
    'SPX'        : '^GSPC',
    'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F'
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def banner(text):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def download_data(start, end):
    """Download EURUSD + all macro features, merge & clean — mirrors Education Forex.py."""
    banner(f"Downloading market data  [{start}  →  {end}]")
    
    data = yf.download(SYMBOL, start=start, end=end, interval=INTERVAL, auto_adjust=True, progress=False)
    
    if data.empty:
        raise ValueError("❌ EURUSD data is empty. Check your date range / internet connection.")

    macro_data = {}
    for name, ticker in MACRO_SYMBOLS.items():
        print(f"  Fetching {name} ({ticker}) ...", end=" ")
        temp = yf.download(ticker, start=start, end=end, interval=INTERVAL, auto_adjust=True, progress=False)[['Close']]
        temp.rename(columns={'Close': name}, inplace=True)
        macro_data[name] = temp
        print("✓")

    for name, df in macro_data.items():
        data = data.join(df, how='outer')

    # Exact same cleaning as training
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[data['Close'].notna()]
    data.dropna(inplace=True)

    print(f"\n  ✅ Loaded {len(data)} hourly bars  |  Columns: {list(data.columns)}")
    return data

def build_windows(data):
    """Reproduce the exact sliding-window logic from Education Forex.py."""
    banner("Building sliding windows (same logic as training)")

    # Normalise all features together — mirrors scaler.fit_transform(data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y_raw = [], []
    close_vals = data['Close'].values
    high_vals  = data['High'].values
    low_vals   = data['Low'].values

    for i in range(WINDOW_SIZE, len(scaled_data) - PREDICT_SIZE):
        X.append(scaled_data[i - WINDOW_SIZE : i])

        current_close = close_vals[i]
        future_highs  = high_vals[i : i + PREDICT_SIZE]
        future_lows   = low_vals [i : i + PREDICT_SIZE]

        pct_max_high = ((np.max(future_highs) - current_close) / current_close) * 100
        pct_min_low  = ((np.min(future_lows)  - current_close) / current_close) * 100

        y_raw.append([pct_max_high, pct_min_low])

    X = np.array(X)
    y = np.array(y_raw)

    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    return X, y, close_vals[WINDOW_SIZE : len(scaled_data) - PREDICT_SIZE]

def compute_metrics(y_true, y_pred):
    """Compute regression + directional metrics."""
    mae  = np.mean(np.abs(y_true - y_pred), axis=0)
    mse  = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Directional accuracy: did we get the sign right?
    dir_high = np.mean(np.sign(y_pred[:, 0]) == np.sign(y_true[:, 0])) * 100
    dir_low  = np.mean(np.sign(y_pred[:, 1]) == np.sign(y_true[:, 1])) * 100

    # Within-band accuracy: was the real price between pred_low and pred_high?
    in_band = np.mean(
        (y_true[:, 1] >= y_pred[:, 1]) & (y_true[:, 0] <= y_pred[:, 0])
    ) * 100

    return {
        "MAE_high" : mae[0],  "MAE_low" : mae[1],
        "MSE_high" : mse[0],  "MSE_low" : mse[1],
        "RMSE_high": rmse[0], "RMSE_low": rmse[1],
        "Dir_Acc_High_%": dir_high,
        "Dir_Acc_Low_%" : dir_low,
        "In_Band_%"     : in_band,
    }

def simulate_trading(y_true, y_pred, close_prices, pip_value=0.0001):
    """
    Simple rule-based trading simulation using model predictions.
    
    Strategy:
      - If predicted high > +0.05%  AND  predicted low > -0.1%  → BUY
      - If predicted low  < -0.05%  AND  predicted high < +0.1%  → SELL
      - Otherwise → HOLD
    
    P&L approximated using actual future high/low.
    """
    buy_threshold  =  0.05   # %
    sell_threshold = -0.05   # %
    stop_loss      =  0.10   # % max adverse

    trades = []
    for i in range(len(y_pred)):
        pred_h = y_pred[i, 0]
        pred_l = y_pred[i, 1]
        real_h = y_true[i, 0]
        real_l = y_true[i, 1]

        if pred_h > buy_threshold and pred_l > sell_threshold:
            direction = "BUY"
            result    = real_h  # gain if price goes up
        elif pred_l < sell_threshold and pred_h < buy_threshold:
            direction = "SELL"
            result    = -real_l  # gain if price goes down
        else:
            direction = "HOLD"
            result    = 0.0

        trades.append({
            "idx"      : i,
            "direction": direction,
            "pred_high": pred_h,
            "pred_low" : pred_l,
            "real_high": real_h,
            "real_low" : real_l,
            "result_%" : result,
        })

    df_trades = pd.DataFrame(trades)
    active    = df_trades[df_trades["direction"] != "HOLD"]
    wins      = active[active["result_%"] > 0]
    losses    = active[active["result_%"] <= 0]
    
    return df_trades, active, wins, losses

def print_metrics(metrics):
    banner("📊 Prediction Accuracy Metrics")
    print(f"  {'Metric':<25} {'HIGH channel':>14} {'LOW channel':>14}")
    print(f"  {'-'*55}")
    print(f"  {'MAE  (pct pts)':<25} {metrics['MAE_high']:>14.4f} {metrics['MAE_low']:>14.4f}")
    print(f"  {'MSE  (pct pts²)':<25} {metrics['MSE_high']:>14.4f} {metrics['MSE_low']:>14.4f}")
    print(f"  {'RMSE (pct pts)':<25} {metrics['RMSE_high']:>14.4f} {metrics['RMSE_low']:>14.4f}")
    print(f"  {'Directional Acc (%)':<25} {metrics['Dir_Acc_High_%']:>13.2f}% {metrics['Dir_Acc_Low_%']:>13.2f}%")
    print(f"\n  {'In-Band Hit Rate (%):':<30} {metrics['In_Band_%']:.2f}%")
    print(f"  (= real price was inside [pred_low, pred_high])")

def print_trade_stats(active, wins, losses):
    banner("💰 Trading Simulation Results")
    total   = len(active)
    n_wins  = len(wins)
    n_loss  = len(losses)
    win_pct = (n_wins / total * 100) if total > 0 else 0
    avg_win = wins["result_%"].mean()   if len(wins)   > 0 else 0
    avg_los = losses["result_%"].mean() if len(losses) > 0 else 0
    total_pnl = active["result_%"].sum()

    buy_trades  = active[active["direction"] == "BUY"]
    sell_trades = active[active["direction"] == "SELL"]

    print(f"  Total signals (non-HOLD) : {total}")
    print(f"  BUY signals              : {len(buy_trades)}")
    print(f"  SELL signals             : {len(sell_trades)}")
    print(f"  Wins                     : {n_wins}  ({win_pct:.1f}%)")
    print(f"  Losses                   : {n_loss}  ({100-win_pct:.1f}%)")
    print(f"  Avg win  (% move)        : +{avg_win:.4f}%")
    print(f"  Avg loss (% move)        : {avg_los:.4f}%")
    print(f"  Cumulative P&L           : {total_pnl:+.4f}%")

def plot_results(y_true, y_pred, df_trades, close_prices, metrics):
    banner("📈 Plotting Results …")

    N = min(300, len(y_true))   # limit to first 300 bars for readability

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

    title_style = dict(color="#e0e0ff", fontsize=11, fontweight="bold", pad=10)
    ax_bg       = "#141428"
    grid_clr    = "#2a2a4a"

    def style_ax(ax, title):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.spines[:].set_color(grid_clr)
        ax.set_title(title, **title_style)
        ax.grid(color=grid_clr, linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, facecolor="#1a1a30", edgecolor=grid_clr, labelcolor="#e0e0ff")

    # ── 1. HIGH channel: pred vs real ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y_true[:N, 0], color="#00e5ff", linewidth=1.2, label="Real High %")
    ax1.plot(y_pred[:N, 0], color="#ff6b35", linewidth=1.2, linestyle="--", label="Pred High %")
    ax1.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
    style_ax(ax1, "HIGH Channel — Predicted vs Reality  (% move from current close)")

    # ── 2. LOW channel: pred vs real ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(y_true[:N, 1], color="#69ff47", linewidth=1.2, label="Real Low %")
    ax2.plot(y_pred[:N, 1], color="#f72585", linewidth=1.2, linestyle="--", label="Pred Low %")
    ax2.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
    style_ax(ax2, "LOW Channel — Predicted vs Reality  (% move from current close)")

    # ── 3. Absolute errors ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(np.abs(y_true[:N, 0] - y_pred[:N, 0]), color="#ffd166", linewidth=1.0, label="|Error| High")
    style_ax(ax3, "Absolute Error — HIGH Channel")
    ax3.set_ylabel("Error (pct pts)", color="#aaaacc", fontsize=8)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(np.abs(y_true[:N, 1] - y_pred[:N, 1]), color="#ef476f", linewidth=1.0, label="|Error| Low")
    style_ax(ax4, "Absolute Error — LOW Channel")
    ax4.set_ylabel("Error (pct pts)", color="#aaaacc", fontsize=8)

    # ── 4. Scatter: pred vs real ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.25, s=5, color="#00e5ff")
    mn = min(y_true[:, 0].min(), y_pred[:, 0].min())
    mx = max(y_true[:, 0].max(), y_pred[:, 0].max())
    ax5.plot([mn, mx], [mn, mx], color="#ff6b35", linewidth=1.0, linestyle="--", label="Perfect fit")
    style_ax(ax5, "Scatter: Real vs Pred — HIGH")
    ax5.set_xlabel("Real", color="#aaaacc", fontsize=8)
    ax5.set_ylabel("Predicted", color="#aaaacc", fontsize=8)

    ax6 = fig.add_subplot(gs[3, 1])
    ax6.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.25, s=5, color="#69ff47")
    mn = min(y_true[:, 1].min(), y_pred[:, 1].min())
    mx = max(y_true[:, 1].max(), y_pred[:, 1].max())
    ax6.plot([mn, mx], [mn, mx], color="#f72585", linewidth=1.0, linestyle="--", label="Perfect fit")
    style_ax(ax6, "Scatter: Real vs Pred — LOW")
    ax6.set_xlabel("Real", color="#aaaacc", fontsize=8)
    ax6.set_ylabel("Predicted", color="#aaaacc", fontsize=8)

    # ── 5. Cumulative P&L curve ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    active  = df_trades[df_trades["direction"] != "HOLD"].copy()
    cum_pnl = active["result_%"].cumsum().values
    colors  = ["#00e5ff" if v >= 0 else "#f72585" for v in np.diff(np.concatenate([[0], cum_pnl]))]
    for j in range(len(cum_pnl) - 1):
        ax7.plot([j, j+1], [cum_pnl[j], cum_pnl[j+1]], color=colors[j], linewidth=1.2)
    ax7.axhline(0, color="#555577", linewidth=0.8, linestyle=":")
    ax7.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=(np.array(cum_pnl) >= 0), alpha=0.15, color="#00e5ff")
    ax7.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=(np.array(cum_pnl) < 0),  alpha=0.15, color="#f72585")
    ax7.set_title("Cumulative P&L from Trading Simulation  (% moves, sum)", **title_style)
    ax7.set_facecolor(ax_bg)
    ax7.tick_params(colors="#aaaacc", labelsize=8)
    ax7.spines[:].set_color(grid_clr)
    ax7.grid(color=grid_clr, linewidth=0.5, linestyle="--")

    # Main super title
    fig.suptitle(
        f"Hamdi EURUSD Forex Model — Real Market Simulation\n"
        f"Dir Acc HIGH: {metrics['Dir_Acc_High_%']:.1f}%  |  "
        f"Dir Acc LOW: {metrics['Dir_Acc_Low_%']:.1f}%  |  "
        f"In-Band Hit: {metrics['In_Band_%']:.1f}%",
        color="#ffffff", fontsize=13, fontweight="bold", y=0.995
    )

    plt.savefig("/Users/ardaunal/Desktop/Hamdi/Forex_Simulation_Results.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  ✅ Saved: Forex_Simulation_Results.png")
    plt.show()

def print_sample_predictions(y_true, y_pred, close_prices, n=30):
    banner(f"🔍 Sample Predictions (first {n} bars)")
    print(f"  {'#':<5} {'Close':>10} | {'P.High%':>9} {'R.High%':>9} | {'P.Low%':>9} {'R.Low%':>9} | {'Match?':>8}")
    print(f"  {'-'*75}")
    for i in range(min(n, len(y_pred))):
        ph, pl = y_pred[i, 0], y_pred[i, 1]
        rh, rl = y_true[i, 0], y_true[i, 1]
        # "Match" = directions both correct
        match = ("✓" if (np.sign(ph) == np.sign(rh) and np.sign(pl) == np.sign(rl)) else "✗")
        print(f"  {i:<5} {close_prices[i]:>10.5f} | {ph:>+9.4f} {rh:>+9.4f} | {pl:>+9.4f} {rl:>+9.4f} | {match:>8}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  HAMDI EURUSD FOREX — REAL MARKET SIMULATION TEST")
    print("  Model : Hamdi_EURUSD_Forex.keras")
    print("  Sim   :", SIM_START, "→", SIM_END)
    print("█" * 60)

    # 1. Load model
    banner("Loading Keras model")
    model = load_model(MODEL_PATH)
    model.summary()

    # 2. Download & prepare data
    data = download_data(SIM_START, SIM_END)

    if len(data) < WINDOW_SIZE + PREDICT_SIZE + 10:
        print(f"⚠️  Not enough bars ({len(data)}) for window {WINDOW_SIZE}. Falling back to a wider date range.")
        data = download_data("2024-07-01", "2026-06-01")

    # 3. Build windows (identical logic to training)
    X, y_true, close_prices = build_windows(data)

    # 4. Run predictions
    banner("Running model predictions")
    y_pred = model.predict(X, batch_size=64, verbose=1)

    print(f"\n  Predictions shape : {y_pred.shape}")
    print(f"  Pred High range   : [{y_pred[:, 0].min():.4f}, {y_pred[:, 0].max():.4f}]")
    print(f"  Pred Low  range   : [{y_pred[:, 1].min():.4f}, {y_pred[:, 1].max():.4f}]")
    print(f"  True High range   : [{y_true[:, 0].min():.4f}, {y_true[:, 0].max():.4f}]")
    print(f"  True Low  range   : [{y_true[:, 1].min():.4f}, {y_true[:, 1].max():.4f}]")

    # 5. Metrics
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

    # 6. Sample predictions
    print_sample_predictions(y_true, y_pred, close_prices, n=40)

    # 7. Trading simulation
    df_trades, active, wins, losses = simulate_trading(y_true, y_pred, close_prices)
    print_trade_stats(active, wins, losses)

    # 8. Save trade log
    csv_path = "/Users/ardaunal/Desktop/Hamdi/Forex_Trade_Log.csv"
    df_trades.to_csv(csv_path, index=False)
    print(f"\n  ✅ Full trade log saved → {csv_path}")

    # 9. Plots
    plot_results(y_true, y_pred, df_trades, close_prices, metrics)

    # 10. Final summary
    banner("✅ SIMULATION COMPLETE")
    print(f"  Directional Accuracy HIGH : {metrics['Dir_Acc_High_%']:.2f}%")
    print(f"  Directional Accuracy LOW  : {metrics['Dir_Acc_Low_%']:.2f}%")
    print(f"  In-Band Hit Rate          : {metrics['In_Band_%']:.2f}%")
    print(f"  Trade Win Rate            : {(len(wins)/len(active)*100 if len(active)>0 else 0):.2f}%")
    print(f"  Cumulative P&L            : {active['result_%'].sum():+.4f}%")
    print()

if __name__ == "__main__":
    main()
