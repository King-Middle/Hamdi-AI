"""
=============================================================
  TEST FOREX 2.0 SIMULATION
  Directional Accuracy & Sequential Account Simulation
  for models trained with Education Forex 2.0.py
=============================================================
"""

import os
import glob
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive background plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = "/Users/ardaunal/Desktop/Hamdi"
SYMBOL = "EURUSD=X"
INTERVAL = "1h"

# 3 months simulation test period
TEST_END = datetime.today()
TEST_START = TEST_END - timedelta(days=92)  # ~3 months

# Extra buffer days to allow for MinMaxScaler fit and window warmup
BUFFER_DAYS = 20
DATA_START = TEST_START - timedelta(days=BUFFER_DAYS)

TEST_END_STR = TEST_END.strftime("%Y-%m-%d")
TEST_START_STR = TEST_START.strftime("%Y-%m-%d")
DATA_START_STR = DATA_START.strftime("%Y-%m-%d")

# 23 Macro Tickers exactly matching Education Forex 2.0.py
MACRO_SYMBOLS = {
    'VIX': '^VIX',
    'US10Y': '^TNX',
    'US3M': '^IRX',
    'US30Y': '^TYX',
    'XAUUSD': 'GC=F',
    'XAGUSD': 'SI=F',
    'WTI': 'USO',
    'Brent': 'BZ=F',
    'NaturalGas': 'NG=F',
    'Copper': 'HG=F',
    'Platinum': 'PL=F',
    'Palladium': 'PA=F',
    'SPX': '^GSPC',
    'DJI': '^DJI',
    'IXIC': '^IXIC',
    'RUT': '^RUT',
    'FTSE': '^FTSE',
    'Nikkei': '^N225',
    'HangSeng': '^HSI',
    'EuroStoxx50': 'FEZ',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
}

def banner(text, char="=", width=64):
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def pick_model():
    """Scan BASE_DIR and subdirectories for Keras models and let user choose one."""
    patterns = [
        os.path.join(BASE_DIR, "Hamdi_*_Forex_2.0.keras"),
        os.path.join(BASE_DIR, "Forex", "Hamdi_*_Forex_2.0.keras"),
        os.path.join(BASE_DIR, "*.keras"),
        os.path.join(BASE_DIR, "**", "*.keras")
    ]
    
    found = []
    for p in patterns:
        found.extend(glob.glob(p, recursive=True))
    
    # Keep unique files
    found = sorted(list(set(found)))
    
    if not found:
        banner("❌ No Keras models found in workspace")
        print(f"Expected folder: {BASE_DIR}")
        raise FileNotFoundError("No models found.")
        
    banner("Available Keras Models for Simulation")
    for idx, path in enumerate(found):
        # Display relative path from BASE_DIR for readability
        rel_path = os.path.relpath(path, BASE_DIR)
        print(f"  [{idx + 1}]  {rel_path}")
        
    while True:
        raw = input(f"\n  Select model (1 to {len(found)}): ").strip()
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(found):
                return found[n - 1]
        print(f"  ⚠️ Invalid choice, please enter a number between 1 and {len(found)}.")

def download_data():
    """Download EURUSD + macro features and align them exactly."""
    banner(f"Downloading market data [{DATA_START_STR} → {TEST_END_STR}]")
    
    data = yf.download(SYMBOL, start=DATA_START_STR, end=TEST_END_STR,
                       interval=INTERVAL, auto_adjust=True, progress=False)
    
    if data.empty:
        raise ValueError("❌ EURUSD data download failed. Check internet connection.")
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Only keep core columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    for name, ticker in MACRO_SYMBOLS.items():
        print(f"  Fetching {name:12s} ({ticker}) ...", end=" ", flush=True)
        try:
            temp = yf.download(ticker, start=DATA_START_STR, end=TEST_END_STR,
                               interval=INTERVAL, auto_adjust=True, progress=False)
            if temp.empty:
                print("⚠️ Empty! Filling with 0.", end=" ")
                data[name] = 0.0
                continue
                
            if isinstance(temp.columns, pd.MultiIndex):
                temp.columns = temp.columns.get_level_values(0)
                
            temp_close = temp['Close']
            
            # Re-align timezones
            if data.index.tzinfo is not None and temp_close.index.tzinfo is None:
                temp_close.index = temp_close.index.tz_localize('UTC')
            elif data.index.tzinfo is None and temp_close.index.tzinfo is not None:
                temp_close.index = temp_close.index.tz_localize(None)
                
            # Alignment using union and forward-fill (robust timezone/holiday matching)
            combined_idx = data.index.union(temp_close.index).sort_values()
            aligned = temp_close.reindex(combined_idx).ffill().bfill().reindex(data.index)
            data[name] = aligned
            print("✓")
        except Exception as e:
            print(f"❌ Error: {e}. Filling with 0.")
            data[name] = 0.0
            
    # Cleaning as in training
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    
    data = data.loc[data['Close'].notna()]
    data.dropna(inplace=True)
    
    print(f"\n  ✅ Loaded {len(data)} hourly bars. Columns: {list(data.columns)}")
    return data

def build_windows(data, window_size, predict_size=4):
    """Normalize and prepare sliding windows exactly as in Education Forex 2.0.py."""
    banner("Preprocessing & Building Sliding Windows")
    
    # 1. Compute percentage difference
    data_diff = data.pct_change().fillna(0)
    
    # 2. Scale using MinMaxScaler range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data_diff)
    
    X, y_lbl, y_pct, timestamps, close_prices = [], [], [], [], []
    
    for i in range(window_size, len(scaled_data) - predict_size):
        pct_change = data_diff['Close'].iloc[i + predict_size] * 100
        
        X.append(scaled_data[i - window_size : i])
        y_lbl.append(1 if pct_change > 0 else 0)
        y_pct.append(pct_change)
        timestamps.append(data.index[i])
        close_prices.append(data['Close'].iloc[i])
        
    X = np.array(X)
    y_lbl = np.array(y_lbl)
    y_pct = np.array(y_pct)
    timestamps = np.array(timestamps)
    close_prices = np.array(close_prices)
    
    print(f"  Total windows built: {len(X)}")
    print(f"  X shape: {X.shape}")
    return X, y_lbl, y_pct, timestamps, close_prices, data_diff

def run_simulation(model, X, y_lbl, y_pct, timestamps, close_prices, data_diff):
    """Run sequential 4-hour non-overlapping trade simulation on the 3-month test window."""
    banner(f"Running Simulation on Last 3 Months [{TEST_START_STR} → {TEST_END_STR}]")
    
    # Align tz-awareness for test start date
    tz_info = timestamps[0].tzinfo if len(timestamps) > 0 else None
    test_start_ts = pd.Timestamp(TEST_START_STR)
    if tz_info is not None:
        test_start_ts = test_start_ts.tz_localize(tz_info)
        
    test_mask = timestamps >= test_start_ts
    test_indices = np.where(test_mask)[0]
    
    if len(test_indices) == 0:
        raise ValueError("No windows fall inside the 3-month test window!")
        
    print(f"  Test range windows: {len(test_indices)} (from {timestamps[test_indices[0]]} to {timestamps[test_indices[-1]]})")
    
    # Predict probabilities for the test set
    X_test = X[test_indices]
    print("  Predicting model outputs ...", end=" ", flush=True)
    y_pred_probs = model.predict(X_test, batch_size=128, verbose=0).flatten()
    print("✓")
    
    # Sequential 4-hour trade simulation
    trades = []
    step = 4
    
    # Loop over test indices with a stride of 4
    for k in range(0, len(test_indices) - 4, step):
        idx = test_indices[k]
        
        # Sigmoid output -> probability
        pred_prob = y_pred_probs[k]
        pred_dir = 1 if pred_prob > 0.5 else 0
        
        # Position opens at the Close of index idx-1 (which is the last completed bar of the input window)
        # Position closes at the Close of index idx+3 (which is exactly 4 hours later)
        entry_time = timestamps[idx - 1]
        entry_price = close_prices[idx - 1]
        
        exit_time = timestamps[idx + 3]
        exit_price = close_prices[idx + 3]
        
        # LONG if prediction probability > 0.5, SHORT otherwise
        if pred_dir == 1:
            direction = "LONG"
            trade_return = (exit_price - entry_price) / entry_price * 100
        else:
            direction = "SHORT"
            trade_return = (entry_price - exit_price) / entry_price * 100
            
        actual_price_up = 1 if exit_price > entry_price else 0
        trade_correct = (pred_dir == actual_price_up)
        
        # Also check model accuracy based on training label definition (change at idx + 4)
        train_label = y_lbl[idx]
        train_label_correct = (pred_dir == train_label)
        
        trades.append({
            "timestamp": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "direction": direction,
            "pred_prob": pred_prob,
            "trade_return_pct": trade_return,
            "trade_correct": int(trade_correct),
            "train_label_correct": int(train_label_correct),
            "noise_filtered": int(abs(y_pct[idx]) >= 0.1) # 0.1 noise threshold from training
        })
        
    df_trades = pd.DataFrame(trades)
    return df_trades

def plot_dashboard(df_trades, model_name):
    """Generate a high-quality dark-themed performance dashboard."""
    banner("Plotting Dashboard ...")
    
    # Calculate performance metrics over time
    df_trades = df_trades.sort_values("timestamp").reset_index(drop=True)
    df_trades["cum_sum_return"] = df_trades["trade_return_pct"].cumsum()
    
    # Capital curve starting at $10,000 with compounded returns
    capital_curve = [10000.0]
    for r in df_trades["trade_return_pct"]:
        capital_curve.append(capital_curve[-1] * (1 + r / 100))
    df_trades["capital"] = capital_curve[1:]
    
    # Calculate drawdowns
    equity_series = pd.Series(capital_curve)
    rolling_max = equity_series.cummax()
    drawdown_series = (equity_series - rolling_max) / rolling_max * 100
    max_dd = drawdown_series.min()
    
    # Create the figure
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#0b0b14")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    
    title_kw = dict(color="#e2e2ec", fontsize=12, fontweight="bold", pad=10)
    ax_bg = "#121222"
    grid_clr = "#252538"
    
    def style_ax(ax, title):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors="#a0a0c0", labelsize=9)
        ax.spines[:].set_color(grid_clr)
        ax.set_title(title, **title_kw)
        ax.grid(color=grid_clr, linewidth=0.5, linestyle="--")
        
    # ── 1. Compounded Account Capital Growth ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_trades["timestamp"], df_trades["capital"], color="#00e5ff", linewidth=1.5, label="Compounded Capital")
    ax1.fill_between(df_trades["timestamp"], df_trades["capital"], 10000.0, 
                     where=(df_trades["capital"] >= 10000.0), alpha=0.15, color="#00e5ff")
                     
    ax1.fill_between(df_trades["timestamp"], df_trades["capital"], 10000.0, 
                     where=(df_trades["capital"] < 10000.0), alpha=0.15, color="#ff2a6d")
                     
    ax1.axhline(10000.0, color="#606080", linestyle=":", linewidth=1.0, label="Starting Balance ($10,000)")
    style_ax(ax1, f"Compounded Account Capital Curve ($) - Max Drawdown: {max_dd:.2f}%")
    ax1.legend(facecolor="#181830", edgecolor=grid_clr, labelcolor="#e0e0ff")
    
    # ── 2. Cumulative Return % (Sum of Trade Returns) ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df_trades["timestamp"], df_trades["cum_sum_return"], color="#69ff47", linewidth=1.2, label="Cumulative Sum")
    ax2.axhline(0, color="#606080", linestyle=":", linewidth=1.0)
    style_ax(ax2, "Cumulative Trade Return (%)")
    
    # ── 3. Trade Outcomes (WIN vs LOSS) ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    wins = (df_trades["trade_return_pct"] > 0).sum()
    losses = (df_trades["trade_return_pct"] <= 0).sum()
    win_rate = wins / len(df_trades) * 100 if len(df_trades) > 0 else 0
    
    sizes = [wins, losses]
    labels = [f"WIN\n{wins} ({win_rate:.1f}%)", f"LOSS\n{losses} ({100-win_rate:.1f}%)"]
    colors = ["#00e5ff", "#ff2a6d"]
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, colors=colors, startangle=140, autopct='%1.1f%%',
        textprops=dict(color='#e0e0ff', fontsize=10), wedgeprops=dict(width=0.4, edgecolor=grid_clr)
    )
    for at in autotexts:
        at.set_color('#ffffff')
        at.set_fontweight('bold')
    ax3.set_facecolor(ax_bg)
    ax3.set_title("Trade Outcome Breakdown (Win Rate)", **title_kw)
    
    # ── 4. LONG vs SHORT P&L Distribution ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    long_pnl = df_trades[df_trades["direction"] == "LONG"]["trade_return_pct"]
    short_pnl = df_trades[df_trades["direction"] == "SHORT"]["trade_return_pct"]
    
    ax4.hist(long_pnl, bins=20, alpha=0.6, color="#00e5ff", label=f"LONG (n={len(long_pnl)})")
    ax4.hist(short_pnl, bins=20, alpha=0.6, color="#ff9e00", label=f"SHORT (n={len(short_pnl)})")
    style_ax(ax4, "PnL Distribution by Trade Direction (%)")
    ax4.legend(facecolor="#181830", edgecolor=grid_clr, labelcolor="#e0e0ff")
    
    # ── 5. Prediction Probability Distribution ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(df_trades["pred_prob"], bins=30, color="#b5179e", alpha=0.7, edgecolor=grid_clr)
    ax5.axvline(0.5, color="#ff2a6d", linestyle="--", linewidth=1.0, label="Decision Threshold (0.5)")
    style_ax(ax5, "Model Prediction Probability Distribution")
    ax5.set_xlabel("Probability (Sigmoid Output)", color="#a0a0c0", fontsize=9)
    ax5.legend(facecolor="#181830", edgecolor=grid_clr, labelcolor="#e0e0ff")
    
    # Main Title
    fig.suptitle(
        f"Hamdi Forex 2.0 Simulation Dashboard\n"
        f"Model: {model_name} | Target: {SYMBOL} (Last 3 Months)\n"
        f"Total Trades: {len(df_trades)} | Trade Win Rate: {win_rate:.2f}% | Final Return: {df_trades['cum_sum_return'].iloc[-1]:+.2f}%",
        color="#ffffff", fontsize=14, fontweight="bold", y=0.97
    )
    
    out_path = os.path.join(BASE_DIR, "Forex_2_0_Simulation_Dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✅ Dashboard saved to: {out_path}")
    
    # Save Trade Log to CSV
    csv_path = os.path.join(BASE_DIR, "Forex_2_0_Trade_Log.csv")
    df_trades.to_csv(csv_path, index=False)
    print(f"  ✅ Trade log saved to: {csv_path}")
    
    return win_rate

def main():
    print("\n" + "█" * 64)
    print("  HAMDI — FOREX 2.0 SIMULATION & TEST ENGINE")
    print(f"  Period: {TEST_START_STR} → {TEST_END_STR} (3 Months)")
    print("█" * 64)
    
    # 1. Pick Keras Model
    try:
        model_path = pick_model()
    except FileNotFoundError:
        return
        
    model_name = os.path.basename(model_path)
    banner(f"Loading Model: {model_name}")
    
    # Load model with compile=False to completely avoid compile/loss mismatches
    model = load_model(model_path, compile=False)
    model.summary()
    
    # 2. Extract Shape Parameters Dynamically
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        window_size = input_shape[1]
        num_features = input_shape[2]
        print(f"\n  Detected parameters:")
        print(f"    - Window Size  : {window_size} (LSTM lag steps)")
        print(f"    - Num Features : {num_features} (inputs expected)")
    except Exception as e:
        print(f"\n  ⚠️ Cannot detect model input shape automatically: {e}")
        print("  Using standard training defaults (window_size=36, num_features=28)")
        window_size = 36
        num_features = 28
        
    # Check features expected
    if num_features != 28:
        print(f"  ⚠️ Warning: Model expects {num_features} features, but standard 2.0 has 28.")
        print("  The simulator will download the 28 standard features. If shape mismatch occurs, the script may fail.")
        
    # 3. Download Data
    data = download_data()
    
    # 4. Preprocessing & Windowing
    X, y_lbl, y_pct, timestamps, close_prices, data_diff = build_windows(data, window_size)
    
    # 5. Run Sequential Simulation
    try:
        df_trades = run_simulation(model, X, y_lbl, y_pct, timestamps, close_prices, data_diff)
    except ValueError as val_err:
        print(f"\n❌ Error during simulation: {val_err}")
        return
        
    if len(df_trades) == 0:
        print("\n❌ Simulation finished with 0 trades.")
        return
        
    # 6. Performance Summary & Dashboard
    win_rate = plot_dashboard(df_trades, model_name)
    
    # 7. Print Final Metrics Table
    total_trades = len(df_trades)
    trade_acc = df_trades["trade_correct"].mean() * 100
    train_acc = df_trades["train_label_correct"].mean() * 100
    
    # Also calculate noise-filtered training label accuracy
    filtered_df = df_trades[df_trades["noise_filtered"] == 1]
    filtered_train_acc = filtered_df["train_label_correct"].mean() * 100 if len(filtered_df) > 0 else 0.0
    
    long_trades = df_trades[df_trades["direction"] == "LONG"]
    short_trades = df_trades[df_trades["direction"] == "SHORT"]
    
    long_win_rate = (long_trades["trade_return_pct"] > 0).mean() * 100 if len(long_trades) > 0 else 0.0
    short_win_rate = (short_trades["trade_return_pct"] > 0).mean() * 100 if len(short_trades) > 0 else 0.0
    
    banner("📊 SIMULATION COMPLETE - PERFORMANCE REPORT", char="▀")
    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  Model Name        : {model_name:<34}│")
    print(f"  │  Sim Period        : {TEST_START_STR} → {TEST_END_STR}         │")
    print(f"  │  Total Trades      : {total_trades:<37}│")
    print(f"  │  LONG / SHORT      : {len(long_trades)} / {len(short_trades):<33}│")
    print(f"  │  Trade Win Rate    : {win_rate:>6.2f}% (LONG={long_win_rate:.1f}%, SHORT={short_win_rate:.1f}%) │")
    print(f"  │  Trade Real Acc    : {trade_acc:>6.2f}% (Price matches prediction)    │")
    print(f"  │  Train Label Acc   : {train_acc:>6.2f}% (Match training targets)      │")
    if len(filtered_df) > 0:
        print(f"  │  Filtered Label Acc: {filtered_train_acc:>6.2f}% (Absolute pct change >= 0.1%)  │")
    print(f"  │  Cumulative Return : {df_trades['trade_return_pct'].sum():>+6.2f}%                             │")
    print(f"  │  Compounded Growth : {(df_trades['capital'].iloc[-1] - 10000.0)/100.0:>+6.2f}%                             │")
    print(f"  └────────────────────────────────────────────────────────┘")
    print()

if __name__ == "__main__":
    main()
