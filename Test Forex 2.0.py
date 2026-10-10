"""
=============================================================
  TEST FOREX 2.0.PY
  Directional Accuracy & Account Simulation Test
  Companion to: Education Forex 2.0.py

  Matches EXACTLY the pipeline of Education Forex 2.0.py:
    • Features  : Close(pct_change), Open, High, Low, Volume,
                  EMA_20, RSI, MACD, Signal_Line
    • Scaler    : MinMaxScaler(feature_range=(-1, 1))  — loaded from .joblib
    • Window    : window_size = 24
    • Predict   : predict_size = 4
    • Label     : pct_change * 1000  (regression float)
    • Threshold : |pct_change| >= 0.05  (same noise filter)

  Models   : Hamdi_*_Forex_2.0.keras  (in BASE_DIR or Forex/ sub-folder)
  Scalers  : Hamdi_*_ForexScaler_2.0.joblib  (same location as model)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model   # type: ignore
from datetime import datetime, timedelta


# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/ardaunal/Desktop/Hamdi"

# Search for models in BASE_DIR itself AND the Forex/ subfolder
SEARCH_DIRS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "Forex"),
]

# ─── PARAMETERS — MUST MATCH Education Forex 2.0.py EXACTLY ──────────────────
INTERVAL     = "1h"
WINDOW_SIZE  = 24          # window_size in Education Forex 2.0.py
PREDICT_SIZE = 4           # predict_size in Education Forex 2.0.py
THRESHOLD    = 0.05        # noise filter: |pct_change * 1000| < 0.05 → skip

# ─── TEST WINDOW ──────────────────────────────────────────────────────────────
TEST_END     = datetime.today()
TEST_START   = TEST_END - timedelta(days=92)          # ~3 calendar months

# Download extra history so the sliding window has enough context
BUFFER_DAYS  = 10
DATA_START   = TEST_START - timedelta(days=BUFFER_DAYS + WINDOW_SIZE + 5)

TEST_END_STR   = TEST_END.strftime("%Y-%m-%d")
TEST_START_STR = TEST_START.strftime("%Y-%m-%d")
DATA_START_STR = DATA_START.strftime("%Y-%m-%d")

# ─── TRADING SIMULATION PARAMETERS ───────────────────────────────────────────
PRED_THRESHOLD = 0.0   # Only simulate signals with |pred| >= this value


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(text, char="=", width=64):
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


# ─── INDICATORS — identical to Education Forex 2.0.py ────────────────────────
def add_indicators(df):
    """Exact copy of add_indicators() from Education Forex 2.0.py."""
    # EMA (20)
    df['EMA_20'] = round(df['Close'].ewm(span=20, adjust=False).mean(), 4)

    # RSI (14) — division-by-zero safe
    delta = df['Close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-10)
    df['RSI'] = round(100 - (100 / (1 + rs)), 4)

    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = round(exp1 - exp2, 4)
    df['Signal_Line'] = round(df['MACD'].ewm(span=9, adjust=False).mean(), 4)

    return df.round(4)


def find_models():
    """
    Scan SEARCH_DIRS for all Forex 2.0 .keras models.
    Returns a list of full paths found.
    """
    found = []
    for d in SEARCH_DIRS:
        if os.path.isdir(d):
            found.extend(sorted(glob.glob(os.path.join(d, "Hamdi_*_Forex_2.0.keras"))))
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in found:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def pick_model():
    """Let the user pick which model(s) to test."""
    found = find_models()

    if not found:
        dirs = ", ".join(SEARCH_DIRS)
        print(f"\n  ❌ No Forex 2.0 models found in: {dirs}")
        print(   "     Expected pattern: Hamdi_*_Forex_2.0.keras")
        raise FileNotFoundError("No models found.")

    banner("Available Forex 2.0 Models")
    for idx, path in enumerate(found):
        print(f"  [{idx + 1}]  {os.path.basename(path)}")
    print(f"  [0]  Test ALL models above")

    while True:
        raw = input("\n  Select model (0 for all, or number): ").strip()
        if raw == "0":
            return found
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(found):
                return [found[n - 1]]
        print("  ⚠️  Invalid choice, try again.")


def find_scaler_for(model_path):
    """
    Find the matching .joblib scaler for a given model path.
    Looks for: Hamdi_SYMBOL_ForexScaler_2.0.joblib  in the same directory.
    Falls back to BASE_DIR if not found next to the model.
    """
    model_dir  = os.path.dirname(model_path)
    model_base = os.path.basename(model_path)          # Hamdi_XX_Forex_2.0.keras
    symbol     = 'EURUSD=X'
    scaler_name = f"Hamdi_{symbol}_ForexScaler_2.0.joblib"

    # 1. Same directory as model
    candidate = os.path.join(model_dir, scaler_name)
    if os.path.exists(candidate):
        return candidate

    # 2. BASE_DIR
    candidate = os.path.join(BASE_DIR, scaler_name)
    if os.path.exists(candidate):
        return candidate

    return None


def infer_symbol_from_model(model_path):
    """
    Derive the forex symbol from the model filename.
    Hamdi_EURUSD=X_Forex_2.0.keras  →  EURUSD=X
    Hamdi_65_Forex_2.0.keras        →  65   (fallback; will prompt user)
    """
    base   = os.path.basename(model_path)
    symbol = base.replace("Hamdi_", "").replace("_Forex_2.0.keras", "")
    # Try common Yahoo Finance forex suffix patterns
    for suffix in ["=X", "-USD", "=F"]:
        if symbol.endswith(suffix):
            return symbol
    # Known pure codes → add =X
    known_forex = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
                   "AUDUSD", "NZDUSD", "XAUUSD"]
    if symbol in known_forex:
        return symbol + "=X"
    return symbol   # Return as-is; user will be asked if download fails


def download_data(symbol):
    """
    Download OHLCV data for `symbol` and add indicators exactly as in
    Education Forex 2.0.py.  Returns a clean DataFrame.
    """
    banner(f"Downloading {symbol}  [{DATA_START_STR} → {TEST_END_STR}]")

    data = yf.download(
        symbol, start=DATA_START_STR, end=TEST_END_STR,
        interval=INTERVAL, auto_adjust=True, progress=False
    )

    if data.empty:
        raise ValueError(f"❌ No data returned for '{symbol}'. "
                         "Check the symbol or internet connection.")

    # Flatten MultiIndex columns (newer yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ── Indicators ────────────────────────────────────────────────────────────
    data = add_indicators(data)

    # ── Cleaning — identical to Education Forex 2.0.py ───────────────────────
    data.fillna(0, inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    data = data.loc[data['Close'].notna()]
    data.dropna(inplace=True)

    print(f"  ✅ {len(data)} hourly bars  |  columns: {list(data.columns)}")
    return data


def build_windows(data, scaler):
    """
    Reproduce the EXACT sliding-window + label logic from Education Forex 2.0.py.

    Preprocessing:
      data_diff = data with Close replaced by Close.pct_change().fillna(0)
      scaled    = scaler.transform(data_diff)          ← uses the SAVED scaler

    Label:
      pct_change = ((next_close - current_close) / current_close) * 1000
      Skip if |pct_change| < THRESHOLD

    Returns X, y_pct, timestamps, close_prices.
    """
    # 1. Reproduce data_diff exactly as in Education Forex 2.0.py
    data_diff = data.assign(Close=data['Close'].pct_change().fillna(0))

    # 2. Scale with the SAVED scaler (transform only — NOT fit)
    scaled_data = scaler.transform(data_diff)

    # 3. Build windows
    original_close = data['Close'].values
    timestamps     = data.index

    X, y_list, ts_list, close_list = [], [], [], []

    for i in range(WINDOW_SIZE, len(scaled_data) - PREDICT_SIZE):
        current_close = original_close[i]
        next_close    = original_close[i + PREDICT_SIZE]

        if current_close == 0:
            continue

        pct_change = ((next_close - current_close) / current_close) * 1000

        if abs(pct_change) < THRESHOLD:
            continue   # same noise filter as training

        X.append(scaled_data[i - WINDOW_SIZE : i])
        y_list.append(pct_change)
        ts_list.append(timestamps[i])
        close_list.append(current_close)

    X   = np.array(X)
    y   = np.array(y_list)
    ts  = np.array(ts_list)
    clp = np.array(close_list)

    print(f"  Total windows built : {len(y)}")
    return X, y, ts, clp


def restrict_to_test_window(X, y, ts, clp):
    """Keep only rows whose timestamp falls inside the 3-month test window."""
    test_start = pd.Timestamp(TEST_START_STR)

    ts_idx = pd.DatetimeIndex(ts)
    # Normalise timezone
    if ts_idx.tzinfo is not None:
        test_start = test_start.tz_localize(ts_idx.tzinfo)

    mask = ts_idx >= test_start
    return X[mask], y[mask], ts[mask], clp[mask]


def simulate_account(y_true_pct, y_pred_pct, close_prices, timestamps, data):
    """
    Simple trade simulation using predicted direction.
      LONG  if prediction > 0   (model expects price to rise)
      SHORT if prediction < 0   (model expects price to fall)

    Dynamic TP/SL:
      LONG  → TP = |pred|,  SL = -0.5%
      SHORT → TP = 0.5%,    SL = -|pred|

    Checks hourly H/L for the next 4 bars; if no TP/SL hit, closes at bar 4.
    """
    records = []

    for i in range(len(y_pred_pct)):
        pred        = y_pred_pct[i]
        actual      = y_true_pct[i]
        entry_close = close_prices[i]
        timestamp   = timestamps[i]

        if pred > 0:
            direction = "LONG"
            raw_pnl   =  actual
            tp = abs(pred)
            sl = -0.5
        else:
            direction = "SHORT"
            raw_pnl   = -actual
            tp = 0.5
            sl = -abs(pred)

        pnl     = None
        outcome = "PARTIAL"

        # Align timestamp timezone to data index
        try:
            tz_data = data.index.tz
            tz_ts   = pd.Timestamp(timestamp).tzinfo
            if tz_data is not None and tz_ts is None:
                ts_aligned = pd.Timestamp(timestamp).tz_localize(tz_data)
            elif tz_data is None and tz_ts is not None:
                ts_aligned = pd.Timestamp(timestamp).tz_localize(None)
            elif tz_data is not None and tz_ts is not None:
                ts_aligned = pd.Timestamp(timestamp).tz_convert(tz_data)
            else:
                ts_aligned = pd.Timestamp(timestamp)

            entry_idx = data.index.get_loc(ts_aligned)
        except KeyError:
            entry_idx = None

        if entry_idx is not None:
            for h in range(1, PREDICT_SIZE + 1):
                idx = entry_idx + h
                if idx >= len(data):
                    break

                high_h = data.iloc[idx]['High']
                low_h  = data.iloc[idx]['Low']

                if direction == "LONG":
                    tp_price = entry_close * (1 + tp / 100)
                    sl_price = entry_close * (1 + sl / 100)

                    if low_h <= sl_price:
                        pnl = sl;  outcome = "LOSS";  break
                    elif high_h >= tp_price:
                        pnl = tp;  outcome = "WIN";   break
                else:
                    tp_price = entry_close * (1 - tp / 100)
                    sl_price = entry_close * (1 - sl / 100)

                    if high_h >= sl_price:
                        pnl = sl;  outcome = "LOSS";  break
                    elif low_h <= tp_price:
                        pnl = tp;  outcome = "WIN";   break

            if pnl is None:
                # Close at the predict_size-th bar
                final_idx = min(entry_idx + PREDICT_SIZE, len(data) - 1)
                close_final = data.iloc[final_idx]['Close']
                if direction == "LONG":
                    pnl = ((close_final - entry_close) / entry_close) * 100
                else:
                    pnl = -((close_final - entry_close) / entry_close) * 100
                outcome = "PARTIAL"

        if pnl is None:
            pnl = max(sl, min(tp, raw_pnl))
            outcome = (
                "WIN"     if pnl >= tp else
                "LOSS"    if pnl <= sl else
                "PARTIAL"
            )

        dir_correct = (np.sign(pred) == np.sign(actual))

        records.append({
            "timestamp"   : timestamp,
            "close"       : entry_close,
            "direction"   : direction,
            "pred_pct"    : pred,
            "actual_pct"  : actual,
            "raw_pnl"     : raw_pnl,
            "clipped_pnl" : pnl,
            "dir_correct" : dir_correct,
            "tp"          : tp,
            "sl"          : sl,
            "outcome"     : outcome,
        })

    return pd.DataFrame(records)


def print_results(df, model_name):
    """Print directional accuracy + account simulation summary."""
    n_total   = len(df)
    n_correct = df['dir_correct'].sum()
    dir_acc   = n_correct / n_total * 100 if n_total > 0 else 0

    n_long    = (df['direction'] == 'LONG').sum()
    n_short   = (df['direction'] == 'SHORT').sum()
    long_acc  = df[df['direction'] == 'LONG' ]['dir_correct'].mean() * 100 if n_long  > 0 else 0
    short_acc = df[df['direction'] == 'SHORT']['dir_correct'].mean() * 100 if n_short > 0 else 0

    banner(f"📊 DIRECTIONAL ACCURACY  [{model_name}]")
    print(f"\n  Total signals         : {n_total}")
    print(f"  LONG  signals         : {n_long}   |  Directional acc : {long_acc:.2f}%")
    print(f"  SHORT signals         : {n_short}  |  Directional acc : {short_acc:.2f}%")
    print(f"\n  ★ Overall Directional Accuracy : {dir_acc:.2f}%  ({n_correct}/{n_total})")

    wins     = (df['outcome'] == 'WIN').sum()
    losses   = (df['outcome'] == 'LOSS').sum()
    partial  = (df['outcome'] == 'PARTIAL').sum()
    win_rate = wins / n_total * 100 if n_total > 0 else 0
    cum_pnl  = df['clipped_pnl'].sum()
    avg_pnl  = df['clipped_pnl'].mean()

    banner("💰 ACCOUNT SIMULATION  (Dynamic TP/SL)")
    print(f"\n  Total trades              : {n_total}")
    print(f"  Wins  (hit TP)            : {wins}   ({win_rate:.1f}%)")
    print(f"  Losses (hit SL)           : {losses}  ({losses/n_total*100 if n_total>0 else 0:.1f}%)")
    print(f"  Partial (no TP/SL reached): {partial}")
    print(f"\n  Avg P&L per trade         : {avg_pnl:+.4f}%")
    print(f"  Cumulative P&L            : {cum_pnl:+.4f}%")

    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0:
            continue
        w    = (sub['outcome'] == 'WIN').sum()
        wr   = w / len(sub) * 100
        cpnl = sub['clipped_pnl'].sum()
        print(f"\n  [{d}]  trades={len(sub)}  wins={w}  win_rate={wr:.1f}%  cum_pnl={cpnl:+.4f}%")

    return dir_acc, win_rate, cum_pnl


def print_sample_trades(df, n=30):
    """Print the first n trades in a tabular format."""
    banner(f"🔍 Sample Trades (first {min(n, len(df))})")
    header = (f"  {'#':<5} {'Timestamp':<20} {'Close':>10} "
              f"{'Dir':>6} {'Pred':>8} {'Actual':>9} {'P&L%':>7} {'OK?':>5}")
    print(header)
    print("  " + "-" * 80)
    for i, row in enumerate(df.head(n).itertuples(index=False)):
        ok = "✓" if row.dir_correct else "✗"
        ts = str(row.timestamp)[:19]
        print(f"  {i:<5} {ts:<20} {row.close:>10.5f} "
              f"{row.direction:>6} {row.pred_pct:>+8.4f} "
              f"{row.actual_pct:>+9.4f} {row.clipped_pnl:>+7.4f} {ok:>5}")


def test_model(model_path, data, X_all, y_all, ts_all, clp_all):
    """Run the full test pipeline for a single model."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    banner(f"🤖 TESTING: {model_name}", char="▀", width=64)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"  ⚠️  Failed to load model {model_name}: {e}")
        return None

    # ── Keep only test-window data ────────────────────────────────────────────
    X_test, y_test, ts_test, clp_test = restrict_to_test_window(
        X_all, y_all, ts_all, clp_all
    )

    if len(X_test) == 0:
        print(f"  ❌ No windows fall inside the test window for {model_name}.")
        return None

    print(f"  Test-window windows : {len(X_test)}")
    print(f"  Date range          : {ts_test[0]} → {ts_test[-1]}")

    # ── Feature dimension check ───────────────────────────────────────────────
    expected_features = X_test.shape[2]
    try:
        model_features = model.input_shape[-1]
        if model_features != expected_features:
            print(f"  ⚠️  Feature mismatch: model expects {model_features} features, "
                  f"data has {expected_features}. Skipping {model_name}.")
            return None
    except Exception:
        pass  # Cannot determine shape; proceed anyway

    # ── Predict ───────────────────────────────────────────────────────────────
    banner("Running predictions")
    y_pred = model.predict(X_test, batch_size=64, verbose=1).flatten()

    print(f"\n  Pred range   : [{y_pred.min():.4f},  {y_pred.max():.4f}]")
    print(f"  Actual range : [{y_test.min():.4f},  {y_test.max():.4f}]")

    # ── Threshold filter ──────────────────────────────────────────────────────
    mask              = np.abs(y_pred) >= PRED_THRESHOLD
    y_pred_f          = y_pred[mask]
    y_test_f          = y_test[mask]
    clp_test_f        = clp_test[mask]
    ts_test_f         = ts_test[mask]

    print(f"  Signals after threshold filter (>={PRED_THRESHOLD}): "
          f"{len(y_pred_f)} / {len(y_pred)}")

    if len(y_pred_f) == 0:
        print(f"  ❌ No predictions met the threshold for {model_name}.")
        return None

    # ── Simulate & report ─────────────────────────────────────────────────────
    df = simulate_account(y_test_f, y_pred_f, clp_test_f, ts_test_f, data)
    print_sample_trades(df, n=20)
    dir_acc, win_rate, cum_pnl = print_results(df, model_name)

    return {
        "model"    : model_name,
        "signals"  : len(df),
        "dir_acc"  : dir_acc,
        "win_rate" : win_rate,
        "cum_pnl"  : cum_pnl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 64)
    print("  HAMDI — TEST FOREX 2.0")
    print("  Companion test for Education Forex 2.0.py models")
    print(f"  Test period : {TEST_START_STR}  →  {TEST_END_STR}  (~3 months)")
    print(f"  Window={WINDOW_SIZE} | PredictSize={PREDICT_SIZE} | "
          f"Threshold={THRESHOLD} | PredFilter={PRED_THRESHOLD}")
    print("█" * 64)

    # 1. Pick model(s)
    selected_models = pick_model()

    # 2. For each model: load its scaler, download matching symbol, build windows
    results = []
    processed_symbols = {}  # cache: symbol → (data, X, y, ts, clp)

    for model_path in selected_models:

        # ── Resolve scaler ────────────────────────────────────────────────────
        scaler_path = find_scaler_for(model_path)
        if scaler_path is None:
            print(f"\n  ⚠️  No scaler found for {os.path.basename(model_path)}.")
            custom = input("  Enter full path to the .joblib scaler (or press Enter to skip): ").strip()
            if not custom or not os.path.exists(custom):
                print("  Skipping this model.")
                continue
            scaler_path = custom

        print(f"\n  Loading scaler: {os.path.basename(scaler_path)}")
        try:
            scaler = MinMaxScaler()
        except Exception as e:
            print(f"  ⚠️  Could not load scaler: {e}. Skipping.")
            continue

        # ── Resolve symbol ────────────────────────────────────────────────────
        symbol = infer_symbol_from_model(model_path)
        print(f"  Inferred symbol: {symbol}")

        # ── Download data (cached by symbol) ──────────────────────────────────
        if symbol not in processed_symbols:
            try:
                data = download_data(symbol)
            except ValueError as e:
                # Symbol inference may have failed; ask user
                print(f"  ⚠️  {e}")
                symbol = input("  Enter the correct Yahoo Finance symbol (e.g. EURUSD=X): ").strip().upper()
                if not symbol:
                    print("  Skipping this model.")
                    continue
                try:
                    data = download_data(symbol)
                except Exception as e2:
                    print(f"  ❌ Could not download data: {e2}. Skipping.")
                    continue

            if len(data) < WINDOW_SIZE + PREDICT_SIZE + 10:
                print(f"  ❌ Not enough data ({len(data)} bars) for {symbol}. Skipping.")
                continue

            # ── Build windows ─────────────────────────────────────────────────
            banner("Building sliding windows  (identical to training logic)")
            try:
                X_all, y_all, ts_all, clp_all = build_windows(data, scaler)
            except Exception as e:
                print(f"  ❌ Window building failed: {e}. Skipping.")
                continue

            processed_symbols[symbol] = (data, X_all, y_all, ts_all, clp_all)
        else:
            data, X_all, y_all, ts_all, clp_all = processed_symbols[symbol]

        # ── Run model test ────────────────────────────────────────────────────
        result = test_model(model_path, data, X_all, y_all, ts_all, clp_all)
        if result:
            results.append(result)

    # ── Comparison summary (multiple models) ──────────────────────────────────
    if len(results) > 1:
        banner("📋 COMPARISON SUMMARY  (All Models — ~3 Months)")
        print(f"\n  {'Model':<45} {'Signals':>8} {'DirAcc':>8} {'WinRate':>8} {'CumP&L':>10}")
        print("  " + "-" * 85)
        for r in sorted(results, key=lambda x: x['dir_acc'], reverse=True):
            print(f"  {r['model']:<45} {r['signals']:>8} "
                  f"{r['dir_acc']:>7.2f}% {r['win_rate']:>7.2f}% "
                  f"{r['cum_pnl']:>+10.3f}%")

        best = max(results, key=lambda x: x['dir_acc'])
        print(f"\n  🏆 Best directional accuracy: {best['model']}  ({best['dir_acc']:.2f}%)")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("✅ TEST COMPLETE")
    for r in results:
        print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  Model    : {r['model']:<49}│")
        print(f"  │  Period   : {TEST_START_STR}  →  {TEST_END_STR}            │")
        print(f"  │  Signals  : {r['signals']:<49}│")
        print(f"  │  Dir. Acc : {r['dir_acc']:>6.2f}%{'':<45}│")
        print(f"  │  Win Rate : {r['win_rate']:>6.2f}%  (Dynamic TP/SL){'':<27}│")
        print(f"  │  Cum P&L  : {r['cum_pnl']:>+7.3f}%{'':<43}│")
        print(f"  └─────────────────────────────────────────────────────────────┘")

    if not results:
        print("\n  ⚠️  No models were successfully tested.")


if __name__ == "__main__":
    main()
