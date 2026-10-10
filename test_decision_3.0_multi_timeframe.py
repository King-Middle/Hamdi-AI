import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model #type:ignore
from datetime import datetime, timedelta
import os
import pytz

# --- Configuration ---
SYMBOL = "BTC-USD"
START_DATE = "2024-03-01"
TRAIN_END_DATE = "2025-12-01"
INTERVAL = '1h'
WINDOW_SIZE = 18  # Unique to Decision 3.0
PREDICT_SIZE = 1

# Timeframes to test (in days)
TIMEFRAMES = {
    "1 Week": 7,
    "2 Weeks": 14,
    "3 Weeks": 21,
    "1 Month": 30,
    "2 Months": 60,
    "4 Months": 120,
    "5 Months": 150
}

# Default model to test
MODEL_PATH = "Hamdi_BTC2_Decision_3.0.keras"

def run_test_for_model(model_path):
    print(f"\n--- Multi-Timeframe Evaluation for: {model_path} ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Downloading data...")
    try:
        df = yf.download(SYMBOL, start=START_DATE, interval=INTERVAL, auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    
    # --- Feature Engineering (Decision 3.0 uses raw OHLCV) ---
    features_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # --- Scaling (MinMaxScaler) ---
    train_mask = (df.index <= TRAIN_END_DATE)
    scaler = MinMaxScaler()
    
    if train_mask.sum() > 0:
        scaler.fit(df.loc[train_mask, features_raw])
        scaled_features = scaler.transform(df[features_raw])
    else:
        scaled_features = scaler.fit_transform(df[features_raw])
        
    df_scaled = pd.DataFrame(scaled_features, columns=features_raw, index=df.index)

    # --- Generate Predictions ---
    print("Generating predictions...")
    X_all = []
    indices_all = []
    
    start_idx = WINDOW_SIZE
    for i in range(start_idx, len(df) - PREDICT_SIZE):
        X_seq = df_scaled.iloc[i - WINDOW_SIZE : i].values
        X_all.append(X_seq)
        indices_all.append(i)
        
    X_all = np.array(X_all)
    predictions = model.predict(X_all, verbose=0).flatten()
    
    # Store results
    trade_results = []
    for idx_in_arr, df_idx in enumerate(indices_all):
        prob = predictions[idx_in_arr]
        current_close = df['Close'].iloc[df_idx]
        future_close = df['Close'].iloc[df_idx + PREDICT_SIZE]
        
        signal = "LONG" if prob > 0.5 else "SHORT"
        confidence = prob if signal == "LONG" else (1 - prob)
        
        win = 0
        if signal == "LONG" and future_close > current_close:
            win = 1
        elif signal == "SHORT" and future_close < current_close:
            win = 1
            
        trade_results.append({
            'timestamp': df.index[df_idx],
            'confidence': confidence,
            'win': win
        })
        
    trade_df = pd.DataFrame(trade_results).set_index('timestamp')
    trade_df.index = pd.to_datetime(trade_df.index, utc=True)
    
    # --- Reporting ---
    now_utc = datetime.now(pytz.utc)
    print("="*105)
    print(f"{'Period':<15} | {'Metric':<25} | {'Trades':<8} | {'Accuracy':<10}")
    print("="*105)
    
    for name, days in TIMEFRAMES.items():
        cutoff = now_utc - timedelta(days=days)
        subset = trade_df[trade_df.index >= cutoff]
        
        if len(subset) == 0:
            print(f"{name:<15} | No data")
            continue
            
        total_trades = len(subset)
        acc_overall = subset['win'].mean()
        
        # Weighted Accuracy
        weighted_acc = (subset['win'] * subset['confidence']).sum() / subset['confidence'].sum()
        
        # Bins
        acc_60 = subset[subset['confidence'] > 0.6]['win'].mean()
        acc_70 = subset[subset['confidence'] > 0.7]['win'].mean()
        acc_80 = subset[subset['confidence'] > 0.8]['win'].mean()
        acc_90 = subset[subset['confidence'] > 0.9]['win'].mean()
        
        count_80 = len(subset[subset['confidence'] > 0.8])
        count_90 = len(subset[subset['confidence'] > 0.9])
        
        print(f"{name:<15} | All Trades (>50%)         | {total_trades:<8} | {acc_overall*100:.2f}%")
        print(f"{'':<15} | WEIGHTED ACCURACY         | {'-':<8} | {weighted_acc*100:.2f}%")
        print(f"{'':<15} | High Conf (>60%)          | {len(subset[subset['confidence']>0.6]):<8} | {acc_60*100:.2f}%")
        print(f"{'':<15} | High Conf (>70%)          | {len(subset[subset['confidence']>0.7]):<8} | {acc_70*100:.2f}%")
        print(f"{'':<15} | High Conf (>80%)          | {count_80:<8} | {acc_80*100:.2f}%")
        print(f"{'':<15} | High Conf (>90%)          | {count_90:<8} | {acc_90*100:.2f}%")
        print("-" * 105)

if __name__ == "__main__":
    # Test both common decision models found in directory
    models_to_test = ["Hamdi_BTC1_Decision_3.0.keras", "Hamdi_BTC2_Decision_3.0.keras"]
    for m in models_to_test:
        if os.path.exists(m):
            run_test_for_model(m)
        else:
            print(f"Skipping {m} (not found)")
