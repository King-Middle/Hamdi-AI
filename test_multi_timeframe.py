import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model #type:ignore
from datetime import datetime, timedelta
import os

# --- Configuration ---
SYMBOL = "BTC-USD"
START_DATE = "2024-03-01"
TRAIN_END_DATE = "2025-12-01"
INTERVAL = '1h'
WINDOW_SIZE = 48
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

MODEL_PATH = "Hamdi_BTCST_Advanced_3.0.keras"

# --- Helper Functions (Same as before) ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def run_test():
    print("--- Multi-Timeframe Model Evaluation ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    try:
        model = load_model(MODEL_PATH)
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

    # --- Feature Engineering ---
    df['RSI'] = compute_rsi(df['Close'])
    macd, signal_line = compute_macd(df['Close'])
    upper_bb, lower_bb = compute_bollinger_bands(df['Close'])
    atr = compute_atr(df['High'], df['Low'], df['Close'])
    sma_50 = df['Close'].rolling(window=50).mean()
    ema_200 = df['Close'].ewm(span=200, adjust=False).mean()

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Volume_Change'] = np.log((df['Volume'] + 1) / (df['Volume'].shift(1) + 1))
    
    df['RSI_Scaled'] = df['RSI'] / 100.0
    df['MACD_Norm'] = macd / df['Close']
    df['Signal_Norm'] = signal_line / df['Close']
    df['BB_Width'] = (upper_bb - lower_bb) / df['Close']
    df['BB_Position'] = (df['Close'] - lower_bb) / (upper_bb - lower_bb)
    df['SMA_50_Dist'] = (df['Close'] - sma_50) / sma_50
    df['EMA_200_Dist'] = (df['Close'] - ema_200) / ema_200
    df['ATR_Pct'] = atr / df['Close']
    
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    df.dropna(inplace=True)

    features = [
        'Log_Return', 'Log_Volume_Change', 'RSI_Scaled', 
        'MACD_Norm', 'Signal_Norm', 
        'BB_Width', 'BB_Position', 
        'SMA_50_Dist', 'EMA_200_Dist', 'ATR_Pct',
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
    ]

    # --- Scaling ---
    train_mask = (df.index <= TRAIN_END_DATE)
    scaler = RobustScaler()
    if train_mask.sum() > 0:
        scaler.fit(df.loc[train_mask, features])
        scaled_features = scaler.transform(df[features])
    else:
        scaled_features = scaler.fit_transform(df[features])
    
    df_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
    
    # --- Generate All Predictions First ---
    # This is more efficient than re-looping
    print("Generating predictions for all valid sequences...")
    
    X_all = []
    indices_all = []
    
    # Start from where we have enough window
    start_idx = WINDOW_SIZE
    
    for i in range(start_idx, len(df) - PREDICT_SIZE):
        X_seq = df_scaled.iloc[i - WINDOW_SIZE : i].values
        X_all.append(X_seq)
        indices_all.append(i)
        
    X_all = np.array(X_all)
    predictions = model.predict(X_all, verbose=0).flatten()
    
    # Store results in a list of dicts
    trade_results = []
    for idx_in_arr, df_idx in enumerate(indices_all):
        prob = predictions[idx_in_arr]
        current_close = df['Close'].iloc[df_idx]
        future_close = df['Close'].iloc[df_idx + PREDICT_SIZE]
        
        signal = "LONG" if prob > 0.5 else "SHORT"
        confidence = prob if signal == "LONG" else (1 - prob)
        
        # Outcome: 1 for Win, 0 for Loss
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
    
    # --- Analyze per Timeframe ---
    import pytz
    now_utc = datetime.now(pytz.utc)
    
    print("\n" + "="*100)
    print(f"{'Period':<15} | {'Metric':<25} | {'Trades':<8} | {'Accuracy':<10}")
    print("="*100)
    
    summary_table = []
    
    for name, days in TIMEFRAMES.items():
        cutoff = now_utc - timedelta(days=days)
        subset = trade_df[trade_df.index >= cutoff]
        
        if len(subset) == 0:
            print(f"{name:<15} | No data")
            continue
            
        # Overall Accuracy
        total_trades = len(subset)
        acc_overall = subset['win'].mean()
        
        # Weighted Accuracy Calculation
        # Formula: Sum(Win * Confidence) / Sum(Confidence)
        # This gives higher weight to high-confidence trades.
        weighted_acc = (subset['win'] * subset['confidence']).sum() / subset['confidence'].sum()
        
        # Breakdown by Confidence
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
        print("-" * 100)
        
        summary_table.append({
            "Period": name,
            "Weighted Score": f"{weighted_acc*100:.2f}%",
            ">80% Acc": f"{acc_80*100:.1f}% ({count_80} trades)",
            ">90% Acc": f"{acc_90*100:.1f}% ({count_90} trades)"
        })

if __name__ == "__main__":
    run_test()
