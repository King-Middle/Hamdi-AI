import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os

# --- Configuration ---
SYMBOL = "BTC-USD"
START_DATE = "2024-03-01"
TRAIN_END_DATE = "2025-12-01"
INTERVAL = '1h'
WINDOW_SIZE = 48
PREDICT_SIZE = 1

# User defined thresholds
# "Noticeable" move for win condition. 
# User asked for "noticebly accept it". 
# The training used 0.8%, but for a "win" in 1h, 0.5% is often considered a good scalp.
# I will report stats for 0.5% and 0.8%.
TARGET_THRESHOLDS = [0.005, 0.008] 

# Model Path
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
    print(f"--- Starting Advanced Accuracy Check (Past 3 Months) ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    print("Loading model...")
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
    print("Computing indicators...")
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
    print(f"Fitting scaler...")
    train_mask = (df.index <= TRAIN_END_DATE)
    
    scaler = RobustScaler()
    if train_mask.sum() > 0:
        scaler.fit(df.loc[train_mask, features])
        scaled_features = scaler.transform(df[features])
    else:
        scaled_features = scaler.fit_transform(df[features])
    
    df_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
    
    # --- Prepare Data for Past 3 Months ---
    # We want to test on the last 90 days.
    cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90)
    print(f"Testing period starting: {cutoff_date}")
    
    # Filter Indices > Cutoff
    # We need to ensure we have enough lookback for windowing
    # So we take the slice of the dataframe that is relevant
    
    # Find the integer index where date > cutoff_date
    # We'll stick to generating ALL sequences and then filtering, it's safer logic-wise
    
    X_all = []
    indices_all = []
    
    # We generate X for validation on the fly to save memory/time or just loop
    # Actually, let's just generate the test batch.
    
    # Find start index
    test_start_idx = df.index.searchsorted(cutoff_date)
    # Ensure variables exist for lookback
    if test_start_idx < WINDOW_SIZE:
        test_start_idx = WINDOW_SIZE
        
    print(f"Processing {len(df) - test_start_idx} potential test candles...")
    
    X_test = []
    test_indices = []
    
    # Collecting data
    for i in range(test_start_idx, len(df) - PREDICT_SIZE):
        X_seq = df_scaled.iloc[i - WINDOW_SIZE : i].values
        X_test.append(X_seq)
        test_indices.append(i)
        
    X_test = np.array(X_test)
    
    if len(X_test) == 0:
        print("No test data found.")
        return

    print("Running predictions...")
    predictions = model.predict(X_test, verbose=0).flatten()
    
    # --- Analysis ---
    print("\nAnalyzing Results...")
    
    results = []
    
    for idx_in_test_array, df_idx in enumerate(test_indices):
        prob = predictions[idx_in_test_array]
        
        # Current Candle (t) info
        current_close = df['Close'].iloc[df_idx]
        
        # Future Candle (t+1) info
        future_high = df['High'].iloc[df_idx + PREDICT_SIZE]
        future_low = df['Low'].iloc[df_idx + PREDICT_SIZE]
        future_close = df['Close'].iloc[df_idx + PREDICT_SIZE]
        
        # Determine Signal
        signal = "LONG" if prob > 0.5 else "SHORT"
        confidence = prob if signal == "LONG" else (1 - prob)
        
        # Evaluate Wins
        # Condition 1: Pure Close-to-Close Win
        is_close_win = False
        if signal == "LONG" and future_close > current_close:
            is_close_win = True
        elif signal == "SHORT" and future_close < current_close:
            is_close_win = True
            
        # Condition 2: Intra-Hour "Noticeable" Win
        # Check against thresholds
        win_hits = {}
        for th in TARGET_THRESHOLDS:
            hit = False
            if signal == "LONG":
                # Did price go up by th%?
                target_price = current_close * (1 + th)
                if future_high >= target_price:
                    hit = True
            else: # SHORT
                # Did price go down by th%?
                target_price = current_close * (1 - th)
                if future_low <= target_price:
                    hit = True
            win_hits[th] = hit

        results.append({
            'timestamp': df.index[df_idx],
            'signal': signal,
            'confidence': confidence,
            'close_win': is_close_win,
            'hits': win_hits
        })
        
    results_df = pd.DataFrame(results)
    
    # --- Reporting ---
    print("\n" + "="*50)
    print(f"TEST RESULTS (Past 90 Days) - {len(results_df)} samples")
    print("="*50)
    
    # Confidence Bins
    bins = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for min_conf in bins:
        subset = results_df[results_df['confidence'] >= min_conf]
        
        if len(subset) == 0:
            continue
            
        print(f"\n--- Confidence >= {min_conf*100:.0f}% ({len(subset)} trades) ---")
        
        # Standard Directional Accuracy (Close-to-Close)
        acc = subset['close_win'].mean()
        print(f"Directional Accuracy (Close > Close): {acc*100:.2f}%")
        
        # Threshold Wins
        for th in TARGET_THRESHOLDS:
            win_rate = subset['hits'].apply(lambda x: x[th]).mean()
            print(f"Intra-Hour Hit Rate (Target {th*100:.1f}%): {win_rate*100:.2f}%")

    print("\nNote: 'Intra-Hour Hit Rate' means the price reached the profit target (High/Low) within the next hour.")

if __name__ == "__main__":
    run_test()
