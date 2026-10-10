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
TRAIN_END_DATE = "2025-12-01" # Matches the original training script
INTERVAL = '1h'
WINDOW_SIZE = 48
PREDICT_SIZE = 1
THRESHOLD = 0.008

# Model Path (Absolute path to be safe, or relative if run from dir)
MODEL_PATH = "Hamdi_BTCST_Advanced_3.0.keras"

# --- Helper Functions (Copied from Education Decision - Advanced 3.0.py) ---

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
    print(f"--- Starting Accuracy Check for Past 3 Months ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Downloading data...")
    # Download up to now
    try:
        df = yf.download(SYMBOL, start=START_DATE, interval=INTERVAL, auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    if df.empty:
        print("Error: Downloaded dataframe is empty.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Downloaded {len(df)} rows. Range: {df.index.min()} to {df.index.max()}")

    # --- Feature Engineering ---
    print("Computing indicators...")
    df['RSI'] = compute_rsi(df['Close'])
    macd, signal_line = compute_macd(df['Close'])
    upper_bb, lower_bb = compute_bollinger_bands(df['Close'])
    atr = compute_atr(df['High'], df['Low'], df['Close'])
    sma_50 = df['Close'].rolling(window=50).mean()
    ema_200 = df['Close'].ewm(span=200, adjust=False).mean()

    # Create Stationary Features
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # Handle zero volume
    df['Log_Volume_Change'] = np.log((df['Volume'] + 1) / (df['Volume'].shift(1) + 1))
    
    df['RSI_Scaled'] = df['RSI'] / 100.0
    df['MACD_Norm'] = macd / df['Close']
    df['Signal_Norm'] = signal_line / df['Close']
    df['BB_Width'] = (upper_bb - lower_bb) / df['Close']
    df['BB_Position'] = (df['Close'] - lower_bb) / (upper_bb - lower_bb)
    df['SMA_50_Dist'] = (df['Close'] - sma_50) / sma_50
    df['EMA_200_Dist'] = (df['Close'] - ema_200) / ema_200
    df['ATR_Pct'] = atr / df['Close']
    
    # Cyclic Time Features
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
    # To properly simulate the trained model, we fit the scaler on the 'training period' 
    # and transform the whole dataset.
    print(f"Fitting scaler on training period (up to {TRAIN_END_DATE})...")
    train_mask = (df.index <= TRAIN_END_DATE)
    
    if train_mask.sum() == 0:
        print("Warning: No data found for training period to fit scaler. Fitting on all data.")
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(df[features])
    else:
        scaler = RobustScaler()
        scaler.fit(df.loc[train_mask, features])
        # Transform ALL data
        scaled_features = scaler.transform(df[features])
    
    df_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
    
    # --- Sequence Creation ---
    X = []
    y = []
    indices = []

    closed_prices = df['Close'].values
    indices_vals = df.index
    
    print("Creating sequences...")
    for i in range(WINDOW_SIZE, len(df_scaled) - PREDICT_SIZE):
        X_seq = df_scaled.iloc[i - WINDOW_SIZE : i].values
        
        current_close = closed_prices[i]
        future_close = closed_prices[i + PREDICT_SIZE]
        pct_change = (future_close - current_close) / current_close
        
        # Apply Threshold Filter
        if abs(pct_change) < THRESHOLD:
            continue
            
        label = 1 if pct_change > 0 else 0
        
        X.append(X_seq)
        y.append(label)
        indices.append(indices_vals[i])

    X = np.array(X)
    y = np.array(y)
    indices = np.array(indices)
    
    print(f"Total samples generated: {len(X)}")

    # --- Slicing for Past 3 Months ---
    # Ensure cutoff_date is timezone-aware (UTC) to match yfinance data
    cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90)
    print(f"Filtering for data since {cutoff_date}...")
    
    mask = indices >= cutoff_date
    X_test = X[mask]
    y_test = y[mask]
    
    if len(X_test) == 0:
        print("No samples found in the past 3 months matching the threshold criteria.")
        return

    print(f"Testing on {len(X_test)} samples (filtered by threshold {THRESHOLD})...")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("="*40)
    print(f"RESULTS FOR PAST 3 MONTHS")
    print(f"Range: {indices[mask].min()} to {indices[mask].max()}")
    print(f"Samples: {len(X_test)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Loss: {loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_test()
