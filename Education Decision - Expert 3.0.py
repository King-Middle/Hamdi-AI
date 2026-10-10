import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Flatten, Activation #type:ignore
from tensorflow.keras.optimizers import AdamW #type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type:ignore
from tensorflow.keras.regularizers import l2 #type:ignore
import matplotlib.pyplot as plt
import os
import signal

# --- Configuration ---
# Expert Model Configuration
SYMBOL = "BTC-USD"
START_DATE = "2024-03-01" 
END_DATE = "2025-12-01"
INTERVAL = '1h'

WINDOW_SIZE = 48  # 2 days context (Optimal for MLP)
PREDICT_SIZE = 1 
THRESHOLD = 0.008 # 0.8% change threshold (Best balance of signal/noise)

EPOCHS = 200     # Higher epochs
BATCH_SIZE = 32  # Smaller batch size for robustness

# --- Helper Functions ---

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

# --- 1. Data Collection & Feature Engineering ---

print("Downloading data...")
try:
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=True)
except Exception as e:
    print(f"Error: {e}")
    df = pd.DataFrame()

if df.empty:
    raise ValueError("No data downloaded!")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"Initial shape: {df.shape}")

# Calculate Core Technical Indicators
df['RSI'] = compute_rsi(df['Close'])
macd, signal_line = compute_macd(df['Close'])
upper_bb, lower_bb = compute_bollinger_bands(df['Close'])
atr = compute_atr(df['High'], df['Low'], df['Close'])
sma_50 = df['Close'].rolling(window=50).mean()
ema_200 = df['Close'].ewm(span=200, adjust=False).mean()

# Stationary Features (Robust for ML)
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

# Time Features
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

print(f"Using {len(features)} Features")

# --- Scaling ---
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df[features])

# --- Sequence Creation ---
X = []
y = []
closed_prices = df['Close'].values

for i in range(WINDOW_SIZE, len(scaled_data) - PREDICT_SIZE):
    X.append(scaled_data[i - WINDOW_SIZE : i])
    current_close = closed_prices[i]
    future_close = closed_prices[i + PREDICT_SIZE]
    pct_change = (future_close - current_close) / current_close
    
    if abs(pct_change) < THRESHOLD:
        X.pop()
        continue
        
    y.append(1 if pct_change > 0 else 0)

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution: 1: {sum(y)}, 0: {len(y)-sum(y)}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=True, random_state=42)

# --- Architecture: Optimized MLP ---
# Uses Flatten + Dense for best performance on technical indicators
input_shape = (X_train.shape[1], X_train.shape[2])

model = Sequential([
    Input(shape=input_shape),
    Flatten(),
    
    Dense(64, use_bias=False, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    
    Dense(32, use_bias=False, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Training Strategy ---
# Solved "Stops at 20" by increasing patience
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=25, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=8, 
    min_lr=1e-6, 
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model_expert.keras", 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# --- Evaluation ---
print("\nEvaluating...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- Visualization ---
try:
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title(f'Accuracy (Final: {accuracy*100:.2f}%)')
    ax2.legend()
    
    plt.show()
except Exception:
    pass

while True:
    save_input = input("Save this EXPERT model? (yes/no)\n").lower()
    if save_input in ['yes', 'y', 'evet']:
        name = input("Enter model suffix name (e.g. 'ExpertV1'): ")
        model.save(f"Hamdi_{name}_Expert_3.0.keras")
        print("Saved.")
        break
    elif save_input in ['no', 'n', 'hayir']:
        break

os.kill(os.getppid(), signal.SIGKILL)
