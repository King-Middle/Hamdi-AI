import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import joblib
import os
import signal


epoch_value = 100

# 1. Data date
start_date = "2020-01-01"
end_date = "2025-11-01"

symbol = "AAPL"

data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)

# Additional macro econommics symbols
macro_symbols = {
    'VIX': '^VIX',        # Volatility index
    'DXY': 'DX-Y.NYB',    # Dolar index
    'US10Y': '^TNX',      # CBOE Interest Rate 10 Years
    'XAUUSD': 'GC=F',     # Gold
    'SPX': '^GSPC',       # S&P 500
    'DJI': '^DJI',        # Dow Jones
    'IXIC': '^IXIC',      # Nasdaq
    'RUT': '^RUT',        # Russell 2000
}

macro_data = {}
for name, ticker in macro_symbols.items():
    temp = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True)[['Close']]
    temp.rename(columns={'Close': name}, inplace=True)
    macro_data[name] = temp

for name, df in macro_data.items():
    data = data.join(df, how='inner')

data.dropna(inplace=True)

# 2. Normalization

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

# 3. Sliding Window and Tagging (Regretion)
window_size = 180
predict_size = 10
X = []
y = []
threshold = 0.5  # Ignore changes below 0.5% (noise filtering)

for i in range(window_size, len(scaled_data) - predict_size):
    original_close = data['Close'].values
    current_close = original_close[i]
    next_close = original_close[i + predict_size]
    if current_close == 0:
        continue  

    pct_change = ((next_close - current_close) / current_close) * 100

    if abs(pct_change) < threshold:
        continue  # Noise filtering

    X.append(scaled_data[i - window_size:i])
    y.append(pct_change)  # Regretion

X = np.array(X)
y = np.array(y)

# 4. Size check
print("Example count:", len(y))
print("X shape:", X.shape)
print("y shape:", y.shape)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=None
)

# 6. Create model (Important!) (Do not change this part!)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1)  # Regretion Output
])



model.compile(
    optimizer='adam',
    loss='mse'
)

# 7. Education
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_value,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 8. Guess for the visual
predictions = model.predict(X_test)

for i in range(len(predictions)):
    print(f"Guess: {predictions[i][0]:.2f} %, Reality: {y_test[i].item():.2f} %")

# Optional: Education graph
plt.figure(figsize=(15, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Value Loss')
plt.title('Training Loss Graph (Save the file if both Train loss and Value loss both decrease!)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Optional: Accuracy graph
plt.figure(figsize=(15, 8))
plt.plot(predictions[:epoch_value], label='Guess')
plt.plot(y_test[:epoch_value], label='Reality')
plt.axhline(y=0, linestyle='--', color='gray', label='0 Line')
plt.title('Accuracy Graph')
plt.xlabel('Epoch')
plt.ylabel('Changes')
plt.legend()
plt.show()

# Optional: Error Graph
plt.figure(figsize=(15, 8))
plt.plot(np.abs(y_test[:epoch_value] - predictions[:epoch_value]), label='Error')
plt.title('Error Graph')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()


while True:
    SaveInput = input("Save it if both Train loss and Value loss both decrease! (Important!)\nSave \n(yes / no)\n").lower()

    if SaveInput == "yes":
        model_symbol = input("Stock Symbol: ").upper()
        model_name = "Hamdi_" + model_symbol + "_Anaconda.keras"
        model.save(model_name)
        break

    elif SaveInput == "no":
        break

    else:
        print("answer as yes or no!")
