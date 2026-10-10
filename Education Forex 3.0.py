import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- AYARLAR ---
symbol = "EURUSD=X"
window_size = 60  
threshold = 0.05 # Sinyal filtresi: çok küçük oynamaları gürültü olarak ata

# 1. VERİ & FEATURE ENGINEERING (Daha derin özellikler)
data = yf.download(symbol, start="2025-01-01", end="2026-06-19", interval='1h', auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
data.ffill(inplace=True); data.bfill(inplace=True)

# RSI ve Volatilite ekliyoruz (Mekanik zekayı artırmak için)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))
data['EMA_200'] = data['Close'].ewm(span=200).mean()
data['Volatility'] = data['Close'].pct_change().rolling(window=14).std()

data.fillna(0, inplace=True)
features = ['Close', 'RSI', 'EMA_200', 'Volatility']
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[features])

# 2. SINIFLANDIRMA (YUKARI: 1, AŞAĞI: 0)
X, y = [], []
close = data['Close'].values
for i in range(window_size, len(data_scaled) - 1):
    # Eğer 1 saat sonra fiyat yükseliyorsa 1, düşüyorsa 0
    if abs(close[i+1] - close[i]) < threshold: continue
    X.append(data_scaled[i-window_size:i])
    y.append(1 if close[i+1] > close[i] else 0)

X, y = np.array(X), np.array(y)

# 3. MODEL: Classification Architect
model = Sequential([
    Input(shape=(window_size, len(features))),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Sınıflandırma için Sigmoid
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 4. EĞİTİM
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, 
          callbacks=[EarlyStopping(monitor='val_accuracy', patience=15, mode='max', restore_best_weights=True)])

# 5. TEST (Net Performans)
print("\n--- Hamdi V4 Sınıflandırma Performansı ---")
test_data = data.iloc[-window_size-100:]
test_scaled = scaler.transform(test_data[features])
X_test, y_test = [], []
close_test = test_data['Close'].values
for i in range(window_size, len(test_scaled)-1):
    X_test.append(test_scaled[i-window_size:i])
    y_test.append(1 if close_test[i+1] > close_test[i] else 0)

preds = (model.predict(np.array(X_test), verbose=0) > 0.5).astype(int)
acc = np.mean(preds.flatten() == np.array(y_test))
print(f"Sınıflandırma Başarı Oranı: %{acc*100:.2f}")

if input("Kaydedilsin mi? (e/h): ").lower() == 'e':
    model.save("Hamdi_V4_Classifier.keras")