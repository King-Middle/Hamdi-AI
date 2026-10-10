import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
import joblib
import os
import signal
import sys
import joblib


epoch_value = 500

loss_limit = 30

# 1. Veri Çek
start_date = "2015-01-01"
end_date = "2025-01-01"

symbol = "aapl"

data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)

# 🔧 Ek makro göstergeler

macro_symbols = {
    # --- Risk & Volatilite ---
    'VIX': '^VIX',             # Volatilite endeksi

    # --- Tahviller & Faiz ---
    'US10Y': '^TNX',           # ABD 10 yıllık tahvil faizi
    'US3M': '^IRX',            # ABD 3 aylık hazine bonosu
    'US30Y': '^TYX',           # ABD 30 yıllık tahvil faizi

    # --- Emtialar ---
    'XAUUSD': 'GC=F',          # Altın
    'XAGUSD': 'SI=F',          # Gümüş
    'WTI': 'USO',              # Ham Petrol (WTI)
    'Brent': 'BZ=F',           # Brent Petrol
    'NaturalGas': 'NG=F',      # Doğalgaz
    'Copper': 'HG=F',          # Bakır
    'Platinum': 'PL=F',        # Platin
    'Palladium': 'PA=F',       # Paladyum

    # --- ABD Endeksleri ---
    'SPX': '^GSPC',            # S&P 500
    'DJI': '^DJI',             # Dow Jones
    'IXIC': '^IXIC',           # Nasdaq
    'RUT': '^RUT',             # Russell 2000

    # --- Global Endeksler ---
    'FTSE': '^FTSE',           # İngiltere FTSE 100
    'Nikkei': '^N225',         # Japonya Nikkei 225
    'HangSeng': '^HSI',        # Hong Kong Hang Seng
    'EuroStoxx50': 'FEZ',# Euro Bölgesi STOXX50
}

macro_data = {}
for name, ticker in macro_symbols.items():
    temp = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True)[['Close']]
    temp.rename(columns={'Close': name}, inplace=True)
    macro_data[name] = temp

for name, df in macro_data.items():
    data = data.join(df, how='inner')

data.dropna(inplace=True)

# 2. Normalize Et

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 30
predict_size = 3
X = []
y = []
threshold = 0.0  # %1 altında değişimleri yok say (gürültü filtreleme)

for i in range(window_size, len(scaled_data) - predict_size):
    original_close = data['Close'].values
    current_close = original_close[i]
    next_close = original_close[i + predict_size]
    if current_close == 0:
        continue  # NaN veya bölünemez durumları atla

    pct_change = ((next_close - current_close) / current_close) * 100

    if abs(pct_change) < threshold:
        continue  # Gürültü say, atla

    X.append(scaled_data[i - window_size:i])
    y.append(pct_change)  # Artık float değer (regresyon)

X = np.array(X)
y = np.array(y)

# 4. Dağılım ve Boyut Kontrolü
print("Örnek sayısı:", len(y))
print("X shape:", X.shape)
print("y shape:", y.shape)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=None
)

# 6. Model Oluştur
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)  # Aktivasyon yok, regresyon çıktısı
])




model.compile(
    optimizer='AdamW',
    loss='huber',
)

# 7. Loss Limit
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=loss_limit,
    restore_best_weights=True
)



lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=1/2,      
    patience=30,       
    min_lr=0.00001,    
)


# 8. Eğitim
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_value,
    callbacks=[early_stop, lr_scheduler],
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 9. Tahmin ve Görsel Kontrol (isteğe bağlı)
predictions = model.predict(X_test)

for i in range(len(predictions)):
    print(f"Guess: {predictions[i][0]:.2f} %, Reality: {y_test[i].item():.2f} %")

# Eğitim grafiği
plt.figure(figsize=(15, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss Graph')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Tutturma grafiği
plt.figure(figsize=(15, 8))
plt.plot(predictions[:epoch_value], label='Guess')
plt.plot(y_test[:epoch_value], label='Reality')
plt.axhline(y=0, linestyle='--', color='gray', label='0 Line')
plt.title('Accuracy Graph')
plt.xlabel('Epoch')
plt.ylabel('Changes')
plt.legend()
plt.show()


# 10. Save Model

while True:
    SaveInput = input("kaydedilsin mi \n(evet / hayır / tekrar / kodu bitir)\n").lower()

    if SaveInput == "evet":
        model_symbol = input("Borsa kodunu girin: ").upper()
        model_name = "Hamdi_" + model_symbol + "_2.2.keras"
        scaler_name = "Hamdi_" + model_symbol + "_2.2.pkl"
        model.save(model_name)
        joblib.dump(scaler, scaler_name)
        pid = os.getppid()
        os.kill(pid, signal.SIGKILL)

    elif SaveInput == "hayır":
        pid = os.getppid()
        os.kill(pid, signal.SIGKILL)

    elif SaveInput == "kodu bitir":
        break

    elif SaveInput == "tekrar":
        os.execv(sys.executable, [sys.executable] + sys.argv)

    else:
        print("evet veya hayır olarak cevap girin!")