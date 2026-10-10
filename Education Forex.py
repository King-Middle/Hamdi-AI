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
import pandas as pd


epoch_value = 500

loss_limit = 30

# 1. Veri Çek
start_date = "2024-07-01"
end_date = "2026-01-01"

symbol = "eurusd=x"

data = yf.download(symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)


# 🔧 Ek makro göstergeler

macro_symbols_forex = {
    'VIX': '^VIX',              
    'US10Y': '^TNX',            
    'DXY': 'UUP',               
    'XAUUSD': 'GC=F',           
    'SPX': '^GSPC',             
    'EuroStoxx50': 'FEZ', 
    'Brent': 'BZ=F'             
}



macro_data = {}
for name, ticker in macro_symbols_forex.items():
    temp = yf.download(ticker, start=start_date, end=end_date, interval='1h', auto_adjust=True)[['Close']]
    temp.rename(columns={'Close': name}, inplace=True)
    macro_data[name] = temp

for name, df in macro_data.items():
    data = data.join(df, how='outer')

#data.dropna(inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.loc[data['Close'].notna()]
data.dropna(inplace=True)

# 2. Normalize Et

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 72
predict_size = 10
X = []
y_raw = []
threshold = 0.1  # %1 altında değişimleri yok say (gürültü filtreleme)

for i in range(window_size, len(scaled_data) - predict_size):
    X.append(scaled_data[i - window_size:i])
    
    current_close = data['Close'].values[i]
    future_highs = data['High'].values[i : i + predict_size]
    future_lows = data['Low'].values[i : i + predict_size]
    
    # Fiyat yerine şu anki fiyata göre yüzde kaç oynadığını hesaplıyoruz
    pct_max_high = ((np.max(future_highs) - current_close) / current_close) * 100
    pct_min_low = ((np.min(future_lows) - current_close) / current_close) * 100
    
    y_raw.append([pct_max_high, pct_min_low])

X = np.array(X)
y = np.array(y_raw)

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
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(2, activation='linear')  # Aktivasyon yok, regresyon çıktısı
])

#model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_EURUSD_Forex.keras')




model.compile(
    optimizer='adamw',
    loss='mse',
)

# 7. Loss Limit
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=loss_limit,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=0.5,      
    patience=20,       
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

for i in range(min(50, len(predictions))): # Terminali boğmamak için ilk 20 tane
    print(f"Tahmin -> High: {predictions[i][0]:.4f}, Low: {predictions[i][1]:.4f} | "
          f"Gerçek -> High: {y_test[i][0]:.4f}, Low: {y_test[i][1]:.4f}")

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

# Error grafiği
plt.figure(figsize=(15, 8))
plt.plot(np.abs(y_test[:epoch_value] - predictions[:epoch_value]), label='Error')
plt.title('Error Graph')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()


# 10. Save Model
while True:
    SaveInput = input("kaydedilsin mi \n(evet / hayır)\n").lower()

    if SaveInput == "evet":
        model_symbol = input("Borsa kodunu girin: ").upper()
        model_name = "Hamdi_" + model_symbol + "_Forex.keras"
        model.save(model_name)
        break

    elif SaveInput == "hayır":
        break

    else:
        print("evet veya hayır olarak cevap girin!")


pid = os.getppid()
os.kill(pid, signal.SIGKILL)