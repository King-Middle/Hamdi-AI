import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type:ignore
import joblib
import os
import signal


NumberOfRepetition = 10

NumberToNote = 0

epoch_value = 100

loss_limit = 10

# 1. Veri Çek
start_date = "2020-01-01"
end_date = "2025-03-01"

symbol = "goog"

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
    'WTI': 'USO',             # Ham Petrol (WTI)
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

    # --- Kripto ---
    'BTC': 'BTC-USD',          # Bitcoin
    'ETH': 'ETH-USD',          # Ethereum
    'SOL': 'SOL-USD',          # Solana
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

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 180
predict_size = 10
X1 = []
y1 = []
threshold = 0.5  # %1 altında değişimleri yok say (gürültü filtreleme)

for i in range (1,NumberOfRepetition+1):
    NumberToNote += 1

    for i in range(window_size, len(scaled_data) - predict_size):
        original_close = data['Close'].values
        current_close = original_close[i]
        next_close = original_close[i + predict_size]
        if current_close == 0:
            continue  # NaN veya bölünemez durumları atla

        pct_change = ((next_close - current_close) / current_close) * 100

        if abs(pct_change) < threshold:
            continue  # Gürültü say, atla

        X1.append(scaled_data[i - window_size:i])
        y1.append(pct_change)  # Artık float değer (regresyon)

    X = np.array(X1)
    y = np.array(y1)

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
        optimizer='adam',
        loss='mse'
    )

    # 7. Loss Limit
    early_stop = EarlyStopping(
        monitor = 'val_loss',
        patience=loss_limit,
        return_best_weights=True
    )

    # 8. Eğitim
    history = model.fit(
        X_train,
        y_train,
        epochs=epoch_value,
        callbacks=early_stop,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 9. Tahmin ve Görsel Kontrol (isteğe bağlı)
    predictions = model.predict(X_test)

    #for i in range(len(predictions)):
        #print(f"Guess: {predictions[i][0]:.2f} %, Reality: {y_test[i].item():.2f} %")

    modelname = "Hamdi_SeriÜretim_"+str(NumberToNote+40)+"_"+symbol.upper()+"-Version_2.0.keras"
    model.save(modelname)
