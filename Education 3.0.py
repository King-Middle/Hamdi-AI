import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, SimpleRNN, GRU, BatchNormalization  # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import AdamW # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import Huber #type: ignore
import joblib
import os
import signal


epoch_value = 100

# 1. Veri Çek
start_date = "2024-03-01"
end_date = "2025-03-01"

symbol = "BTC-USD"

data = yf.download(symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)

# 🔧 Ek makro göstergeler

data.dropna(inplace=True)

# 2. Normalize Et

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 18
predict_size = 1
X = []
y = []
threshold = 0.5  # %1 altında değişimleri yok say (gürültü filtreleme)

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
    LSTM(64),
    Dense(1)  # Aktivasyon yok, regresyon çıktısı
])

#model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_BTC2_Anaconda_3.0.keras')




model.compile(
    optimizer='AdamW',
    loss='mae'
)

# 7. Eğitim
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_value,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 8. Tahmin ve Görsel Kontrol (isteğe bağlı)
predictions = model.predict(X_test)

for i in range(len(predictions)):
    print(f"Guess: {predictions[i][0]:.2f} %, Reality: {y_test[i].item():.2f} %")

# Opsiyonel: Eğitim grafiği
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



DirectionList = np.sign(predictions)
DirectionListReality = np.sign(y_test)
DirectionAccuracyList = []
for i in range(len(DirectionList)):
    if DirectionListReality[i] == DirectionList[i]:
        DirectionAccuracyList.append(1)
    else:
        DirectionAccuracyList.append(0)

# Direction grafiği
plt.figure(figsize=(15, 8))
plt.plot(DirectionAccuracyList[:epoch_value], label='Accuracy')
plt.title(f"Direction Graph\nAccuracy: {(DirectionAccuracyList.count(1)/len(DirectionAccuracyList)*100):.2f}%")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



while True:
    SaveInput = input("kaydedilsin mi \n(evet / hayır)\n").lower()

    if SaveInput == "evet":
        model_symbol = input("Borsa kodunu girin: ").upper()
        model_name = "Hamdi_" + model_symbol + "_Anaconda_3.0.keras"
        model.save(model_name)
        break

    elif SaveInput == "hayır":
        break

    else:
        print("evet veya hayır olarak cevap girin!")


os.kill(os.getppid(), signal.SIGKILL)