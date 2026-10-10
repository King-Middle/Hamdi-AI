import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import AdamW, Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
import joblib
import os
import signal
import sys


epoch_value = 10

loss_limit = 30

# 1. Veri Çek

data = pd.read_csv('BTCUSD_15m_ready.csv', sep=';')

needed_collumns = [
    'Open_pct', 
    'High_pct', 
    'Low_pct', 
    'Close_pct', 
    'Volume_pct', 
    'ATR_pct',
    'MACD', 
    'RSI',
    'STOCHk',
    'BBP'
]

X_raw = data[needed_collumns].values
y_raw = data['Target'].values

data = data[needed_collumns]

# 2. Sliding Window için ham veri kullan (scaling sonraya alındı)


# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 96
X = []
y = []
threshold = 0  # %1 altında değişimleri yok say (gürültü filtreleme)

data_values = data.values

for i in range(window_size, len(data)):
    if abs(y_raw[i - 1]) < threshold:
        continue  # Gürültü say, atla

    X.append(data_values[i - window_size:i])
    y.append(y_raw[i - 1])

X = np.array(X)
y = np.array(y)

# 4. Dağılım ve Boyut Kontrolü
print("Örnek sayısı:", len(y))
print("X shape:", X.shape)
print("y shape:", y.shape)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=None
)

# Normalize Et (split sonrası — sadece train'e fit)
scaler = MinMaxScaler()
n_train, w, f = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n_train, w, f)
n_test = X_test.shape[0]
X_test = scaler.transform(X_test.reshape(-1, f)).reshape(n_test, w, f)

# 6. Model Oluştur
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(16),
    Dense(1)
])




model.compile(
    optimizer=AdamW(learning_rate=0.001),
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
    patience=10,       
    min_lr=0.00001,    
)



# 8. Eğitim
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_value,
    callbacks=[early_stop, lr_scheduler],
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 9. Tahmin ve Görsel Kontrol (isteğe bağlı)
predictions = model.predict(X_test)

for i in range(30):
    print(f"Guess: {predictions[i][0]:.4f} %, Reality: {y_test[i].item():.4f} %")

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
plt.plot(predictions[:200], label='Guess')
plt.plot(y_test[:200], label='Reality')
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
        model_name = "Hamdi_" + model_symbol + "_5.0.keras"
        model.save(model_name)
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