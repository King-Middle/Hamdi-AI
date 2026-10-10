import yfinance as yf
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, SimpleRNN, GRU, BatchNormalization, Dropout, LeakyReLU, Bidirectional, LayerNormalization, GaussianNoise # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, AdamW # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
import joblib
import os
import sys
import signal
import pandas as pd
from datetime import datetime
#import pandas_ta as ta


epoch_value = 500

loss_limit = 30

# 1. Veri Çek
start_date = "2025-01-01"
end_date = "2026-03-01"

symbol = "EURUSD=X"

data = yf.download(symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)

def add_indicators(df):
    # EMA (20 günlük)
    df['EMA_20'] = round(df['Close'].ewm(span=20, adjust=False).mean(), 4)
    
    # RSI (14 günlük) - Güvenli hesaplama
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # + 1e-10 ekleyerek sıfıra bölünmeyi önledik
    rs = gain / (loss + 1e-10) 
    df['RSI'] = round(100 - (100 / (1 + rs)), 4)
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = round(exp1 - exp2, 4)
    df['Signal_Line'] = round(df['MACD'].ewm(span=9, adjust=False).mean(), 4)
    
    return df.round(4)

data = add_indicators(data)

data.fillna(0, inplace=True)

data.ffill(inplace=True)
data.bfill(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.loc[data['Close'].notna()]
data.dropna(inplace=True)

# 2. Normalize Et
data_diff = data.assign(Close=data['Close'].pct_change().fillna(0))

# Scaler'ı -1 ile 1 arasına çekiyoruz
scaler = MinMaxScaler(feature_range=(-1, 1)) 
scaled_data = scaler.fit_transform(data_diff)

# 3. Sliding Window ve Etiketleme
window_size = 24
predict_size = 4
X = []
y = []
threshold = 0.05  # Senin istediğin gibi sabit tuttuk

for i in range(window_size, len(scaled_data) - predict_size):
    original_close = data['Close'].values
    current_close = original_close[i]
    next_close = original_close[i + predict_size]
    if current_close == 0:
        continue  # NaN veya bölünemez durumları atla

    pct_change = ((next_close - current_close) / current_close) * 1000

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
    X, y, test_size=0.2, shuffle=False, random_state=None
)

y_train = y_train.flatten()
y_test = y_test.flatten()

# 6. Model Oluştur
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Dense(128),
    Dropout(0.2),
    Dense(64),
    Dense(1)
])



model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss='huber',
    metrics=['mae']
)

# 7. Loss Limit
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=loss_limit,
    restore_best_weights=True,
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


# test
predictions = model.predict(X_test)


for i in range(min(50, len(predictions))):
    print(f"Guess: {predictions[i][0]:.2f}, Reality: {y_test[i]:.2f}")

AccuracyScore = 0

for i in range(len(predictions)):
    if np.sign(predictions[i]) == np.sign(y_test[i]):
        AccuracyScore += 1

print("***********************")
print(f"Accuracy: {(AccuracyScore/len(predictions)*100):.2f}")
print("***********************")


FinalAccuracyScore = 0

final_test_data = yf.download(symbol, start='2026-03-01', end='2026-06-15', interval='1h', auto_adjust=True)

final_test_data = add_indicators(final_test_data)
final_test_data.fillna(0, inplace=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
final_test_data = final_test_data.assign(Close=final_test_data['Close'].pct_change().fillna(0))

scaled_data_final = scaler.transform(final_test_data)

X_final = []
y_final = []

for i in range(window_size, len(final_test_data) - predict_size):
    X_final.append(scaled_data_final[i - window_size:i]) 
    
    current_close = final_test_data['Close'].values[i]
    next_close = final_test_data['Close'].values[i + predict_size]
    y_final.append((next_close - current_close) / current_close)

X_final = np.array(X_final)
y_final = np.array(y_final)

final_predictions = model.predict(X_final)

correct_predictions = 0
for i in range(len(final_predictions)):
    if np.sign(final_predictions[i][0]) == np.sign(y_final[i]):
        correct_predictions += 1

final_accuracy = (correct_predictions / len(final_predictions)) * 100

print("***********************")
print(f"Real Accuracy: {final_accuracy:.2f}%")
print("***********************")



# Eğitim grafiği
plt.figure(figsize=(15, 8))
plt.plot(history.history['mae'], label='Train Loss')
plt.plot(history.history['val_mae'], label='Val Loss')
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
        model_name = "Hamdi_" + model_symbol + "_Forex_2.0.keras"
        model.save(model_name)
        joblib.dump(scaler, 'Hamdi_' + model_symbol + '_ForexScaler_2.0.joblib')
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

