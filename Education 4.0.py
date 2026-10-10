import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout, BatchNormalization, LayerNormalization, Bidirectional, Conv1D, GlobalAveragePooling1D, MultiHeadAttention, Add, Concatenate # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, AdamW # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
import joblib
import os
import signal
import sys
import tensorflow.keras.backend as K  # type: ignore


epoch_value = 500

loss_limit = 30

# 1. Veri Çek
start_date = "2015-01-01"
end_date = "2025-01-01"

symbol = "aapl"

data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)

data['Open'] = data['Open'].pct_change()
data['Volume'] = data['Volume'].pct_change()
data['High'] = data['High'].pct_change()
data['Low'] = data['Low'].pct_change()


# Calculate 4 extra features directly (RSI, MACD, MACD 1st degree derivative, MACD 2nd degree derivative)
close = data['Close']
if hasattr(close, 'squeeze'):
    close = close.squeeze()

# 1. RSI
delta = close.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
rs = avg_gain / (avg_loss + 1e-9)
data['RSI'] = 100 - (100 / (1 + rs))

# 2. MACD
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26

# 3. MACD 1st degree derivative
data['MACD_1st_derivative'] = data['MACD'].diff()

# 4. MACD 2nd degree derivative
data['MACD_2nd_derivative'] = data['MACD_1st_derivative'].diff()

data.dropna(inplace=True)

# 2. Normalize Et

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(data[['Close']].values)

# 3. Sliding Window ve Etiketleme (Regresyon için)
window_size = 20
predict_size = 1
X = []
y = []
threshold = 0.6  # %0.6 altında değişimleri yok say (gürültü filtreleme)
upper_threshold = 8.0

for i in range(window_size, len(scaled_data) - predict_size):
    original_close = data['Close'].values
    current_close = original_close[i]
    next_close = original_close[i + predict_size]
    if current_close == 0:
        continue  # NaN veya bölünemez durumları atla

    pct_change = ((next_close - current_close) / current_close) * 100

    if abs(pct_change) < threshold or abs(pct_change) > upper_threshold:
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

# Pure Directional Sign Classification Loss Function
def directional_classification_loss(y_true, y_pred):
    target_sign = K.sign(y_true)
    pred_sign = K.tanh(y_pred * 3.0)
    
    sign_mse = K.square(target_sign - pred_sign)
    sign_mismatch = K.cast(K.less(target_sign * y_pred, 0.0), K.floatx())
    
    return sign_mse + sign_mismatch * 5.0

# 6. Model Oluştur (ResNet TCN + 8-Head Attention + Bi-GRU Directional Sign Classifier)
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
normed_inputs = LayerNormalization(axis=-1)(inputs)

# 1. Residual TCN Block (Dilated Convolutions)
c1 = Conv1D(64, kernel_size=3, dilation_rate=1, activation='elu', padding='same')(normed_inputs)
c2 = Conv1D(64, kernel_size=3, dilation_rate=2, activation='elu', padding='same')(c1)
c2 = Dropout(0.2)(c2)
tcn_out = Add()([c1, c2])
tcn_out = LayerNormalization()(tcn_out)

# 2. 8-Head Self-Attention Block
attn = MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.1)(tcn_out, tcn_out)
attn_out = Add()([tcn_out, attn])
attn_out = LayerNormalization()(attn_out)

# 3. Bidirectional GRU Sequence Layer
g1 = Bidirectional(GRU(32, return_sequences=False))(attn_out)
g1 = Dropout(0.2)(g1)

# 4. Dense Directional Classifier Head
d1 = Dense(32, activation='elu')(g1)
d1 = Dropout(0.1)(d1)
outputs = Dense(1)(d1)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss=directional_classification_loss,
    metrics=['mae']
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
    patience=10,       
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

# 9. Model Değerlendirme (Evaluate) ve Tahmin
print("\n--- Test/Validation Verisi Değerlendirmesi ---")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)

print(f"\n[Değerlendirme Sonuçları]")
print(f"Test Huber Loss : {test_loss:.4f}")
print(f"Test MAE (Ortalama Hata) : %{test_mae:.2f}")

# Tahmin alma
predictions = model.predict(X_test)

# Yön Doğruluğu (Fiyatın yönü doğru tahmin edildi mi?)
correct_direction = np.sign(predictions.flatten()) == np.sign(y_test)
directional_accuracy = np.mean(correct_direction) * 100

print(f"Yön Doğruluğu (Directional Accuracy): %{directional_accuracy:.2f}")


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


SaveInput = input("kaydedilsin mi \n(evet / hayır / tekrar / kodu bitir)\n").lower()

if SaveInput == "evet":
    model_symbol = input("Borsa kodunu girin: ").upper()
    model_name = "Hamdi_" + model_symbol + "_4.0.keras"
    model.save(model_name)
    #joblib.dump(scaler, 'Hamdi_' + model_symbol + '_4.0.joblib')
    pid = os.getppid()
    os.kill(pid, signal.SIGKILL)

elif SaveInput == "hayır":
    pid = os.getppid()
    os.kill(pid, signal.SIGKILL)

elif SaveInput == "tekrar":
    os.execv(sys.executable, [sys.executable] + sys.argv)