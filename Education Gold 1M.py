import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, GRU, Flatten # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
import os
import signal
import sys
import pandas as pd
import joblib


epoch_value = 500

loss_limit = 30

# 1. Veri Çek (random contiguous chunk for speed)
sample_size = 1000000  # number of contiguous rows to use

# Count total lines without loading the whole file into memory
with open('XAU_1m_data copy.csv', 'r') as f:
    total_lines = sum(1 for _ in f) - 1  # subtract header

if total_lines <= sample_size:
    data = pd.read_csv('XAU_1m_data copy.csv', sep=';', index_col='Date', parse_dates=True)
else:
    start_row = np.random.randint(0, total_lines - sample_size)
    print(f"Sampling {sample_size} contiguous rows starting at row {start_row} (out of {total_lines:,} total)")
    data = pd.read_csv(
        'XAU_1m_data copy.csv',
        sep=';',
        index_col='Date',
        parse_dates=True,
        skiprows=range(1, start_row + 1),  # skip rows after header
        nrows=sample_size,
    )

data.dropna(inplace=True)

# 2. Sliding Window parametreleri
window_size = 100
predict_size = 8
threshold = 0  # %1 altında değişimleri yok say (gürültü filtreleme)

# 3. Train/Test Split (Ham veri üzerinde index bazlı böl)
# Scaler'ı sadece train verisine fit etmek için önce split yapıyoruz
total_usable = len(data) - window_size - predict_size
split_index = int(total_usable * 0.8) + window_size  # ham verideki bölme noktası

train_data_raw = data.iloc[:split_index + predict_size]  # train kısmı (window için önceki veriler dahil)
test_data_raw = data.iloc[:len(data)]  # test kısmı tüm veriyi görebilir (window geçmişi için)

# 4. Scaler'ı sadece train verisine fit et (Data Leakage önleme)
scaler = MinMaxScaler()
scaler.fit(train_data_raw)

# Tüm veriyi transform et (ama scaler sadece train'den öğrendi)
scaled_data = scaler.transform(data)

# 5. Sliding Window ve Etiketleme (Regresyon için)
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

original_close = data['Close'].values

for i in range(window_size, len(scaled_data) - predict_size):
    current_close = original_close[i]
    next_close = original_close[i + predict_size]
    if current_close == 0:
        continue  # NaN veya bölünemez durumları atla

    pct_change = ((next_close - current_close) / current_close) * 100

    if abs(pct_change) < threshold:
        continue  # Gürültü say, atla

    label = 1 if pct_change > 0 else 0  # 1 = up, 0 = down

    if i < split_index:
        X_train_list.append(scaled_data[i - window_size:i])
        y_train_list.append(label)
    else:
        X_test_list.append(scaled_data[i - window_size:i])
        y_test_list.append(label)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

# 6. Dağılım ve Boyut Kontrolü
print("Train örnek sayısı:", len(y_train))
print("Test örnek sayısı:", len(y_test))
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 7. Model Oluştur
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  
])




model.compile(
    optimizer='AdamW',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# 8. Loss Limit
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


# 9. Eğitim
history = model.fit(
    X_train,
    y_train,
    epochs=epoch_value,
    callbacks=[early_stop, lr_scheduler],
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 10. Tahmin ve Görsel Kontrol (isteğe bağlı)
predictions = model.predict(X_test)

pred_classes = (predictions.flatten() > 0.5).astype(int)
for i in range(len(predictions)):
    print(f"Guess: {'UP' if pred_classes[i] else 'DOWN'} ({predictions[i][0]:.2f}), Reality: {'UP' if y_test[i] else 'DOWN'}")

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
plt.plot(pred_classes[:epoch_value], label='Guess')
plt.plot(y_test[:epoch_value], label='Reality')
plt.axhline(y=0.5, linestyle='--', color='gray', label='Threshold (0.5)')
plt.title('Classification Graph')
plt.xlabel('Sample')
plt.ylabel('Class (0=Down, 1=Up)')
plt.legend()
plt.show()

# 11. Classification Accuracy Evaluation
print("\n" + "="*60)
print("CLASSIFICATION ACCURACY EVALUATION")
print("="*60)

correct = np.sum(pred_classes == y_test)
total_predictions = len(y_test)
overall_accuracy = (correct / total_predictions) * 100

# Detaylı analiz
true_up = np.sum(y_test == 1)
true_down = np.sum(y_test == 0)

pred_up_correct = np.sum((pred_classes == 1) & (y_test == 1))
pred_down_correct = np.sum((pred_classes == 0) & (y_test == 0))

up_accuracy = (pred_up_correct / true_up * 100) if true_up > 0 else 0
down_accuracy = (pred_down_correct / true_down * 100) if true_down > 0 else 0

print(f"\nToplam Test Örneği: {total_predictions}")
print(f"Doğru Tahmin: {correct}")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print(f"\nYukarı (Up) Örnekler: {true_up}  |  Doğru Tahmin: {pred_up_correct}  |  Accuracy: {up_accuracy:.2f}%")
print(f"Aşağı (Down) Örnekler: {true_down}  |  Doğru Tahmin: {pred_down_correct}  |  Accuracy: {down_accuracy:.2f}%")
print("="*60)

# Classification Accuracy Bar Chart
plt.figure(figsize=(8, 5))
categories = ['Overall', 'Up (Yukarı)', 'Down (Aşağı)']
accuracies = [overall_accuracy, up_accuracy, down_accuracy]
colors = ['#4A90D9', '#2ECC71', '#E74C3C']
bars = plt.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=0.8)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%',
             ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 110)
plt.axhline(y=50, linestyle='--', color='gray', alpha=0.7, label='Random (50%)')
plt.title('Classification Accuracy')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

# 12. Save Model

while True:
    SaveInput = input("kaydedilsin mi \n(evet / hayır / tekrar / kodu bitir)\n").lower()

    if SaveInput == "evet":
        model_symbol = input("Borsa kodunu girin: ").upper()
        model_name = "Hamdi_" + model_symbol + "_Gold.keras"
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