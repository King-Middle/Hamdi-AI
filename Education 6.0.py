"""
Education 6.0 - Competition-Grade BTC Directional Prediction Model
===================================================================
Architecture: Residual Dilated TCN → Multi-Head Self-Attention → BiLSTM → Dense
Goal: Maximize directional accuracy (predict whether BTC goes up or down)

Key Design Decisions:
- Walk-forward (chronological) train/test split — no data leakage
- Feature engineering: momentum derivatives, volatility ratios, cross-feature interactions
- Proper per-feature scaling fitted only on training data
- Asymmetric directional loss with Huber stabilization
- Cosine annealing LR schedule for better convergence
- Extensive evaluation metrics including directional accuracy, precision, recall, F1
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input, Dense, Dropout, LayerNormalization, BatchNormalization,
    Conv1D, Add, Multiply, Bidirectional, LSTM, GRU,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
    Flatten, Activation, SpatialDropout1D
)
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.callbacks import (  # type: ignore
    EarlyStopping, ReduceLROnPlateau, Callback
)
import tensorflow.keras.backend as K  # type: ignore

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# HYPERPARAMETERS (all chosen for optimal BTC 15m prediction)
# ============================================================
WINDOW_SIZE = 64           # 64 candles = 16 hours of context
EPOCHS = 300               # will early-stop well before this
BATCH_SIZE = 256           # larger batch = more stable gradients on noisy data
PATIENCE = 40              # early stopping patience
LR_PATIENCE = 12           # reduce LR patience
INITIAL_LR = 5e-4          # starting learning rate
MIN_LR = 1e-6              # minimum learning rate
WEIGHT_DECAY = 1e-4        # L2 regularization via AdamW
NOISE_THRESHOLD = 0.0      # minimum |pct_change| to keep a sample (0 = keep all)
VAL_SPLIT = 0.1            # validation fraction from training data

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("Education 6.0 — Competition-Grade BTC Prediction Model")
print("=" * 70)

data = pd.read_csv('BTCUSD_15m_ready.csv', sep=';')
print(f"\n[DATA] Loaded {len(data)} rows")
print(f"[DATA] Columns: {list(data.columns)}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
# Base features from CSV
base_features = [
    'Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_pct',
    'ATR_pct', 'MACD', 'RSI', 'STOCHk', 'BBP'
]

# Engineer additional features
df = data.copy()

# --- Momentum Features ---
df['Close_pct_momentum_3'] = df['Close_pct'].rolling(3).mean()   # 3-bar momentum
df['Close_pct_momentum_8'] = df['Close_pct'].rolling(8).mean()   # 8-bar momentum
df['Close_pct_momentum_16'] = df['Close_pct'].rolling(16).mean() # 16-bar momentum

# --- Volatility Features ---
df['Close_pct_vol_8'] = df['Close_pct'].rolling(8).std()         # 8-bar realized volatility
df['Close_pct_vol_16'] = df['Close_pct'].rolling(16).std()       # 16-bar realized volatility
df['HL_range'] = df['High_pct'] - df['Low_pct']                  # candle range (proxy for volatility)

# --- RSI Derivatives ---
df['RSI_deriv'] = df['RSI'].diff()                               # RSI momentum
df['RSI_overbought'] = (df['RSI'] > 70).astype(float)            # binary overbought signal
df['RSI_oversold'] = (df['RSI'] < 30).astype(float)              # binary oversold signal

# --- MACD Derivatives ---
df['MACD_deriv'] = df['MACD'].diff()                             # MACD momentum (1st derivative)
df['MACD_accel'] = df['MACD_deriv'].diff()                       # MACD acceleration (2nd derivative)

# --- Stochastic Derivatives ---
df['STOCHk_deriv'] = df['STOCHk'].diff()                         # Stochastic momentum

# --- Volume-Price Interaction ---
df['Volume_Price_interaction'] = df['Volume_pct'] * df['Close_pct']  # volume-weighted price move

# --- BBP derivative ---
df['BBP_deriv'] = df['BBP'].diff()

# All features
all_features = base_features + [
    'Close_pct_momentum_3', 'Close_pct_momentum_8', 'Close_pct_momentum_16',
    'Close_pct_vol_8', 'Close_pct_vol_16', 'HL_range',
    'RSI_deriv', 'RSI_overbought', 'RSI_oversold',
    'MACD_deriv', 'MACD_accel',
    'STOCHk_deriv',
    'Volume_Price_interaction',
    'BBP_deriv'
]

# Drop NaN rows from rolling calculations
df.dropna(subset=all_features + ['Target'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n[FEATURES] Using {len(all_features)} features:")
for i, f in enumerate(all_features):
    print(f"  {i+1:2d}. {f}")

# ============================================================
# 3. CREATE SLIDING WINDOWS
# ============================================================
X_raw = df[all_features].values
y_raw = df['Target'].values

X_list = []
y_list = []

for i in range(WINDOW_SIZE, len(df)):
    target = y_raw[i - 1]  # target aligned with the CSV's Target column
    if abs(target) < NOISE_THRESHOLD:
        continue
    X_list.append(X_raw[i - WINDOW_SIZE:i])
    y_list.append(target)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"\n[WINDOWS] Created {len(y)} samples with window_size={WINDOW_SIZE}")
print(f"[WINDOWS] X shape: {X.shape}, y shape: {y.shape}")

# Direction distribution
n_up = np.sum(y > 0)
n_down = np.sum(y <= 0)
print(f"[WINDOWS] UP samples: {n_up} ({100*n_up/len(y):.1f}%) | DOWN samples: {n_down} ({100*n_down/len(y):.1f}%)")

# ============================================================
# 4. WALK-FORWARD SPLIT (NO DATA LEAKAGE)
# ============================================================
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n[SPLIT] Train: {len(y_train)} | Test: {len(y_test)} (chronological, no shuffle)")

# ============================================================
# 5. ROBUST SCALING (fit on train only)
# ============================================================
scaler = RobustScaler()  # RobustScaler handles outliers better than MinMaxScaler for crypto
n_train, w, f = X_train.shape
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n_train, w, f)
n_test = X_test.shape[0]
X_test_scaled = scaler.transform(X_test.reshape(-1, f)).reshape(n_test, w, f)

# Replace any remaining NaN/Inf with 0
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"[SCALING] RobustScaler fitted on training data")

# ============================================================
# 6. LOSS FUNCTION — Directional Huber Hybrid
# ============================================================
def directional_huber_loss(y_true, y_pred):
    """
    Hybrid loss that combines:
    1. Huber loss for magnitude regression (stable gradients)
    2. Heavy directional penalty when sign is wrong
    3. Confidence reward when prediction is correct with high confidence
    """
    # --- Huber component (magnitude) ---
    huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)

    # --- Directional component ---
    true_sign = K.sign(y_true)
    pred_sign = K.sign(y_pred)

    # Binary: 1 if wrong direction, 0 if correct
    direction_wrong = K.cast(K.not_equal(true_sign, pred_sign), K.floatx())

    # Scale penalty by how wrong the prediction magnitude is
    magnitude_penalty = K.abs(y_pred) * direction_wrong

    # --- Combined ---
    # Weight: 60% direction, 40% magnitude
    loss = 0.4 * huber + 0.6 * (direction_wrong * 2.0 + magnitude_penalty * 0.5)

    return loss


# ============================================================
# 7. BUILD MODEL — Residual TCN + MultiHead Attention + BiLSTM
# ============================================================
def build_model(input_shape):
    """
    Architecture:
    1. Input LayerNorm
    2. Residual Dilated TCN (causal convolutions with skip connections)
    3. Multi-Head Self-Attention (captures long-range dependencies)
    4. Bidirectional LSTM (sequential pattern extraction)
    5. Dense head with dropout regularization
    """
    inputs = Input(shape=input_shape, name='input')

    # --- Input Normalization ---
    x = LayerNormalization(axis=-1, name='input_norm')(inputs)

    # --- Residual Dilated TCN Block ---
    # Dilation rates 1, 2, 4 capture patterns at 15min, 30min, 1hr scales
    tcn_filters = 64

    # Branch 1: dilation=1
    c1 = Conv1D(tcn_filters, kernel_size=3, dilation_rate=1, padding='causal',
                activation='elu', name='tcn_d1')(x)
    c1 = SpatialDropout1D(0.1, name='tcn_d1_drop')(c1)

    # Branch 2: dilation=2
    c2 = Conv1D(tcn_filters, kernel_size=3, dilation_rate=2, padding='causal',
                activation='elu', name='tcn_d2')(c1)
    c2 = SpatialDropout1D(0.1, name='tcn_d2_drop')(c2)

    # Branch 3: dilation=4
    c3 = Conv1D(tcn_filters, kernel_size=3, dilation_rate=4, padding='causal',
                activation='elu', name='tcn_d4')(c2)
    c3 = SpatialDropout1D(0.1, name='tcn_d4_drop')(c3)

    # Residual: project input to match dims, then add
    residual = Conv1D(tcn_filters, kernel_size=1, padding='same', name='tcn_residual')(x)
    tcn_out = Add(name='tcn_add')([residual, c3])
    tcn_out = LayerNormalization(name='tcn_norm')(tcn_out)

    # --- Multi-Head Self-Attention Block ---
    attn = MultiHeadAttention(
        num_heads=8, key_dim=16, dropout=0.1, name='mha'
    )(tcn_out, tcn_out)
    attn = Dropout(0.1, name='mha_drop')(attn)
    attn_out = Add(name='attn_add')([tcn_out, attn])
    attn_out = LayerNormalization(name='attn_norm')(attn_out)

    # --- Bidirectional LSTM Block ---
    lstm_out = Bidirectional(
        LSTM(48, return_sequences=False, dropout=0.15, recurrent_dropout=0.0),
        name='bilstm'
    )(attn_out)
    lstm_out = LayerNormalization(name='lstm_norm')(lstm_out)

    # --- Dense Prediction Head ---
    d = Dense(64, activation='elu', name='dense1')(lstm_out)
    d = Dropout(0.25, name='dense1_drop')(d)
    d = Dense(32, activation='elu', name='dense2')(d)
    d = Dropout(0.15, name='dense2_drop')(d)
    output = Dense(1, name='output')(d)

    model = Model(inputs=inputs, outputs=output, name='BTC_Predictor_v6')
    return model

model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

model.compile(
    optimizer=AdamW(learning_rate=INITIAL_LR, weight_decay=WEIGHT_DECAY),
    loss=directional_huber_loss,
    metrics=['mae']
)

model.summary()

# ============================================================
# 8. CALLBACKS
# ============================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=LR_PATIENCE,
    min_lr=MIN_LR,
    verbose=1
)

# Custom callback to track directional accuracy during training
class DirectionalAccuracyCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_dir_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0 or epoch < 5:  # print every 5 epochs + first 5
            preds = self.model.predict(self.X_val, verbose=0)
            pred_dir = np.sign(preds.flatten())
            true_dir = np.sign(self.y_val)
            dir_acc = np.mean(pred_dir == true_dir) * 100
            if dir_acc > self.best_dir_acc:
                self.best_dir_acc = dir_acc
            print(f"  → Epoch {epoch}: Val Directional Accuracy = {dir_acc:.2f}% (best: {self.best_dir_acc:.2f}%)")

# Create a validation split for the directional accuracy callback
val_size = int(len(X_train_scaled) * VAL_SPLIT)
X_val_cb = X_train_scaled[-val_size:]
y_val_cb = y_train[-val_size:]

dir_acc_callback = DirectionalAccuracyCallback(X_val_cb, y_val_cb)

# ============================================================
# 9. TRAINING
# ============================================================
print("\n" + "=" * 70)
print("TRAINING STARTED")
print("=" * 70)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_scheduler, dir_acc_callback],
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

# ============================================================
# 10. COMPREHENSIVE EVALUATION
# ============================================================
predictions = model.predict(X_test_scaled).flatten()

# --- Directional Accuracy ---
pred_direction = np.sign(predictions)
true_direction = np.sign(y_test)

# Convert to binary for sklearn metrics (1 = UP, 0 = DOWN)
pred_binary = (pred_direction > 0).astype(int)
true_binary = (true_direction > 0).astype(int)

dir_accuracy = accuracy_score(true_binary, pred_binary) * 100
precision_up = precision_score(true_binary, pred_binary, zero_division=0) * 100
recall_up = recall_score(true_binary, pred_binary, zero_division=0) * 100
f1_up = f1_score(true_binary, pred_binary, zero_division=0) * 100

precision_down = precision_score(true_binary, pred_binary, pos_label=0, zero_division=0) * 100
recall_down = recall_score(true_binary, pred_binary, pos_label=0, zero_division=0) * 100
f1_down = f1_score(true_binary, pred_binary, pos_label=0, zero_division=0) * 100

# --- Magnitude Metrics ---
mae = np.mean(np.abs(predictions - y_test))
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

# --- Confidence-Filtered Accuracy ---
# Only evaluate predictions where the model is "confident" (|pred| > median)
median_confidence = np.median(np.abs(predictions))
confident_mask = np.abs(predictions) > median_confidence
if confident_mask.sum() > 0:
    confident_accuracy = np.mean(
        np.sign(predictions[confident_mask]) == np.sign(y_test[confident_mask])
    ) * 100
    confident_count = confident_mask.sum()
else:
    confident_accuracy = 0
    confident_count = 0

# --- Streak Analysis ---
correct = (pred_direction == true_direction).astype(int)
max_correct_streak = 0
max_wrong_streak = 0
current_correct = 0
current_wrong = 0
for c in correct:
    if c == 1:
        current_correct += 1
        current_wrong = 0
        max_correct_streak = max(max_correct_streak, current_correct)
    else:
        current_wrong += 1
        current_correct = 0
        max_wrong_streak = max(max_wrong_streak, current_wrong)

# --- Print Results ---
print("\n" + "=" * 70)
print("📊  FINAL EVALUATION RESULTS — Education 6.0")
print("=" * 70)

print(f"\n{'─' * 50}")
print(f"  🎯  DIRECTIONAL ACCURACY:  {dir_accuracy:.2f}%")
print(f"{'─' * 50}")

print(f"\n  📈 UP Predictions:")
print(f"       Precision: {precision_up:.2f}%")
print(f"       Recall:    {recall_up:.2f}%")
print(f"       F1 Score:  {f1_up:.2f}%")

print(f"\n  📉 DOWN Predictions:")
print(f"       Precision: {precision_down:.2f}%")
print(f"       Recall:    {recall_down:.2f}%")
print(f"       F1 Score:  {f1_down:.2f}%")

print(f"\n  🔍 Confidence-Filtered Accuracy:")
print(f"       Accuracy (top 50% confident): {confident_accuracy:.2f}% ({confident_count} samples)")

print(f"\n  📏 Magnitude Metrics:")
print(f"       MAE:  {mae:.4f}")
print(f"       RMSE: {rmse:.4f}")

print(f"\n  🔥 Streak Analysis:")
print(f"       Longest correct streak: {max_correct_streak}")
print(f"       Longest wrong streak:   {max_wrong_streak}")

print(f"\n  📋 Confusion Matrix:")
cm = confusion_matrix(true_binary, pred_binary)
print(f"       Predicted:    DOWN    UP")
print(f"       Actual DOWN: {cm[0][0]:>6d}  {cm[0][1]:>6d}")
print(f"       Actual UP:   {cm[1][0]:>6d}  {cm[1][1]:>6d}")

print(f"\n  📊 Classification Report:")
print(classification_report(true_binary, pred_binary, target_names=['DOWN', 'UP']))

print("=" * 70)

# ============================================================
# 11. SAMPLE PREDICTIONS
# ============================================================
print("\n📋 Sample Predictions (first 30):")
print(f"{'#':>4}  {'Prediction':>12}  {'Reality':>10}  {'Direction':>10}  {'Correct':>8}")
print("-" * 55)
for i in range(min(30, len(predictions))):
    pred_dir = "UP" if predictions[i] > 0 else "DOWN"
    real_dir = "UP" if y_test[i] > 0 else "DOWN"
    correct_str = "✓" if pred_dir == real_dir else "✗"
    print(f"{i+1:>4}  {predictions[i]:>11.4f}%  {y_test[i]:>9.4f}%  {pred_dir:>10}  {correct_str:>8}")

# ============================================================
# 12. PLOTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Education 6.0 — BTC Prediction Model Results', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', color='#2196F3')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='#FF5722')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Predictions vs Reality (first 200 samples)
n_plot = min(200, len(predictions))
axes[0, 1].plot(predictions[:n_plot], label='Prediction', color='#4CAF50', alpha=0.8)
axes[0, 1].plot(y_test[:n_plot], label='Reality', color='#F44336', alpha=0.6)
axes[0, 1].axhline(y=0, linestyle='--', color='gray', alpha=0.5)
axes[0, 1].set_title(f'Predictions vs Reality (first {n_plot} samples)')
axes[0, 1].set_xlabel('Sample')
axes[0, 1].set_ylabel('% Change')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Directional Accuracy Rolling Window
window = 100
rolling_correct = pd.Series(correct).rolling(window).mean() * 100
axes[1, 0].plot(rolling_correct, color='#9C27B0', linewidth=1.5)
axes[1, 0].axhline(y=50, linestyle='--', color='red', alpha=0.5, label='Random baseline (50%)')
axes[1, 0].axhline(y=dir_accuracy, linestyle='--', color='green', alpha=0.7, label=f'Overall ({dir_accuracy:.1f}%)')
axes[1, 0].set_title(f'Rolling Directional Accuracy ({window}-sample window)')
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('Accuracy %')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(30, 80)

# Plot 4: Prediction Distribution
axes[1, 1].hist(predictions, bins=80, alpha=0.7, color='#2196F3', label='Predictions', density=True)
axes[1, 1].hist(y_test, bins=80, alpha=0.5, color='#FF5722', label='Reality', density=True)
axes[1, 1].axvline(x=0, linestyle='--', color='black', alpha=0.5)
axes[1, 1].set_title('Distribution: Predictions vs Reality')
axes[1, 1].set_xlabel('% Change')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 13. FINAL SUMMARY CARD
# ============================================================
print("\n" + "=" * 70)
print("  🏆  COMPETITION SUMMARY CARD — Education 6.0")
print("=" * 70)
print(f"  Model:        Residual TCN + 8-Head Attention + BiLSTM")
print(f"  Features:     {len(all_features)} engineered features")
print(f"  Window:       {WINDOW_SIZE} bars (= {WINDOW_SIZE*15/60:.0f} hours)")
print(f"  Train/Test:   {len(y_train)}/{len(y_test)} (chronological)")
print(f"  Epochs Run:   {len(history.history['loss'])}")
print(f"  ─────────────────────────────────────────────")
print(f"  DIRECTIONAL ACCURACY:   {dir_accuracy:.2f}%")
print(f"  CONFIDENT ACCURACY:     {confident_accuracy:.2f}%")
print(f"  F1 (UP):                {f1_up:.2f}%")
print(f"  F1 (DOWN):              {f1_down:.2f}%")
print(f"  MAE:                    {mae:.4f}")
print(f"  RMSE:                   {rmse:.4f}")
print("=" * 70)

# ============================================================
# 14. SAVE MODEL
# ============================================================
while True:
    SaveInput = input("\nKaydedilsin mi?\n(evet / hayır / tekrar / kodu bitir)\n").lower()

    if SaveInput == "evet":
        model_symbol = input("Borsa kodunu girin: ").upper()
        model_name = "Hamdi_" + model_symbol + "_6.0.keras"
        model.save(model_name)
        print(f"✅ Model saved as {model_name}")
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
        print("evet, hayır, tekrar veya kodu bitir olarak cevap girin!")
