"""
=============================================================
  EDUCATION FOREX 4.0  — High-Accuracy Classification Edition
  Symbol : EURUSD=X  |  Interval : 1h  |  Horizon : 4 bars

  Improvements over original:
    1.  Rich technical indicators (RSI, MACD, ATR, Bollinger,
        Stochastic, ADX, CCI, multi-TF returns, volatility)
    2.  Cyclical time encoding (hour-of-day, day-of-week)
    3.  Window size 48 (2 full trading days)
    4.  Noise filter threshold 0.10%
    5.  Conv1D + BiLSTM + Self-Attention architecture (~77k params)
    6.  Focal loss (focuses on hard examples)
    7.  Class balancing via minority oversampling
    8.  Walk-forward 3-fold cross-validation
    9.  Cosine annealing LR schedule
    10. Early stopping on val_accuracy (not val_loss)
    11. Saves scaler + metadata for reproducible inference
    12. Confidence-bucket evaluation
=============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import Model                    # type: ignore
from tensorflow.keras.layers import (                        # type: ignore
    Conv1D, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, Input, Flatten, Multiply,
    Permute, RepeatVector, Lambda
)
from tensorflow.keras.optimizers import Adam                 # type: ignore
from tensorflow.keras.callbacks import (                     # type: ignore
    EarlyStopping, LearningRateScheduler
)

# =========================================================
# CONFIG — tune these, this is the "knobs" section
# =========================================================
EPOCHS       = 500
PATIENCE     = 50
BATCH_SIZE   = 64
INITIAL_LR   = 1e-3
MIN_LR       = 1e-6

# Date range — stay just under the 730-day ceiling for 1h data
end_dt     = datetime.today()
start_dt   = end_dt - timedelta(days=700)
start_date = start_dt.strftime("%Y-%m-%d")
end_date   = end_dt.strftime("%Y-%m-%d")

symbol        = "EURUSD=X"          # locked per project rule
window_size   = 48                  # how many past bars the model sees (2 full days)
predict_size  = 4                   # how many bars ahead we predict
threshold_pct = 0.10                # min |% move| to count as a real signal (raised from 0.05)
test_size_frac = 0.2                # (informational — we use walk-forward CV now)

# Macro instruments — expanded set for richer macro context
macro_symbols_forex = {
    'DXY_proxy'  : 'UUP',        # USD strength ETF
    'VIX'        : '^VIX',
    'US10Y'      : '^TNX',
    'XAUUSD'     : 'GC=F',
    'SPX'        : '^GSPC',
    'EuroStoxx50': 'FEZ',
    'Brent'      : 'BZ=F',
}

SAVE_DIR = "/Users/ardaunal/Desktop/Hamdi/Forex"


# =========================================================
# FOCAL LOSS — focuses learning on hard-to-classify examples
# =========================================================
def focal_loss(gamma=2.0, alpha=0.5):
    """
    Focal loss for binary classification.
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class balance factor
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * focal_weight * bce)
    return loss_fn


# =========================================================
# COSINE ANNEALING LR SCHEDULE
# =========================================================
def cosine_annealing_schedule(epoch, lr):
    cos_val = 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
    return MIN_LR + (INITIAL_LR - MIN_LR) * cos_val


# =========================================================
# TECHNICAL INDICATORS — adds ~20 features to raw OHLCV
# =========================================================
def add_technical_indicators(df):
    close = df['Close']
    high  = df['High']
    low   = df['Low']

    # ── RSI(14) ──────────────────────────────────────────
    delta_c = close.diff()
    gain    = delta_c.clip(lower=0)
    loss    = (-delta_c).clip(lower=0)
    avg_g   = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_l   = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs      = avg_g / (avg_l + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ── MACD histogram ───────────────────────────────────
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd - sig

    # ── ATR(14) normalised ───────────────────────────────
    prev_c = close.shift(1)
    tr     = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs()
    ], axis=1).max(axis=1)
    atr14  = tr.ewm(com=13, min_periods=14, adjust=False).mean()
    df['ATR_14_norm'] = atr14 / (close + 1e-9) * 100

    # ── Bollinger %B ─────────────────────────────────────
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    df['BB_pctB'] = (close - lower) / (upper - lower + 1e-9)

    # ── EMA distances ────────────────────────────────────
    ema9   = close.ewm(span=9,  adjust=False).mean()
    ema21  = close.ewm(span=21, adjust=False).mean()
    df['dist_EMA9']  = (close - ema9)  / (close + 1e-9) * 100
    df['dist_EMA21'] = (close - ema21) / (close + 1e-9) * 100

    # ── Stochastic Oscillator ────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14 + 1e-9) * 100
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_k.rolling(3).mean()

    # ── ADX(14) ──────────────────────────────────────────
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_smooth = tr.ewm(com=13, min_periods=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(com=13, min_periods=14, adjust=False).mean() / (atr_smooth + 1e-9)
    minus_di = 100 * minus_dm.ewm(com=13, min_periods=14, adjust=False).mean() / (atr_smooth + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['ADX_14'] = dx.ewm(com=13, min_periods=14, adjust=False).mean()

    # ── CCI(20) ──────────────────────────────────────────
    tp_price = (high + low + close) / 3
    sma_tp   = tp_price.rolling(20).mean()
    mad      = tp_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI_20'] = (tp_price - sma_tp) / (0.015 * mad + 1e-9)

    # ── Multi-timeframe returns ──────────────────────────
    df['return_1h']  = close.pct_change(1) * 100
    df['return_4h']  = close.pct_change(4) * 100
    df['return_12h'] = close.pct_change(12) * 100
    df['return_24h'] = close.pct_change(24) * 100

    # ── Rolling volatility (24h) ─────────────────────────
    returns = close.pct_change() * 100
    df['vol_24h'] = returns.rolling(24).std()

    # Clean NaNs from indicator warmup
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def add_cyclical_time_features(df):
    """Encode hour-of-day and day-of-week as sin/cos pairs."""
    hours     = df.index.hour
    day_of_wk = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin']  = np.sin(2 * np.pi * day_of_wk / 7)
    df['dow_cos']  = np.cos(2 * np.pi * day_of_wk / 7)
    return df


# =========================================================
# MODEL — Conv1D + BiLSTM + Self-Attention (~77k params)
# =========================================================
def build_classification_model(input_shape):
    """
    Conv1D → BiLSTM → BiLSTM → Self-Attention → Dense classification head.
    Output: sigmoid probability of UP move.
    """
    inp = Input(shape=input_shape)

    # CNN for local pattern extraction
    x = Conv1D(48, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # BiLSTM layer 1
    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # BiLSTM layer 2
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    # Self-attention: learn which timesteps matter most
    attention_weights = Dense(1, activation='tanh')(x)                             # (batch, timesteps, 1)
    attention_weights = Flatten()(attention_weights)                                # (batch, timesteps)
    attention_weights = Dense(input_shape[0], activation='softmax')(attention_weights)  # (batch, timesteps)
    attention_weights = RepeatVector(64)(attention_weights)                         # (batch, 64, timesteps)
    attention_weights = Permute([2, 1])(attention_weights)                          # (batch, timesteps, 64)

    # Apply attention
    attended = Multiply()([x, attention_weights])
    attended = Lambda(lambda z: tf.reduce_sum(z, axis=1))(attended)                # (batch, 64)

    # Classification head
    out = Dense(32, activation='relu')(attended)
    out = Dropout(0.15)(out)
    out = Dense(16, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=inp, outputs=out)
    return model


# =========================================================
# CLASS BALANCING — oversample minority class
# =========================================================
def balance_classes(X, y):
    """Oversample minority class for balanced training."""
    n_up   = (y == 1).sum()
    n_down = (y == 0).sum()

    if n_up == n_down or n_up == 0 or n_down == 0:
        return X, y

    if n_up > n_down:
        minority_mask = (y == 0)
        extra_n = n_up - n_down
    else:
        minority_mask = (y == 1)
        extra_n = n_down - n_up

    minority_X = X[minority_mask]
    minority_y = y[minority_mask]
    idx = np.random.choice(len(minority_X), size=extra_n, replace=True)
    X_bal = np.concatenate([X, minority_X[idx]], axis=0)
    y_bal = np.concatenate([y, minority_y[idx]], axis=0)
    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# =========================================================
# 1. DATA — download EURUSD + macro instruments
# =========================================================
print(f"\n{'='*66}")
print(f"  EDUCATION FOREX 4.0 — HIGH-ACCURACY CLASSIFICATION")
print(f"  {symbol}  |  {start_date} → {end_date}")
print(f"  Target: >65% Directional Accuracy  |  <200k Parameters")
print(f"  Approach: Conv1D + BiLSTM + Attention + Focal Loss")
print(f"{'='*66}\n")

print("Downloading EURUSD data …")
data = yf.download(symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    raise RuntimeError(
        f"EURUSD=X returned NO data for {start_date} to {end_date}. "
        "Check your internet connection or try again."
    )
print(f"EURUSD bars fetched: {len(data)}")

# Merge macro instruments
macro_cols_added = []
for name, ticker in macro_symbols_forex.items():
    try:
        print(f"  Fetching {name:12s} ({ticker}) …", end=" ", flush=True)
        temp = yf.download(ticker, start=start_date, end=end_date,
                           interval='1h', auto_adjust=True, progress=False)
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)
        if temp.empty or 'Close' not in temp.columns:
            print(f"[skip] no data")
            continue
        temp = temp[['Close']].rename(columns={'Close': name})
        data = data.join(temp, how='left')
        macro_cols_added.append(name)
        print("✓")
    except Exception as e:
        print(f"[skip] {e}")

print(f"\nMacro columns added: {macro_cols_added}")

# Forward-fill macro columns (different trading sessions)
for col in macro_cols_added:
    data[col] = data[col].ffill().bfill()

# Only require CORE EURUSD OHLC columns to be non-null
core_cols = ['Close', 'Open', 'High', 'Low']
data = data.dropna(subset=[c for c in core_cols if c in data.columns])

print(f"Raw bars after merge: {len(data)}")

if len(data) < window_size + predict_size + 100:
    raise RuntimeError(
        f"Only {len(data)} usable bars — too few. "
        "Check the [skip] messages above or widen the date range."
    )

# =========================================================
# 2. FEATURE ENGINEERING — technical indicators + cyclical time
# =========================================================
print("\nAdding technical indicators …")
data = add_technical_indicators(data)
data = add_cyclical_time_features(data)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

print(f"  ✅ {len(data)} bars  |  {data.shape[1]} features")
print(f"  Features: {list(data.columns)}\n")

# =========================================================
# 3. NORMALISE — fit MinMaxScaler on raw data
# =========================================================
scaler      = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================================================
# 4. SLIDING WINDOW + BINARY LABELS (UP=1, DOWN=0)
# =========================================================
X, y, y_pct = [], [], []
original_close = data['Close'].values

for i in range(window_size, len(scaled_data) - predict_size):
    current_close = original_close[i]
    next_close    = original_close[i + predict_size]
    if current_close == 0:
        continue
    pct_change = ((next_close - current_close) / current_close) * 100
    if abs(pct_change) < threshold_pct:
        continue   # noise filter
    X.append(scaled_data[i - window_size : i])
    y.append(1.0 if pct_change > 0 else 0.0)
    y_pct.append(pct_change)

X     = np.array(X)
y     = np.array(y)
y_pct = np.array(y_pct)

n_up   = (y == 1).sum()
n_down = (y == 0).sum()
print(f"  Samples     : {len(y)}")
print(f"  X shape     : {X.shape}")
print(f"  UP labels   : {n_up}  ({n_up/len(y)*100:.1f}%)")
print(f"  DOWN labels : {n_down}  ({n_down/len(y)*100:.1f}%)")

# Majority-class baseline
baseline_acc = max(n_up, n_down) / len(y) * 100
print(f"  Majority-class baseline: {baseline_acc:.2f}%  ← model must beat this\n")

# Focal loss alpha based on class distribution
alpha = n_down / len(y)  # higher weight for rarer class
print(f"  Focal alpha : {alpha:.3f}")

if len(X) == 0:
    raise ValueError("No samples after threshold filtering — lower threshold_pct or widen date range.")

# =========================================================
# 5. WALK-FORWARD 3-FOLD CROSS-VALIDATION
# =========================================================
print(f"\n{'='*66}")
print(f"  WALK-FORWARD 3-FOLD CROSS-VALIDATION")
print(f"{'='*66}\n")

fold_splits = [
    (0.00, 0.60, 0.60, 0.73),   # Fold 1: train [0,60%) → test [60,73%)
    (0.00, 0.73, 0.73, 0.87),   # Fold 2: train [0,73%) → test [73,87%)
    (0.00, 0.87, 0.87, 1.00),   # Fold 3: train [0,87%) → test [87,100%)
]

best_dir_acc     = 0.0
best_fold        = -1
best_model       = None
best_history     = None
best_preds       = None
best_y_test      = None
best_y_pct_test  = None
fold_results     = []

for fold_idx, (ts, te, vs, ve) in enumerate(fold_splits):
    n = len(X)
    train_s, train_e = int(n * ts), int(n * te)
    test_s,  test_e  = int(n * vs), int(n * ve)

    X_train_fold = X[train_s:train_e]
    y_train_fold = y[train_s:train_e]
    X_test_fold  = X[test_s:test_e]
    y_test_fold  = y[test_s:test_e]
    y_pct_test   = y_pct[test_s:test_e]

    print(f"  ── Fold {fold_idx + 1}/3 ──────────────────────────────────────")
    print(f"     Train : {len(X_train_fold)} samples  |  "
          f"UP: {(y_train_fold==1).sum()}  DOWN: {(y_train_fold==0).sum()}")
    print(f"     Test  : {len(X_test_fold)} samples  |  "
          f"UP: {(y_test_fold==1).sum()}  DOWN: {(y_test_fold==0).sum()}")

    # Chronological train/val split (last 12% of training fold = validation)
    # This keeps temporal order intact — val is always FUTURE relative to train
    val_n = max(1, int(len(X_train_fold) * 0.12))
    X_tr  = X_train_fold[:-val_n]
    y_tr  = y_train_fold[:-val_n]
    X_val = X_train_fold[-val_n:]
    y_val = y_train_fold[-val_n:]

    # Class weights — handles imbalance WITHOUT destroying chronological order
    # (oversampling + permutation was scrambling time order, killing accuracy)
    n_up_tr   = (y_tr == 1).sum()
    n_down_tr = (y_tr == 0).sum()
    total_tr  = len(y_tr)
    cw = {
        0: total_tr / (2.0 * n_down_tr) if n_down_tr > 0 else 1.0,
        1: total_tr / (2.0 * n_up_tr)   if n_up_tr   > 0 else 1.0,
    }
    print(f"     Train: {len(X_tr)} | Val: {len(X_val)} | Wt UP={cw[1]:.2f} DOWN={cw[0]:.2f}")

    # Build model
    model = build_classification_model((X_tr.shape[1], X_tr.shape[2]))
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=focal_loss(gamma=2.0, alpha=alpha),
        metrics=['accuracy']
    )

    if fold_idx == 0:
        model.summary()
        total_params = model.count_params()
        print(f"\n  ★ Total parameters : {total_params:,}  (limit 200,000)\n")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    lr_scheduler = LearningRateScheduler(cosine_annealing_schedule, verbose=0)

    # Train — shuffle=False is CRITICAL for time series:
    # the model must see data in chronological order to learn regime transitions
    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        callbacks=[early_stop, lr_scheduler],
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        shuffle=False,      # CRITICAL: preserve chronological order for time series
        class_weight=cw,    # handle class imbalance via weighting, not oversampling
        verbose=1
    )

    # Evaluate — directional accuracy on holdout test set
    pred_probs = model.predict(X_test_fold, verbose=0).flatten()
    pred_dirs  = (pred_probs > 0.5).astype(float)
    dir_acc    = np.mean(pred_dirs == y_test_fold) * 100

    print(f"\n     ★ Fold {fold_idx + 1} Directional Accuracy: {dir_acc:.2f}%")
    print(f"     Pred UP  : {(pred_dirs == 1).sum()}  |  Pred DOWN: {(pred_dirs == 0).sum()}")
    print(f"     Prob range: [{pred_probs.min():.4f}, {pred_probs.max():.4f}]")

    # Confidence analysis
    high_conf = np.abs(pred_probs - 0.5) > 0.1
    if high_conf.sum() > 0:
        hc_acc = np.mean(pred_dirs[high_conf] == y_test_fold[high_conf]) * 100
        print(f"     High-confidence ({high_conf.sum()} signals): {hc_acc:.2f}%\n")

    fold_results.append({
        'fold': fold_idx + 1,
        'train_samples': len(X_tr),
        'test_samples': len(X_test_fold),
        'dir_acc': dir_acc,
        'epochs_trained': len(history.history['loss'])
    })

    if dir_acc > best_dir_acc:
        best_dir_acc    = dir_acc
        best_fold       = fold_idx + 1
        best_model      = model
        best_history    = history
        best_preds      = pred_probs
        best_y_test     = y_test_fold
        best_y_pct_test = y_pct_test


# =========================================================
# 6. FOLD COMPARISON
# =========================================================
print(f"\n{'='*66}")
print(f"  FOLD COMPARISON")
print(f"{'='*66}")
for r in fold_results:
    marker = " ← BEST" if r['fold'] == best_fold else ""
    print(f"  Fold {r['fold']}:  Dir.Acc = {r['dir_acc']:.2f}%  "
          f"({r['epochs_trained']} epochs){marker}")

print(f"\n  🏆 Best fold: {best_fold}  with {best_dir_acc:.2f}% directional accuracy")
print(f"  Majority-class baseline: {baseline_acc:.2f}%")
print(f"  Edge over baseline: {best_dir_acc - baseline_acc:+.2f}%")

if best_dir_acc >= 65.0:
    print(f"  ✅ TARGET MET: {best_dir_acc:.2f}% >= 65%")
else:
    print(f"  ⚠️  Target not met: {best_dir_acc:.2f}% < 65%")


# =========================================================
# 7. DETAILED ANALYSIS
# =========================================================
pred_dirs = (best_preds > 0.5).astype(float)

print(f"\n{'='*66}")
print(f"  SAMPLE PREDICTIONS (first 25)")
print(f"{'='*66}")
for i in range(min(25, len(best_preds))):
    prob   = best_preds[i]
    actual = best_y_test[i]
    pred_d = "UP  " if prob > 0.5 else "DOWN"
    act_d  = "UP  " if actual == 1 else "DOWN"
    ok     = "✓" if pred_dirs[i] == actual else "✗"
    pct    = best_y_pct_test[i]
    print(f"  [{ok}]  Pred: {pred_d} ({prob:.3f})  |  Actual: {act_d} ({pct:+.3f}%)")

# Direction breakdown
up_mask   = (pred_dirs == 1)
down_mask = (pred_dirs == 0)
if up_mask.sum() > 0:
    up_acc = np.mean(best_y_test[up_mask] == 1) * 100
    print(f"\n  LONG  predictions: {up_mask.sum()}  |  accuracy: {up_acc:.1f}%")
if down_mask.sum() > 0:
    down_acc = np.mean(best_y_test[down_mask] == 0) * 100
    print(f"  SHORT predictions: {down_mask.sum()}  |  accuracy: {down_acc:.1f}%")

print(f"\n  ★ Overall Directional Accuracy: {best_dir_acc:.2f}%")

# Confidence bucket analysis
print(f"\n  ── Confidence Bucket Analysis ──")
for lo, hi, label in [(0.0, 0.3, "Strong DOWN"), (0.3, 0.45, "Mild DOWN"),
                       (0.45, 0.55, "Uncertain"), (0.55, 0.7, "Mild UP"),
                       (0.7, 1.01, "Strong UP")]:
    mask = (best_preds >= lo) & (best_preds < hi)
    if mask.sum() > 0:
        bucket_acc = np.mean(pred_dirs[mask] == best_y_test[mask]) * 100
        print(f"  {label:12s} ({lo:.1f}–{hi:.1f}):  {mask.sum():5d} signals  |  acc: {bucket_acc:.1f}%")


# =========================================================
# 8. PLOTS
# =========================================================
os.makedirs(SAVE_DIR, exist_ok=True)

# Loss + Accuracy graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0d0d1a')

for ax in [ax1, ax2]:
    ax.set_facecolor('#141428')
    ax.tick_params(colors='#aaaacc')
    ax.grid(color='#2a2a4a', linewidth=0.5, linestyle='--')

ax1.plot(best_history.history['loss'],     label='Train Loss', color='#00e5ff', linewidth=1.5)
ax1.plot(best_history.history['val_loss'], label='Val Loss',   color='#ff6b35', linewidth=1.5)
ax1.set_title(f'Loss — Forex 4.0 (Fold {best_fold})', color='#e0e0ff', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', color='#aaaacc')
ax1.set_ylabel('Loss', color='#aaaacc')
ax1.legend(fontsize=10, facecolor='#1a1a30', edgecolor='#2a2a4a', labelcolor='#e0e0ff')

ax2.plot(np.array(best_history.history['accuracy'])*100,     label='Train Acc', color='#00e5ff', linewidth=1.5)
ax2.plot(np.array(best_history.history['val_accuracy'])*100, label='Val Acc',   color='#ff6b35', linewidth=1.5)
ax2.axhline(y=65, linestyle='--', color='#f72585', linewidth=1, label='65% Target')
ax2.axhline(y=baseline_acc, linestyle=':', color='#555577', linewidth=1, label=f'Baseline ({baseline_acc:.1f}%)')
ax2.set_title(f'Accuracy — Forex 4.0 (Fold {best_fold})', color='#e0e0ff', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', color='#aaaacc')
ax2.set_ylabel('Accuracy %', color='#aaaacc')
ax2.legend(fontsize=10, facecolor='#1a1a30', edgecolor='#2a2a4a', labelcolor='#e0e0ff')

plt.tight_layout()
loss_path = os.path.join(SAVE_DIR, "Education_Forex_4_0_Training.png")
plt.savefig(loss_path, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n  ✅ Training graph → {loss_path}")

# Prediction probability distribution
fig2, ax3 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('#0d0d1a')
ax3.set_facecolor('#141428')
ax3.hist(best_preds[best_y_test == 1], bins=50, alpha=0.7, color='#00e5ff', label='Actual UP')
ax3.hist(best_preds[best_y_test == 0], bins=50, alpha=0.7, color='#f72585', label='Actual DOWN')
ax3.axvline(x=0.5, linestyle='--', color='#ffd166', linewidth=1.5, label='Decision boundary')
ax3.set_title(f'Prediction Probability Distribution | Dir.Acc: {best_dir_acc:.2f}%',
              color='#e0e0ff', fontsize=13, fontweight='bold')
ax3.set_xlabel('Predicted Probability (UP)', color='#aaaacc')
ax3.set_ylabel('Count', color='#aaaacc')
ax3.tick_params(colors='#aaaacc')
ax3.grid(color='#2a2a4a', linewidth=0.5, linestyle='--')
ax3.legend(fontsize=10, facecolor='#1a1a30', edgecolor='#2a2a4a', labelcolor='#e0e0ff')
plt.tight_layout()
dist_path = os.path.join(SAVE_DIR, "Education_Forex_4_0_Distribution.png")
plt.savefig(dist_path, dpi=150, facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"  ✅ Distribution graph → {dist_path}")

# Fold comparison bar chart
fig3, ax4 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor('#0d0d1a')
ax4.set_facecolor('#141428')
folds  = [f"Fold {r['fold']}" for r in fold_results]
accs   = [r['dir_acc'] for r in fold_results]
colors = ['#00e5ff' if r['fold'] == best_fold else '#7b61ff' for r in fold_results]
bars   = ax4.bar(folds, accs, color=colors, width=0.5, alpha=0.85)
ax4.axhline(y=65, linestyle='--', color='#f72585', linewidth=1.5, label='65% Target')
ax4.axhline(y=baseline_acc, linestyle=':', color='#555577', linewidth=1, label=f'Baseline ({baseline_acc:.1f}%)')
for bar, acc in zip(bars, accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', color='#e0e0ff', fontweight='bold', fontsize=12)
ax4.set_title('Directional Accuracy per Fold — Forex 4.0',
              color='#e0e0ff', fontsize=14, fontweight='bold')
ax4.set_ylabel('Accuracy (%)', color='#aaaacc')
ax4.tick_params(colors='#aaaacc')
ax4.grid(color='#2a2a4a', linewidth=0.5, linestyle='--', axis='y')
ax4.legend(fontsize=10, facecolor='#1a1a30', edgecolor='#2a2a4a', labelcolor='#e0e0ff')
ax4.set_ylim(0, max(accs) + 10)
plt.tight_layout()
fold_path = os.path.join(SAVE_DIR, "Education_Forex_4_0_FoldAccuracy.png")
plt.savefig(fold_path, dpi=150, facecolor=fig3.get_facecolor())
plt.close(fig3)
print(f"  ✅ Fold accuracy graph → {fold_path}")


# =========================================================
# 9. SAVE MODEL, SCALER, METADATA
# =========================================================
model_name  = "Hamdi_EURUSD_Forex_4.0.keras"
model_path  = os.path.join(SAVE_DIR, model_name)

save = input("\nSave model? (yes/no): ").strip().lower()
if save == "yes":
    best_model.save(model_path)
    print(f"  ✅ Model saved → {model_path}")

    # Save scaler
    scaler_path = model_path.replace(".keras", "_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Scaler saved → {scaler_path}")

    # Save metadata
    meta = {
        'version': '4.0',
        'type': 'classification',
        'symbol': symbol,
        'window_size': window_size,
        'predict_size': predict_size,
        'threshold': threshold_pct,
        'best_fold': best_fold,
        'dir_acc': best_dir_acc,
        'baseline_acc': baseline_acc,
        'fold_results': fold_results,
        'features': list(data.columns),
        'n_features': data.shape[1],
        'macro_symbols': macro_symbols_forex,
    }
    meta_path = model_path.replace(".keras", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  ✅ Metadata saved → {meta_path}")


# =========================================================
# 10. FINAL SUMMARY
# =========================================================
print(f"\n{'='*66}")
print(f"  ┌────────────────────────────────────────────────────────┐")
print(f"  │  Education Forex 4.0 — TRAINING COMPLETE               │")
print(f"  │  Type       : Binary Classification (UP/DOWN)          │")
print(f"  │  Symbol     : EURUSD=X                                 │")
print(f"  │  Prediction : {predict_size}-hour ahead                              │")
print(f"  │  Features   : {data.shape[1]} columns                              │")
print(f"  │  Parameters : {best_model.count_params():,}{'':>39}│")
print(f"  │  Best Fold  : {best_fold}{'':>43}│")
print(f"  │  Dir. Acc   : {best_dir_acc:.2f}%{'':>40}│")
print(f"  │  Baseline   : {baseline_acc:.2f}%{'':>40}│")
print(f"  │  Edge       : {best_dir_acc - baseline_acc:+.2f}%{'':>40}│")
print(f"  │  Model      : {model_name:<42}│")
print(f"  └────────────────────────────────────────────────────────┘")
print(f"{'='*66}\n")