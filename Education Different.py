"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     EDUCATION DIFFERENT — STOCK MARKET AI                      ║
║                   Advanced Ensemble Prediction Framework v3.0                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Strategy  : Tri-layer Ensemble (LSTM-Attention + XGBoost + LightGBM + RF)    ║
║  Validation: Walk-Forward (TimeSeriesSplit) — zero look-ahead bias             ║
║  Features  : 60+ technical indicators + price-action + volatility regimes      ║
║  Signals   : Classification (BUY / HOLD / SELL) with confidence %             ║
║  Backtest  : Sharpe Ratio, Max Drawdown, Win Rate, Calmar Ratio               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

DISCLAIMER: This code is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.
            Stock markets are inherently unpredictable. Never invest
            solely based on algorithmic predictions.

Required libraries — install once with:
    pip install yfinance pandas numpy scikit-learn xgboost lightgbm
    pip install tensorflow keras matplotlib seaborn joblib tqdm
    pip install ta pandas-ta scipy statsmodels
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import logging
import json
import time
import math
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import yfinance as yf
import ta
# pyrefly: ignore [missing-import]
import pandas_ta as pta

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EducationDifferent")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random; random.seed(SEED)

OUTPUT_DIR = Path("education_different_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR  = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR   = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

CFG = {
    # Data
    "ticker"                  : "AAPL",
    "start_date"              : "2015-01-01",
    "end_date"                : datetime.today().strftime("%Y-%m-%d"),
    "interval"                : "1d",

    # Target
    "predict_horizon"         : 5,
    "classification_threshold": 0.005,

    # LSTM
    "lookback"                : 60,
    "lstm_units"              : [128, 64],
    "attention_heads"         : 4,
    "dropout_rate"            : 0.3,
    "lstm_epochs"             : 100,
    "lstm_batch_size"         : 32,
    "lstm_lr"                 : 1e-3,

    # Tree models
    "xgb_n_estimators"        : 800,
    "lgb_n_estimators"        : 800,
    "rf_n_estimators"         : 400,

    # Walk-forward
    "n_splits"                : 5,
    "test_size_days"          : 252,

    # Indicators
    "rsi_period"              : 14,
    "macd_fast"               : 12,
    "macd_slow"               : 26,
    "macd_signal"             : 9,
    "bb_period"               : 20,
    "bb_std"                  : 2,
    "atr_period"              : 14,
    "stoch_period"            : 14,
    "cci_period"              : 20,
    "williams_period"         : 14,
    "mfi_period"              : 14,
    "adx_period"              : 14,
    "ema_periods"             : [5, 9, 12, 20, 26, 50, 100, 200],
    "sma_periods"             : [5, 10, 20, 50, 200],

    # Stacking
    "meta_model"              : "logistic",

    # Backtest
    "initial_capital"         : 100_000.0,
    "transaction_cost"        : 0.001,
    "stop_loss_pct"           : 0.05,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1: DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════
class DataAcquisition:

    @staticmethod
    def download(ticker, start, end, interval="1d", max_retries=3):
        logger.info(f"Downloading {ticker} [{start} → {end}]")
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    ticker, start=start, end=end,
                    interval=interval, progress=False, auto_adjust=True
                )
                if df.empty:
                    raise ValueError(f"No data for {ticker}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                logger.info(f"  ✓  {len(df)} rows downloaded.")
                return df
            except Exception as exc:
                logger.warning(f"Attempt {attempt}/{max_retries}: {exc}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to download {ticker}.")

    @staticmethod
    def validate(df):
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[list(required)].copy()
        df.sort_index(inplace=True)
        df = df[df["Volume"] > 0]
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: FEATURE ENGINEERING  (60+ indicators)
# ══════════════════════════════════════════════════════════════════════════════
class FeatureEngineer:

    def __init__(self, cfg):
        self.cfg = cfg

    def _price_action(self, df):
        df["returns_1d"]  = df["Close"].pct_change(1)
        df["returns_3d"]  = df["Close"].pct_change(3)
        df["returns_5d"]  = df["Close"].pct_change(5)
        df["returns_10d"] = df["Close"].pct_change(10)
        df["returns_20d"] = df["Close"].pct_change(20)
        df["log_return"]  = np.log(df["Close"] / df["Close"].shift(1))
        df["hl_ratio"]    = (df["High"] - df["Low"]) / df["Close"]
        df["co_ratio"]    = (df["Close"] - df["Open"]) / df["Open"]
        df["true_range"]  = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"]  - df["Close"].shift(1))
            )
        )
        df["gap_pct"]     = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        return df

    def _volume_features(self, df):
        df["vol_change"] = df["Volume"].pct_change(1)
        df["vol_sma5"]   = df["Volume"].rolling(5).mean()
        df["vol_sma20"]  = df["Volume"].rolling(20).mean()
        df["vol_ratio"]  = df["Volume"] / (df["vol_sma20"] + 1e-9)
        df["vwap"]       = (df["Close"] * df["Volume"]).rolling(20).sum() / \
                            (df["Volume"].rolling(20).sum() + 1e-9)
        df["obv"]        = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["cmf"]        = ta.volume.chaikin_money_flow(
                               df["High"], df["Low"], df["Close"], df["Volume"], window=20)
        df["mfi"]        = ta.volume.money_flow_index(
                               df["High"], df["Low"], df["Close"], df["Volume"],
                               window=self.cfg["mfi_period"])
        df["force_idx"]  = df["returns_1d"] * df["Volume"]
        return df

    def _moving_averages(self, df):
        for p in self.cfg["ema_periods"]:
            df[f"ema_{p}"] = ta.trend.ema_indicator(df["Close"], window=p)
        for p in self.cfg["sma_periods"]:
            df[f"sma_{p}"] = ta.trend.sma_indicator(df["Close"], window=p)
        df["golden_cross"]    = (df["ema_50"] > df["ema_200"]).astype(int)
        df["price_vs_ema200"] = (df["Close"] - df["ema_200"]) / (df["ema_200"] + 1e-9)
        df["price_vs_sma50"]  = (df["Close"] - df["sma_50"])  / (df["sma_50"]  + 1e-9)
        # Hull MA proxy
        half = max(2, int(20 / 2))
        wma1 = df["Close"].rolling(half).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(),
            raw=True)
        wma2 = df["Close"].rolling(20).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(),
            raw=True)
        hull_raw = 2 * wma1 - wma2
        sq = max(2, int(np.sqrt(20)))
        df["hma_20"] = hull_raw.rolling(sq).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(),
            raw=True)
        # DEMA / TEMA
        e1 = df["Close"].ewm(span=20, adjust=False).mean()
        e2 = e1.ewm(span=20, adjust=False).mean()
        e3 = e2.ewm(span=20, adjust=False).mean()
        df["dema_20"] = 2 * e1 - e2
        df["tema_20"] = 3 * e1 - 3 * e2 + e3
        return df

    def _momentum(self, df):
        df["rsi"]         = ta.momentum.rsi(df["Close"], window=self.cfg["rsi_period"])
        df["rsi_7"]       = ta.momentum.rsi(df["Close"], window=7)
        df["rsi_21"]      = ta.momentum.rsi(df["Close"], window=21)
        df["rsi_norm"]    = (df["rsi"] - 50) / 50
        sr = ta.momentum.StochRSIIndicator(df["Close"], window=14, smooth1=3, smooth2=3)
        df["stoch_rsi_k"] = sr.stochrsi_k()
        df["stoch_rsi_d"] = sr.stochrsi_d()
        macd = ta.trend.MACD(df["Close"],
                              window_slow=self.cfg["macd_slow"],
                              window_fast=self.cfg["macd_fast"],
                              window_sign=self.cfg["macd_signal"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()
        df["macd_cross"]  = np.where(df["macd"] > df["macd_signal"], 1, -1)
        stoch = ta.momentum.StochasticOscillator(
                    df["High"], df["Low"], df["Close"],
                    window=self.cfg["stoch_period"], smooth_window=3)
        df["stoch_k"]     = stoch.stoch()
        df["stoch_d"]     = stoch.stoch_signal()
        df["williams_r"]  = ta.momentum.williams_r(
                                df["High"], df["Low"], df["Close"],
                                lbp=self.cfg["williams_period"])
        df["cci"]         = ta.trend.cci(df["High"], df["Low"], df["Close"],
                                          window=self.cfg["cci_period"])
        df["roc_5"]       = ta.momentum.roc(df["Close"], window=5)
        df["roc_10"]      = ta.momentum.roc(df["Close"], window=10)
        df["roc_20"]      = ta.momentum.roc(df["Close"], window=20)
        df["awesome_osc"] = ta.momentum.awesome_oscillator(df["High"], df["Low"])
        df["tsi"]         = ta.momentum.tsi(df["Close"])
        df["ult_osc"]     = ta.momentum.ultimate_oscillator(df["High"], df["Low"], df["Close"])
        return df

    def _volatility(self, df):
        bb = ta.volatility.BollingerBands(df["Close"],
                                           window=self.cfg["bb_period"],
                                           window_dev=self.cfg["bb_std"])
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"]  = bb.bollinger_lband()
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-9)
        df["bb_pct"]    = bb.bollinger_pband()
        df["atr"]       = ta.volatility.average_true_range(
                              df["High"], df["Low"], df["Close"],
                              window=self.cfg["atr_period"])
        df["atr_pct"]   = df["atr"] / (df["Close"] + 1e-9)
        kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=20)
        df["kc_upper"]  = kc.keltner_channel_hband()
        df["kc_lower"]  = kc.keltner_channel_lband()
        df["kc_pct"]    = kc.keltner_channel_pband()
        dc = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"], window=20)
        df["dc_upper"]  = dc.donchian_channel_hband()
        df["dc_lower"]  = dc.donchian_channel_lband()
        df["dc_width"]  = (df["dc_upper"] - df["dc_lower"]) / (df["Close"] + 1e-9)
        df["hv_10"]     = df["log_return"].rolling(10).std() * np.sqrt(252)
        df["hv_20"]     = df["log_return"].rolling(20).std() * np.sqrt(252)
        df["hv_60"]     = df["log_return"].rolling(60).std() * np.sqrt(252)
        df["squeeze"]   = (df["bb_width"] < df["dc_width"]).astype(int)
        return df

    def _trend(self, df):
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"],
                                     window=self.cfg["adx_period"])
        df["adx"]       = adx.adx()
        df["adx_p"]     = adx.adx_pos()
        df["adx_n"]     = adx.adx_neg()
        df["di_diff"]   = df["adx_p"] - df["adx_n"]
        aroon = ta.trend.AroonIndicator(df["High"], df["Low"], window=25)
        df["aroon_up"]  = aroon.aroon_up()
        df["aroon_dn"]  = aroon.aroon_down()
        df["aroon_osc"] = aroon.aroon_indicator()
        psar = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"])
        df["psar"]      = psar.psar()
        df["psar_bull"] = (df["Close"] > df["psar"]).astype(int)
        df["tenkan"]    = (df["High"].rolling(9).max()  + df["Low"].rolling(9).min())  / 2
        df["kijun"]     = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
        df["ich_sig"]   = (df["Close"] > df["tenkan"]).astype(int)
        return df

    def _statistical(self, df):
        for w in [10, 20, 60]:
            rm  = df["Close"].rolling(w).mean()
            rs  = df["Close"].rolling(w).std()
            df[f"zscore_{w}"] = (df["Close"] - rm) / (rs + 1e-9)
        df["skew_20"] = df["log_return"].rolling(20).skew()
        df["kurt_20"] = df["log_return"].rolling(20).kurt()

        def hurst(series):
            s = np.array(series)
            if len(s) < 20 or np.std(s) < 1e-10:
                return 0.5
            lags = range(2, min(20, len(s) // 2))
            tau  = [np.std(np.subtract(s[l:], s[:-l])) for l in lags]
            if any(t <= 0 for t in tau):
                return 0.5
            try:
                reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                return float(np.clip(reg[0], 0, 1))
            except Exception:
                return 0.5

        df["hurst"]     = df["log_return"].rolling(100).apply(hurst, raw=False)
        df["ar1_resid"] = df["log_return"] - \
                           df["log_return"].shift(1) * \
                           df["log_return"].rolling(20).corr(df["log_return"].shift(1))
        df["regime"]    = np.where(df["adx"] > 25, 1,
                          np.where(df["hurst"] < 0.45, -1, 0))
        return df

    def _calendar(self, df):
        df["day_of_week"]    = df.index.dayofweek
        df["day_of_month"]   = df.index.day
        df["week_of_year"]   = df.index.isocalendar().week.astype(int)
        df["month"]          = df.index.month
        df["quarter"]        = df.index.quarter
        df["is_month_end"]   = df.index.is_month_end.astype(int)
        df["is_qtr_end"]     = df.index.is_quarter_end.astype(int)
        return df

    def _targets(self, df):
        h   = self.cfg["predict_horizon"]
        thr = self.cfg["classification_threshold"]
        # Future cumulative log-return
        df["target_return"] = df["log_return"].shift(-h).rolling(h).sum()
        # 3-class: SELL=0, HOLD=1, BUY=2
        df["target_class"]  = np.where(df["target_return"] >  thr, 2,
                               np.where(df["target_return"] < -thr, 0, 1))
        df["target_binary"] = (df["target_return"] > 0).astype(int)
        return df

    def transform(self, df):
        logger.info("Engineering features …")
        df = self._price_action(df)
        df = self._volume_features(df)
        df = self._moving_averages(df)
        df = self._momentum(df)
        df = self._volatility(df)
        df = self._trend(df)
        df = self._statistical(df)
        df = self._calendar(df)
        df = self._targets(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        logger.info(f"  ✓  {df.shape[1]} columns × {df.shape[0]} rows")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: DATA SPLITTING & SCALING
# ══════════════════════════════════════════════════════════════════════════════
def get_feature_cols(df):
    exclude = {"Open", "High", "Low", "Close", "Volume",
               "target_return", "target_class", "target_binary"}
    return [c for c in df.columns if c not in exclude]


class FeatureScaler:
    def __init__(self, method="robust"):
        self.method  = method
        self._scaler = None

    def fit(self, X):
        if self.method == "minmax":
            self._scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.method == "standard":
            self._scaler = StandardScaler()
        else:
            self._scaler = RobustScaler()
        self._scaler.fit(X)
        return self

    def transform(self, X):
        return self._scaler.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        joblib.dump(self._scaler, path)

    def load(self, path):
        self._scaler = joblib.load(path)
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: LSTM WITH MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
def build_lstm_attention(n_features, lookback, units, n_heads, dropout, n_classes, lr):
    inp = layers.Input(shape=(lookback, n_features), name="seq_in")

    x = layers.Bidirectional(
            layers.LSTM(units[0], return_sequences=True,
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm1")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(units[1], return_sequences=True,
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="lstm2")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    attn = layers.MultiHeadAttention(
               num_heads=n_heads,
               key_dim=max(1, units[1] // n_heads),
               name="mhsa")(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    x   = layers.Concatenate()([avg, mx])

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(n_classes, activation="softmax", name="out")(x)

    model = Model(inp, out, name="LSTMAttention")
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys)


def train_lstm(cfg, X_train_sc, y_train, X_val_sc, y_val, n_features, n_classes=3):
    lookback = cfg["lookback"]
    X_seq, y_seq     = make_sequences(X_train_sc, y_train, lookback)
    X_vseq, y_vseq   = make_sequences(
        np.vstack([X_train_sc[-lookback:], X_val_sc]),
        np.concatenate([y_train[-lookback:], y_val]),
        lookback
    )
    model = build_lstm_attention(
        n_features, lookback,
        cfg["lstm_units"], cfg["attention_heads"],
        cfg["dropout_rate"], n_classes, cfg["lstm_lr"]
    )
    cbs = [
        EarlyStopping(monitor="val_loss", patience=15,
                       restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                           patience=7, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        X_seq, y_seq,
        validation_data=(X_vseq, y_vseq),
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch_size"],
        callbacks=cbs,
        verbose=0
    )
    logger.info(f"  LSTM trained: {len(history.epoch)} epochs.")
    return model, history


def predict_lstm(model, X_context_sc, X_test_sc, lookback):
    combined = np.vstack([X_context_sc, X_test_sc])
    seqs = np.array([
        combined[i - lookback: i]
        for i in range(lookback, len(combined))
    ], dtype=np.float32)
    return model.predict(seqs, verbose=0)   # probabilities


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: TREE ENSEMBLE  (XGBoost + LightGBM + Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
def build_tree_models(cfg):
    xgb_model = xgb.XGBClassifier(
        n_estimators=cfg["xgb_n_estimators"], max_depth=6,
        learning_rate=0.05, subsample=0.80, colsample_bytree=0.80,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0, min_child_weight=3,
        eval_metric="mlogloss",
        random_state=SEED, n_jobs=-1, tree_method="hist"
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=cfg["lgb_n_estimators"], num_leaves=63,
        learning_rate=0.05, subsample=0.80, colsample_bytree=0.80,
        reg_alpha=0.05, reg_lambda=1.0, min_child_samples=20,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    rf_model = RandomForestClassifier(
        n_estimators=cfg["rf_n_estimators"], max_depth=10,
        min_samples_split=10, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    return xgb_model, lgb_model, rf_model


def train_trees(cfg, xgb_m, lgb_m, rf_m, X_tr, y_tr, X_va, y_va):
    logger.info("  Training XGBoost …")
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    logger.info("  Training LightGBM …")
    lgb_cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=lgb_cbs)

    logger.info("  Training Random Forest …")
    rf_m.fit(X_tr, y_tr)
    return xgb_m, lgb_m, rf_m


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: STACKING META-LEARNER
# ══════════════════════════════════════════════════════════════════════════════
def stack_features(lstm_p, xgb_p, lgb_p, rf_p):
    parts = []
    if lstm_p is not None:
        parts.append(lstm_p)
    parts += [xgb_p, lgb_p, rf_p]
    return np.hstack(parts)


def build_meta(cfg):
    if cfg["meta_model"] == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=SEED,
                                   n_jobs=-1, solver="lbfgs")
    elif cfg["meta_model"] == "xgb":
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=SEED, n_jobs=-1, tree_method="hist"
        )
    else:
        return lgb.LGBMClassifier(n_estimators=200, num_leaves=15,
                                   learning_rate=0.05, random_state=SEED,
                                   n_jobs=-1, verbose=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def cls_metrics(y_true, y_pred, y_proba=None):
    res = {
        "accuracy"        : accuracy_score(y_true, y_pred),
        "precision_macro" : precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro"    : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro"        : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted"     : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist(),
        "report"          : classification_report(y_true, y_pred,
                                target_names=["SELL","HOLD","BUY"], zero_division=0),
    }
    if y_proba is not None and y_proba.shape[1] == 3:
        try:
            res["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            res["roc_auc"] = 0.0
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 8: BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(price_series, signals, probas, cfg):
    capital    = cfg["initial_capital"]
    tc         = cfg["transaction_cost"]
    sl         = cfg["stop_loss_pct"]
    prices     = price_series.values
    records    = []
    shares, cash, entry_price, max_val = 0.0, capital, 0.0, capital

    for i in range(1, len(prices)):
        sig  = int(signals[i])
        conf = float(probas[i, sig]) if probas is not None else 1.0
        px   = prices[i]

        # Trailing stop-loss
        if shares > 0 and px < entry_price * (1 - sl):
            cash   += shares * px * (1 - tc)
            shares  = 0.0
            sig     = 1

        if conf >= 0.55:
            if sig == 2 and shares == 0:
                buy_qty = cash / px
                cost    = buy_qty * px * (1 + tc)
                if cost <= cash:
                    shares, cash, entry_price = buy_qty, cash - cost, px
            elif sig == 0 and shares > 0:
                cash, shares, entry_price = cash + shares * px * (1 - tc), 0.0, 0.0

        pv  = cash + shares * px
        max_val = max(max_val, pv)
        records.append({
            "date": price_series.index[i], "price": px,
            "signal": sig, "confidence": conf,
            "portfolio_val": pv, "drawdown": (max_val - pv) / max_val,
        })

    bt = pd.DataFrame(records).set_index("date")
    bt["ret"] = bt["portfolio_val"].pct_change().fillna(0)
    return bt


def backtest_stats(bt, rfr=0.04):
    dr = bt["ret"]
    ann_ret = dr.mean() * 252
    ann_vol = dr.std()  * np.sqrt(252)
    sharpe  = (ann_ret - rfr) / (ann_vol + 1e-9)
    max_dd  = bt["drawdown"].max()
    total   = (bt["portfolio_val"].iloc[-1] / bt["portfolio_val"].iloc[0]) - 1
    return {
        "total_return_pct" : total   * 100,
        "annual_return_pct": ann_ret * 100,
        "annual_vol_pct"   : ann_vol * 100,
        "sharpe_ratio"     : sharpe,
        "max_drawdown_pct" : max_dd  * 100,
        "calmar_ratio"     : ann_ret / (max_dd + 1e-9),
        "win_rate_pct"     : (dr > 0).mean() * 100,
        "n_trades"         : int((bt["signal"].diff().abs() > 0).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 9: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def plot_signals(price, signals, ticker):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(price.index, price.values, color="#C0C0C0", lw=1.2, label="Close")
    buy_i  = signals[signals == 2].index
    sell_i = signals[signals == 0].index
    ax.scatter(buy_i,  price.loc[buy_i],  color="#51CF66", s=55, marker="^",
                zorder=3, label="BUY")
    ax.scatter(sell_i, price.loc[sell_i], color="#FF6B6B", s=55, marker="v",
                zorder=3, label="SELL")
    ax.set_title(f"{ticker} — Buy/Sell Signals", fontsize=14, fontweight="bold")
    ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_signals.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


def plot_equity(bt, price, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 1.5]})
    bh = (price / price.iloc[0]) * CFG["initial_capital"]
    axes[0].plot(bt.index, bt["portfolio_val"], color="#00D4FF", lw=2, label="Strategy")
    axes[0].plot(bh.index, bh.values, color="#FFD43B", lw=1.3, ls="--", label="Buy&Hold")
    axes[0].fill_between(bt.index, bt["portfolio_val"], CFG["initial_capital"],
                          alpha=0.12, color="#00D4FF")
    axes[0].set_ylabel("Portfolio ($)"); axes[0].legend()
    axes[0].set_title(f"{ticker} — Equity Curve", fontsize=14, fontweight="bold")
    axes[1].bar(bt.index, bt["ret"] * 100,
                 color=np.where(bt["ret"] >= 0, "#51CF66", "#FF6B6B"), width=1)
    axes[1].set_ylabel("Return (%)"); axes[1].axhline(0, color="w", lw=0.5)
    axes[2].fill_between(bt.index, -bt["drawdown"] * 100, 0,
                          color="#FF6B6B", alpha=0.7)
    axes[2].set_ylabel("Drawdown (%)"); axes[2].set_xlabel("Date")
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_equity.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


def plot_feature_importance(fi_df, ticker, top_n=30):
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(top["feature"][::-1], top["avg"][::-1],
             color=sns.color_palette("magma", len(top))[::-1])
    ax.set_title(f"Top {top_n} Feature Importances — {ticker}",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Importance")
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_feature_importance.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


def plot_confusion_matrix(cm, ticker):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                 xticklabels=["SELL","HOLD","BUY"],
                 yticklabels=["SELL","HOLD","BUY"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {ticker}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_cm.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


def plot_walk_forward(fold_metrics, ticker):
    folds  = [m["fold"]   for m in fold_metrics]
    acc    = [m["accuracy"] for m in fold_metrics]
    f1     = [m["f1_macro"] for m in fold_metrics]
    sharpe = [m["backtest"]["sharpe_ratio"] for m in fold_metrics]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, col in zip(
            axes, [acc, f1, sharpe],
            ["Accuracy", "F1-Macro", "Sharpe Ratio"],
            ["#00D4FF", "#845EF7", "#51CF66"]):
        ax.bar(folds, vals, color=col, alpha=0.85)
        ax.axhline(np.mean(vals), color="w", ls="--", lw=1.5,
                    label=f"μ={np.mean(vals):.3f}")
        ax.set_xlabel("Fold"); ax.set_title(title, fontweight="bold")
        ax.legend()
    fig.suptitle(f"Walk-Forward Performance — {ticker}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_walk_forward.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


def plot_lstm_history(history, ticker):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], color="#00D4FF", label="Train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], color="#FF6B6B", label="Val")
    axes[0].set_title("Loss", fontweight="bold"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], color="#51CF66", label="Train")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], color="#FFD43B", label="Val")
    axes[1].set_title("Accuracy", fontweight="bold"); axes[1].legend()
    plt.tight_layout()
    path = PLOT_DIR / f"{ticker}_lstm_history.png"
    plt.savefig(path, dpi=150); plt.close(fig)
    logger.info(f"  Chart → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 10: PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class StockPredictionPipeline:
    """
    Complete end-to-end pipeline:
      load → features → walk-forward train → backtest → predict → save
    """

    def __init__(self, ticker=None, cfg=None):
        self.cfg     = {**CFG, **(cfg or {})}
        self.ticker  = (ticker or self.cfg["ticker"]).upper()
        self.cfg["ticker"] = self.ticker

        self.features_df = None
        self.fc          = []

        # Will be set after final fold training
        self._lstm      = None
        self._xgb       = None
        self._lgb       = None
        self._rf        = None
        self._meta      = None
        self._scaler    = None
        self._ctx_sc    = None   # last-lookback scaled rows (context for LSTM)

        self.fold_metrics = []

    # ── Step 1 ────────────────────────────────────────────────────────────────
    def load_and_engineer(self):
        logger.info("\n" + "═" * 70)
        logger.info(f"  Loading & engineering features for {self.ticker}")
        raw            = DataAcquisition.download(self.ticker, self.cfg["start_date"],
                                                   self.cfg["end_date"], self.cfg["interval"])
        validated      = DataAcquisition.validate(raw)
        fe             = FeatureEngineer(self.cfg)
        self.features_df = fe.transform(validated)
        self.fc          = get_feature_cols(self.features_df)
        return self

    # ── Step 2 ────────────────────────────────────────────────────────────────
    def walk_forward_train(self):
        logger.info("\n" + "═" * 70)
        logger.info(f"  Walk-Forward Training ({self.cfg['n_splits']} folds)")
        df       = self.features_df
        fc       = self.fc
        lookback = self.cfg["lookback"]
        tscv     = TimeSeriesSplit(n_splits=self.cfg["n_splits"],
                                    test_size=self.cfg["test_size_days"])

        all_true, all_pred = [], []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(df[fc].values)):
            train_df = df.iloc[tr_idx]
            test_df  = df.iloc[te_idx]
            logger.info(f"\n  ── Fold {fold+1}/{self.cfg['n_splits']} "
                         f"[train={len(train_df)} | test={len(test_df)}] ──")

            X_tr = train_df[fc].values
            y_tr = train_df["target_class"].values.astype(int)
            X_te = test_df[fc].values
            y_te = test_df["target_class"].values.astype(int)

            # Internal 85/15 split for validation inside the fold
            split      = int(len(X_tr) * 0.85)
            X_tri, X_va = X_tr[:split], X_tr[split:]
            y_tri, y_va = y_tr[:split], y_tr[split:]

            # Scale
            scaler   = FeatureScaler("robust")
            X_tri_sc = scaler.fit_transform(X_tri)
            X_va_sc  = scaler.transform(X_va)
            X_te_sc  = scaler.transform(X_te)

            # ── Trees ──────────────────────────────────────────────────────
            xgb_m, lgb_m, rf_m = build_tree_models(self.cfg)
            train_trees(self.cfg, xgb_m, lgb_m, rf_m,
                         X_tri_sc, y_tri, X_va_sc, y_va)

            xgb_va = xgb_m.predict_proba(X_va_sc)
            lgb_va = lgb_m.predict_proba(X_va_sc)
            rf_va  = rf_m.predict_proba(X_va_sc)
            xgb_te = xgb_m.predict_proba(X_te_sc)
            lgb_te = lgb_m.predict_proba(X_te_sc)
            rf_te  = rf_m.predict_proba(X_te_sc)

            # ── LSTM ───────────────────────────────────────────────────────
            lstm, hist = train_lstm(self.cfg, X_tri_sc, y_tri, X_va_sc, y_va,
                                     len(fc))
            ctx_sc = X_tri_sc[-lookback:]
            lstm_te = predict_lstm(lstm, ctx_sc, X_te_sc, lookback)

            # Align lengths (LSTM may produce fewer rows)
            min_te = min(len(y_te), lstm_te.shape[0],
                          xgb_te.shape[0], lgb_te.shape[0], rf_te.shape[0])
            y_te    = y_te[-min_te:]
            lstm_te = lstm_te[-min_te:]
            xgb_te  = xgb_te[-min_te:]
            lgb_te  = lgb_te[-min_te:]
            rf_te   = rf_te[-min_te:]
            dates_te = test_df.index[-min_te:]
            prices_te = test_df["Close"].values[-min_te:]

            # LSTM val probas (align)
            lstm_va = predict_lstm(lstm, ctx_sc,
                                    X_va_sc, lookback)
            min_va  = min(len(y_va), lstm_va.shape[0],
                           xgb_va.shape[0], lgb_va.shape[0], rf_va.shape[0])
            y_va_al   = y_va[-min_va:]
            lstm_va   = lstm_va[-min_va:]
            xgb_va_al = xgb_va[-min_va:]
            lgb_va_al = lgb_va[-min_va:]
            rf_va_al  = rf_va[-min_va:]

            # ── Meta-learner ───────────────────────────────────────────────
            meta = build_meta(self.cfg)
            meta.fit(stack_features(lstm_va, xgb_va_al, lgb_va_al, rf_va_al), y_va_al)

            stack_te    = stack_features(lstm_te, xgb_te, lgb_te, rf_te)
            pred_labels = meta.predict(stack_te)
            pred_probas = meta.predict_proba(stack_te)

            # ── Metrics ───────────────────────────────────────────────────
            m = cls_metrics(y_te, pred_labels, pred_probas)
            logger.info(f"  Accuracy={m['accuracy']:.4f}  "
                         f"F1={m['f1_macro']:.4f}  "
                         f"AUC={m.get('roc_auc', 0):.4f}")
            logger.info(m["report"])

            # ── Backtest ──────────────────────────────────────────────────
            price_s  = pd.Series(prices_te, index=dates_te)
            bt_df    = run_backtest(price_s, pred_labels, pred_probas, self.cfg)
            bt_stat  = backtest_stats(bt_df)
            logger.info(f"  Sharpe={bt_stat['sharpe_ratio']:.3f}  "
                         f"Ret={bt_stat['total_return_pct']:.2f}%  "
                         f"MaxDD={bt_stat['max_drawdown_pct']:.2f}%")

            fold_rec = {"fold": fold + 1, **m, "backtest": bt_stat}
            self.fold_metrics.append(fold_rec)
            all_true.append(y_te)
            all_pred.append(pred_labels)

            # Keep last fold's models
            if fold == self.cfg["n_splits"] - 1:
                self._lstm    = lstm
                self._history = hist
                self._xgb     = xgb_m
                self._lgb     = lgb_m
                self._rf      = rf_m
                self._meta    = meta
                self._scaler  = scaler
                self._ctx_sc  = ctx_sc

        # ── Aggregate ─────────────────────────────────────────────────────
        logger.info("\n" + "═" * 70)
        logger.info("  AGGREGATE WALK-FORWARD RESULTS")
        agg = cls_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        logger.info(f"  Overall Accuracy: {agg['accuracy']:.4f}  "
                     f"F1-Macro: {agg['f1_macro']:.4f}")
        avg_s = np.mean([m["backtest"]["sharpe_ratio"]    for m in self.fold_metrics])
        avg_r = np.mean([m["backtest"]["total_return_pct"] for m in self.fold_metrics])
        avg_d = np.mean([m["backtest"]["max_drawdown_pct"] for m in self.fold_metrics])
        logger.info(f"  Avg Sharpe={avg_s:.3f}  "
                     f"Avg Ret={avg_r:.2f}%  "
                     f"Avg MaxDD={avg_d:.2f}%")

        plot_walk_forward(self.fold_metrics, self.ticker)
        plot_confusion_matrix(agg["confusion_matrix"], self.ticker)
        if self._history:
            plot_lstm_history(self._history, self.ticker)

        # Feature importance
        fi = pd.DataFrame({
            "feature": self.fc,
            "xgb":     self._xgb.feature_importances_,
            "lgb":     self._lgb.feature_importances_,
            "rf":      self._rf.feature_importances_,
        })
        fi["avg"] = fi[["xgb","lgb","rf"]].mean(axis=1)
        fi = fi.sort_values("avg", ascending=False).reset_index(drop=True)
        plot_feature_importance(fi, self.ticker)
        fi.to_csv(OUTPUT_DIR / f"{self.ticker}_feature_importance.csv", index=False)

        return self

    # ── Step 3: Full final backtest ───────────────────────────────────────────
    def full_backtest(self):
        logger.info("\n" + "═" * 70)
        logger.info("  Full Backtest (last 2 trading years)")
        df       = self.features_df
        fc       = self.fc
        lookback = self.cfg["lookback"]

        n_test   = 504                            # ~2 trading years
        test_df  = df.iloc[-n_test:]
        X_te     = test_df[fc].values
        X_te_sc  = self._scaler.transform(X_te)

        lstm_p   = predict_lstm(self._lstm, self._ctx_sc, X_te_sc, lookback)
        min_len  = min(len(test_df), lstm_p.shape[0])
        lstm_p   = lstm_p[-min_len:]
        X_te_sc2 = X_te_sc[-min_len:]
        xgb_p    = self._xgb.predict_proba(X_te_sc2)
        lgb_p    = self._lgb.predict_proba(X_te_sc2)
        rf_p     = self._rf.predict_proba(X_te_sc2)

        stack    = stack_features(lstm_p, xgb_p, lgb_p, rf_p)
        preds    = self._meta.predict(stack)
        probas   = self._meta.predict_proba(stack)

        prices   = pd.Series(test_df["Close"].values[-min_len:],
                              index=test_df.index[-min_len:])
        bt       = run_backtest(prices, preds, probas, self.cfg)
        stats    = backtest_stats(bt)

        logger.info("\n  ══ FINAL BACKTEST SUMMARY ══")
        for k, v in stats.items():
            logger.info(f"    {k:<25} : {v:.4f}")

        sig_s = pd.Series(preds, index=prices.index)
        plot_signals(prices, sig_s, self.ticker)
        plot_equity(bt, prices, self.ticker)

        bt.to_csv(OUTPUT_DIR / f"{self.ticker}_backtest.csv")
        return bt, stats

    # ── Step 4: Latest prediction ─────────────────────────────────────────────
    def predict_latest(self):
        logger.info("\n" + "═" * 70)
        logger.info("  Generating Latest Prediction")
        df       = self.features_df
        fc       = self.fc
        lookback = self.cfg["lookback"]

        X_last_sc = self._scaler.transform(df[fc].values[-1:])
        ctx_plus  = np.vstack([self._ctx_sc, X_last_sc])
        seq       = ctx_plus[-lookback:][np.newaxis, ...].astype(np.float32)
        lstm_p    = self._lstm.predict(seq, verbose=0)
        xgb_p     = self._xgb.predict_proba(X_last_sc)
        lgb_p     = self._lgb.predict_proba(X_last_sc)
        rf_p      = self._rf.predict_proba(X_last_sc)
        stack     = stack_features(lstm_p, xgb_p, lgb_p, rf_p)
        pred_lbl  = self._meta.predict(stack)[0]
        pred_prob = self._meta.predict_proba(stack)[0]

        label_map = {0: "SELL 🔴", 1: "HOLD 🟡", 2: "BUY  🟢"}
        signal    = int(pred_lbl)
        conf      = float(pred_prob[signal]) * 100
        last_px   = float(df["Close"].iloc[-1])
        last_dt   = df.index[-1].strftime("%Y-%m-%d")
        h         = self.cfg["predict_horizon"]

        logger.info(f"\n  ╔══════════════════════════════════════════════╗")
        logger.info(f"  ║  🤖  EDUCATION DIFFERENT PREDICTION          ║")
        logger.info(f"  ║  Ticker     : {self.ticker:<6}                       ║")
        logger.info(f"  ║  As-of date : {last_dt}                  ║")
        logger.info(f"  ║  Last Close : ${last_px:>8.2f}                    ║")
        logger.info(f"  ║  Horizon    : {h} trading days                ║")
        logger.info(f"  ╠══════════════════════════════════════════════╣")
        logger.info(f"  ║  Signal     : {label_map[signal]:>36}  ║")
        logger.info(f"  ║  Confidence : {conf:>6.2f}%                         ║")
        logger.info(f"  ║  SELL prob  : {pred_prob[0]*100:>6.2f}%                         ║")
        logger.info(f"  ║  HOLD prob  : {pred_prob[1]*100:>6.2f}%                         ║")
        logger.info(f"  ║  BUY  prob  : {pred_prob[2]*100:>6.2f}%                         ║")
        logger.info(f"  ╚══════════════════════════════════════════════╝\n")

        result = {
            "ticker"    : self.ticker,
            "date"      : last_dt,
            "close"     : last_px,
            "signal"    : label_map[signal],
            "signal_int": signal,
            "confidence": conf,
            "proba_sell": pred_prob[0],
            "proba_hold": pred_prob[1],
            "proba_buy" : pred_prob[2],
        }
        with open(OUTPUT_DIR / f"{self.ticker}_prediction.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    def save_all(self):
        logger.info("\n" + "═" * 70)
        logger.info("  Saving models …")
        self._lstm.save(str(MODEL_DIR / f"{self.ticker}_lstm"))
        joblib.dump({"xgb": self._xgb, "lgb": self._lgb, "rf": self._rf},
                     MODEL_DIR / f"{self.ticker}_trees.pkl")
        joblib.dump(self._meta,    MODEL_DIR / f"{self.ticker}_meta.pkl")
        self._scaler.save(          MODEL_DIR / f"{self.ticker}_scaler.pkl")
        with open(OUTPUT_DIR / f"{self.ticker}_config.json", "w") as f:
            json.dump(self.cfg, f, indent=2, default=str)
        with open(OUTPUT_DIR / f"{self.ticker}_fold_metrics.json", "w") as f:
            json.dump(self.fold_metrics, f, indent=2, default=str)
        logger.info("  ✓  All artifacts saved.")

    # ── Master run ────────────────────────────────────────────────────────────
    def run(self):
        t0 = time.time()
        logger.info("\n" + "╔" + "═" * 68 + "╗")
        logger.info("║" + " EDUCATION DIFFERENT — STOCK MARKET AI ".center(68) + "║")
        logger.info("║" + f" Ticker: {self.ticker}  |  Horizon: {self.cfg['predict_horizon']}d  |  Folds: {self.cfg['n_splits']} ".center(68) + "║")
        logger.info("╚" + "═" * 68 + "╝")

        self.load_and_engineer()
        self.walk_forward_train()
        bt_df, bt_stats = self.full_backtest()
        prediction = self.predict_latest()
        self.save_all()

        elapsed = time.time() - t0
        logger.info(f"\n  ✅  Done in {elapsed/60:.1f} min — outputs: {OUTPUT_DIR.resolve()}")
        return {"prediction": prediction, "backtest": bt_stats,
                "fold_metrics": self.fold_metrics}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 11: MULTI-TICKER LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
def run_multiple_tickers(tickers, cfg_override=None):
    """
    Train & predict for a list of tickers and return a sorted leaderboard.

    Example:
        board = run_multiple_tickers(["AAPL","MSFT","TSLA","NVDA"])
    """
    summary = []
    for tk in tickers:
        logger.info(f"\n\n{'='*70}\n  TICKER: {tk}\n{'='*70}")
        try:
            pipe = StockPredictionPipeline(ticker=tk, cfg=cfg_override)
            res  = pipe.run()
            summary.append({
                "ticker"        : tk,
                "signal"        : res["prediction"]["signal"],
                "confidence_pct": res["prediction"]["confidence"],
                "sharpe"        : res["backtest"]["sharpe_ratio"],
                "total_ret_pct" : res["backtest"]["total_return_pct"],
                "max_dd_pct"    : res["backtest"]["max_drawdown_pct"],
                "win_rate_pct"  : res["backtest"]["win_rate_pct"],
            })
        except Exception as exc:
            logger.error(f"  ✗ Failed for {tk}: {exc}")

    board = pd.DataFrame(summary).sort_values("sharpe", ascending=False)
    board.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    logger.info(f"\n  📊  Leaderboard:\n{board.to_string(index=False)}")
    return board


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    ─────────────────────────────────────────────────────────────────────────
    USAGE EXAMPLES
    ─────────────────────────────────────────────────────────────────────────
    [A] Single ticker (default AAPL):
          python "Education Different.py"

    [B] Custom ticker via CLI argument:
          python "Education Different.py" TSLA

    [C] Multiple tickers — leaderboard mode:
          Set MODE = "multi" below and edit TICKERS list.

    [D] Override config (e.g., faster run for testing):
          cfg_override = {"lstm_epochs": 20, "xgb_n_estimators": 200}
          pipe = StockPredictionPipeline(ticker="AAPL", cfg=cfg_override)
          pipe.run()
    ─────────────────────────────────────────────────────────────────────────
    """

    # ── Optional CLI override ─────────────────────────────────────────────────
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else CFG["ticker"]

    # ── Mode: "single" or "multi" ─────────────────────────────────────────────
    MODE    = "single"
    TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META"]

    if MODE == "single":
        pipe    = StockPredictionPipeline(ticker=ticker_arg)
        results = pipe.run()

        print("\n" + "=" * 55)
        print("  FINAL PREDICTION SUMMARY")
        print("=" * 55)
        p = results["prediction"]
        b = results["backtest"]
        print(f"  Ticker       : {p['ticker']}")
        print(f"  Date         : {p['date']}")
        print(f"  Last Close   : ${p['close']:.2f}")
        print(f"  Signal       : {p['signal']}")
        print(f"  Confidence   : {p['confidence']:.2f}%")
        print(f"  Sharpe Ratio : {b['sharpe_ratio']:.3f}")
        print(f"  Total Return : {b['total_return_pct']:.2f}%")
        print(f"  Max Drawdown : {b['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate     : {b['win_rate_pct']:.2f}%")
        print("=" * 55)

    elif MODE == "multi":
        board = run_multiple_tickers(TICKERS)
        print("\n" + "=" * 70)
        print("  MULTI-TICKER LEADERBOARD (sorted by Sharpe Ratio)")
        print("=" * 70)
        print(board.to_string(index=False))
