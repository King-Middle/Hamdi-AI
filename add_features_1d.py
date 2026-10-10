"""
Script to compute and add technical indicator features to XAU_1d_data.csv:
- ATR_14    : Average True Range (14-period, Wilder's smoothing)
- MACD_Hist : MACD Histogram (12/26/9 EMA)
- Log_Return: Logarithmic return of Close price
- BB_Width  : Bollinger Band Width ((Upper - Lower) / Middle * 100, 20-period, 2 std)
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

FILE_PATH = Path(__file__).parent / "XAU_1d_data.csv"
BACKUP_PATH = Path(__file__).parent / "XAU_1d_data.csv.bak"

def main():
    print(f"Reading {FILE_PATH} ...")
    df = pd.read_csv(FILE_PATH, sep=";")
    print(f"Loaded {len(df):,} rows. Existing columns: {list(df.columns)}")

    # 1. Backup original file
    if not BACKUP_PATH.exists():
        shutil.copyfile(FILE_PATH, BACKUP_PATH)
        print(f"Created backup at {BACKUP_PATH.name}")

    # Ensure correct sort order
    df["_dt"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df.sort_values("_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["_dt"], inplace=True)

    # 2. Average True Range (ATR_14) - Wilder's smoothing (alpha = 1/14)
    high_low = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # 3. MACD Histogram (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - signal

    # 4. Log Returns
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 5. Bollinger Band Width (20-period, 2 std, as % of middle band)
    window = 20
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    df["BB_Width"] = (upper_band - lower_band) / rolling_mean * 100

    # Save enriched file
    print(f"Writing updated data to {FILE_PATH} ...")
    df.to_csv(FILE_PATH, sep=";", index=False)
    print("Successfully updated XAU_1d_data.csv!")

    print("\nNew columns and non-null summary:")
    new_cols = ["ATR_14", "MACD_Hist", "Log_Return", "BB_Width"]
    for col in new_cols:
        val_sample = df[col].dropna().iloc[:3].tolist()
        val_latest = df[col].dropna().iloc[-3].tolist() if len(df[col].dropna()) >= 3 else []
        print(f"  {col:12s} -> First 3: {val_sample}, Latest: {df[col].iloc[-1]:.4f}")

    print("\nFirst 5 rows with new features (after warmup):")
    print(df.dropna().head(5)[["Date", "Close"] + new_cols].to_string(index=False))

    print("\nLast 5 rows:")
    print(df.tail(5)[["Date", "Close"] + new_cols].to_string(index=False))

if __name__ == "__main__":
    main()
