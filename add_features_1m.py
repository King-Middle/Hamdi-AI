"""
Script to compute and add technical indicator features to XAU_1m_data copy.csv:
- BB_Upper      : Bollinger Upper Band (20-period SMA + 2 * rolling std)
- BB_Middle     : Bollinger Middle Band (20-period SMA)
- BB_Lower      : Bollinger Lower Band (20-period SMA - 2 * rolling std)
- BB_Width      : Bollinger Band Width ((Upper - Lower) / Middle * 100)
- StdDev_15     : Rolling Standard Deviation (15-period)
- StdDev_Deriv  : Derivative (diff) of StdDev_15
"""

import shutil
from pathlib import Path
import pandas as pd

FILE_PATH = Path(__file__).parent / "XAU_1m_data copy.csv"
BACKUP_PATH = Path(__file__).parent / "XAU_1m_data copy.csv.bak"


def main():
    # Read from backup if it exists (clean data without prior feature columns)
    source = BACKUP_PATH if BACKUP_PATH.exists() else FILE_PATH
    print(f"Reading {source} ...")
    df = pd.read_csv(source, sep=";")
    print(f"Loaded {len(df):,} rows. Existing columns: {list(df.columns)}")

    # Backup original file if not already done
    if not BACKUP_PATH.exists():
        shutil.copyfile(FILE_PATH, BACKUP_PATH)
        print(f"Created backup at {BACKUP_PATH.name}")

    # Ensure correct sort order
    df["_dt"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df.sort_values("_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["_dt"], inplace=True)

    # ── Bollinger Bands (period=20, deviation=2) ──────────────────────
    bb_period = 20
    bb_dev = 2

    bb_middle = df["Close"].rolling(bb_period).mean()
    bb_std = df["Close"].rolling(bb_period).std()

    df["BB_Upper"] = bb_middle + bb_dev * bb_std
    df["BB_Middle"] = bb_middle
    df["BB_Lower"] = bb_middle - bb_dev * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"] * 100

    # ── Standard Deviation (period=15) + Derivative ───────────────────
    sd_period = 15

    df["StdDev_15"] = df["Close"].rolling(sd_period).std()
    df["StdDev_Deriv"] = df["StdDev_15"].diff()

    # Save enriched file
    print(f"Writing updated data to {FILE_PATH} ...")
    df.to_csv(FILE_PATH, sep=";", index=False)
    print("Successfully updated XAU_1m_data copy.csv!")

    # Summary
    new_cols = ["BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
                "StdDev_15", "StdDev_Deriv"]

    print("\nNew columns and non-null summary:")
    for col in new_cols:
        non_null = df[col].notna().sum()
        sample = df[col].dropna().iloc[:3].tolist() if non_null >= 3 else []
        latest = f"{df[col].iloc[-1]:.6f}" if non_null > 0 else "N/A"
        print(f"  {col:14s} -> non-null: {non_null:,}, First 3: {sample}, Latest: {latest}")

    print("\nFirst 5 rows with new features (after warmup):")
    print(df.dropna().head(5)[["Date", "Close"] + new_cols].to_string(index=False))

    print("\nLast 5 rows:")
    print(df.tail(5)[["Date", "Close"] + new_cols].to_string(index=False))


if __name__ == "__main__":
    main()
