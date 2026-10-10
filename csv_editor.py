import pandas as pd
import numpy as np

# 1. Yüzdelik veriyi yüklüyoruz
df = pd.read_csv('BTCUSD_15m_ready.csv', sep=';')





df.to_csv('BTCUSD_15m_ready.csv', index=False, sep=';')

print("1 saatlik (4 mumluk) hedef başarıyla tanımlandı!")
print(df.head())
print(df.tail())