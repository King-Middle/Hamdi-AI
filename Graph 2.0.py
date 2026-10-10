from yfinance import download
import pandas as pd
from datetime import date, timedelta, datetime
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import os
import tensorflow
import signal


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


Symbol = 'goog'

Symbol = Symbol.upper()


guess_dates = []
predicted_prices = []

macro_symbols = {
    # --- Risk & Volatilite ---
    'VIX': '^VIX',             # Volatilite endeksi

    # --- Tahviller & Faiz ---
    'US10Y': '^TNX',           # ABD 10 yıllık tahvil faizi
    'US3M': '^IRX',            # ABD 3 aylık hazine bonosu
    'US30Y': '^TYX',           # ABD 30 yıllık tahvil faizi

    # --- Emtialar ---
    'XAUUSD': 'GC=F',          # Altın
    'XAGUSD': 'SI=F',          # Gümüş
    'WTI': 'CL=F',             # Ham Petrol (WTI)
    'Brent': 'BZ=F',           # Brent Petrol
    'NaturalGas': 'NG=F',      # Doğalgaz
    'Copper': 'HG=F',          # Bakır
    'Platinum': 'PL=F',        # Platin
    'Palladium': 'PA=F',       # Paladyum

    # --- ABD Endeksleri ---
    'SPX': '^GSPC',            # S&P 500
    'DJI': '^DJI',             # Dow Jones
    'IXIC': '^IXIC',           # Nasdaq
    'RUT': '^RUT',             # Russell 2000

    # --- Global Endeksler ---
    'FTSE': '^FTSE',           # İngiltere FTSE 100
    'Nikkei': '^N225',         # Japonya Nikkei 225
    'HangSeng': '^HSI',        # Hong Kong Hang Seng
    'EuroStoxx50': 'FEZ',# Euro Bölgesi STOXX50

    # --- Kripto ---
    'BTC': 'BTC-USD',          # Bitcoin
    'ETH': 'ETH-USD',          # Ethereum
    'SOL': 'SOL-USD',          # Solana
}

predicted_dates = []
predicted_prices = []


# 🔹 1. Tarihler





def Hamdi(sembol, date):

  today = date
  if today == date.today():
    today_hour = datetime.now()

    if today_hour.hour < 3:
      today = today + timedelta(0)
    else:
      today = today + timedelta(1)

  else:
    today = today + timedelta(1)




  back_date = 180
  start_date = today - timedelta(back_date)
  end_date = today
  guess_date = today + timedelta(9)

  stock_symbol = sembol

  # 🔹 2. Hisse ve makro verileri çek
  stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
  price_on_date = float(stock_df["Close"].iloc[-1])

  for name, ticker in macro_symbols.items():
      temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)[['Close']]
      temp.rename(columns={'Close': name}, inplace=True)
      stock_df = stock_df.join(temp, how='inner')

  # 🔹 3. Eksik satırları temizle
  stock_df.fillna(method='ffill')


  # 🔹 4. Özellikleri seç
  features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(macro_symbols.keys())
  data = stock_df[features].copy()

  # 🔹 5. Normalizasyon

  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)

  # 🔹 6. Inputu modele uygun hale getir
  X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

  # 🔹 7. Modeli yükle ve tahmin yap

  if sembol == 'BTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_2.0.keras')
  elif sembol == 'TSM':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSM_Anaconda_2.0.keras')
  elif sembol == 'GOOG':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_GOOG_Anaconda_2.0.keras')
  elif sembol == 'AAPL':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AAPL_Anaconda_2.0.keras')
  elif sembol == 'ETH-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_2.0.keras')
  elif sembol == 'MSFT':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_MSFT_Anaconda_2.0.keras')
  elif sembol == 'LTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_2.0.keras')
    decimals = 2
  elif sembol == 'AMZN':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AMZN_Anaconda_2.0.keras')
  elif sembol == 'LINK-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_2.0.keras')
    decimals = 2
  elif sembol == 'TSLA':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSLA_Anaconda_2.0.keras')
  elif sembol == 'META':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_META_Anaconda_2.0.keras')
  elif sembol == 'AVGO':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AVGO_Anaconda_2.0.keras')
  elif sembol == 'UBER':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_UBER_Anaconda_2.0.keras')
  elif sembol == 'DASH':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_DASH_Anaconda_2.0.keras')
  elif sembol == 'NFLX':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_NFLX_Anaconda_2.0.keras')
  elif sembol == 'NVDA':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_NVDA_Anaconda_2.0.keras')
  elif sembol == 'DUOL':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_DUOL_Anaconda_2.0.keras')
  else:
    print("Hatalı sembol")
    return

  prediction = model.predict(X_input, verbose=0)[0][0]

  predicted_price = price_on_date * (1 + prediction / 100)

  print(f"\n{sembol}: {int(price_on_date):,} $".replace(",", "."))


  if prediction < -0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[91m{prediction:.2f} % ▼\033[0m  -  {predicted_price:,.0f} $".replace(",", "."))
  elif prediction > 0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[92m{prediction:.2f} % ▲\033[0m  -  {predicted_price:,.0f} $".replace(",", "."))
  else:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[89m0.00 % \033[0m  -  {predicted_price:.0f:,.0f} $".replace(",", "."))

  predicted_prices.append(predicted_price)

#*******************************************************************************

guess_dates.append(date.today())
special_stock_df = download(Symbol, start = date.today()-timedelta(1), end = date.today()+timedelta(1), interval='1d', auto_adjust=True, progress=False)
special_stock_df.fillna(method='ffill')
special_price_on_date = float(special_stock_df["Close"].iloc[-1])
predicted_prices.append(special_price_on_date)

for i in range(10, -1, -1):
  changing_date = date.today() - timedelta(i)
  Hamdi(Symbol, changing_date)
  changing_date = changing_date + timedelta(10)
  guess_dates.append(changing_date)


plt.figure(figsize=(15, 8))
plt.title(Symbol)
plt.plot(guess_dates, predicted_prices, marker='o')
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.axhline(y=special_price_on_date, linestyle='--', color = 'orange')
plt.grid(True)
plt.xticks(guess_dates, [d.strftime("%Y-%m-%d") for d in guess_dates], rotation=45)
plt.legend()
plt.show()


pid = os.getppid()
os.kill(pid, signal.SIGKILL)