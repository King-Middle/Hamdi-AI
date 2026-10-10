from yfinance import download
import pandas as pd
from datetime import date, timedelta, datetime
from tensorflow.keras.models import load_model # type: ignore #type:
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import time
import warnings
import os
import tensorflow
import signal


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

macro_symbols = {
    'VIX': '^VIX',        # Volatility index
    'DXY': 'DX-Y.NYB',    # Dolar index
    'US10Y': '^TNX',      # CBOE Interest Rate 10 Years
    'XAUUSD': 'GC=F',     # Gold
    'SPX': '^GSPC',       # S&P 500
    'DJI': '^DJI',        # Dow Jones
    'IXIC': '^IXIC',      # Nasdaq
    'RUT': '^RUT',        # Russell 2000
}

# 1. Dates

today = date.today()
#today = date(2025, 11, 1)



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

def Hamdi(sembol):

  stock_symbol = sembol

  decimals = 0

  # 2. Stock and Macro economics
  stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
  price_on_date = float(stock_df["Close"].iloc[-1])

  for name, ticker in macro_symbols.items():
      temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)[['Close']]
      temp.rename(columns={'Close': name}, inplace=True)
      stock_df = stock_df.join(temp, how='inner')

  # 3. Delete the missing lines
  stock_df.dropna(inplace=True)

  # 4. Features
  features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(macro_symbols.keys())
  data = stock_df[features].copy()

  # 5. Normalization

  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)

  # 6. Inputu modele uygun hale getir
  X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

  # 7. Modeli yükle ve tahmin yap

  if sembol == 'BTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda.keras')
  elif sembol == 'TSM':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSM_Anaconda.keras')
  elif sembol == 'GOOG':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_GOOG_Anaconda.keras')
  elif sembol == 'AAPL':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AAPL_Anaconda.keras')
  elif sembol == 'NVDA':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_NVDA_Anaconda.keras')
  elif sembol == 'ETH-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda.keras')
  elif sembol == 'MSFT':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_MSFT_Anaconda.keras')
  elif sembol == 'LTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda.keras')
    decimals = 2
  elif sembol == 'AMZN':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AMZN_Anaconda.keras')
  elif sembol == 'LINK-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda.keras')
    decimals = 2
  elif sembol == 'TSLA':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSLA_Anaconda.keras')
  else:
    print("Hatalı sembol")
  
  model = load_model(os.path.join(os.path.dirname(__file__), "Models", "Hamdi_NVDA_Anaconda.keras"))

  prediction = model.predict(X_input, verbose=0)[0][0]

  predicted_price = price_on_date * (1 + prediction / 100)

  print(f"\n{sembol}: {price_on_date:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))


  if prediction < -0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[91m{prediction:.2f} % ▼\033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))
  elif prediction > 0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[92m{prediction:.2f} % ▲\033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))
  else:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[89m0.00 % \033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))

#*******************************************************************************

Hamdi('NVDA')


input()

pid = os.getppid()
os.kill(pid, signal.SIGKILL)