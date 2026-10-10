from yfinance import download
import pandas as pd
from datetime import date, timedelta, datetime
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import time
import warnings
import os
import tensorflow # type: ignore
import signal


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

today_hour = datetime.now() - timedelta(hours=0)


back_hour = 72
start_date = today_hour - timedelta(hours=back_hour)
end_date = today_hour
guess_date = today_hour + timedelta(hours=10)

file_name = ''

def Hamdi(sembol):


  stock_symbol = sembol

  decimals = 4

  # 🔹 2. Hisse ve makro verileri çek
  stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1h', auto_adjust=True, progress=False)
  stock_df2 = download(stock_symbol, start=today_hour - timedelta(hours=1), end=today_hour + timedelta(hours=1), interval='1h', auto_adjust=True, progress=False)

  def add_indicators(df):
    # EMA (20 günlük)
    df['EMA_20'] = round(df['Close'].ewm(span=20, adjust=False).mean(), 4)
    
    # RSI (14 günlük) - Güvenli hesaplama
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # + 1e-10 ekleyerek sıfıra bölünmeyi önledik
    rs = gain / (loss + 1e-10) 
    df['RSI'] = round(100 - (100 / (1 + rs)), 4)
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = round(exp1 - exp2, 4)
    df['Signal_Line'] = round(df['MACD'].ewm(span=9, adjust=False).mean(), 4)
    
    return df.round(4)

  stock_df = add_indicators(stock_df)

  price_on_date = None
  if not stock_df2.empty:
    try:
      if isinstance(stock_df2.columns, pd.MultiIndex):
        close_series = stock_df2["Close"].iloc[:, 0] if isinstance(stock_df2["Close"], pd.DataFrame) else stock_df2["Close"]
      else:
        close_series = stock_df2["Close"]
      close_series = close_series.dropna()
      if not close_series.empty:
        price_on_date = float(close_series.iloc[-1])
    except Exception:
      pass

  if price_on_date is None:
    try:
      if isinstance(stock_df.columns, pd.MultiIndex):
        close_series = stock_df["Close"].iloc[:, 0] if isinstance(stock_df["Close"], pd.DataFrame) else stock_df["Close"]
      else:
        close_series = stock_df["Close"]
      close_series = close_series.dropna()
      if not close_series.empty:
        price_on_date = float(close_series.iloc[-1])
    except Exception:
      pass

  if price_on_date is None:
    raise ValueError(f"Could not retrieve close price for {stock_symbol}")


  # 🔹 3. Eksik satırları temizle
  if isinstance(stock_df.columns, pd.MultiIndex):
    has_close = stock_df[('Close', stock_symbol)].notna()
  else:
    has_close = stock_df['Close'].notna()

  stock_df.ffill(inplace=True)
  stock_df.bfill(inplace=True)

  stock_df = stock_df.loc[has_close]

  if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = stock_df.columns.get_level_values(0)


  features = ['Open', 'High', 'Low', 'Close', 'Volume']
  data = stock_df[features].copy()

  data.dropna(inplace=True)

  data = data.assign(Close=data['Close'].pct_change().fillna(0))

  # 🔹 5. Normalizasyon

  scaler = joblib.load('Hamdi_' + file_name + '_ForexScaler_2.0.joblib')
  scaled_data = scaler.fit_transform(data)

  # 🔹 6. Inputu modele uygun hale getir
  X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

  # 🔹 7. Modeli yükle ve tahmin yap

  model = load_model('/Users/ardaunal/Desktop/Hamdi/Forex/Hamdi_' + file_name + '_Forex_2.0.keras')

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

Hamdi('EURUSD=X')



print("The End\n\n")
input()


pid = os.getppid()
os.kill(pid, signal.SIGKILL)
