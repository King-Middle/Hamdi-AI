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
import tensorflow
import signal


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

macro_symbols = {}


today = date.today()
#today = date(2025, 5, 1)


if today == date.today():
  today_hour = datetime.now()

  if today_hour.hour < 3:
    today = today + timedelta(0)
  else:
    today = today + timedelta(1)

else:
  today = today + timedelta(1)


back_date = 18
start_date = today - timedelta(hours=back_date)
end_date = today
guess_date = today + timedelta(9)



def Hamdi(sembol):


  stock_symbol = sembol

  decimals = 0

  # 🔹 2. Hisse ve makro verileri çek
  stock_df = download(stock_symbol, period='1d', interval='1h', auto_adjust=True, progress=False)
  stock_df = stock_df.tail(18)
  stock_df2 = download(stock_symbol, start=date.today()- timedelta(1), end=date.today() + timedelta(1), interval='1m', auto_adjust=True, progress=False)
  price_on_date = float(stock_df2["Close"].iloc[-1])

  # 🔹 3. Eksik satırları temizle
  stock_df.dropna(inplace=True)

  # 🔹 4. Özellikleri seç
  features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(macro_symbols.keys())
  data = stock_df[features].copy()

  # 🔹 5. Normalizasyon

  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)

  # 🔹 6. Inputu modele uygun hale getir
  X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

  # 🔹 7. Modeli yükle ve tahmin yap

  model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_BTC2_Decision_3.0.keras')

  prediction = model.predict(X_input, verbose=0)[0][0]

  # Binary classification: prediction is probability (0-1)
  # > 0.5 = Increase, < 0.5 = Decrease
  decision = "Increase" if prediction > 0.5 else "Decrease"
  confidence = prediction if prediction > 0.5 else (1 - prediction)

  print(f"\n{sembol}: {price_on_date:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))

  if decision == "Decrease":
    print(f"{sembol}: \033[91m{decision} (Confidence: {confidence:.2%}) ▼\033[0m\n")
  else:
    print(f"{sembol}: \033[92m{decision} (Confidence: {confidence:.2%}) ▲\033[0m\n")

#*******************************************************************************

Hamdi('BTC-USD')



print("The End\n\n")
input()


pid = os.getppid()
os.kill(pid, signal.SIGKILL)