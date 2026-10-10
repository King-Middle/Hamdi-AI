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
import matplotlib.pyplot as plt
import signal


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


ChooseTheSymbolForOnce = 'AAPL'


Test_Memory = []
Test_Memory_Percentage = []

TheRealPricesForATime = download(ChooseTheSymbolForOnce, start=date(2025, 1, 1), end=date(2025, 10, 1), interval = '1d')['Close']


TheRealPricesForATime = TheRealPricesForATime.asfreq('D')       # Her günü göster
TheRealPricesForATime = TheRealPricesForATime.fillna(method='ffill')  # Eksik günlerde bir önceki değeri kullan


JustThePrices = TheRealPricesForATime.squeeze().tolist()

JustThePrices = [round(x) for x in JustThePrices]

print (len(JustThePrices))


macro_symbols = {
    'VIX': '^VIX',        # Volatilite endeksi
    'DXY': 'DX-Y.NYB',    # Dolar endeksi
    'US10Y': '^TNX',      # ABD 10 yıllık tahvili
    'XAUUSD': 'GC=F',     # Altın
    'SPX': '^GSPC',       # S&P 500
    'DJI': '^DJI',        # Dow Jones
    'IXIC': '^IXIC',      # Nasdaq
    'RUT': '^RUT',        # Russell 2000
}




def Hamdi(sembol, todaydate):


  today = todaydate



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

  decimals = 0

  # 🔹 2. Hisse ve makro verileri çek
  stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
  price_on_date = float(stock_df["Close"].iloc[-1])

  for name, ticker in macro_symbols.items():
      temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)[['Close']]
      temp.rename(columns={'Close': name}, inplace=True)
      stock_df = stock_df.join(temp, how='inner')

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
  
  if sembol == 'BTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_1.0.keras')
  elif sembol == 'TSM':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSM_Anaconda_1.0.keras')
  elif sembol == 'GOOG':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_GOOG_Anaconda_1.0.keras')
  elif sembol == 'AAPL':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AAPL_Anaconda_1.0.keras')
  elif sembol == 'ETH-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_1.0.keras')
  elif sembol == 'MSFT':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_MSFT_Anaconda_1.0.keras')
  elif sembol == 'LTC-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_1.0.keras')
    decimals = 2
  elif sembol == 'AMZN':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_GOOG_Anaconda_1.0.keras')
  elif sembol == 'LINK-USD':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_BTC_Anaconda_1.0.keras')
    decimals = 2
  elif sembol == 'TSLA':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_TSLA_Anaconda_1.0.keras')
  elif sembol == 'META':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_META_Anaconda_1.0.keras')
  elif sembol == 'AVGO':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_AVGO_Anaconda_1.0.keras')
  elif sembol == 'UBER':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_UBER_Anaconda_1.0.keras')
  elif sembol == 'DASH':
    model = load_model('/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_DASH_Anaconda_1.0.keras')
  else:
    print("Hatalı sembol")
    return

  prediction = model.predict(X_input, verbose=0)[0][0]

  predicted_price = price_on_date * (1 + prediction / 100)

  print(f"\n{sembol}: {price_on_date:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))


  if prediction < -0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[91m{prediction:.2f} % ▼\033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))
  elif prediction > 0.01:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[92m{prediction:.2f} % ▲\033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))
  else:
    print(f"{sembol}: {guess_date} tahmini fiyat: \033[89m0.00 % \033[0m  -  {predicted_price:,.{decimals}f} $\n".replace(",", "?").replace(".", ",").replace("?", "."))
  

  Test_Memory.append(predicted_price)
  Test_Memory_Percentage.append(prediction)

#*******************************************************************************


for i in range(1, 10):
    for a in range(1, 22, 10):
        Hamdi(ChooseTheSymbolForOnce, date(2025, i, a))


        TheOneToLookAt = (3 * (i - 1) + (a//10) + 1) * 10 # Gerçek verinin kaçıncı karakterine bakılacak = i + a + (10 gün sonrasından başlanıyor)

        print (TheOneToLookAt)

        TheResult1 = ((JustThePrices[TheOneToLookAt-5] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult2 = ((JustThePrices[TheOneToLookAt-4] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult3 = ((JustThePrices[TheOneToLookAt-3] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult4 = ((JustThePrices[TheOneToLookAt-2] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult5 = ((JustThePrices[TheOneToLookAt-1] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult6 = ((JustThePrices[TheOneToLookAt] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma   Normal değer
        TheResult7 = ((JustThePrices[TheOneToLookAt+1] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult8 = ((JustThePrices[TheOneToLookAt+2] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResult9 = ((JustThePrices[TheOneToLookAt+3] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResultA = ((JustThePrices[TheOneToLookAt+4] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
        TheResultB = ((JustThePrices[TheOneToLookAt+5] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma

        AddingScore = 0

        PlusMinusFiveDaysList = [TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA, TheResultB]   # ± 5 gün hesaplama ve doğru güne bakma

        ClosestDay = min(PlusMinusFiveDaysList, key=lambda x: abs(x - Test_Memory_Percentage[-1]))   # Sıradaki 5 gün içerisinden en mantıklı olanı bulma fonksiyonu

        TheResult = ClosestDay   # Ana sonucu en yakın güne eşitleme


        # Puan hesaplama

        if (TheResult > 0 and Test_Memory_Percentage[-1] > 0) or (TheResult < 0 and Test_Memory_Percentage[-1] < 0):
            AddingScore = 0.5
            if (TheResult / Test_Memory_Percentage[-1] < 5 or Test_Memory_Percentage[-1] / TheResult < 5):
                AddingScore = 1
        elif (abs(abs(TheResult) - abs(Test_Memory_Percentage[-1])) < 2):
            AddingScore = 0.5
            
        else:
            AddingScore = 0

        TestScore += AddingScore


TestScore = int(TestScore/ len(Test_Memory) * 100)  # Yüzde olarak hesaplama

print("Accuracy: ",TestScore,"%")



x_dates_real = TheRealPricesForATime.index

# Tahmin noktaları 10 günde bir (+1 eklenmiş)
step = len(x_dates_real) // len(Test_Memory) + 0

# Tahmin noktalarını al
x_dates_guess_shifted = x_dates_real[step::step]  # bir adım kaydırma
# Boyut uyuşmazlığını önle
min_len = min(len(x_dates_guess_shifted), len(Test_Memory))
x_dates_guess_shifted = x_dates_guess_shifted[:min_len]
Test_Memory_trimmed = Test_Memory[:min_len]

plt.figure(figsize=(15, 8))
plt.plot(x_dates_real, TheRealPricesForATime, label='Reality (daily)', color='red')
plt.plot(x_dates_guess_shifted, Test_Memory_trimmed, label='Guess', color='blue', marker='o')

plt.title('Test Graph')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


pid = os.getppid()
os.kill(pid, signal.SIGKILL)