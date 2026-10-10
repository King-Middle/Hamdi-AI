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
import math
from tqdm import tqdm


tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

print()

# ==============================================================================
# DIRECTIONAL ACCURACY CONFIGURATION
# ==============================================================================
# Select the mode for checking directional accuracy:
# 'strict'   : Compare the prediction direction with the exact 10-day actual price change (TheResult6).
# 'flexible' : Compare the prediction direction with the flexible timing result (TheResult).
# 'any_day'  : Check if the prediction direction matches the sign of at least one day in the +/- 5 days window.
DIRECTION_ACCURACY_MODE = 'any_day' 
# ==============================================================================

simulated_today = date.today()

modelList = [
    '/Users/ardaunal/Desktop/Hamdi/Important Modelz/Hamdi_SeriÜretim_14_AAPL-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/King of the Court/Hamdi_SeriÜretim_15_2.0.keras'
]


SymbolList = [
    'aapl', 'msft', 'amzn', 'googl', 'goog', 
    'nvda', 'meta', 'tsla', 'jpm', 'jnj'
]

#SymbolList = ['aapl', 'msft', 'amzn', 'googl', 'goog', 'nvda', 'avgo', 'tsla', 'meta', 'wmt','tsm']


scoreDict = {}


for iothernumber in range(0,len(modelList)):
    
    AllTestScores = []
    for x in SymbolList:
        model = load_model(modelList[iothernumber])
        Symbol = x


        ChooseTheSymbolForOnce = Symbol.upper()


        ModelForOnce = 0

        TestScore = 0

        TheAmountDone = 0

        Test_Memory = []
        Test_Memory_Percentage = []

        start_date = date(2024, 5, 1)
        end_date = simulated_today

        TheRealPricesForATime = download(ChooseTheSymbolForOnce, start=date(2025, 1, 1), end=end_date, interval = '1d', timeout = 30, progress=False)['Close']

        TheRealPricesForATime = TheRealPricesForATime.asfreq('D')       # Her günü göster
        TheRealPricesForATime = TheRealPricesForATime.fillna(method='ffill')  # Eksik günlerde bir önceki değeri kullan


        JustThePrices = TheRealPricesForATime.squeeze().tolist()

        JustThePrices = [round(x) for x in JustThePrices]


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
            'WTI': 'USO',             # Ham Petrol (WTI)
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



        stock_df = download(Symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 30)
        price_on_date = float(stock_df["Close"].iloc[-1])

        for name, ticker in macro_symbols.items():
            temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 30)[['Close']]
            temp.rename(columns={'Close': name}, inplace=True)
            stock_df = stock_df.join(temp, how='inner')

        stock_df.dropna(inplace=True)



        def Hamdi(sembol, todaydate):

        
            global ModelForOnce
            global TheAmountDone
            global model
            global stock_df

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
            stock_dfSpecial = stock_df.loc[start_date:end_date].copy()

            # 🔹 4. Özellikleri seç
            features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(macro_symbols.keys())
            data = stock_dfSpecial[features].copy()

            # 🔹 5. Normalizasyon

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # 🔹 6. Inputu modele uygun hale getir
            X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

            # 🔹 7. Modeli yükle ve tahmin yap




            prediction = model.predict(X_input, verbose=0)[0][0]

            predicted_price = price_on_date * (1 + prediction / 100)

            

            Test_Memory.append(predicted_price)
            Test_Memory_Percentage.append(prediction)

            TheAmountDone += 1

        #*******************************************************************************


        FullPointsFactorLimit = 1000
        HalfPointsFactorLimit = 2000

        FullPointsFactorLimitInverse = 0
        HalfPointsFactorLimitInverse = 0


        for b in range(2025, end_date.year + 1):
            if b == end_date.year:                              # Determine the last month
                last_month = end_date.month - 1
            else:
                last_month = 12
            for i in range(1, last_month + 1):
                for a in range(1, 22, 10):
                    Hamdi(ChooseTheSymbolForOnce, date(b, i, a))
                    
                    TheOneToLookAt = (3 * (i - 1) + (a//10) + 1) * 10 # Gerçek verinin kaçıncı karakterine bakılacak = i + a + (10 gün sonrasından başlanıyor)

                    try:
                        TheResultMinus1 = ((JustThePrices[TheOneToLookAt-7] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult0 = ((JustThePrices[TheOneToLookAt-6] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
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
                        TheResultC = ((JustThePrices[TheOneToLookAt+6] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResultD = ((JustThePrices[TheOneToLookAt+7] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                    except:
                        False

                    AddingScore = 0

                    PlusMinusFiveDaysList = [TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA, TheResultB]   # ± 5 gün hesaplama ve doğru güne bakma

                    PlusMinusFiveDaysListOtherVersion = [TheResultMinus1,TheResult0, TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA, TheResultB, TheResultC, TheResultD] # Ekstra liste 1 gün önce ve sonrayı bulabilmek için

                    ClosestDay = min(PlusMinusFiveDaysList, key=lambda x: abs(x - Test_Memory_Percentage[-1]))   # Sıradaki 5 gün içerisinden en mantıklı olanı bulma fonksiyonu

                    indexOfClosestDay = PlusMinusFiveDaysList.index(ClosestDay)

                    indexOfClosestDayOtherVersion = indexOfClosestDay + 2

                    TheResult = np.mean([ClosestDay, TheResult6, ClosestDay, ClosestDay])   # Ana sonucu en yakın güne eşitleme
                    

                    # ------------------------------------------------------------------
                    # Puan hesaplama (Directional Accuracy Only)
                    # ------------------------------------------------------------------
                    pred_pct = Test_Memory_Percentage[-1]

                    if DIRECTION_ACCURACY_MODE == 'strict':
                        target_val = TheResult6
                    elif DIRECTION_ACCURACY_MODE == 'flexible':
                        target_val = TheResult
                    else: # 'any_day'
                        target_val = None

                    if target_val is not None:
                        is_correct = (target_val > 0 and pred_pct > 0) or \
                                     (target_val < 0 and pred_pct < 0) or \
                                     (target_val == 0 and pred_pct == 0)
                    else:
                        is_correct = any((day > 0 and pred_pct > 0) or (day < 0 and pred_pct < 0) or (day == 0 and pred_pct == 0) for day in PlusMinusFiveDaysList)

                    AddingScore = 1.0 if is_correct else 0.0
                    TestScore += AddingScore

                    # Determine directions for display
                    pred_dir = "INCREASE 📈" if pred_pct > 0 else ("DECREASE 📉" if pred_pct < 0 else "NO CHANGE ➡️")
                    real_dir = "INCREASE 📈" if (target_val if target_val is not None else ClosestDay) > 0 else \
                               ("DECREASE 📉" if (target_val if target_val is not None else ClosestDay) < 0 else "NO CHANGE ➡️")
                    match_status = "✅ CORRECT" if is_correct else "❌ WRONG"

                    print(f"[{ChooseTheSymbolForOnce}] Date: {date(b, i, a)} | Guess: {pred_dir} ({pred_pct:+.2f}%) | Reality: {real_dir} ({(target_val if target_val is not None else ClosestDay):+.2f}%) | {match_status}")






        TestScore = int(TestScore/ len(Test_Memory) * 100)  # Yüzde olarak hesaplama

        AllTestScores.append(TestScore)

    AllTestScores = sum(AllTestScores)/len(AllTestScores)

    scoreDict[iothernumber+1] = AllTestScores


    print(iothernumber+1)
    print("*****************")
    print("Accuracy:",AllTestScores,"%")
    print("*****************")


plt.figure(figsize=(15,8))
plt.plot(list(scoreDict.keys()), list(scoreDict.values()))
plt.title = x
plt.xlabel('models')
plt.ylabel('TestScores')
plt.grid()
plt.legend()
plt.show()