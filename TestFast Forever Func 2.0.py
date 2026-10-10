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

modelList = [
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_10_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_11_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_12_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_13_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_14_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_15_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_16_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_17_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_18_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_19_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_20_PLTR-Version_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_SeriÜretim_15_2.0.keras',
    '/Users/ardaunal/Desktop/Hamdi/Hamdi_Anaconda/Hamdi_GOOGDENEME_Anaconda_2.0.keras'
]


SymbolList = [
    'aapl', 'msft', 'amzn', 'googl', 'goog', 'nvda',
    'meta', 'tsla', 'brk.b', 'jpm', 'jnj', 'v', 'ma',
    'unh', 'xom', 'pg', 'lly', 'avgo', 'hd', 'ko',
    'pep', 'csco', 'nflx', 'adbe', 'crm', 'wmt',
    'mcd', 'cvx', 'intc', 'tsm'
]

for x in SymbolList:
    Symbol = x
    scoreDict = {}
    for iothernumber in range(0,len(modelList)):
        model = load_model(modelList[iothernumber])


        ChooseTheSymbolForOnce = Symbol.upper()


        ModelForOnce = 0

        TestScore = 0

        TheAmountDone = 0

        Test_Memory = []
        Test_Memory_Percentage = []

        start_date = date(2024, 5, 15)
        end_date = date.today()

        TheRealPricesForATime = download(ChooseTheSymbolForOnce, start=date(2025, 1, 1), end=end_date, interval = '1d', timeout = 30)['Close']

        TheRealPricesForATime = TheRealPricesForATime.asfreq('D')       # Her günü göster
        TheRealPricesForATime = TheRealPricesForATime.fillna(method='ffill')  # Eksik günlerde bir önceki değeri kullan


        JustThePrices = TheRealPricesForATime.squeeze().tolist()

        JustThePrices = [round(x) for x in JustThePrices]


        print("\n",ChooseTheSymbolForOnce,"\n")

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



        for i in range(1, end_date.month):
            for a in range(1, 22, 10):
                Hamdi(ChooseTheSymbolForOnce, date(2025, i, a))
                
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

                # Günlük sapmalara ekstra puan vermesini önlemek

                DeletingScore = 0

                if (indexOfClosestDay != 0 and indexOfClosestDay != 10):
                    FindingDeletingScoreWay1 = (abs(PlusMinusFiveDaysList[indexOfClosestDay] - PlusMinusFiveDaysList[indexOfClosestDay - 1]) - 2) * 0.05
                    FindingDeletingScoreWay2 = (abs(PlusMinusFiveDaysList[indexOfClosestDay] - PlusMinusFiveDaysList[indexOfClosestDay + 1]) - 2) * 0.05
                    DeletingScore = min(FindingDeletingScoreWay1, FindingDeletingScoreWay2, 0.25)                

                elif  (indexOfClosestDay == 0):
                    DeletingScore = (abs(PlusMinusFiveDaysList[indexOfClosestDay] - PlusMinusFiveDaysList[indexOfClosestDay + 1]) - 3) * 0.025
                    DeletingScore = min(DeletingScore, 0.25)

                else:
                    DeletingScore = (abs(PlusMinusFiveDaysList[indexOfClosestDay] - PlusMinusFiveDaysList[indexOfClosestDay - 1]) - 3) * 0.025
                    DeletingScore = min(DeletingScore, 0.25)
                
                if (DeletingScore < 0):
                    DeletingScore = 0


                # Puan hesaplama

                if (TheResult > 0 and Test_Memory_Percentage[-1] > 0) or (TheResult < 0 and Test_Memory_Percentage[-1] < 0):
                    if (TheResult / Test_Memory_Percentage[-1] < 2 and TheResult / Test_Memory_Percentage[-1] > 1/2):
                        AddingScore = 1
                    elif (TheResult / Test_Memory_Percentage[-1] < 3 and TheResult / Test_Memory_Percentage[-1] > 1/3):
                        if (abs(TheResult) > abs(Test_Memory_Percentage[-1])):
                            AddingScore = 3 - (TheResult / Test_Memory_Percentage[-1])
                        else:
                            AddingScore = 3 - (Test_Memory_Percentage[-1] / TheResult)
                        AddingScore *= 1


                elif (abs(abs(TheResult) - abs(Test_Memory_Percentage[-1])) < 1):
                    AddingScore = 1
                
                elif (abs(abs(TheResult) - abs(Test_Memory_Percentage[-1])) < 3):
                    AddingScore = 3 - (abs(abs(TheResult) - abs(Test_Memory_Percentage[-1])))
                    AddingScore *= 1/2
                    
                else:
                    AddingScore = 0

                if not((ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-1] >= 0) and (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-2] >= 0) or (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+1] >= 0) and (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+2] >= 0)):
                    AddingScore -= abs(abs(np.mean([PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-1], PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+1]])) - abs(ClosestDay)) * 0.1


                if (AddingScore > 0.5):
                    AddingScore -= DeletingScore
                
                AddingScore = max([AddingScore, 0])
                TestScore += AddingScore



        TestScore = int(TestScore/ len(Test_Memory) * 100)  # Yüzde olarak hesaplama

        scoreDict[iothernumber] = TestScore


        print(iothernumber+1)
        print("\n*****************\n")
        print("Accuracy:",TestScore,"%")
        print("\n*****************\n")


    plt.figure(figsize=(15,8))
    plt.plot(list(scoreDict.keys()), list(scoreDict.values()))
    plt.title = x
    plt.xlabel('models')
    plt.ylabel('TestScores')
    plt.grid()
    plt.legend()
    plt.show()