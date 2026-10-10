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



#   -- -- -- --| Test HyperParameters |-- -- -- --

simulated_today = date.today()
check_daily_deviations = False
check_directional_accuracy = False
give_points_for_opposite_direction = False
show_closest_day_statistic = True
show_every_stock_accuracy = True
show_accuracy_graph = True
show_prediction_graphs = False

#   -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --




modelList = [
    '/Users/ardaunal/Desktop/Hamdi/King of the Court/Hamdi_SeriÜretim_15_2.0.keras',
]


SymbolList = [
    'aapl', 'msft', 'amzn', 'googl', 'goog', 
    'nvda', 'meta', 'tsla', 'jpm', 'jnj'
]


sp500_symbols = [
    'PAYC', 'PAYX', 'PBCT', 'PCAR', 'PCG',
    'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL',
    'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF',
    'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB',
    'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP',
    'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TROW', 'TRV', 'TSCO', 'TSLA',
    'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC',
    'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WM',
    'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS'
]

#SymbolList = sp500_symbols


# FXIFY Asset List
#SymbolList = [
#    'AAPL', 'ABNB', 'AMZN', 'BABA', 'COIN',
#    'CSCO', 'DIS', 'EBAY', 'META', 'GOOG',
#    'HD', 'JPM', 'MSFT', 'NKE', 'QCOM',
#    'SBUX', 'TSLA', 'UBER', 'WMT', 'XOM'
#]


#SymbolList = ['aapl', 'goog', 'nvda', 'avgo', 'wmt', 'v']

scoreDict = {}

predictionsDict = {}

pricesDict = {}

ClosestDayList = []


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


        if not check_directional_accuracy:
            FullPointsFactorLimit = 3
            HalfPointsFactorLimit = 6
        else:
            FullPointsFactorLimit = 1000000000
            HalfPointsFactorLimit = 2000000000

        if give_points_for_opposite_direction:
            FullPointsFactorLimitInverse = 1
            HalfPointsFactorLimitInverse = 2
        else:
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
                        TheResult1 = ((JustThePrices[TheOneToLookAt-9] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult2 = ((JustThePrices[TheOneToLookAt-8] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult3 = ((JustThePrices[TheOneToLookAt-7] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult4 = ((JustThePrices[TheOneToLookAt-6] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult5 = ((JustThePrices[TheOneToLookAt-5] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult6 = ((JustThePrices[TheOneToLookAt-4] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma   
                        TheResult7 = ((JustThePrices[TheOneToLookAt-3] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult8 = ((JustThePrices[TheOneToLookAt-2] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResult9 = ((JustThePrices[TheOneToLookAt-1] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                        TheResultA = ((JustThePrices[TheOneToLookAt] / JustThePrices[TheOneToLookAt -10]) - 1) * 100  # Değişimi bulma
                    except:
                        continue

                    AddingScore = 0

                    PlusMinusFiveDaysList = [TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA]   # 10 güne kadarki bütün değişimleri listeleme

                    ClosestDay = min(PlusMinusFiveDaysList, key=lambda x: abs(x - Test_Memory_Percentage[-1]))   # 10 güne kadarki en yakın sonucu bulma fonksiyonu

                    indexOfClosestDay = PlusMinusFiveDaysList.index(ClosestDay)

                    TheResult = ClosestDay



                    ClosestDayList.append(indexOfClosestDay - 9)

                    # Günlük sapmalara ekstra puan vermesini önlemek

                    DeletingScore = 0

                    if check_daily_deviations: 

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
                        if (TheResult / Test_Memory_Percentage[-1] < FullPointsFactorLimit and TheResult / Test_Memory_Percentage[-1] > 1/FullPointsFactorLimit):
                            AddingScore = 1
                        elif (TheResult / Test_Memory_Percentage[-1] < HalfPointsFactorLimit and TheResult / Test_Memory_Percentage[-1] > 1/HalfPointsFactorLimit):
                            if (abs(TheResult) > abs(Test_Memory_Percentage[-1])):
                                AddingScore = HalfPointsFactorLimit - (TheResult / Test_Memory_Percentage[-1])
                            else:
                                AddingScore = HalfPointsFactorLimit - (Test_Memory_Percentage[-1] / TheResult)
                            AddingScore /= HalfPointsFactorLimit - FullPointsFactorLimit


                    elif (abs(TheResult - Test_Memory_Percentage[-1]) < FullPointsFactorLimitInverse):
                        AddingScore = 1
                    
                    elif (abs(TheResult - Test_Memory_Percentage[-1]) < HalfPointsFactorLimitInverse):
                        AddingScore = HalfPointsFactorLimitInverse - (abs(TheResult - Test_Memory_Percentage[-1]))
                        AddingScore *= 1/(HalfPointsFactorLimitInverse - FullPointsFactorLimitInverse)
                        
                    else:
                        AddingScore = 0


                    if (AddingScore > 0.5):
                        AddingScore -= DeletingScore
                    
                    AddingScore = max([AddingScore, 0])
                    TestScore += AddingScore

                    
        predictionsDict[ChooseTheSymbolForOnce] = Test_Memory

        pricesDict[ChooseTheSymbolForOnce] = JustThePrices


        TestScore = int(TestScore/ len(Test_Memory) * 100)  # Yüzde olarak hesaplama
        
        if show_every_stock_accuracy:
            print(f"{x} - {TestScore}")

        AllTestScores.append(TestScore)

    AllTestScores = sum(AllTestScores)/len(AllTestScores)

    scoreDict[iothernumber+1] = AllTestScores

    print("*****************")
    print("Accuracy:",AllTestScores,"%")
    print("*****************")


    if show_closest_day_statistic:
        ClosestDayDict = {}

        for i in range (-9, 1):
            ClosestDayDict[i] = ClosestDayList.count(i)

        for i in ClosestDayDict.keys():
            print(f"{abs(i)} {"days before" if i < 0 else "that day" if i == 0 else "days later"} --> {ClosestDayDict[i]}")


timeListPred = []
timeListReal = []
for i in range (1, len(Test_Memory)):
    timeListPred.append(i * 10 + 10)

for i in range (1, len(JustThePrices)):
    timeListReal.append(i)

if show_prediction_graphs:
    for i in list(predictionsDict.keys()):
        plt.figure(figsize=(15,8))
        plt.plot(timeListPred, list(predictionsDict[i]), marker='o')
        plt.plot(timeListReal, list(pricesDict[i]))
        plt.title = "Predictions"
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid()
        plt.legend()
        plt.show()


if show_accuracy_graph:
    plt.figure(figsize=(15,8))
    plt.plot(list(scoreDict.keys()), list(scoreDict.values()), marker='o')
    plt.title = x
    plt.xlabel('models')
    plt.ylabel('TestScores')
    plt.grid()
    plt.legend()
    plt.show()