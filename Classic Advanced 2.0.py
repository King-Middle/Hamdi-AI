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

print("")

tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

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


today = date.today()
#today = date(2026, 11, 11)


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

model = load_model('/Users/ardaunal/Desktop/Hamdi/Important Modelz/Hamdi_SeriÜretim_15_2.0.keras')

def Hamdi(sembol):


    stock_symbol = sembol

    decimals = 0

    # 🔹 2. Hisse ve makro verileri çek
    stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
    stock_df2 = download(stock_symbol, start=date.today()- timedelta(1), end=date.today() + timedelta(1), interval='1d', auto_adjust=True, progress=False)
    price_on_date = float(stock_df2["Close"].iloc[-1])


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

    prediction = model.predict(X_input, verbose=0)[0][0]

    predicted_price = price_on_date * (1 + prediction / 100)

    print(f"\n{sembol}: {price_on_date:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))


    if prediction < -0.01:
        print(f"{sembol}: {guess_date} tahmini fiyat: \033[91m{prediction:.2f} % ▼\033[0m  -  {predicted_price:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))
    elif prediction > 0.01:
        print(f"{sembol}: {guess_date} tahmini fiyat: \033[92m{prediction:.2f} % ▲\033[0m  -  {predicted_price:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))
    else:
        print(f"{sembol}: {guess_date} tahmini fiyat: \033[89m0.00 % \033[0m  -  {predicted_price:,.{decimals}f} $".replace(",", "?").replace(".", ",").replace("?", "."))




#******************************************************************************************************************************************************************************************************************************************




def HamdiTest(sembol):
    Symbol = sembol

    TestScore = 0

    Test_Memory = []
    Test_Memory_Percentage = []

    TheRealPricesForATime = download(Symbol, start=date(2025, 1, 1), end=date(2025, 10, 15), interval = '1d', timeout = 30, progress=False)['Close']

    TheRealPricesForATime = TheRealPricesForATime.asfreq('D')       # Her günü göster
    TheRealPricesForATime = TheRealPricesForATime.fillna(method='ffill')  # Eksik günlerde bir önceki değeri kullan


    JustThePrices = TheRealPricesForATime.squeeze().tolist()

    JustThePrices = [round(x) for x in JustThePrices]


    def HamdiForTest(sembol, todaydate):
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
        stock_df = download(stock_symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 30)
        price_on_date = float(stock_df["Close"].iloc[-1])

        for name, ticker in macro_symbols.items():
            temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 30)[['Close']]
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


        prediction = model.predict(X_input, verbose=0)[0][0]

        predicted_price = price_on_date * (1 + prediction / 100)

        Test_Memory.append(predicted_price)
        Test_Memory_Percentage.append(prediction)


    FullPointsFactorLimit = 3
    HalfPointsFactorLimit = 6

    FullPointsFactorLimitInverse = 1
    HalfPointsFactorLimitInverse = 3

    for i in range(1, 13):
        for a in range(1, 22, 10):
            HamdiForTest(Symbol, date(2025, i, a))
            
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

            PlusMinusFiveDaysList = [TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9]   # ± 5 gün hesaplama ve doğru güne bakma

            PlusMinusFiveDaysListOtherVersion = [TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA, TheResultB] # Ekstra liste 1 gün önce ve sonrayı bulabilmek için

            ClosestDay = min(PlusMinusFiveDaysList, key=lambda x: abs(x - Test_Memory_Percentage[-1]))   # Sıradaki 5 gün içerisinden en mantıklı olanı bulma fonksiyonu

            indexOfClosestDay = PlusMinusFiveDaysList.index(ClosestDay)

            indexOfClosestDayOtherVersion = indexOfClosestDay + 2

            TheResult = np.mean([ClosestDay, TheResult6, ClosestDay])   # Ana sonucu en yakın güne eşitleme

            # Günlük sapmalara ekstra puan vermesini önlemek

            DeletingScore = 0

            if (indexOfClosestDay != 0 and indexOfClosestDay != -1):
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
                    AddingScore *= 1


            elif (abs(TheResult - Test_Memory_Percentage[-1]) < FullPointsFactorLimitInverse):
                AddingScore = 1
            
            elif (abs(TheResult - Test_Memory_Percentage[-1]) < HalfPointsFactorLimitInverse):
                AddingScore = HalfPointsFactorLimitInverse - (abs(TheResult - Test_Memory_Percentage[-1]))
                AddingScore *= 1/(HalfPointsFactorLimitInverse - FullPointsFactorLimitInverse)
                
            else:
                AddingScore = 0

            if not((ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-1] >= 0) and (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-2] >= 0) or (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+1] >= 0) and (ClosestDay >= 0) == (PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+2] >= 0)):
                AddingScore -= abs(abs(np.mean([PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion-1], PlusMinusFiveDaysListOtherVersion[indexOfClosestDayOtherVersion+1]])) - abs(ClosestDay)) * 0.1


            if (AddingScore > 0.5):
                AddingScore -= DeletingScore
            
            AddingScore = max([AddingScore, 0])
            TestScore += AddingScore

    for i in range(1, end_date.month):
        for a in range(1, 22, 10):
            HamdiForTest(Symbol, date(2026, i, a))
            
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

            PlusMinusFiveDaysList = [TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9]   # ± 5 gün hesaplama ve doğru güne bakma

            PlusMinusFiveDaysListOtherVersion = [TheResult1, TheResult2, TheResult3, TheResult4, TheResult5, TheResult6, TheResult7, TheResult8, TheResult9, TheResultA, TheResultB] # Ekstra liste 1 gün önce ve sonrayı bulabilmek için

            ClosestDay = min(PlusMinusFiveDaysList, key=lambda x: abs(x - Test_Memory_Percentage[-1]))   # Sıradaki 5 gün içerisinden en mantıklı olanı bulma fonksiyonu

            indexOfClosestDay = PlusMinusFiveDaysList.index(ClosestDay)

            indexOfClosestDayOtherVersion = indexOfClosestDay + 2

            TheResult = np.mean([ClosestDay, TheResult6, ClosestDay])   # Ana sonucu en yakın güne eşitleme

            # Günlük sapmalara ekstra puan vermesini önlemek

            DeletingScore = 0

            if (indexOfClosestDay != 0 and indexOfClosestDay != -1):
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
                if (TheResult / Test_Memory_Percentage[-1] < 3 and TheResult / Test_Memory_Percentage[-1] > 1/3):
                    AddingScore = 1
                elif (TheResult / Test_Memory_Percentage[-1] < 6 and TheResult / Test_Memory_Percentage[-1] > 1/6):
                    if (abs(TheResult) > abs(Test_Memory_Percentage[-1])):
                        AddingScore = 6 - (TheResult / Test_Memory_Percentage[-1])
                    else:
                        AddingScore = 6 - (Test_Memory_Percentage[-1] / TheResult)
                    AddingScore *= 1/3


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

    print("Accuracy:",TestScore,"%")




#******************************************************************************************************************************************************************************************************************************************




SymbolValue = 0
while SymbolValue != "":
    SymbolValue = input("Stock Symbol: ").upper()
    HamdiTest(SymbolValue)
    Hamdi(SymbolValue)
    print("\n")




pid = os.getppid()
os.kill(pid, signal.SIGKILL)