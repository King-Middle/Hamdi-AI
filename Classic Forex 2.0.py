
# ***********************************************
# This code is under maintance (macro symbols and indicators change)
# ***********************************************




from yfinance import download
import pandas as pd
from datetime import date, timedelta, datetime
from tensorflow.keras.models import load_model #type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import warnings
import os
import tensorflow
import signal

tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


file_name = ''

today = date.today() - timedelta(hours=4)
today_hour = datetime.now() - timedelta(hours=4)
if today_hour.hour >= 3: today = today + timedelta(1)

back_date = 36
start_date = today_hour - timedelta(hours=back_date)
end_date = today
guess_date = today_hour + timedelta(hours=4)

def resample_to_hourly(df):
    """Veriyi saatlik mum yapısına dönüştürür, sütun isimlerini kontrol eder."""
    # Eğer MultiIndex geliyorsa (yfinance bazen böyle yapar), sadece ilk seviyeyi alalım
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Sütunları küçük harfe çevirip eşleştirme yapalım (Open -> open gibi)
    df.columns = [c.capitalize() for c in df.columns]
    
    return df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

def Hamdi(sembol):
    stock_symbol = sembol
    decimals = 4

    # 1. Veriyi çek ve saatliğe çevir
    raw_df = download(stock_symbol, start=start_date, end=end_date, interval='15m', auto_adjust=True, progress=False)
    stock_df = resample_to_hourly(raw_df)

    price_on_date = stock_df['Close'].iloc[-1]

    # 2. Makro verileri çek ve resample et
    for name, ticker in macro_symbols_forex.items():
        try:
            temp_raw = download(ticker, start=start_date, end=end_date, interval='15m', auto_adjust=True, progress=False)
            temp = resample_to_hourly(temp_raw)[['Close']]
            temp.rename(columns={'Close': name}, inplace=True)
            stock_df = stock_df.join(temp, how='outer')
        except:
            continue

    stock_df.ffill(inplace=True)
    stock_df.bfill(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(macro_symbols_forex.keys())
    data = stock_df[features].copy().dropna()

    # 3. SCALER: Burada 'fit_transform' kullanmalısın, çünkü 'transform' tek başına çalışmaz!
    scaler = joblib.load('/Users/ardaunal/Desktop/Hamdi/Scalers/Hamdi_' + file_name + '_ForexScaler_2.0.joblib')
    scaled_data = scaler.fit_transform(data) # 'fit' ve 'transform' aynı anda yapıldı!

    X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])

    # 4. Tahmin


    model = load_model('/Users/ardaunal/Desktop/Hamdi/Forex/Hamdi_' + file_name + '_Forex_2.0.keras')



    prediction = model.predict(X_input, verbose=0)[0][0]
    predicted_price = price_on_date * (1 + prediction / 100)

    print(f"\n{sembol}: {price_on_date:,.{decimals}f} $".replace(".", ","))
    
    color = "\033[92m" if prediction > 0.01 else ("\033[91m" if prediction < -0.01 else "\033[89m")
    symbol_dir = "▲" if prediction > 0.01 else ("▼" if prediction < -0.01 else "")
    
    print(f"{sembol}: {guess_date} tahmini: {color}{prediction:.2f} % {symbol_dir}\033[0m - {predicted_price:,.{decimals}f} $\n".replace(".", ","))


Hamdi('EURUSD=X')


print("The End\n\n")

input()

pid = os.getppid()
os.kill(pid, signal.SIGKILL)