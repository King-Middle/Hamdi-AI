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
import random



tensorflow.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

print()

# --- Strategy Parameters ---
starting_date = date(2026, 1, 1)
simulated_today = date.today()
STARTING_MONEY = 25000
RISK_PROFIT_RATIO = 1
UNDER_3_PCT_TP_FACTOR = 1      # Multiplier for TP limit when absolute prediction is < 3%
MIN_PREDICTION_PCT = 0       # Minimum absolute prediction percentage required to open a trade (e.g., 1.0%)
OPPOSITE_DIR_REQUIRED = False  # Only open a position if prediction direction is opposite of market change
OPPOSITE_USE_TODAYS_CHANGE = True  # If True, checks today's market change; if False, checks yesterday's change
TP_WAIT_DAYS = 0               # Days to wait before opening a new trade after a Take Profit (TP) closure
SL_WAIT_DAYS = 0               # Days to wait before opening a new trade after a Stop Loss (SL) closure
BIG_CHANGE_PAUSE_ENABLED = False  # Boolean to turn the big change pause tactic on or off in the simulation
BIG_CHANGE_MIN_PCT = 5       # Minimum percentage change (between or during days) considered as a big change
BIG_CHANGE_WAIT_DAYS = 2       # Amount of days to wait/pause trading after a big change
USE_ATR_STOP_LOSS = True        # If True, sets Stop Loss to ATR_SL_MULTIPLIER * ATR instead of percentage prediction
ATR_SL_MULTIPLIER = 1.5        # Multiplier for ATR when USE_ATR_STOP_LOSS is True (e.g., 1.5x ATR)
ATR_WINDOW = 14                # Lookback period (in days) for ATR calculation
PLOT_ASSET_TRADES = False       # If True, plots individual asset trades. If False, skips plotting.
GUESS = 'real'                 # 'real' -> real model, 'random' -> random guesses, 'positive' -> only positive guesses
save_to_file = True           # saving to a file
file_name = 'secure_list_18_FXIFY.txt' # the file name
min_profit_percent_to_save = 1  # the least amount of profit percent the stock should reach monthly
max_risk_ratio = 80            # the maximum real win loss risk ratio value accepted
needed_win_per_stock = 0.0
# ---------------------------


month = (simulated_today.day - starting_date.day - 10) / 30 + (simulated_today.month - starting_date.month) + (simulated_today.year - starting_date.year) * 12


if save_to_file: 
    with open (file_name, 'a') as f:
        pass

profit_list = 0

def compute_atr(high_series, low_series, close_series, window=14):
    """
    Computes the Average True Range (ATR) over a given window period.
    Uses min_periods=1 so that initial values are calculated gracefully without NaNs.
    """
    high_low = high_series - low_series
    high_close = (high_series - close_series.shift()).abs()
    low_close = (low_series - close_series.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean()

def check_which_hit_first(symbol, trade_date, tp_price, sl_price, is_long):
    """
    Determines whether TP or SL was hit first on a given day.
    Uses 5-minute candles for maximum granularity (available ~60 days back),
    falling back to 1-hour candles for older historical data.
    When both levels are hit within the same candle, conservatively assumes
    SL was hit first (worst-case for the trader) instead of using an unreliable
    bullish/bearish directional heuristic.
    """
    start_str = trade_date.strftime('%Y-%m-%d')
    end_str = (trade_date + timedelta(days=1)).strftime('%Y-%m-%d')

    def _scan_candles(df):
        """Scan OHLC candles and return the first target hit."""
        for _, row in df.iterrows():
            c_open  = float(row['Open'])
            c_high  = float(row['High'])
            c_low   = float(row['Low'])

            tp_hit = (c_high >= tp_price) if is_long else (c_low  <= tp_price)
            sl_hit = (c_low  <= sl_price) if is_long else (c_high >= sl_price)

            if tp_hit and not sl_hit:
                # Only TP was reached — check for gap open past TP
                if is_long and c_open >= tp_price:
                    return "TP", c_open
                elif not is_long and c_open <= tp_price:
                    return "TP", c_open
                return "TP", float(tp_price)

            if sl_hit and not tp_hit:
                # Only SL was reached — check for gap open past SL
                if is_long and c_open <= sl_price:
                    return "SL", c_open
                elif not is_long and c_open >= sl_price:
                    return "SL", c_open
                return "SL", float(sl_price)

            if tp_hit and sl_hit:
                # Both levels inside the same candle.
                # Conservative assumption: SL was hit first (worst case for the trader).
                # This avoids the unreliable bullish/bearish directional heuristic.
                if is_long and c_open <= sl_price:
                    return "SL", c_open
                elif not is_long and c_open >= sl_price:
                    return "SL", c_open
                return "SL", float(sl_price)

        return None, None  # No target hit in this candle set

    # --- Try 5-minute candles first (granular, available ~60 days back) ---
    try:
        df_5m = download(symbol, start=start_str, end=end_str, interval='5m', progress=False, timeout=10)
        if not df_5m.empty:
            if isinstance(df_5m.columns, pd.MultiIndex):
                df_5m.columns = df_5m.columns.droplevel('Ticker')
            result, price = _scan_candles(df_5m)
            if result is not None:
                return result, price
    except Exception:
        pass

    # --- Fall back to 1-hour candles for older historical data ---
    try:
        df_1h = download(symbol, start=start_str, end=end_str, interval='1h', progress=False, timeout=10)
        if not df_1h.empty:
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.droplevel('Ticker')
            result, price = _scan_candles(df_1h)
            if result is not None:
                return result, price
    except Exception:
        pass

    # --- Conservative fallback if no data available at all ---
    return "SL", float(sl_price)

modelList = [
    '/Users/ardaunal/Desktop/Hamdi/King of the Court/Hamdi_SeriÜretim_15_2.0.keras',
]


scalerList = []

for i in modelList:
    scaler_directory = i[:-5] + 'pkl'

SymbolList = [  
    'GC=F'
]


scoreDict = {}


ClosestDayList = []


for iothernumber in range(0,len(modelList)):
    scoreDict = {}
    AllTestScores = []
    for x in SymbolList:
        try:
            model = load_model(modelList[iothernumber])
            Symbol = x

            try:
                scaler = joblib.load(scalerList[iothernumber])
            except:
                scaler = MinMaxScaler()

            ChooseTheSymbolForOnce = Symbol.upper()


            ModelForOnce = 0

            TestScore = 0

            TheAmountDone = 0

            Test_Memory = []
            Test_Memory_Percentage = []

            start_date = starting_date - timedelta(days=210)
            end_date = simulated_today

            TheRealPricesForATime = download(ChooseTheSymbolForOnce, start=starting_date, end=end_date, interval = '1d', timeout = 40, progress=False)
            if TheRealPricesForATime.empty:
                print(f"⚠️ Skipping {ChooseTheSymbolForOnce}: download returned empty data (possibly delisted)")
                continue
            if isinstance(TheRealPricesForATime.columns, pd.MultiIndex):
                TheRealPricesForATime.columns = TheRealPricesForATime.columns.droplevel('Ticker')

            # Using actual trading days only without rounding or weekend calendar padding
            JustThePrices = [float(x) for x in TheRealPricesForATime['Close'].squeeze().tolist()]
            JustTheHighs = [float(x) for x in TheRealPricesForATime['High'].squeeze().tolist()]
            JustTheLows = [float(x) for x in TheRealPricesForATime['Low'].squeeze().tolist()]
            JustTheOpens = [float(x) for x in TheRealPricesForATime['Open'].squeeze().tolist()]
            atr_series = compute_atr(TheRealPricesForATime['High'], TheRealPricesForATime['Low'], TheRealPricesForATime['Close'], window=ATR_WINDOW)
            JustTheATRs = [float(x) for x in atr_series.squeeze().tolist()]


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
                'WTI': 'USO',              # Ham Petrol (WTI)
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



            stock_df = download(Symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 40)
            if stock_df.empty:
                print(f"⚠️ Skipping {Symbol}: download returned empty data (possibly delisted)")
                continue
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = stock_df.columns.droplevel('Ticker')
            price_on_date = float(stock_df["Close"].iloc[-1])

            for name, ticker in macro_symbols.items():
                temp = download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False, timeout = 40)
                if isinstance(temp.columns, pd.MultiIndex):
                    temp.columns = temp.columns.droplevel('Ticker')
                temp = temp[['Close']]
                temp.rename(columns={'Close': name}, inplace=True)
                stock_df = stock_df.join(temp, how='left')

            stock_df.ffill(inplace=True)
            stock_df.bfill(inplace=True)



            def Hamdi(sembol, todaydate):

            
                global ModelForOnce
                global TheAmountDone
                global model
                global stock_df

                # Use data strictly up to yesterday's close (todaydate - timedelta(1)) to prevent lookahead bias
                today = todaydate - timedelta(1)


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

                scaled_data = scaler.fit_transform(data)

                # 🔹 6. Inputu modele uygun hale getir
                X_input = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])  # (1, zaman, özellik)

                # 🔹 7. Modeli yükle ve tahmin yap



                if GUESS == 'real':
                    prediction = model.predict(X_input, verbose=0)[0][0]
                elif GUESS == 'positive':
                    prediction = random.uniform(1, 2)
                elif GUESS == 'random':
                    prediction = random.uniform(-2, 2)

                predicted_price = price_on_date * (1 + prediction / 100)

                

                Test_Memory.append(predicted_price)
                Test_Memory_Percentage.append(prediction)

                TheAmountDone += 1

            #*******************************************************************************
            # TRADING SIMULATION
            #*******************************************************************************
            current_money = STARTING_MONEY
            balance_history = [current_money]
            date_history = []

            trades_count = 0
            tp_hits = 0
            sl_hits = 0
            time_closes = 0
            total_money_earned = 0.0
            total_money_lost = 0.0
            trades_info = []
            current_idx = 0
            end_idx = len(JustThePrices)

            last_exit_idx = None
            last_exit_price = None

            tp_distance_list = []
            sl_distance_list = []

            while current_idx < end_idx - 1:
                # Determine entry price: exact moment previous position closed if same day, or market open
                if last_exit_idx == current_idx and last_exit_price is not None:
                    P_entry = last_exit_price
                else:
                    P_entry = JustTheOpens[current_idx]

                current_timestamp = TheRealPricesForATime.index[current_idx]
                current_step_date = current_timestamp.to_pydatetime().date()
                Hamdi(ChooseTheSymbolForOnce, current_step_date)

                prediction = Test_Memory_Percentage[-1]

                if prediction == 0.0 or abs(prediction) < MIN_PREDICTION_PCT:
                    current_idx += 1
                    continue

                if OPPOSITE_DIR_REQUIRED:
                    if OPPOSITE_USE_TODAYS_CHANGE:
                        if current_idx < 1:
                            current_idx += 1
                            continue
                        # Real-time change from yesterday's close to current moment of entry (P_entry)
                        market_change = P_entry - JustThePrices[current_idx - 1]
                    else:
                        if current_idx < 2:
                            current_idx += 1
                            continue
                        market_change = JustThePrices[current_idx - 1] - JustThePrices[current_idx - 2]

                    if prediction > 0 and market_change >= 0:
                        current_idx += 1
                        continue
                    if prediction < 0 and market_change <= 0:
                        current_idx += 1
                        continue

                if BIG_CHANGE_PAUSE_ENABLED:
                    has_big_change = False
                    lookback_start = max(1, current_idx - BIG_CHANGE_WAIT_DAYS)
                    for idx_check in range(lookback_start, current_idx + 1):
                        prev_close = JustThePrices[idx_check - 1]
                        curr_close = JustThePrices[idx_check]
                        curr_open = JustTheOpens[idx_check]
                        curr_high = JustTheHighs[idx_check]
                        curr_low = JustTheLows[idx_check]

                        # Between days (interday): Close-to-Close or Close-to-Open gap
                        between_days_pct = abs(curr_close - prev_close) / prev_close * 100.0 if prev_close != 0 else 0
                        gap_pct = abs(curr_open - prev_close) / prev_close * 100.0 if prev_close != 0 else 0

                        # During days (intraday): High-to-Low or Open-to-Close
                        during_days_pct = abs(curr_high - curr_low) / curr_low * 100.0 if curr_low != 0 else 0
                        open_close_pct = abs(curr_close - curr_open) / curr_open * 100.0 if curr_open != 0 else 0

                        if max(between_days_pct, gap_pct, during_days_pct, open_close_pct) >= BIG_CHANGE_MIN_PCT:
                            has_big_change = True
                            break

                    if has_big_change:
                        current_idx += 1
                        continue

                trades_count += 1
                is_long = prediction > 0
                T_pct = abs(prediction) / 100.0
                if abs(prediction) < 3.0:
                    T_pct *= UNDER_3_PCT_TP_FACTOR

                if USE_ATR_STOP_LOSS:
                    # Use ATR computed strictly up to yesterday's close to prevent lookahead bias
                    atr_val = JustTheATRs[current_idx - 1] if current_idx > 0 else JustTheATRs[0]
                    sl_distance = ATR_SL_MULTIPLIER * atr_val
                    if is_long:
                        tp_price = P_entry * (1.0 + T_pct)
                        sl_price = max(0.01, P_entry - sl_distance)
                    else:
                        tp_price = P_entry * (1.0 - T_pct)
                        sl_price = P_entry + sl_distance
                else:
                    if is_long:
                        tp_price = P_entry * (1.0 + T_pct)
                        sl_price = P_entry * (1.0 - T_pct * RISK_PROFIT_RATIO)
                    else:
                        tp_price = P_entry * (1.0 - T_pct)
                        sl_price = P_entry * (1.0 + T_pct * RISK_PROFIT_RATIO)

                trade_closed = False
                close_reason = ""
                close_price = None
                exit_idx = current_idx + 10
                exit_idx_final = exit_idx

                # Check day-by-day Close prices between current_idx + 1 and exit_idx
                for idx in range(current_idx + 1, exit_idx + 1):
                    if idx >= end_idx:
                        trade_closed = True
                        close_reason = "Time"
                        close_price = JustThePrices[-1]
                        current_idx = end_idx
                        exit_idx_final = end_idx - 1
                        break

                    daily_open = JustTheOpens[idx]
                    daily_high = JustTheHighs[idx]
                    daily_low = JustTheLows[idx]

                    tp_hit = (daily_high >= tp_price) if is_long else (daily_low <= tp_price)
                    sl_hit = (daily_low <= sl_price) if is_long else (daily_high >= sl_price)

                    if tp_hit and sl_hit:
                        # Both hit on the same day. Find out which happened first and actual execution price.
                        current_date_of_idx = TheRealPricesForATime.index[idx].to_pydatetime().date()
                        first_hit, actual_exec_price = check_which_hit_first(ChooseTheSymbolForOnce, current_date_of_idx, tp_price, sl_price, is_long)
                        trade_closed = True
                        close_reason = first_hit

                        # Check daily open gap override
                        if is_long:
                            if first_hit == "TP" and daily_open >= tp_price:
                                actual_exec_price = daily_open
                            elif first_hit == "SL" and daily_open <= sl_price:
                                actual_exec_price = daily_open
                        else:
                            if first_hit == "TP" and daily_open <= tp_price:
                                actual_exec_price = daily_open
                            elif first_hit == "SL" and daily_open >= sl_price:
                                actual_exec_price = daily_open

                        close_price = actual_exec_price
                        current_idx = idx + TP_WAIT_DAYS if first_hit == "TP" else idx + SL_WAIT_DAYS
                        exit_idx_final = idx
                        break
                    elif tp_hit:
                        trade_closed = True
                        close_reason = "TP"
                        if is_long and daily_open >= tp_price:
                            close_price = daily_open
                        elif not is_long and daily_open <= tp_price:
                            close_price = daily_open
                        else:
                            close_price = tp_price
                        current_idx = idx + TP_WAIT_DAYS
                        exit_idx_final = idx
                        break
                    elif sl_hit:
                        trade_closed = True
                        close_reason = "SL"
                        if is_long and daily_open <= sl_price:
                            close_price = daily_open
                        elif not is_long and daily_open >= sl_price:
                            close_price = daily_open
                        else:
                            close_price = sl_price
                        current_idx = idx + SL_WAIT_DAYS
                        exit_idx_final = idx
                        break

                if not trade_closed:
                    close_reason = "Time"
                    close_price = JustThePrices[exit_idx]
                    current_idx = exit_idx
                    exit_idx_final = exit_idx

                last_exit_idx = exit_idx_final
                last_exit_price = close_price

                # Update money based on trade outcome and actual close price
                if is_long:
                    return_pct = (close_price - P_entry) / P_entry
                else:
                    return_pct = (P_entry - close_price) / P_entry

                if close_reason == "TP":
                    tp_hits += 1
                elif close_reason == "SL":
                    sl_hits += 1
                else: # Time
                    time_closes += 1

                balance_before = current_money
                current_money = current_money * (1.0 + return_pct)
                current_money = max(current_money, 0.0)

                dollar_change = current_money - balance_before
                if dollar_change > 0:
                    total_money_earned += dollar_change
                else:
                    total_money_lost += abs(dollar_change)

                balance_history.append(current_money)
                date_history.append(current_step_date)

                trades_info.append({
                    'entry_date': current_timestamp,
                    'exit_date': TheRealPricesForATime.index[exit_idx_final],
                    'entry_price': P_entry,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'close_price': close_price,
                    'close_reason': close_reason,
                    'is_long': is_long
                })


                tp_distance_list.append(abs(P_entry-tp_price))
                sl_distance_list.append(abs(P_entry-sl_price))

            # Output detailed stats for this symbol
            profit = current_money - STARTING_MONEY
            profit_percent = (profit / STARTING_MONEY) * 100.0

            win_loss_ratio = tp_hits / (tp_hits + sl_hits) if (tp_hits + sl_hits) > 0 else 0.0

            ideal_risk_win_loss_ratio = sum(sl_distance_list) / (sum(tp_distance_list) + sum(sl_distance_list))

            win_loss_difference_final = (needed_win_per_stock + (1 - needed_win_per_stock) * ideal_risk_win_loss_ratio) - ideal_risk_win_loss_ratio

            if ideal_risk_win_loss_ratio + win_loss_difference_final < win_loss_ratio:
                sign = '🟢🟢🟢'
            else:
                sign = '🔴🔴🔴'

            monthly_profit_percent = profit_percent / month


            if save_to_file:
                if ideal_risk_win_loss_ratio + win_loss_difference_final < win_loss_ratio and monthly_profit_percent > min_profit_percent_to_save and ideal_risk_win_loss_ratio * 100 < max_risk_ratio:
                    with open (file_name, 'r') as f:
                        file_context = f.read()
                    if Symbol not in [s.strip() for s in file_context.split(',')]:
                        with open (file_name, 'a') as f:
                            f.write(f'{Symbol}, ')
                        file_evaluation = '\033[92m🟩Symbol added🟩\033[0m'
                    else:
                        file_evaluation = '\033[93m🟨Symbol already exists in the file🟨\033[0m'
                else:
                    file_evaluation = '\033[91m🟥Symbol failed🟥\033[0m'
            else:
                if ideal_risk_win_loss_ratio + win_loss_difference_final < win_loss_ratio and monthly_profit_percent > min_profit_percent_to_save and ideal_risk_win_loss_ratio * 100 < max_risk_ratio:
                    closed_note = '🟩'
                else:
                    closed_note = '🟥'

                file_evaluation = f'File saving feature is closed ({closed_note})'


            print("=" * 50)
            print(f"SYMBOL: {ChooseTheSymbolForOnce}")
            print("-" * 50)
            print(f"Final Money:      ${current_money:,.2f}")
            print(f"Profit/Loss:      ${profit:+,.2f} ({profit_percent:+.2f}%)")
            print(f"Risk Ratio:       {(ideal_risk_win_loss_ratio * 100):.2f}%")
            print(f"Win-Loss Ratio:   {(win_loss_ratio * 100):.2f}% {sign}")
            print(f"TP Hits (Wins):   {tp_hits}")
            print(f"SL Hits (Losses): {sl_hits}")
            print(f"Closed by Time:   {time_closes}")
            print(f'{file_evaluation}')
            print("=" * 50)


            profit_list += profit

            # Save data for plotting
            scoreDict[ChooseTheSymbolForOnce] = (date_history, balance_history)

            # Plot this asset's trades on top of close prices
            if PLOT_ASSET_TRADES:
                plt.figure(figsize=(15, 8))
                plt.plot(TheRealPricesForATime.index, JustThePrices, label='Close Price', color='black', alpha=0.3)
                
                tp_legend_added = False
                sl_legend_added = False
                time_legend_added = False
                entry_legend_added = False
                
                for t in trades_info:
                    # Draw TP and SL levels as horizontal lines across the trade duration
                    plt.plot([t['entry_date'], t['exit_date']], [t['tp_price'], t['tp_price']], color='green', alpha=0.6)
                    plt.plot([t['entry_date'], t['exit_date']], [t['sl_price'], t['sl_price']], color='red', alpha=0.6)
                    
                    # Draw segment connecting entry and exit
                    line_color = 'green' if t['is_long'] else 'red'
                    plt.plot([t['entry_date'], t['exit_date']], [t['entry_price'], t['close_price']], color=line_color, linestyle='-', alpha=0.4)
                    
                    # Plot entry point
                    ent_lbl = 'Trade Entry' if not entry_legend_added else ''
                    entry_legend_added = True
                    plt.scatter(t['entry_date'], t['entry_price'], color='blue', marker='o', s=35, label=ent_lbl)
                    
                    # Plot exit point color-coded by reason
                    if t['close_reason'] == "TP":
                        lbl = "TP Hit (Win)" if not tp_legend_added else ""
                        tp_legend_added = True
                        plt.scatter(t['exit_date'], t['close_price'], color='green', marker='^', s=80, label=lbl, zorder=5)
                    elif t['close_reason'] == "SL":
                        lbl = "SL Hit (Loss)" if not sl_legend_added else ""
                        sl_legend_added = True
                        plt.scatter(t['exit_date'], t['close_price'], color='red', marker='v', s=80, label=lbl, zorder=5)
                    else: # Time
                        lbl = "Closed by Time" if not time_legend_added else ""
                        time_legend_added = True
                        plt.scatter(t['exit_date'], t['close_price'], color='orange', marker='s', s=60, label=lbl, zorder=5)
                        
                plt.title(f"{ChooseTheSymbolForOnce} - Trade Execution & Price History")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.grid(True)
                plt.legend()
                plt.show()

        except:
            pass

    # Plotting code for this model
    plt.figure(figsize=(15, 8))
    for symbol, (dates, balances) in scoreDict.items():
        plt.plot(dates, balances[1:], label=symbol, marker='o', markersize=3)
    plt.title(f"Simulation Equity Curves - Model {iothernumber+1}")
    plt.xlabel("Date")
    plt.ylabel("Balance ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

profit_list = profit_list/len(SymbolList)    # Calculating the month for monthly profit
print("*********************************")
print(f"Total Profit: ${(profit_list / STARTING_MONEY * 100):.1f}% for {len(SymbolList)} stocks")
print(f"Monthly Profit: ${((profit_list / month) / STARTING_MONEY * 100):.1f}% for {len(SymbolList)} stocks")
print("*********************************")