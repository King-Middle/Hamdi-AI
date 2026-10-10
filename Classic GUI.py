import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model # type: ignore
from datetime import datetime, timedelta
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import csv
import sys

# --- Configuration ---
SYMBOL = "BTC-USD"
MODEL_PATH = "/Users/ardaunal/Desktop/Hamdi/Hamdi_BTCST_Advanced_3.0.keras"
HISTORY_FILE = "/Users/ardaunal/Desktop/Hamdi/prediction_history.csv"
WINDOW_SIZE = 48

# --- Colors & Style ---
BG_COLOR = "#0c0d10"
ACCENT_COLOR = "#1f2229"
TEXT_COLOR = "#ffffff"
UP_COLOR = "#00ff7f"
DOWN_COLOR = "#ff4d4d"
BTN_COLOR = "#1692e4" # Blue color as requested

# --- Helper Functions ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

class HamdiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hamdi BTC AI Predictor v3.0")
        self.root.geometry("1000x750")
        self.root.configure(bg=BG_COLOR)
        
        self.model = None
        self.loading_model = False
        self.is_refreshing = False
        
        self.setup_ui()
        self.init_history()
        self.load_model_async()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        header_frame.pack(fill="x")
        
        tk.Label(header_frame, text="Hamdi BTC AI Predictor", font=("Helvetica", 24, "bold"), 
                 bg=BG_COLOR, fg=TEXT_COLOR).pack()
        tk.Label(header_frame, text="1-Hour BTC-USD Forecast", font=("Helvetica", 12), 
                 bg=BG_COLOR, fg="#888888").pack()
        
        # Main Dashboard
        dash_frame = tk.Frame(self.root, bg=BG_COLOR, pady=10)
        dash_frame.pack(fill="both", expand=True, padx=20)
        
        # Stats Side (Left)
        left_panel = tk.Frame(dash_frame, bg=ACCENT_COLOR, padx=20, pady=20, bd=0, highlightthickness=0)
        left_panel.place(relx=0.05, rely=0.05, relwidth=0.3, relheight=0.4)
        
        self.price_val = tk.Label(left_panel, text="-.-- $", font=("Helvetica", 24, "bold"), 
                                  bg=ACCENT_COLOR, fg=TEXT_COLOR)
        self.price_val.pack(anchor="w", pady=(0, 10))
        
        tk.Label(left_panel, text="PREDICTION", font=("Helvetica", 10), bg=ACCENT_COLOR, fg="#888888").pack(anchor="w")
        self.pred_val = tk.Label(left_panel, text="WAITING", font=("Helvetica", 28, "bold"), 
                                 bg=ACCENT_COLOR, fg="#555555")
        self.pred_val.pack(anchor="w")
        
        tk.Label(left_panel, text="CONFIDENCE", font=("Helvetica", 10), bg=ACCENT_COLOR, fg="#888888").pack(anchor="w", pady=(10, 0))
        self.conf_bar = ttk.Progressbar(left_panel, orient="horizontal", length=200, mode="determinate")
        self.conf_bar.pack(anchor="w", pady=5)
        self.conf_label = tk.Label(left_panel, text="-%", font=("Helvetica", 12), bg=ACCENT_COLOR, fg=TEXT_COLOR)
        self.conf_label.pack(anchor="w")
        
        # Custom button using Label for better color support on Mac
        self.refresh_btn = tk.Label(left_panel, text="REFRESH", 
                                    bg=BTN_COLOR, fg=TEXT_COLOR, font=("Helvetica", 10, "bold"), 
                                    padx=20, pady=10, cursor="hand2")
        self.refresh_btn.pack(side="bottom", fill="x", pady=10)
        self.refresh_btn.bind("<Button-1>", lambda e: self.on_refresh())
        
        # Chart Side (Right/Bottom)
        self.chart_frame = tk.Frame(dash_frame, bg=BG_COLOR)
        self.chart_frame.place(relx=0.45, rely=0.1, relwidth=0.5, relheight=0.6)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)
        self.ax.tick_params(colors=TEXT_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color("#444444")
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                              anchor="w", bg=ACCENT_COLOR, fg="#888888", padx=10)
        status_bar.pack(side="bottom", fill="x")

    def init_history(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'price', 'prediction', 'confidence'])

    def load_model_async(self):
        self.status_var.set("Loading Keras model... please wait.")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            if not os.path.exists(MODEL_PATH):
                self.root.after(0, lambda: messagebox.showerror("Error", f"Model not found at {MODEL_PATH}"))
                self.status_var.set("Model not found.")
                return
                
            self.model = load_model(MODEL_PATH)
            self.root.after(0, lambda: self.status_var.set("Model loaded. Ready."))
            self.root.after(0, self.on_refresh) # Auto refresh on load
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error loading model: {e}"))

    def on_refresh(self):
        if self.is_refreshing or not self.model:
            return
        
        self.is_refreshing = True
        self.refresh_btn.config(text="REFRESHING...", bg="#444444")
        self.status_var.set("Fetching latest market data...")
        
        threading.Thread(target=self._predict_and_update, daemon=True).start()

    def _predict_and_update(self):
        try:
            # 1. Fetch Data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5) # 5 days for indicators
            
            df = yf.download(SYMBOL, start=start_date, end=end_date, interval='1h', auto_adjust=True, progress=False)
            
            if len(df) < WINDOW_SIZE + 50:
                self.root.after(0, lambda: self.status_var.set("Error: Not enough data points."))
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            current_price = float(df['Close'].iloc[-1])
            
            # 2. Features
            df['RSI'] = compute_rsi(df['Close'])
            macd, signal_line = compute_macd(df['Close'])
            upper_bb, lower_bb = compute_bollinger_bands(df['Close'])
            atr = compute_atr(df['High'], df['Low'], df['Close'])
            sma_50 = df['Close'].rolling(window=50).mean()
            ema_200 = df['Close'].ewm(span=200, adjust=False).mean()

            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Log_Volume_Change'] = np.log((df['Volume'] + 1) / (df['Volume'].shift(1) + 1))
            df['RSI_Scaled'] = df['RSI'] / 100.0
            df['MACD_Norm'] = macd / df['Close']
            df['Signal_Norm'] = signal_line / df['Close']
            df['BB_Width'] = (upper_bb - lower_bb) / df['Close']
            df['BB_Position'] = (df['Close'] - lower_bb) / (upper_bb - lower_bb)
            df['SMA_50_Dist'] = (df['Close'] - sma_50) / sma_50
            df['EMA_200_Dist'] = (df['Close'] - ema_200) / ema_200
            df['ATR_Pct'] = atr / df['Close']
            
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

            df.dropna(inplace=True)
            
            features = [
                'Log_Return', 'Log_Volume_Change', 'RSI_Scaled', 
                'MACD_Norm', 'Signal_Norm', 
                'BB_Width', 'BB_Position', 
                'SMA_50_Dist', 'EMA_200_Dist', 'ATR_Pct',
                'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos'
            ]
            
            # 3. Scaling
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(df[features].values)
            
            # 4. Prediction
            X_input = scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(features))
            prediction = self.model.predict(X_input, verbose=0)[0][0]
            
            decision = "UP" if prediction > 0.5 else "DOWN"
            confidence = prediction if prediction > 0.5 else (1 - prediction)
            
            # 5. Save History
            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), current_price, decision, f"{confidence:.2%}"])
            
            # 6. Update UI
            self.root.after(0, lambda: self.update_ui_results(current_price, decision, confidence, df))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error during refresh: {e}"))
        finally:
            self.root.after(0, self._finish_refresh)

    def update_ui_results(self, price, decision, confidence, df):
        self.price_val.config(text=f"{price:,.2f} $")
        
        if decision == "UP":
            self.pred_val.config(text="UP ▲", fg=UP_COLOR)
        else:
            self.pred_val.config(text="DOWN ▼", fg=DOWN_COLOR)
            
        self.conf_bar['value'] = confidence * 100
        self.conf_label.config(text=f"{confidence:.2%}")
        
        # Update Chart - 24 hours only
        self.ax.clear()
        self.ax.plot(df.index[-24:], df['Close'].iloc[-24:], color="#3498db", linewidth=2, label="BTC Price")
        
        # Style chart - smaller font sizes
        self.ax.set_title("BTC-USD Last 24 Hours", color=TEXT_COLOR, fontsize=8)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig.autofmt_xdate()
        self.ax.grid(True, alpha=0.1, color="#ffffff")
        
        # Labels - smaller
        self.ax.tick_params(colors="#888888", labelsize=7)
        
        self.canvas.draw()
        self.status_var.set(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    def _finish_refresh(self):
        self.is_refreshing = False
        self.refresh_btn.config(text="REFRESH", bg=BTN_COLOR)

if __name__ == "__main__":
    root = tk.Tk()
    app = HamdiApp(root)
    root.mainloop()
