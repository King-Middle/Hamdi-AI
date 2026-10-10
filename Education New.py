import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. Introduction and Setup ---
print("Welcome to the AI Stock Predictor!")
print("This script uses a Long Short-Term Memory (LSTM) neural network to predict the next day's stock price.")
print("Disclaimer: This is for educational purposes only and not financial advice.\n")

# --- 2. Get User Input ---
ticker = input("Enter the stock ticker symbol (e.g., AAPL, GOOG): ").upper()

# --- 3. Fetch and Prepare Data ---
try:
    # Download historical data
    data = yf.download(ticker, start='2020-01-01', end='2025-10-20')
    if data.empty:
        print(f"No data found for ticker {ticker}. Please check the symbol and try again.")
        exit()

    print(f"\nSuccessfully downloaded data for {ticker}.")

    # We'll use the 'Close' price for our prediction
    close_prices = data['Close'].values.reshape(-1, 1)

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Create sequences of data for the LSTM model
    # We'll use the last 60 days of data to predict the next day
    prediction_days = 60
    x_train, y_train = [], []

    for i in range(prediction_days, len(scaled_prices)):
        x_train.append(scaled_prices[i-prediction_days:i, 0])
        y_train.append(scaled_prices[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

except Exception as e:
    print(f"An error occurred while fetching or preparing data: {e}")
    exit()

# --- 4. Build the LSTM Model ---
print("\nBuilding the LSTM model...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- 5. Train the Model ---
print("Training the model... (This may take a few minutes)")
model.fit(x_train, y_train, batch_size=1, epochs=1)

# --- 6. Make a Prediction ---
print("\nMaking a prediction for the next day...")

# Get the last 60 days of data to make the prediction
last_60_days = scaled_prices[-prediction_days:]
x_test = np.array([last_60_days.reshape(prediction_days,)])
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price_scaled = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# --- 7. Display the Result ---
last_actual_price = close_prices[-1][0]
print(f"The last closing price for {ticker} was: ${last_actual_price:.2f}")
print(f"The predicted closing price for the next trading day is: ${predicted_price[0][0]:.2f}")

# --- 8. (Optional) Save the Model ---
save_model = input("\nDo you want to save the trained model? (y/n): ").lower()
if save_model == 'y':
    model_path = f"Hamdi_{ticker}_Education.keras"
    model.save(model_path)
    print(f"Model saved to {os.path.join(os.getcwd(), model_path)}")

print("\nThank you for using the AI Stock Predictor!")
