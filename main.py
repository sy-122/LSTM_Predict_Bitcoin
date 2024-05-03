import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import os


def fetch_data(start_date, end_date, ticker):
    """Fetch data from Yahoo Finance."""
    data = yf.download(ticker, start_date, end_date)
    return data


def prepare_data(data, prediction_days):
    """Prepare data for model training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    if len(scaled_data) <= prediction_days:
        raise ValueError("Prediction days exceed the length of data.")

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler



def build_model(input_shape):
    """Build LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def plot_results(index, actual_prices, prediction_prices, crypto_currency):
    """Plot actual and predicted prices."""
    plt.plot(index, actual_prices, color='black', label='Actual Prices')
    plt.plot(index, prediction_prices, color='green', label='Prediction Prices')
    plt.title(f'{crypto_currency} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Corrected formatting
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Adjust the interval as needed
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for better readability
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Set the environment variable
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    start_date = dt.datetime(2016, 1, 1)
    end_date = dt.datetime.now()
    crypto_currency = 'BTC'
    against_currency = 'GBP'
    ticker_symbol = f'{crypto_currency}-{against_currency}'

    # Fetch data
    data = fetch_data(start_date, end_date, ticker_symbol)

    # Prepare data
    prediction_days = 60
    x_train, y_train, scaler = prepare_data(data, prediction_days)

    # Build model
    input_shape = (x_train.shape[1], 1)
    model = build_model(input_shape)

    # Train model
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

    # Test the model
    test_start = dt.datetime(2020, 1, 1)
    test_end = end_date
    test_data = fetch_data(test_start, test_end, ticker_symbol)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = np.array([model_inputs[i - prediction_days:i, 0] for i in range(prediction_days, len(model_inputs))])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # Plot results
    test_data.index = pd.to_datetime(test_data.index)
    plot_results(test_data.index, actual_prices, prediction_prices, crypto_currency)

