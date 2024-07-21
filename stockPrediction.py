import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import sys

# python stockPrediction.py

if len(sys.argv) < 3:
    print("Usage: python stockPrediction.py <stock_symbol> <days_back>")
    sys.exit(1)

STOCK_SYMBOL = sys.argv[1]
DAYS_BACK = int(sys.argv[2])



def download_stock_data(symbol, start_date, end_date):
    #Download stock data from Yahoo Finance.
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True) #    &
    return data



def plot_candlestick_chart(data):
    # Plot candlestick chart
    figure = go.Figure(data=[go.Candlestick(x=data['Date'],
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'])])
    figure.update_layout(title=f"{STOCK_SYMBOL} Stock Price Analysis", xaxis_rangeslider_visible=False)
    figure.show()

def prepare_data_for_model(data):
    # Prepare data for LSTM model.

    # x = [Open, High, Low, Volume]
    # => y = Close

    x = data[['Open', 'High', 'Low', 'Volume']].to_numpy()
    y = data['Close'].to_numpy().reshape(-1, 1)
    return train_test_split(x, y, test_size=0.2, random_state=42)

def build_and_train_model(x_train, y_train):
    #Build and train LSTM model.
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=30)
    return model

def main():
    #date range
    start_date = (date.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    end_date = date.today().strftime("%Y-%m-%d")

    data = download_stock_data(STOCK_SYMBOL, start_date, end_date)
    #draw candlestick chart
    plot_candlestick_chart(data)

    print(data.corr()["Close"].sort_values(ascending=False))

    x_train, x_test, y_train, y_test = prepare_data_for_model(data)
    model = build_and_train_model(x_train, y_train)

    # Predict with a sample feature set
    # [Open, High, Low, Volume]
    features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
    # Output = predicted closing price
    print ("Predicted closing price: ")
    print(model.predict(features))
    #save to file
    with open('prediction_result.txt', 'w') as file:
        file.write(str(model.predict(features)))
    print("Completed")

    with open('fetched_data.csv', 'w') as file:
        data.to_csv(file, index=False)

if __name__ == "__main__":
    main()