import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import streamlit as st
from plotly import graph_objs as go

# get data frame
start = '2014-01-01'
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Application")

stock = st.text_input("Enter the stock symbol", value="GOOG")

n_years = st.slider("Years of prediction:", 1, 10)
days = n_years * 365

@st.cache_data
def get_data(stock, start, today):
    data = yf.download(stock, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = get_data(stock, start, today)
data_load_state.text("Loading data...done!")

st.subheader('Data Frame')
st.write(data.tail())

st.subheader('Summary of the data')
st.write(data.describe())

st.subheader('Closing Price vs Time Chart')

def plot_chart_price(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_price(data)

st.subheader('Volume vs Time Chart')

def plot_chart_volume(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Volume'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_volume(data)

st.subheader('Average of Various Stock')

ma = [25, 50, 75, 100, 125, 150, 175, 200]

selected_ma = st.selectbox("Select Moving Average", ma)

if 'selected_ma' not in st.session_state:
    st.session_state.selected_ma = []

if st.button('Add MA'):
    if selected_ma not in st.session_state.selected_ma:
        st.session_state.selected_ma.append(selected_ma)
        st.success("Added: {}".format(selected_ma))

def plot_chart_ma(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    for ma in st.session_state.selected_ma:
        data['MA'+str(ma)] = data['Close'].rolling(window=ma).mean()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MA'+str(ma)], name='MA'+str(ma)))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_ma(data)

st.subheader('Daily Return vs Time Chart')

def plot_chart_return(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Daily Return'], name='Daily Return'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_return(data)

st.subheader('Daily Return Histogram')

def plot_chart_return_hist(data):
    data['Daily Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data['Daily Return'].dropna(), 
        name='Daily Return',
        xbins=dict(
            start=-10,  # Adjust the start value as per the new unit
            end=10,     # Adjust the end value as per the new unit
            size=0.25    # Adjust the bin size to control the distance between bars
        ),
        marker=dict(line=dict(width=0.5))  # Adds a small line between the bars for better distinction
    ))
    fig.update_layout(
        title_text='Daily Return Histogram',
        xaxis_title='Daily Return (%)',  # Update the x-axis title to reflect the new unit
        yaxis_title='Frequency',
        xaxis=dict(
            tickmode='linear',
            tick0=-10,
            dtick=1  
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=20  
        ),
        xaxis_rangeslider_visible=True,
        height=400,  
        bargap=0.1  
    )
    st.plotly_chart(fig)

plot_chart_return_hist(data)

# Train the model and make prediction
from sklearn.preprocessing import MinMaxScaler

close_data = data.filter(['Close'])
close_data = close_data.values
print(close_data)
train_data_len = int(np.ceil(len(close_data)*0.9))
print(len(close_data))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)
print(scaled_data)

train_data = scaled_data[0:train_data_len, :]

print("number of training data: ", len(train_data))
print("train_data: ", train_data)

x_train = []
y_train = []

for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # add one more dimension of the input data
print(x_train.shape)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

def train_and_predict(x_train, y_train, x_test):
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    predictions = model.predict(x_test)
    return predictions

# Test data
test_data = scaled_data[train_data_len - 100:, :]

x_test = []
y_test = close_data[train_data_len:, :]

for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = train_and_predict(x_train, y_train, x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)   
print(rmse)

# Plot the data
train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = predictions

# visualize the data
print(valid)

st.subheader('Prediction vs Actual Data')

def plot_chart_prediction(train, valid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=train['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Predictions'], name='Predictions'))
    fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Close'], name='Actual Price'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_prediction(train, valid)
