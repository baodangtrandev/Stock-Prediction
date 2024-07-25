import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM


st.title("Stock Prediction Application")

if 'data' not in st.session_state:
    st.warning("No data available. Please run the visualization script first.")
else:
    st.write("Data loaded successfully for prediction.")

data = st.session_state.data

close_data = data.filter(['Close'])
close_data = close_data.values
train_data_len = int(np.ceil(len(close_data) * 0.9))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

train_data = scaled_data[0:train_data_len, :]

x_train = []
y_train = []

for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # add one more dimension of the input data

@st.cache_resource
def create_and_train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

model = create_and_train_model(x_train, y_train)

# Test data
test_data = scaled_data[train_data_len - 100:, :]

x_test = []
y_test = close_data[train_data_len:, :]

for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
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