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

stock = st.text_input("Enter the stock symbol",value="GOOG")

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
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Vloume'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_chart_volume(data)
