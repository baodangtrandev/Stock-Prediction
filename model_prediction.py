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

st.subheader('Average of Various Stock')

ma = [25,50,75,100,125,150,175,200]

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
            dtick=1  # Adjust this value for more granular ticks (e.g., 0.01 for 0.01 steps)
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=20  # Adjust this value for more granular ticks (e.g., 0.01 for 0.01 steps)
        ),
        xaxis_rangeslider_visible=True,
        height=400,  # Adjust the height of the figure
        bargap=0.1   # Adjust the gap between bars
    )
    st.plotly_chart(fig)

plot_chart_return_hist(data)
