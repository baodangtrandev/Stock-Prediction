import streamlit as st
import yfinance as yf
from datetime import date
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from plotly import graph_objs as go

st.title("Stock Prediction Application")

# Get data frame
start = '2014-01-01'
today = date.today().strftime("%Y-%m-%d")

stock = st.text_input("Enter the stock symbol", value="GOOG")

@st.cache_data
def get_data(stock, start, today):
    data = yf.download(stock, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
st.session_state.data = get_data(stock, start, today)
data_load_state.text("Loading data...done!")

data = st.session_state.data
pr = ProfileReport(data, title="Pandas Profiling Report")
st_profile_report(pr)

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
