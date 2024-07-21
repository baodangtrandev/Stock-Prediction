import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import subprocess
import os
import plotly.graph_objects as go
#result

# streamlit run host.py


#title
st.title('Stock Price Prediction')
#description
st.write('A LSTM model to predict the stock price of companies using historical data.')

################## select company ##################
#get tickers
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table = pd.read_html(url) #read html tables
df = table[0]
companyName = df['Security'].tolist() #company name
companySymbol = df['Symbol'].tolist() #company symbol
maps = dict(zip(companyName, companySymbol)) #[Apple, AAPL]

if 'selected_companies' not in st.session_state:
    st.session_state.selected_companies = []

new_company = st.selectbox('Select a Company', companyName)
if st.button('Add Company'):
    #if the selected company is not in the list -> add
    if new_company not in st.session_state.selected_companies:
        st.session_state.selected_companies.append(new_company)
        st.success(f"Added {new_company} for prediction")

for company in st.session_state.selected_companies:
    st.write(f"- {company}")

################## select date ##################
# selectedBeginDay = st.date_input('Select Begin Date', pd.to_datetime('2024-01-01'))
# selectedEndDay = st.date_input('Select End Date', pd.to_datetime('2024-12-31'))

selectedDays = st.slider('Select number of rollback days', min_value=1, max_value=3650, value=30, step=1)
st.write(f'You selected {selectedDays} days')

#combine 2 files
if st.button('Run Predictions'):
    data_dict = {}
    for company in st.session_state.selected_companies:
        symbol = maps[company]
        st.write(f"Running prediction for {company}...")
        # Run stockPrediction.py for each company
        subprocess.run(["python", "stockPrediction.py", symbol, str(selectedDays)], check=True)

        # Display results
        st.write(f"Stock Prediction for {company} is completed.")
        with open('prediction_result.txt', 'r') as file:
            prediction_result = file.read()
        st.write(f"Predicted closing price of {company} is: {prediction_result}")
        # Fetch data
        with open('fetched_data.csv', 'r') as file:
            data = pd.read_csv(file)
        data_dict[company] = data

    # Create the comparison chart
    fig = go.Figure()
    for company, data in data_dict.items():
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=company
        ))

    fig.update_layout(
        title='Stock Price Comparison',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        xaxis_type="date",
        xaxis_rangeslider=dict(
            visible=True,
            thickness=0.05,
            bgcolor="lightgray"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    #line chart for close price
    fig_close_price = go.Figure()
    for company, data in data_dict.items():
        fig_close_price.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name=company
        ))

    fig_close_price.update_layout(
        title='Close Price Comparison',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_type="date",
        xaxis_rangeslider_visible = True
    )

    st.plotly_chart(fig_close_price, use_container_width=True)






# test
# print("---------ALL-----------")
# for i, t in enumerate(table):
#     print(f"Table {i}:")
#     print(t.head())
#     print("\n")

# dfT = table[1]
# dataT = dfT['Date']
# print(dataT)
# selectTest = st.selectbox('Select Date', dataT)
# print("---------TABLE 1-----------")
# dfT = table[1]
# print(dfT.head())
#streamlit <select>
# stock_name1 = st.selectbox('Select Company 1', companyName)
# stock_name2 = st.selectbox('Select Company 2', companyName)
# stock_name3 = st.selectbox('Select Company 3', companyName)
#maps
# if stock_name1:
#     selectedCompany1 = maps[stock_name1]
#     st.write(selectedCompany1)
# if stock_name2:
#     selectedCompany2 = maps[stock_name2]
#     st.write(selectedCompany2)
# if stock_name3:
#     selectedCompany3 = maps[stock_name3]
#     st.write(selectedCompany3)


# =>  selectedCompany