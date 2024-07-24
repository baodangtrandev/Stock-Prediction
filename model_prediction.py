

import streamlit as st


visualize_page = st.Page("visualize_data.py",title="Visualize Data", icon="📊")
prediction_page = st.Page("prediction_data.py",title="Model Prediction", icon="📈")

pg = st.navigation(
                {
                    "Visualize Data": [visualize_page],
                    "Model Prediction": [prediction_page],
                })
pg.run()

# st.subheader('Data Frame')
# st.write(data.tail())

# st.subheader('Summary of the data')
# st.write(data.describe())

# st.subheader('Closing Price vs Time Chart')

# def plot_chart_price(data):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
#     fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_chart_price(data)

# st.subheader('Volume vs Time Chart')

# def plot_chart_volume(data):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name='Volume'))
#     fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_chart_volume(data)



# st.subheader('Daily Return vs Time Chart')

# def plot_chart_return(data):
#     data['Daily Return'] = data['Adj Close'].pct_change()
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=data['Date'], y=data['Daily Return'], name='Daily Return'))
#     fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_chart_return(data)

# st.subheader('Daily Return Histogram')

# def plot_chart_return_hist(data):
#     data['Daily Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage
#     fig = go.Figure()
#     fig.add_trace(go.Histogram(
#         x=data['Daily Return'].dropna(), 
#         name='Daily Return',
#         xbins=dict(
#             start=-10,  # Adjust the start value as per the new unit
#             end=10,     # Adjust the end value as per the new unit
#             size=0.25    # Adjust the bin size to control the distance between bars
#         ),
#         marker=dict(line=dict(width=0.5))  # Adds a small line between the bars for better distinction
#     ))
#     fig.update_layout(
#         title_text='Daily Return Histogram',
#         xaxis_title='Daily Return (%)',  # Update the x-axis title to reflect the new unit
#         yaxis_title='Frequency',
#         xaxis=dict(
#             tickmode='linear',
#             tick0=-10,
#             dtick=1  
#         ),
#         yaxis=dict(
#             tickmode='linear',
#             tick0=0,
#             dtick=20  
#         ),
#         xaxis_rangeslider_visible=True,
#         height=400,  
#         bargap=0.1  
#     )
#     st.plotly_chart(fig)

# plot_chart_return_hist(data)

# # Train the model and make prediction

