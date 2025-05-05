import streamlit as st
import pandas as pd
from fbprophet import Prophet

# Title of the app
st.title("📊 AI-Powered Demand Forecasting")

# Upload dataset
uploaded_file = st.file_uploader("Upload your historical sales data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Prepare data for Prophet
    df.rename(columns={"date": "ds", "sales": "y"}, inplace=True)
    
    # Train forecasting model
    model = Prophet()
    model.fit(df)

    # Create future dates
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Display forecast
    st.write("📈 Forecast Data:", forecast.tail())
    st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))
