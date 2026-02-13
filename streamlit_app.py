
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Apple Stock Interactive Dashboard", layout="wide")

st.title("Apple Stock Forecasting & Analysis Dashboard (SARIMA)")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("P639_DATASET.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("sarima_model.pkl")

model = load_model()

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Dashboard Controls")

start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())

filtered_df = df.loc[str(start_date):str(end_date)]

show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_volatility = st.sidebar.checkbox("Show Volatility", True)
show_events = st.sidebar.checkbox("Show Major Events", True)

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# ==========================================================
# PRICE + TREND
# ==========================================================

st.header("Price & Trend Analysis")

fig_price = go.Figure()

fig_price.add_trace(go.Scatter(
    x=filtered_df.index,
    y=filtered_df['Close'],
    mode='lines',
    name='Close Price'
))

if show_ma:
    filtered_df['MA20'] = filtered_df['Close'].rolling(20).mean()
    filtered_df['MA100'] = filtered_df['Close'].rolling(100).mean()

    fig_price.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['MA20'],
        mode='lines',
        name='20-Day MA'
    ))

    fig_price.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['MA100'],
        mode='lines',
        name='100-Day MA'
    ))

if show_events:
    fig_price.add_vline(x="2020-03-01", line_dash="dash", line_color="red")

fig_price.update_layout(title="Interactive Price Trend")
st.plotly_chart(fig_price, use_container_width=True)

# ==========================================================
# VOLATILITY
# ==========================================================

if show_volatility:
    st.header("Volatility Analysis")

    filtered_df['Returns'] = filtered_df['Close'].pct_change()
    filtered_df['Rolling_Std'] = filtered_df['Returns'].rolling(20).std()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['Rolling_Std'],
        mode='lines',
        name='20-Day Rolling Volatility'
    ))

    fig_vol.update_layout(title="Market Risk Indicator")
    st.plotly_chart(fig_vol, use_container_width=True)

# ==========================================================
# SEASONALITY
# ==========================================================

st.header("Seasonality & Decomposition")

decomp_type = st.selectbox("Decomposition Model", ["additive", "multiplicative"])

decomposition = seasonal_decompose(
    filtered_df['Close'],
    model=decomp_type,
    period=30
)

fig_season = go.Figure()
fig_season.add_trace(go.Scatter(
    x=filtered_df.index,
    y=decomposition.seasonal,
    mode='lines',
    name='Seasonal Component'
))

fig_season.update_layout(title="Seasonality Pattern")
st.plotly_chart(fig_season, use_container_width=True)

# ==========================================================
# MODEL EVALUATION
# ==========================================================

st.header("Model Evaluation")

train_size = int(len(df) * 0.8)
train = df['Close'][:train_size]
test = df['Close'][train_size:]

pred = model.predict(start=len(train), end=len(df)-1)
rmse = np.sqrt(mean_squared_error(test, pred))

col1, col2 = st.columns(2)

col1.metric("SARIMA RMSE", round(rmse, 2))
col2.metric("Forecast Horizon Selected", forecast_days)

# ==========================================================
# FORECASTING
# ==========================================================

st.header("Future Forecast")

if st.button("Generate Forecast"):

    forecast = model.forecast(steps=forecast_days)

    last_date = df.index[-1]
    forecast_index = pd.date_range(
        start=last_date,
        periods=forecast_days+1,
        freq="D"
    )[1:]

    forecast_series = pd.Series(forecast.values, index=forecast_index)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines',
        name='Forecast'
    ))

    fig_forecast.update_layout(title="Future Price Forecast (SARIMA)")

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.success("Forecast generated successfully")
