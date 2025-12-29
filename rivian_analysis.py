import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Rivian (RIVN) Stock Technical Analysis")

@st.cache_data(ttl=300)
def fetch_data():
    ticker = "RIVN"
    stock = yf.Ticker(ticker)
    hist = yf.download(ticker, period="1y", interval="1d")
    info = stock.info
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("regularMarketPreviousClose")
    volume = info.get("volume")
    market_cap = info.get("marketCap")
    return hist, current_price, previous_close, volume, market_cap

hist, current_price, previous_close, volume, market_cap = fetch_data()

st.subheader("Current Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
col2.metric("Previous Close", f"${previous_close:.2f}" if previous_close else "N/A")
col3.metric("Volume", f"{volume:,}" if volume else "N/A")
col4.metric("Market Cap", f"${market_cap / 1e9:.2f}B" if market_cap else "N/A")

if not hist.empty:
    # Simple indicators
    delta = hist['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    hist['RSI_14'] = 100 - (100 / (1 + rs))

    exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = exp12 - exp26
    hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

    hist['SMA_50'] = hist['Close'].rolling(50).mean()
    hist['SMA_200'] = hist['Close'].rolling(200).mean()

    rolling_mean = hist['Close'].rolling(20).mean()
    rolling_std = hist['Close'].rolling(20).std()
    hist['BB_Upper'] = rolling_mean + (rolling_std * 2)
    hist['BB_Lower'] = rolling_mean - (rolling_std * 2)

    st.subheader("Historical Data (Last 10 Days)")
    st.dataframe(hist[["Open", "High", "Low", "Close", "Volume"]].tail(10))

    st.subheader("Price Chart with Indicators")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist.index, hist['Close'], label='Close', color='blue')
    ax.plot(hist.index, hist['SMA_50'], label='50-day SMA', color='orange')
    ax.plot(hist.index, hist['SMA_200'], label='200-day SMA', color='red')
    ax.plot(hist.index, hist['BB_Upper'], label='Upper BB', color='green', linestyle='--')
    ax.plot(hist.index, hist['BB_Lower'], label='Lower BB', color='green', linestyle='--')
    ax.fill_between(hist.index, hist['BB_Upper'], hist['BB_Lower'], alpha=0.1, color='green')
    ax.set_title("RIVN Price with SMAs and Bollinger Bands")
    ax.legend()
    st.pyplot(fig)

    st.subheader("RSI (14)")
    fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
    ax_rsi.plot(hist.index, hist['RSI_14'], color='purple')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(30, color='green', linestyle='--')
    ax_rsi.set_title("RSI")
    st.pyplot(fig_rsi)

    st.subheader("MACD")
    fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
    ax_macd.plot(hist.index, hist['MACD'], label='MACD', color='blue')
    ax_macd.plot(hist.index, hist['MACD_Signal'], label='Signal', color='orange')
    ax_macd.bar(hist.index, hist['MACD'] - hist['MACD_Signal'], label='Histogram', color='gray', alpha=0.5)
    ax_macd.set_title("MACD")
    ax_macd.legend()
    st.pyplot(fig_macd)

    latest = hist.iloc[-1]
    st.subheader("Quick Insights")
    if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
        st.write("**Bullish:** Price above 50-day and 200-day SMA")
    if latest['RSI_14'] > 70:
        st.write("**Overbought (RSI > 70)**")
    elif latest['RSI_14'] < 30:
        st.write("**Oversold (RSI < 30)**")
    if latest['MACD'] > latest['MACD_Signal']:
        st.write("**Bullish MACD crossover**")

st.caption(f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}. Not financial advice.")