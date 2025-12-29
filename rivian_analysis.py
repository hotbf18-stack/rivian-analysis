import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === FIX FOR YAHOO FINANCE BLOCKING ON STREAMLIT CLOUD ===
# Set a real browser User-Agent to avoid rate limiting / blocking
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Apply headers to yfinance session
session = yf.shared._session or yf.utils.get_yf_session()
session.headers.update(headers)

# Clear any cached errors from previous failed attempts
yf.shared._DFS = {}
yf.shared._ERRORS = {}

# ================================================

st.title("🚗 Rivian (RIVN) Stock Technical Analysis")

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data():
    ticker = "RIVN"
    try:
        # Download 1 year of daily data
        hist = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        
        # Get current market info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist['Close'].iloc[-1] if not hist.empty else None
        previous_close = info.get("regularMarketPreviousClose")
        volume = info.get("volume")
        market_cap = info.get("marketCap")
        
        return hist, current_price, previous_close, volume, market_cap
    
    except Exception as e:
        st.error(f"Could not fetch data: {str(e)}")
        return pd.DataFrame(), None, None, None, None

hist, current_price, previous_close, volume, market_cap = fetch_data()

# Current Metrics
st.subheader("📊 Current Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
col2.metric("Previous Close", f"${previous_close:.2f}" if previous_close else "N/A")
col3.metric("Volume", f"{volume:,}" if volume else "N/A")
col4.metric("Market Cap", f"${market_cap / 1e9:.2f}B" if market_cap else "N/A")

if hist.empty:
    st.warning("No stock data available right now. Please try again in a few minutes.")
    st.stop()

# Calculate Technical Indicators
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

hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

rolling_mean = hist['Close'].rolling(window=20).mean()
rolling_std = hist['Close'].rolling(window=20).std()
hist['BB_Upper'] = rolling_mean + (rolling_std * 2)
hist['BB_Lower'] = rolling_mean - (rolling_std * 2)

# Recent Data
st.subheader("📅 Recent Price Data")
st.dataframe(hist[["Open", "High", "Low", "Close", "Volume"]].tail(10).round(2))

# Price Chart
st.subheader("📈 Price Chart with Indicators")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist.index, hist['Close'], label='Close Price', color='blue', linewidth=2)
ax.plot(hist.index, hist['SMA_50'], label='50-day SMA', color='orange')
ax.plot(hist.index, hist['SMA_200'], label='200-day SMA', color='red')
ax.plot(hist.index, hist['BB_Upper'], label='Upper Bollinger', color='green', linestyle='--', alpha=0.8)
ax.plot(hist.index, hist['BB_Lower'], label='Lower Bollinger', color='green', linestyle='--', alpha=0.8)
ax.fill_between(hist.index, hist['BB_Lower'], hist['BB_Upper'], color='green', alpha=0.1)
ax.set_title("RIVN Price with Moving Averages & Bollinger Bands")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# RSI
st.subheader("🔄 RSI (14-day)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
ax_rsi.plot(hist.index, hist['RSI_14'], color='purple', linewidth=2)
ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought')
ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold')
ax_rsi.set_ylim(0, 100)
ax_rsi.set_title("Relative Strength Index (RSI)")
ax_rsi.legend()
ax_rsi.grid(True, alpha=0.3)
st.pyplot(fig_rsi)

# MACD
st.subheader("📉 MACD")
fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
ax_macd.plot(hist.index, hist['MACD'], label='MACD Line', color='blue')
ax_macd.plot(hist.index, hist['MACD_Signal'], label='Signal Line', color='orange')
ax_macd.bar(hist.index, hist['MACD'] - hist['MACD_Signal'], label='Histogram', color='gray', alpha=0.6)
ax_macd.axhline(0, color='black', linewidth=0.8)
ax_macd.set_title("MACD Indicator")
ax_macd.legend()
ax_macd.grid(True, alpha=0.3)
st.pyplot(fig_macd)

# Quick Insights
st.subheader("💡 Quick Technical Outlook")
latest = hist.iloc[-1]
insights = []

if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
    insights.append("🟢 **Strong Bullish Trend**")
elif latest['Close'] > latest['SMA_50']:
    insights.append("🟡 **Moderate Bullish**")

if latest['RSI_14'] > 70:
    insights.append("🔴 **Overbought** – Risk of pullback")
elif latest['RSI_14'] < 30:
    insights.append("🟢 **Oversold** – Possible rebound")

if latest['MACD'] > latest['MACD_Signal']:
    insights.append("🟢 **Bullish Momentum**")
else:
    insights.append("🔴 **Bearish Momentum**")

for insight in insights:
    st.write(insight)

if not insights:
    st.write("⚪ Neutral market signals")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from Yahoo Finance | Not financial advice")
