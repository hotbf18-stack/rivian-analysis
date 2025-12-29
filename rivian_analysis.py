import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests  # Needed for custom session

# === PROVEN YFINANCE FIX FOR STREAMLIT CLOUD (2025) ===
# Create a custom requests session with browser-like headers
session = requests.Session()
session.headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Force yfinance to use this session
yf.shared._session = session

# Clear any old caches/errors
yf.shared._DFS = {}
yf.shared._ERRORS = {}

# ================================================

st.title("🚗 Rivian (RIVN) Stock Technical Analysis")

@st.cache_data(ttl=3600)  # Refresh max once per hour
def fetch_data():
    ticker = "RIVN"
    try:
        hist = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if hist.empty:
            raise ValueError("No data returned")
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist['Close'].iloc[-1]
        previous_close = info.get("regularMarketPreviousClose")
        volume = info.get("volume")
        market_cap = info.get("marketCap")
        
        return hist, current_price, previous_close, volume, market_cap
    
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        st.info("Temporary Yahoo issue – refresh in a minute.")
        return pd.DataFrame(), None, None, None, None

hist, current_price, previous_close, volume, market_cap = fetch_data()

st.subheader("📊 Current Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
col2.metric("Previous Close", f"${previous_close:.2f}" if previous_close else "N/A")
col3.metric("Volume", f"{volume:,}" if volume else "N/A")
col4.metric("Market Cap", f"${market_cap / 1e9:.2f}B" if market_cap else "N/A")

if hist.empty:
    st.stop()

# Indicators
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

# Clean NaNs for plotting
hist_plot = hist.dropna()

st.subheader("📅 Recent Data")
st.dataframe(hist.tail(10)[["Open", "High", "Low", "Close", "Volume"]].round(2))

st.subheader("📈 Price Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist_plot.index, hist_plot['Close'], label='Close', color='blue', linewidth=2)
ax.plot(hist_plot.index, hist_plot['SMA_50'], label='50-day SMA', color='orange')
ax.plot(hist_plot.index, hist_plot['SMA_200'], label='200-day SMA', color='red')
ax.plot(hist_plot.index, hist_plot['BB_Upper'], label='Upper BB', color='green', linestyle='--')
ax.plot(hist_plot.index, hist_plot['BB_Lower'], label='Lower BB', color='green', linestyle='--')
ax.fill_between(hist_plot.index, hist_plot['BB_Lower'], hist_plot['BB_Upper'], alpha=0.1, color='green')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("🔄 RSI (14)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
ax_rsi.plot(hist_plot.index, hist_plot['RSI_14'], color='purple', linewidth=2)
ax_rsi.axhline(70, color='red', linestyle='--')
ax_rsi.axhline(30, color='green', linestyle='--')
ax_rsi.set_ylim(0, 100)
ax_rsi.grid(alpha=0.3)
st.pyplot(fig_rsi)

st.subheader("📉 MACD")
fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
ax_macd.plot(hist_plot.index, hist_plot['MACD'], label='MACD', color='blue')
ax_macd.plot(hist_plot.index, hist_plot['MACD_Signal'], label='Signal', color='orange')
ax_macd.bar(hist_plot.index, hist_plot['MACD'] - hist_plot['MACD_Signal'], color='gray', alpha=0.6)
ax_macd.axhline(0, color='black', linewidth=0.8)
ax_macd.legend()
ax_macd.grid(alpha=0.3)
st.pyplot(fig_macd)

st.subheader("💡 Quick Insights")
latest = hist_plot.iloc[-1]
insights = []
if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
    insights.append("🟢 Strong Bullish Trend")
if latest['RSI_14'] > 70:
    insights.append("🔴 Overbought")
elif latest['RSI_14'] < 30:
    insights.append("🟢 Oversold")
if latest['MACD'] > latest['MACD_Signal']:
    insights.append("🟢 Bullish Momentum")
else:
    insights.append("🔴 Bearish Momentum")

for insight in insights:
    st.write(insight)

st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data: Yahoo Finance via yfinance | Not financial advice")
