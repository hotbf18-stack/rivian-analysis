import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from polygon import RESTClient

# Polygon client (API key is auto-configured in many environments; if not, add yours)
client = RESTClient()  # If needed: RESTClient(api_key="YOUR_KEY_HERE")

st.title("🚗 Rivian (RIVN) Stock Technical Analysis")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data():
    ticker = "RIVN"
    try:
        # Historical data (1 year daily bars)
        today = date.today()
        one_year_ago = today - timedelta(days=365)
        hist = client.get_aggs(ticker, 1, "day", one_year_ago, today)
        if not hist:
            raise Exception("No historical data returned")
        
        # Convert to DataFrame
        df = pd.DataFrame(hist)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Snapshot for current metrics
        snapshot = client.get_snapshot_all('stocks', [ticker])[0]
        details = client.get_ticker_details(ticker)
        
        current_price = snapshot.last_trade.price if snapshot.last_trade else snapshot.min.close
        previous_close = snapshot.prev_day.close
        volume = snapshot.day.volume
        market_cap = details.market_cap
        
        return df, current_price, previous_close, volume, market_cap
    
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        st.info("Try again in a few minutes or check API limits.")
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
    st.warning("No data available right now.")
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

hist = hist.dropna()  # Clean up NaNs

# Recent Data
st.subheader("📅 Recent Price Data")
st.dataframe(hist[["Open", "High", "Low", "Close", "Volume"]].tail(10).round(2))

# Price Chart
st.subheader("📈 Price Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist.index, hist['Close'], label='Close', color='blue')
ax.plot(hist.index, hist['SMA_50'], label='50-day SMA', color='orange')
ax.plot(hist.index, hist['SMA_200'], label='200-day SMA', color='red')
ax.plot(hist.index, hist['BB_Upper'], label='Upper BB', color='green', linestyle='--')
ax.plot(hist.index, hist['BB_Lower'], label='Lower BB', color='green', linestyle='--')
ax.fill_between(hist.index, hist['BB_Lower'], hist['BB_Upper'], color='green', alpha=0.1)
ax.set_title("RIVN Price with Indicators")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# RSI
st.subheader("🔄 RSI (14)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
ax_rsi.plot(hist.index, hist['RSI_14'], color='purple')
ax_rsi.axhline(70, color='red', linestyle='--')
ax_rsi.axhline(30, color='green', linestyle='--')
ax_rsi.set_ylim(0, 100)
ax_rsi.set_title("RSI")
ax_rsi.grid(alpha=0.3)
st.pyplot(fig_rsi)

# MACD
st.subheader("📉 MACD")
fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
ax_macd.plot(hist.index, hist['MACD'], label='MACD', color='blue')
ax_macd.plot(hist.index, hist['MACD_Signal'], label='Signal', color='orange')
ax_macd.bar(hist.index, hist['MACD'] - hist['MACD_Signal'], color='gray', alpha=0.6)
ax_macd.axhline(0, color='black')
ax_macd.set_title("MACD")
ax_macd.legend()
ax_macd.grid(alpha=0.3)
st.pyplot(fig_macd)

# Insights
st.subheader("💡 Quick Insights")
latest = hist.iloc[-1]
insights = []
if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
    insights.append("🟢 Strong bullish trend")
if latest['RSI_14'] > 70:
    insights.append("🔴 Overbought – possible pullback")
elif latest['RSI_14'] < 30:
    insights.append("🟢 Oversold – possible rebound")
if latest['MACD'] > latest['MACD_Signal']:
    insights.append("🟢 Bullish momentum")
else:
    insights.append("🔴 Bearish momentum")

for i in insights:
    st.write(i)

st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: Polygon API | Not investment advice")
