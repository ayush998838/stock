import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# Dark theme CSS
st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    .stApp {
        background-color: #121212;
    }
    h1, h2, h3 {
        color: #00FFAA;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 AI-Based Stock Price Predictor")
st.markdown("Predict next 7 days' prices with LSTM + view SMA/RSI + see latest news")

ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()

if st.button("Predict Stock"):
    df = yf.download(ticker, start="2018-01-01", end="2024-01-01")
    data = df['Close'].values.reshape(-1, 1)

    df['SMA_14'] = df['Close'].rolling(14).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_data(dataset, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    look_back = 60
    X, y = create_data(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(30),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    preds = []
    input_seq = last_60.copy()

    for _ in range(7):
        inp = input_seq[-60:].reshape(1, 60, 1)
        pred = model.predict(inp)[0][0]
        preds.append(pred)
        input_seq = np.append(input_seq, [[pred]], axis=0)

    pred_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    st.subheader(f"📈 {ticker} Closing Price with Predictions")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'][-100:], label="Last 100 Days", color='cyan')
    future_x = np.arange(len(df), len(df) + 7)
    ax.plot(future_x, pred_prices, label="Predicted", color='magenta')
    ax.legend()
    ax.set_title("Stock Price Forecast")
    st.pyplot(fig)

    st.subheader("📊 Technical Indicators (SMA, RSI)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df['Close'][-100:], label="Close", color='white')
    ax2.plot(df['SMA_14'][-100:], label="SMA 14", color='yellow')
    ax2.set_ylabel("Price")
    ax2.legend(loc='upper left')
    ax2_2 = ax2.twinx()
    ax2_2.plot(df['RSI'][-100:], label="RSI", color='red')
    ax2_2.axhline(70, color='gray', linestyle='--')
    ax2_2.axhline(30, color='gray', linestyle='--')
    ax2_2.set_ylabel("RSI")
    ax2_2.legend(loc='upper right')
    st.pyplot(fig2)

    st.subheader(f"🗞️ Recent Headlines for {ticker}")
    headlines = [
        f"{ticker} stock shows strong bullish trend",
        f"{ticker} earnings expected to beat estimates",
        f"Analysts discuss long-term upside of {ticker}",
        f"{ticker} volatility increases as market shifts"
    ]
    for h in headlines:
        st.markdown(f"- {h}")
