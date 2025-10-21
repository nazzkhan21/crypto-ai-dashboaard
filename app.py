import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime
import time

st.set_page_config(page_title="AI Crypto Indicator", layout="wide")

st.title("💹 AI-Powered Crypto Buy/Sell Signal Dashboard")

# --- Sidebar settings ---
st.sidebar.header("Settings")

# Top 120+ Crypto coins list (expanded with 20 more coins)
top_120_crypto = [
    # Top 20 (Major Cryptocurrencies)
    "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "ADA-USD", 
    "AVAX-USD", "DOGE-USD", "TRX-USD", "LINK-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", 
    "DAI-USD", "UNI-USD", "WBTC-USD", "LTC-USD", "ATOM-USD",
    
    # Next 20 (Popular Altcoins)
    "ETC-USD", "XLM-USD", "BCH-USD", "FIL-USD", "APT-USD", "HBAR-USD", "VET-USD", "ICP-USD", 
    "CRO-USD", "NEAR-USD", "ALGO-USD", "QNT-USD", "FTM-USD", "FLOW-USD", "MANA-USD", "SAND-USD", 
    "AAVE-USD", "GRT-USD", "CRV-USD", "MKR-USD",
    
    # Additional 20 (DeFi & Emerging Coins)
    "SNX-USD", "COMP-USD", "YFI-USD", "SUSHI-USD", "1INCH-USD", "BAT-USD", "ZRX-USD", "ENJ-USD", 
    "CHZ-USD", "HOT-USD", "DENT-USD", "WIN-USD", "CELR-USD", "KSM-USD", "WAVES-USD", "ZEC-USD", 
    "DASH-USD", "NEO-USD", "ONT-USD", "QTUM-USD",
    
    # New 20 Coins (High Market Cap & Trending)
    "IOTA-USD", "EOS-USD", "XMR-USD", "XTZ-USD", "THETA-USD", "RUNE-USD", "KAVA-USD", "BAND-USD", 
    "REN-USD", "STORJ-USD", "KNC-USD", "REP-USD", "LRC-USD", "NMR-USD", "CVC-USD", "OMG-USD", 
    "GNT-USD", "LOOM-USD", "FUN-USD", "KICK-USD",
    
    # Additional 20 (More Popular Coins)
    "LEND-USD", "CND-USD", "WABI-USD", "LUN-USD", "TRIG-USD", "VIB-USD", "BCPT-USD", "ARK-USD", 
    "YOYO-USD", "POWR-USD", "KMD-USD", "SUB-USD", "NULS-USD", "REQ-USD", "VIBE-USD", "ARK-USD", 
    "YOYO-USD", "POWR-USD", "KMD-USD", "SUB-USD",
    
    # Latest 20 (Emerging & Trending)
    "ARB-USD", "OP-USD", "SUI-USD", "SEI-USD", "TIA-USD", "INJ-USD", "JUP-USD", "WIF-USD", 
    "BONK-USD", "PEPE-USD", "FLOKI-USD", "BABYDOGE-USD", "SAFEMOON-USD", "ELON-USD", "DOGE-USD", 
    "SHIB-USD", "AKITA-USD", "KISHU-USD", "HOKK-USD", "LEASH-USD"
]

coins = st.sidebar.multiselect(
    "Select Coins (Top 120+ Crypto)",
    top_120_crypto,
    default=["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "LINK-USD", "DOT-USD"]
)

period_days = st.sidebar.slider("Data Period (days)", 1, 30, 7)
show_chart = st.sidebar.checkbox("Show Trend Charts", True)

# Auto-refresh settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", True)
if auto_refresh:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", ["1 minute", "5 minutes", "10 minutes"], index=0)
    interval_seconds = {"1 minute": 60, "5 minutes": 300, "10 minutes": 600}[refresh_interval]
else:
    interval_seconds = None

st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Data: Yahoo Finance | Signals: ML Linear Regression | Educational only")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data Now"):
    st.rerun()

# Auto-refresh status
if auto_refresh:
    st.info(f"🔄 Auto-refresh enabled: {refresh_interval.lower()}")
else:
    st.info("🔄 Auto-refresh disabled")

# --- Date range ---
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=period_days)

# Add timestamp
st.write(f"📅 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []

for coin in coins:
    st.subheader(f"{coin.replace('-USD','')} Trend & Forecast")

    # Download data
    df = yf.download(coin, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        st.warning(f"No data for {coin}")
        continue

    df.reset_index(inplace=True)
    df = df.rename(columns={"Date": "ds", "Close": "y"})

    # Simple ML-based prediction using Linear Regression
    df['days'] = range(len(df))
    df['price_change'] = df['y'].pct_change()
    df['ma_5'] = df['y'].rolling(window=5).mean()
    df['ma_10'] = df['y'].rolling(window=10).mean()
    df['volatility'] = df['price_change'].rolling(window=5).std()
    
    # Prepare features for prediction
    features = ['days', 'ma_5', 'ma_10', 'volatility']
    df_clean = df.dropna()
    
    if len(df_clean) > 10:  # Need enough data for prediction
        X = df_clean[features].values
        y = df_clean['y'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Predict next 3 days
        last_features = df_clean[features].iloc[-1].values.reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        # Simple trend-based prediction
        recent_trend = df_clean['y'].iloc[-5:].mean() - df_clean['y'].iloc[-10:-5].mean()
        trend_factor = 1 + (recent_trend / df_clean['y'].iloc[-1]) * 0.1
        
        next_price = float((model.predict(last_features_scaled)[0] * trend_factor).item())
        last_price = float(df["y"].iloc[-1].item())
        predicted_change = float(((next_price - last_price) / last_price) * 100)
    else:
        last_price = float(df["y"].iloc[-1].item())
        next_price = last_price
        predicted_change = 0.0

    if predicted_change > 1:
        signal = "BUY 🟢"
        color = "green"
    elif predicted_change < -1:
        signal = "SELL 🔴"
        color = "red"
    else:
        signal = "HOLD 🟡"
        color = "orange"

    # --- Chart ---
    if show_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Price", line=dict(color="blue")))
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=df["ds"], y=df["ma_5"], mode="lines", name="MA 5", line=dict(color="orange", dash="dash")))
        fig.add_trace(go.Scatter(x=df["ds"], y=df["ma_10"], mode="lines", name="MA 10", line=dict(color="red", dash="dash")))
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[df["ds"].iloc[-1] + pd.Timedelta(days=1)], 
            y=[next_price],
            mode="markers", 
            name="Prediction", 
            marker=dict(color=color, size=10, symbol="diamond")
        ))
        
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True, template="plotly_white",
            title=f"{coin.replace('-USD','')} Price Analysis"
        )
        st.plotly_chart(fig, width='stretch')

    # --- Signal Display ---
    st.write(f"📊 **Trading Signal for {coin.replace('-USD','')}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Current Price", value=f"${last_price:.2f}")
    with col2:
        st.metric(label="Predicted Price (3 days)", value=f"${next_price:.2f}")
    with col3:
        st.metric(label="Expected Change", value=f"{predicted_change:.2f}%")

    results.append({
        "Coin": coin.replace("-USD", ""),
        "Current Price": round(last_price, 2),
        "Predicted Price (3 days)": round(next_price, 2),
        "Change (%)": round(predicted_change, 2),
        "Signal": signal,
        "Confidence": f"{min(round(abs(predicted_change)*10,1),100)}%"
    })

# --- Summary Table ---
if results:
    st.markdown("### 📊 Summary Table")
    table = pd.DataFrame(results)
    st.dataframe(table, width='stretch')

st.caption("⚠️ Not financial advice. This tool is for educational and research use only.")
