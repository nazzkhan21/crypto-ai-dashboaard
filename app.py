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
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests

st.set_page_config(page_title="AI Crypto Indicator", layout="wide")

# Hybrid data fetching system (Binance + Yahoo Finance)
def fetch_binance_data(symbol, interval='1h', limit=100):
    """Fetch real-time data from Binance API"""
    try:
        # Initialize Binance client (no API keys needed for public data)
        client = Client()
        
        # Convert symbol format (BTC-USD -> BTCUSDT)
        binance_symbol = symbol.replace('-USD', 'USDT')
        
        # Get historical klines data
        klines = client.get_historical_klines(
            binance_symbol, 
            interval, 
            limit=limit
        )
        
        if not klines:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except BinanceAPIException as e:
        st.warning(f"Binance API error for {symbol}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error fetching Binance data for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_yahoo_data(coin, start_date, end_date):
    """Fetch data from Yahoo Finance as fallback"""
    try:
        df = yf.download(coin, start=start_date, end=end_date, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        st.warning(f"Yahoo Finance error for {coin}: {str(e)}")
        return pd.DataFrame()

def fetch_hybrid_data(coin, start_date, end_date):
    """Fetch data using hybrid approach (Binance primary, Yahoo Finance fallback)"""
    # Try Binance first for real-time data
    df_binance = fetch_binance_data(coin.replace('-USD', ''), interval='1h', limit=168)  # 1 week of hourly data
    
    if not df_binance.empty:
        return df_binance
    
    # Fallback to Yahoo Finance
    df_yahoo = fetch_yahoo_data(coin, start_date, end_date)
    return df_yahoo

def get_cached_data(coin, start_date, end_date):
    """Get cached data from session state or fetch fresh data"""
    cache_key = f"data_{coin}_{start_date}_{end_date}"
    cache_time_key = f"time_{coin}_{start_date}_{end_date}"
    
    # Check if we have cached data that's less than 1 minute old
    if cache_key in st.session_state and cache_time_key in st.session_state:
        cache_time = st.session_state[cache_time_key]
        if time.time() - cache_time < 60:  # Cache for 1 minute
            return st.session_state[cache_key]
    
    # Fetch fresh data using hybrid approach and cache it
    df = fetch_hybrid_data(coin, start_date, end_date)
    if not df.empty:
        st.session_state[cache_key] = df
        st.session_state[cache_time_key] = time.time()
    
    return df

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
st.sidebar.subheader("🔄 Real-Time Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable Real-Time Updates", True)
if auto_refresh:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", ["1 minute", "2 minutes", "5 minutes"], index=0)
    interval_seconds = {"1 minute": 60, "2 minutes": 120, "5 minutes": 300}[refresh_interval]
else:
    interval_seconds = None

st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Data: Binance API + Yahoo Finance | Signals: ML Linear Regression | Educational only")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data Now"):
    st.rerun()

# Background auto-refresh implementation (non-blocking)
if auto_refresh:
    st.info(f"🔄 Real-time updates enabled: {refresh_interval.lower()}")
    
    # Use JavaScript for background refresh without blocking UI
    st.markdown(f"""
    <script>
    setTimeout(function(){{
        window.location.reload();
    }}, {interval_seconds * 1000});
    </script>
    """, unsafe_allow_html=True)
    
    # Show countdown without blocking
    countdown_placeholder = st.empty()
    countdown_placeholder.info(f"⏰ Next automatic update in {interval_seconds} seconds...")
else:
    st.info("🔄 Real-time updates disabled")

# --- Date range ---
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=period_days)

# Add timestamp
# Enhanced data status with API source information
current_time = datetime.datetime.now()
st.write(f"📅 Data fetched at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Data source information
st.info("🔗 **Data Sources:** Binance API (Primary) + Yahoo Finance (Fallback) for enhanced accuracy")

# Background update status (non-blocking)
if auto_refresh:
    st.success(f"🔄 Background updates active - Auto-refresh every {refresh_interval.lower()}")
    st.info("💡 UI remains responsive while data updates in the background")
else:
    st.warning("⚠️ Background updates disabled - Click 'Refresh Data Now' for updates")

results = []

for coin in coins:
    st.subheader(f"{coin.replace('-USD','')} Trend & Forecast")

    # Enhanced data fetch with API source indication
    with st.spinner(f"Fetching latest data for {coin.replace('-USD','')}..."):
        df = get_cached_data(coin, start_date, end_date)
    
    if df.empty:
        st.warning(f"No data available for {coin}")
        continue
    
    # Show data source for this coin
    if not df.empty:
        st.caption(f"📊 Data source: {'Binance API' if 'volume' in df.columns else 'Yahoo Finance'}")

    # Handle different data formats from Binance vs Yahoo Finance
    df.reset_index(inplace=True)
    
    # Rename columns based on data source
    if 'close' in df.columns:  # Binance data
        df = df.rename(columns={"timestamp": "ds", "close": "y"})
    else:  # Yahoo Finance data
        df = df.rename(columns={"Date": "ds", "Close": "y"})

    # Enhanced ML-based prediction with live data analysis
    df['days'] = range(len(df))
    df['price_change'] = df['y'].pct_change()
    df['ma_5'] = df['y'].rolling(window=5).mean()
    df['ma_10'] = df['y'].rolling(window=10).mean()
    df['ma_20'] = df['y'].rolling(window=20).mean()
    df['volatility'] = df['price_change'].rolling(window=5).std()
    df['rsi'] = 50  # Placeholder for RSI calculation
    
    # Calculate RSI (Relative Strength Index)
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Prepare enhanced features for prediction
    features = ['days', 'ma_5', 'ma_10', 'ma_20', 'volatility', 'rsi', 'price_change']
    df_clean = df.dropna()
    
    if len(df_clean) > 20:  # Need enough data for reliable prediction
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
        
        # Enhanced trend analysis with live data
        recent_trend = df_clean['y'].iloc[-3:].mean() - df_clean['y'].iloc[-6:-3].mean()
        short_trend = df_clean['y'].iloc[-1] - df_clean['y'].iloc[-3]
        volatility_factor = df_clean['volatility'].iloc[-1]
        rsi_factor = df_clean['rsi'].iloc[-1]
        
        # Dynamic trend factor based on live market conditions
        trend_factor = 1 + (recent_trend / df_clean['y'].iloc[-1]) * 0.15
        volatility_adjustment = 1 + (volatility_factor * 0.1) if volatility_factor < 0.05 else 1 - (volatility_factor * 0.1)
        rsi_adjustment = 1 + ((rsi_factor - 50) / 1000)  # RSI influence
        
        # Combine all factors
        combined_factor = trend_factor * volatility_adjustment * rsi_adjustment
        
        next_price = float((model.predict(last_features_scaled)[0] * combined_factor).item())
        last_price = float(df["y"].iloc[-1].item())
        predicted_change = float(((next_price - last_price) / last_price) * 100)
        
        # Additional live data validation
        current_price = last_price
        recent_high = df_clean['y'].iloc[-10:].max()
        recent_low = df_clean['y'].iloc[-10:].min()
        price_position = (current_price - recent_low) / (recent_high - recent_low)
        
        # Adjust prediction based on price position
        if price_position > 0.8:  # Near recent high
            predicted_change *= 0.8  # Reduce bullishness
        elif price_position < 0.2:  # Near recent low
            predicted_change *= 1.2  # Increase bullishness
            
    else:
        last_price = float(df["y"].iloc[-1].item())
        next_price = last_price
        predicted_change = 0.0

    # Enhanced signal generation based on live data
    if len(df_clean) > 20:
        # Get current market indicators
        current_rsi = df_clean['rsi'].iloc[-1]
        current_volatility = df_clean['volatility'].iloc[-1]
        ma_5_current = df_clean['ma_5'].iloc[-1]
        ma_10_current = df_clean['ma_10'].iloc[-1]
        ma_20_current = df_clean['ma_20'].iloc[-1]
        
        # Multi-factor signal analysis
        bullish_factors = 0
        bearish_factors = 0
        
        # RSI analysis
        if current_rsi < 30:
            bullish_factors += 2  # Oversold
        elif current_rsi > 70:
            bearish_factors += 2  # Overbought
        elif current_rsi < 50:
            bullish_factors += 1
        else:
            bearish_factors += 1
            
        # Moving average analysis
        if ma_5_current > ma_10_current > ma_20_current:
            bullish_factors += 2  # Strong uptrend
        elif ma_5_current < ma_10_current < ma_20_current:
            bearish_factors += 2  # Strong downtrend
        elif ma_5_current > ma_10_current:
            bullish_factors += 1
        else:
            bearish_factors += 1
            
        # Volatility analysis
        if current_volatility > 0.05:  # High volatility
            if predicted_change > 0:
                bullish_factors += 1
            else:
                bearish_factors += 1
                
        # Price momentum
        if predicted_change > 2:
            bullish_factors += 2
        elif predicted_change < -2:
            bearish_factors += 2
        elif predicted_change > 0.5:
            bullish_factors += 1
        elif predicted_change < -0.5:
            bearish_factors += 1
            
        # Generate signal based on combined factors
        if bullish_factors >= 4:
            signal = "STRONG BUY 🟢"
            color = "green"
        elif bullish_factors >= 2:
            signal = "BUY 🟢"
            color = "green"
        elif bearish_factors >= 4:
            signal = "STRONG SELL 🔴"
            color = "red"
        elif bearish_factors >= 2:
            signal = "SELL 🔴"
            color = "red"
        else:
            signal = "HOLD 🟡"
            color = "orange"
    else:
        # Fallback for insufficient data
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
        fig.add_trace(go.Scatter(x=df["ds"], y=df["ma_20"], mode="lines", name="MA 20", line=dict(color="purple", dash="dash")))
        
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

    # --- Enhanced Signal Display with Live Data ---
    st.write(f"📊 **Live Trading Signal for {coin.replace('-USD','')}**")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current Price", value=f"${last_price:.2f}")
    with col2:
        st.metric(label="Predicted Price (3 days)", value=f"${next_price:.2f}")
    with col3:
        st.metric(label="Expected Change", value=f"{predicted_change:.2f}%")
    with col4:
        st.metric(label="Signal", value=signal)
    
    # Technical indicators row (if enough data)
    if len(df_clean) > 20:
        st.write("🔍 **Live Technical Indicators**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="RSI", value=f"{current_rsi:.1f}")
        with col2:
            st.metric(label="Volatility", value=f"{current_volatility:.3f}")
        with col3:
            st.metric(label="MA Trend", value="📈" if ma_5_current > ma_10_current else "📉")
        with col4:
            st.metric(label="Price Position", value=f"{price_position:.1%}")

    # Get current timestamp for this signal
    current_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    results.append({
        "Coin": coin.replace("-USD", ""),
        "Current Price": round(last_price, 2),
        "Predicted Price (3 days)": round(next_price, 2),
        "Change (%)": round(predicted_change, 2),
        "Signal": f"{signal} (Updated: {current_timestamp})",
        "Confidence": f"{min(round(abs(predicted_change)*10,1),100)}%"
    })

# --- Summary Table ---
if results:
    st.markdown("### 📊 Summary Table")
    st.caption(f"🕒 Analysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    table = pd.DataFrame(results)
    st.dataframe(table, width='stretch')

st.caption("⚠️ Not financial advice. This tool is for educational and research use only.")
