# 💹 AI-Powered Crypto Buy/Sell Dashboard

A comprehensive cryptocurrency analysis dashboard powered by machine learning and real-time data from Yahoo Finance.

## 🚀 Features

- **Top 100+ Cryptocurrencies** - Analyze BTC, ETH, SOL, ADA, AVAX, DOGE, and many more
- **AI-Powered Predictions** - Machine learning-based price forecasting using Linear Regression
- **Real-Time Data** - Live data from Yahoo Finance with auto-refresh capabilities
- **Technical Analysis** - Moving averages, volatility analysis, and trend indicators
- **Buy/Sell Signals** - Automated trading recommendations based on ML predictions
- **Interactive Charts** - Beautiful Plotly visualizations with multiple indicators
- **Historical Simulation** - Backtesting with ROI calculations and accuracy metrics
- **Auto-Refresh** - Configurable refresh intervals (1 min, 5 min, 10 min)

## 📊 Dashboard Components

1. **Coin Selection** - Choose from 100+ cryptocurrencies
2. **Price Analysis** - Real-time price charts with technical indicators
3. **ML Predictions** - 3-day price forecasts using scikit-learn
4. **Trading Signals** - BUY/SELL/HOLD recommendations with confidence levels
5. **Performance Metrics** - Historical accuracy and simulated ROI
6. **Summary Table** - Comprehensive overview of all selected coins

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data**: Yahoo Finance API (yfinance)
- **ML**: scikit-learn (Linear Regression)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 📈 How It Works

1. **Data Collection** - Fetches real-time crypto data from Yahoo Finance
2. **Feature Engineering** - Creates technical indicators (MA, volatility, trends)
3. **ML Training** - Trains Linear Regression model on historical data
4. **Prediction** - Generates price forecasts for next 3 days
5. **Signal Generation** - Creates BUY/SELL/HOLD recommendations
6. **Visualization** - Displays results in interactive charts and tables

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not financial advice**. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always do your own research and consult with financial advisors before making investment decisions.

## 🔧 Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📱 Deployment

This app is designed to be deployed on Streamlit Community Cloud. Simply connect your GitHub repository to Streamlit Cloud for automatic deployment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.
