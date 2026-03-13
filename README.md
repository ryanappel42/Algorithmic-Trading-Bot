# ML Algorithmic Trading Bot 🤖📈

An end-to-end algorithmic trading system that uses machine learning to generate 
buy/sell signals for US equities and executes automated paper trades via the 
Alpaca API.

## Live Demo
> Bot runs automatically every weekday at 9:30 AM EST via Railway cloud deployment

---

## Results
| Metric | Value |
|--------|-------|
| Buy Signal Accuracy | 67.1% |
| Strategy Max Drawdown | -10.9% |
| Buy & Hold Max Drawdown | -16.6% |
| Training Data | 5 years (2019-2024) |
| Test Period | 1 year (2024) |

---

## How It Works

1. **Data Pipeline** — Pulls daily OHLCV price data from Yahoo Finance
2. **Feature Engineering** — Computes 13 technical indicators as ML features
3. **ML Model** — XGBoost classifier predicts 5-day price direction
4. **Risk Management** — Only trades when model confidence exceeds 60%
5. **Execution** — Places paper trades automatically via Alpaca API
6. **Scheduling** — Runs every weekday at market open via Railway cron job

---

## Technical Indicators (Features)
- RSI (Relative Strength Index)
- MACD + Signal Line
- Bollinger Bands (Upper + Lower)
- 20-day and 50-day Moving Averages
- MA Crossover signal
- 20-day Volume Moving Average
- 1-day, 5-day, 20-day Returns
- 20-day Rolling Volatility

---

## Tech Stack
- **Python 3.14**
- **XGBoost** — gradient boosted classifier
- **pandas + numpy** — data processing
- **ta** — technical indicator library
- **yfinance** — market data
- **Alpaca Trade API** — paper trade execution
- **Streamlit** — interactive dashboard
- **Railway** — cloud deployment + scheduling
- **GitHub Actions** — CI/CD

---

## Project Structure
```
algo-trading-bot/
├── Paper_Trader.py       # main trading bot
├── app.py                # Streamlit dashboard
├── models/               # saved XGBoost model
├── requirements.txt      # dependencies
├── Procfile              # Railway deployment config
└── README.md
```

---

## Running Locally
```bash
# Clone the repo
git clone https://github.com/ryanappel42/algo-trading-bot.git
cd algo-trading-bot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Alpaca API keys
echo "ALPACA_API_KEY=your_key" > .env
echo "ALPACA_SECRET_KEY=your_secret" >> .env

# Run the trading bot
python Paper_Trader.py

# Or launch the dashboard
streamlit run app.py
```

---

## Dashboard
The Streamlit dashboard allows interactive analysis of any US equity ticker:
- Live ML buy/sell signals plotted on price chart
- Strategy vs buy & hold performance comparison
- Feature importance visualization
- Latest signal with model confidence score

---

## Disclaimer
This project is for educational purposes only. Not financial advice. 
All trading is done with paper money — no real funds are at risk.

---

## Author
Built by Ryan Appel | Fintech & Big Data Analytics