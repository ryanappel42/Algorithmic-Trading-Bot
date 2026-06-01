# ML Algorithmic Trading Bot 🤖📈

> An automated trading system that uses machine learning to analyze 20 large-cap US stocks every morning, identify high-probability trade opportunities, and execute paper trades automatically — all without any human intervention.

## 🖥️ Live Dashboard
**[View the live trading dashboard here →](https://algo-trading-bot-nqb2tcb5uqdq3fdasvznda.streamlit.app/)**

*Shows real-time portfolio performance, individual stock charts with ML signals, and complete trade history.*

---

## What This Project Does

Every weekday at 9:30 AM EST, this bot wakes up on a cloud server and:

1. Scans all 20 watchlist stocks and collects every signal
2. Executes all SELL signals first — freeing up capital before any buying
3. Ranks all BUY signals by model confidence — highest conviction first
4. Buys down the ranked list until the 40% portfolio cap is hit
5. Updates this README automatically with the latest results

No human involvement required. It runs, trades, and reports entirely on its own.

---

## Live Performance — Auto-updated June 01, 2026 03:13 PM EST

| Metric | Value |
|--------|-------|
| Portfolio Value | $101,105.89 |
| Total P&L | $+1,105.89 (+1.11%) |
| Daily P&L | 📈 $+0.00 (+0.00%) |
| Cash Available | $57,874.09 |
| Open Positions | 11 |
| Total Trades Executed | 50 |

📈 Bot has been live since March 27, 2026

### Current Open Positions
| Ticker | Shares | Value | Unrealized P&L |
|--------|--------|-------|----------------|
| AMZN | 3 | $791.16 | $-22.32 |
| BAC | 232 | $11,958.44 | $+75.37 |
| GOOGL | 2 | $755.88 | $-9.82 |
| HD | 44 | $13,588.96 | $+179.10 |
| JPM | 10 | $2,962.50 | $-40.72 |
| MA | 2 | $992.18 | $-2.48 |
| META | 1 | $606.67 | $-17.35 |
| NFLX | 43 | $3,696.93 | $-129.23 |
| PYPL | 136 | $6,156.99 | $+178.46 |
| UNH | 2 | $758.04 | $+0.78 |
| V | 3 | $964.05 | $-24.03 |


---

## Backtest Results

| Metric | Value |
|--------|-------|
| Buy Signal Accuracy | 67.1% |
| Strategy Max Drawdown | -10.9% |
| Buy & Hold Max Drawdown | -16.6% |
| Training Data | 5 years (2019-2024) |
| Training Rows | 14,600 across 10 stocks |
| Test Period | 1 year (2024) |
| Paper Profit (live) | $60+ since March 2026 |

---

## Watchlist

The bot monitors these 20 large-cap stocks daily — selected for their stable price action, high liquidity, and similarity to the model's training data:

| | | | | |
|-|-|-|-|-|
| AAPL | MSFT | GOOGL | AMZN | META |
| NVDA | V | JPM | ORCL | COST |
| ADBE | CRM | AMD | NFLX | PYPL |
| MA | UNH | HD | BAC | QCOM |

These stocks were chosen because they share characteristics with the model's training data — large market cap, high daily trading volume, and relatively stable price behavior compared to smaller or more speculative stocks.

---

## How The Machine Learning Works

### Plain English
Every stock gets evaluated using 13 technical indicators — mathematical calculations based on a stock's price and volume history. These indicators answer questions like: Is this stock overbought right now? Is momentum accelerating or slowing down? Is trading volume unusually high, suggesting something significant is happening?

The machine learning model was trained on 5 years of historical data across 10 large-cap stocks. It learned which combinations of these indicators have historically predicted price increases over the next 5 trading days. When it sees a similar pattern today, it generates a BUY or SELL signal along with a confidence score between 0% and 100%.

### The Technical Details

The model is a **gradient boosted decision tree classifier (XGBoost)** trained on 14,600 labeled examples. Each example consists of 13 features computed from daily OHLCV (Open, High, Low, Close, Volume) price data:

```python
# Computing technical indicators from raw price data
df["rsi"]         = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["macd"]        = ta.trend.MACD(df["Close"]).macd()
df["macd_sig"]    = ta.trend.MACD(df["Close"]).macd_signal()
df["bb_high"]     = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
df["bb_low"]      = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
df["vol_ma"]      = df["Volume"].rolling(20).mean()
df["returns_1d"]  = df["Close"].pct_change(1)
df["returns_5d"]  = df["Close"].pct_change(5)
df["returns_20d"] = df["Close"].pct_change(20)
df["volatility"]  = df["returns_1d"].rolling(20).std()
df["ma_20"]       = df["Close"].rolling(20).mean()
df["ma_50"]       = df["Close"].rolling(50).mean()
df["ma_cross"]    = (df["ma_20"] > df["ma_50"]).astype(int)

# Target: 1 if price is higher 5 trading days from now, else 0
df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
```

Training used a strict **chronological train/test split** to prevent data leakage — a common mistake in financial ML that produces artificially inflated backtest results. The data was never shuffled randomly.

```python
# Time series split — always train on past, test on future
split   = int(len(df) * 0.8)
X_train = df[features].iloc[:split]   # 2019–2023
X_test  = df[features].iloc[split:]   # 2024 only
```

---

## Risk Management

### Plain English
The bot never goes all in. It sizes each trade based on how confident the model is and never deploys more than 40% of the total portfolio at once. Think of it like a poker player who bets bigger when they have a stronger hand — but always keeps chips in reserve.

### The Rules

| Confidence Level | Position Size | Dollar Amount |
|-----------------|---------------|---------------|
| 60% – 65% | 1% of portfolio | ~$1,000 |
| 65% – 70% | 2% of portfolio | ~$2,000 |
| 70% – 80% | 3% of portfolio | ~$3,000 |
| 80%+ | 5% of portfolio | ~$5,000 |

**Additional protections:**
- **60% minimum confidence** — weak signals are ignored entirely
- **40% max portfolio exposure** — always keeps 60% as a cash buffer
- **Dollar cost averaging** — adds to existing positions at 65%+ confidence even if temporarily at a loss
- **Full position exit** — closes the entire position at once when a SELL signal fires
- **Auto-scaling** — as the portfolio grows, position sizes grow proportionally

```python
# Conviction-weighted position sizing
def get_position_size(confidence, price, portfolio_value):
    if confidence >= 0.80:
        dollars = portfolio_value * 0.05
    elif confidence >= 0.70:
        dollars = portfolio_value * 0.03
    elif confidence >= 0.65:
        dollars = portfolio_value * 0.02
    else:
        dollars = portfolio_value * 0.01
    return max(1, int(dollars / price))
```

---

## How a Trade Decision Is Made

Every morning the bot runs in three distinct passes:

### Pass 1 — Collect All Signals
The bot scans all 20 stocks simultaneously and collects every BUY and SELL signal before executing a single trade. This ensures no capital is committed until the full picture is known.

### Pass 2 — Execute All Sells First
All SELL signals are executed immediately, freeing up capital before any buying begins. This ensures the portfolio always has maximum available capital for the best opportunities of the day.

### Pass 3 — Execute Buys in Confidence Order
BUY signals are ranked highest to lowest confidence and executed in that order. The highest conviction trade always gets filled first. If the 40% cap is hit mid-list, remaining signals are skipped and logged.

```
Example morning ranked buy list:
  #1 NVDA   Confidence: 84.2% → Bought first
  #2 AAPL   Confidence: 79.1% → Bought second
  #3 MA     Confidence: 71.3% → Bought third
  #4 JPM    Confidence: 63.2% → Cap hit, skipped
  #5 BAC    Confidence: 61.8% → Cap hit, skipped
```

**The full decision chain for each buy:**
```
Confidence above 60%?
        → No  : Skip — signal too weak
        → Yes : Portfolio below 40% exposure?
                        → No  : Stop — cap reached
                        → Yes : Already own this stock?
                                        → No  : Buy immediately
                                        → Yes : Is position profitable?
                                                        → Yes : Add more shares
                                                        → No  : Confidence above 65%?
                                                                        → Yes : Dollar cost average
                                                                        → No  : Hold and wait

SELL signal + own the stock → Sell entire position immediately (Pass 2)
```

---

## Infrastructure

### How It All Connects

```
9:30 AM EST — Railway cloud server wakes up
        ↓
yfinance → downloads 1 year of price data for all 20 stocks
        ↓
Feature engineering → computes 13 technical indicators per stock
        ↓
XGBoost model → generates BUY/SELL signal + confidence score for all 20 stocks
        ↓
Pass 1 — collect all signals
        ↓
Pass 2 — execute all SELL signals first
        ↓
Pass 3 — rank BUY signals by confidence, execute highest first until 40% cap
        ↓
9:45 AM EST — GitHub Actions updates this README with latest stats
```

### Tech Stack
- **Python 3.14** — core language
- **XGBoost** — gradient boosted ML classifier
- **pandas + numpy** — data processing and feature engineering
- **ta** — technical indicator calculations
- **yfinance** — free real-time market data
- **Alpaca Trade API** — automated paper trade execution
- **Railway** — cloud deployment and daily scheduling
- **GitHub Actions** — automated README performance updates
- **Streamlit** — interactive live dashboard

---

## Project Structure

```
algo-trading-bot/
├── Paper_Trader.py              # main trading bot
├── dashboard.py                 # live Streamlit dashboard
├── app.py                       # original analysis dashboard
├── multi_stock_training.ipynb   # model training notebook
├── update_readme.py             # auto README updater
├── models/
│   └── aapl_xgb_model.joblib   # trained XGBoost model
├── .github/
│   └── workflows/
│       └── update_readme.yml   # GitHub Actions workflow
├── requirements.txt
├── Procfile
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
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your Alpaca paper trading keys
# ALPACA_API_KEY=your_key
# ALPACA_SECRET_KEY=your_secret

# Run the trading bot
python Paper_Trader.py

# Or launch the live dashboard
streamlit run dashboard.py
```

---

## Model Retraining

The model is retrained every 6 months on fresh data to prevent model drift — the gradual decline in accuracy as market conditions evolve beyond what the model was trained on. To retrain, open `multi_stock_training.ipynb` and update the end date to include recent data, then push the new model file to GitHub.

**Next scheduled retraining: October 2026**

---

## Disclaimer

This project is for educational purposes only and is not financial advice. All trading uses paper money — no real funds are at risk at any point.

---

## Author

**Ryan Appel** | Fintech & Big Data Analytics | Virginia Tech Pamplin College of Business