import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
if not os.getenv("ALPACA_API_KEY"):
    load_dotenv(r"C:\Users\ryanc\OneDrive\Desktop\algo-trading-project\.env")

# ── Credentials ────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

print("API_KEY found:", API_KEY is not None)
print("SECRET_KEY found:", SECRET_KEY is not None)

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Verify connection ──────────────────────────────────────────────────────
account         = api.get_account()
portfolio_value = float(account.portfolio_value)
print(f"Account status : {account.status}")
print(f"Portfolio value: ${portfolio_value:,.2f}")
print(f"Buying power   : ${float(account.buying_power):,.2f}")
print(f"Cash           : ${float(account.cash):,.2f}")

# ── Feature engineering ────────────────────────────────────────────────────
def get_features(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty:
                raise ValueError("Empty dataframe")
            df.columns = df.columns.get_level_values(0)
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
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1}/3 failed for {ticker}: {e}")
            time.sleep(5)
    return pd.DataFrame()

# ── Load model ─────────────────────────────────────────────────────────────
features = ["rsi","macd","macd_sig","bb_high","bb_low",
            "vol_ma","returns_1d","returns_5d","returns_20d",
            "volatility","ma_20","ma_50","ma_cross"]

model = joblib.load("models/aapl_xgb_model.joblib")
print("\nModel loaded successfully")

# ── Get signal ─────────────────────────────────────────────────────────────
def get_signal(ticker):
    try:
        df = get_features(ticker)
        if df.empty or len(df) < 50:
            print(f"  ⚠ Not enough data for {ticker}, skipping")
            return None
        latest = df[features].iloc[-1:]
        if latest.empty:
            print(f"  ⚠ No features for {ticker}, skipping")
            return None
        pred       = model.predict(latest)[0]
        prob       = model.predict_proba(latest)[0]
        price      = df["Close"].iloc[-1]
        confidence = prob[1] if pred == 1 else prob[0]
        return {
            "ticker"    : ticker,
            "signal"    : "BUY" if pred == 1 else "SELL",
            "confidence": confidence,
            "price"     : price,
            "time"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"  ✗ Error getting signal for {ticker}: {e}")
        return None

# ── Position sizing ────────────────────────────────────────────────────────
def get_position_size(confidence, price, portfolio_value):
    if confidence >= 0.80:
        dollars = portfolio_value * 0.05   # 5% of portfolio
    elif confidence >= 0.70:
        dollars = portfolio_value * 0.03   # 3% of portfolio
    elif confidence >= 0.65:
        dollars = portfolio_value * 0.02   # 2% of portfolio
    else:
        dollars = portfolio_value * 0.01   # 1% of portfolio
    qty = max(1, int(dollars / price))
    return qty, dollars

# ── Portfolio exposure ─────────────────────────────────────────────────────
def get_portfolio_exposure():
    positions      = api.list_positions()
    total_invested = sum(float(p.market_value) for p in positions)
    return total_invested

# ── Place trade ────────────────────────────────────────────────────────────
def place_trade(ticker, signal, confidence, portfolio_value):
    try:
        # Check total portfolio exposure
        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        max_exposure   = 0.40  # never invest more than 40% of portfolio

        try:
            position      = api.get_position(ticker)
            has_position  = True
            current_qty   = int(float(position.qty))
            unrealized_pl = float(position.unrealized_pl)
        except:
            has_position  = False
            current_qty   = 0
            unrealized_pl = 0

        if signal == "BUY" and confidence > 0.60:

            # Check exposure limit before buying
            if exposure_pct >= max_exposure:
                print(f"  — Portfolio at {exposure_pct:.1%} exposure (max {max_exposure:.0%}) — skipping {ticker}")
                return None

            # Get position size based on conviction
            qty, dollars = get_position_size(confidence, 
                           get_signal(ticker)["price"] if not has_position else float(api.get_position(ticker).current_price),
                           portfolio_value)

            if not has_position:
                order = api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = "buy",
                    type          = "market",
                    time_in_force = "day"
                )
                print(f"  ✓ BUY {qty} share(s) of {ticker} (~${dollars:,.0f}) — portfolio {exposure_pct:.1%} deployed")
                return order

            elif unrealized_pl > 0:
                order = api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = "buy",
                    type          = "market",
                    time_in_force = "day"
                )
                print(f"  ✓ Adding {qty} share(s) to {ticker} (~${dollars:,.0f}) — portfolio {exposure_pct:.1%} deployed")
                return order

            else:
                print(f"  — Position exists but not profitable yet (P&L: ${unrealized_pl:.2f})")
                return None

        elif signal == "SELL" and has_position:
            order = api.submit_order(
                symbol        = ticker,
                qty           = current_qty,
                side          = "sell",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ SELL closing full position ({current_qty} shares) of {ticker}")
            return order

        else:
            print(f"  — No action: signal={signal}, has_position={has_position}, confidence={confidence:.1%}")
            return None

    except Exception as e:
        print(f"  ✗ Order failed: {e}")
        return None

# ── Portfolio summary ──────────────────────────────────────────────────────
def print_portfolio():
    print("\n── Current Positions ─────────────────────")
    positions = api.list_positions()
    if not positions:
        print("  No open positions")
    total_pnl      = 0
    total_invested = 0
    for p in positions:
        pnl            = float(p.unrealized_pl)
        qty            = int(float(p.qty))
        value          = float(p.market_value)
        total_pnl     += pnl
        total_invested += value
        print(f"  {p.symbol:<6} {qty} shares | Value: ${value:,.2f} | P&L: ${pnl:+.2f}")
    exposure = (total_invested / portfolio_value) * 100
    print(f"\n  Total invested    : ${total_invested:,.2f} ({exposure:.1f}% of portfolio)")
    print(f"  Total P&L         : ${total_pnl:+.2f}")
    print(f"  Cash remaining    : ${float(account.cash):,.2f}")
    print(f"  Max exposure limit: ${portfolio_value * 0.40:,.2f} (40%)")
    print("\n── Recent Orders ──────────────────────────")
    orders = api.list_orders(status="all", limit=10)
    for o in orders:
        print(f"  {o.symbol} {o.side.upper()} {o.qty} shares — {o.status} @ {o.created_at}")

# ── Main ───────────────────────────────────────────────────────────────────
WATCHLIST = [
    # Original 10
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "V",    "JPM",   "ORCL", "COST",
    # New 10
    "ADBE", "CRM",  "AMD",   "NFLX", "PYPL",
    "MA",   "UNH",  "HD",    "BAC",  "QCOM",
]

print("\n── Running Signal Check ───────────────────")
for ticker in WATCHLIST:
    print(f"\nAnalyzing {ticker}...")
    signal_data = get_signal(ticker)
    if signal_data is None:
        print(f"  — Skipping {ticker}")
        continue
    print(f"  Signal    : {signal_data['signal']}")
    print(f"  Confidence: {signal_data['confidence']:.1%}")
    print(f"  Price     : ${signal_data['price']:.2f}")
    place_trade(ticker, signal_data["signal"], signal_data["confidence"], portfolio_value)

print_portfolio()
print("\nDone. Run this script daily to execute your strategy.")