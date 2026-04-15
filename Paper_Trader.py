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
load_dotenv()  # works on Railway via environment variables
# fallback for local development
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
account = api.get_account()
print(f"Account status : {account.status}")
print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
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

# ── Place trade ────────────────────────────────────────────────────────────
def place_trade(ticker, signal, confidence, qty=1):
    try:
        try:
            api.get_position(ticker)
            has_position = True
        except:
            has_position = False

        if signal == "BUY" and not has_position and confidence > 0.60:
            order = api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "buy",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ BUY order placed: {qty} share(s) of {ticker}")
            return order
        elif signal == "SELL" and has_position:
            order = api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "sell",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ SELL order placed: {qty} share(s) of {ticker}")
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
    for p in positions:
        pnl = float(p.unrealized_pl)
        print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f} | P&L: ${pnl:+.2f}")
    print("\n── Recent Orders ──────────────────────────")
    orders = api.list_orders(status="all", limit=5)
    for o in orders:
        print(f"  {o.symbol} {o.side.upper()} {o.qty} shares — {o.status} @ {o.created_at}")

# ── Main ───────────────────────────────────────────────────────────────────
WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
             "NVDA", "V", "JPM", "ORCL", "COST"]

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
    place_trade(ticker, signal_data["signal"], signal_data["confidence"], qty=1)

print_portfolio()
print("\nDone. Run this script daily to execute your strategy.")