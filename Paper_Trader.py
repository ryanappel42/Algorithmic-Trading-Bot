import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib
import time
import warnings
import contextlib
import io
from datetime import datetime, timedelta
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

# ── Risk parameters ─────────────────────────────────────────────────────────
TAKE_PROFIT_PCT       = 0.10   # exit at +10%
STOP_LOSS_PCT         = -0.07  # exit at -7%
HIGH_VOL_THRESHOLD    = 0.025  # 2.5% daily volatility = "high volatility" stock
HIGH_VOL_MIN_CONF     = 0.65   # high-vol stocks need 65%+ confidence instead of 60%
HIGH_VOL_SIZE_FACTOR  = 0.6    # high-vol stocks get 60% of normal position size
EARNINGS_BLACKOUT_DAYS = 3     # no new buys within 3 trading days of earnings

# ── Feature engineering ────────────────────────────────────────────────────
def get_features(ticker):
    for attempt in range(3):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = yf.download(
                        ticker,
                        period="1y",
                        progress=False,
                        auto_adjust=True
                    )
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
            return None
    return None

# ── Earnings date check ─────────────────────────────────────────────────────
def is_near_earnings(ticker, blackout_days=EARNINGS_BLACKOUT_DAYS):
    """
    Returns True if the ticker has an earnings date within the next
    `blackout_days` trading days. Fails safe — if the earnings date can't
    be determined, assumes NOT near earnings so the bot doesn't stall out.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tk = yf.Ticker(ticker)
                edates = tk.get_earnings_dates(limit=4)
        if edates is None or edates.empty:
            return False, None

        today = pd.Timestamp.now(tz=edates.index.tz) if edates.index.tz else pd.Timestamp.now()
        upcoming = edates[edates.index >= today]
        if upcoming.empty:
            return False, None

        next_earnings = upcoming.index.min()
        days_until     = (next_earnings.tz_localize(None) - pd.Timestamp.now()).days

        if 0 <= days_until <= blackout_days:
            return True, next_earnings.strftime("%Y-%m-%d")
        return False, next_earnings.strftime("%Y-%m-%d")
    except Exception:
        # Fail safe: unknown earnings date should not block trading
        return False, None

# ── Load model ─────────────────────────────────────────────────────────────
features = ["rsi","macd","macd_sig","bb_high","bb_low",
            "vol_ma","returns_1d","returns_5d","returns_20d",
            "volatility","ma_20","ma_50","ma_cross"]

model = xgb.XGBClassifier()
model.load_model("models/aapl_xgb_model.ubj")
print("\nModel loaded successfully")

# ── Get signal ─────────────────────────────────────────────────────────────
def get_signal(ticker):
    try:
        df = get_features(ticker)
        if df is None or df.empty or len(df) < 50:
            return None, ["  ⚠ Not enough data, skipping"]
        latest      = df[features].iloc[-1:]
        pred        = model.predict(latest)[0]
        prob        = model.predict_proba(latest)[0]
        price       = df["Close"].iloc[-1]
        confidence  = prob[1] if pred == 1 else prob[0]
        volatility  = df["volatility"].iloc[-1]
        return {
            "ticker"     : ticker,
            "signal"     : "BUY" if pred == 1 else "SELL",
            "confidence" : confidence,
            "price"      : price,
            "volatility" : volatility,
            "time"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, None
    except Exception as e:
        return None, [f"  ✗ Error: {e}"]

# ── Position sizing — now volatility-aware ─────────────────────────────────
def get_position_size(confidence, price, portfolio_value, volatility):
    if confidence >= 0.80:
        dollars = portfolio_value * 0.05
    elif confidence >= 0.70:
        dollars = portfolio_value * 0.03
    elif confidence >= 0.65:
        dollars = portfolio_value * 0.02
    else:
        dollars = portfolio_value * 0.01

    # Volatility haircut — high-volatility stocks get a smaller dollar bet
    # per unit of confidence, since tail risk per dollar is higher
    is_high_vol = volatility >= HIGH_VOL_THRESHOLD
    if is_high_vol:
        dollars *= HIGH_VOL_SIZE_FACTOR

    qty = max(1, int(dollars / price))
    return qty, dollars, is_high_vol

# ── Portfolio exposure ─────────────────────────────────────────────────────
def get_portfolio_exposure():
    positions      = api.list_positions()
    total_invested = sum(float(p.market_value) for p in positions)
    return total_invested

# ── Execute sell ───────────────────────────────────────────────────────────
def execute_sell(ticker, reason="SELL signal"):
    try:
        position      = api.get_position(ticker)
        current_qty   = int(float(position.qty))
        unrealized_pl = float(position.unrealized_pl)
        order = api.submit_order(
            symbol        = ticker,
            qty           = current_qty,
            side          = "sell",
            type          = "market",
            time_in_force = "day"
        )
        print(f"  ✓ SELL {current_qty} shares of {ticker} | Reason: {reason} | P&L: ${unrealized_pl:+.2f}")
        return order
    except Exception as e:
        print(f"  ✗ Sell failed for {ticker}: {e}")
        return None

# ── Take profit / stop loss checks ──────────────────────────────────────────
def check_take_profit(ticker):
    try:
        position        = api.get_position(ticker)
        unrealized_plpc = float(position.unrealized_plpc)
        unrealized_pl   = float(position.unrealized_pl)
        if unrealized_plpc >= TAKE_PROFIT_PCT:
            print(f"  💰 Take profit triggered for {ticker} — up {unrealized_plpc:.1%} (${unrealized_pl:+.2f})")
            execute_sell(ticker, reason=f"Take profit at {unrealized_plpc:.1%}")
            return True
        return False
    except:
        return False

def check_stop_loss(ticker):
    try:
        position        = api.get_position(ticker)
        unrealized_plpc = float(position.unrealized_plpc)
        unrealized_pl   = float(position.unrealized_pl)
        if unrealized_plpc <= STOP_LOSS_PCT:
            print(f"  🛑 Stop loss triggered for {ticker} — down {unrealized_plpc:.1%} (${unrealized_pl:+.2f})")
            execute_sell(ticker, reason=f"Stop loss at {unrealized_plpc:.1%}")
            return True
        return False
    except:
        return False

# ── Execute buy ────────────────────────────────────────────────────────────
def execute_buy(ticker, confidence, price, portfolio_value, volatility):
    try:
        # ── Volatility-adjusted confidence gate ────────────────────────────
        is_high_vol  = volatility >= HIGH_VOL_THRESHOLD
        min_conf     = HIGH_VOL_MIN_CONF if is_high_vol else 0.60

        if confidence < min_conf:
            tag = "high-vol " if is_high_vol else ""
            print(f"  — Confidence {confidence:.1%} below {tag}threshold {min_conf:.0%} — skipping {ticker}")
            return None

        # ── Earnings blackout check ─────────────────────────────────────────
        near_earnings, edate = is_near_earnings(ticker)
        if near_earnings:
            print(f"  📅 Earnings blackout — {ticker} reports around {edate}, skipping new buy")
            return None

        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        max_exposure   = 0.40

        if exposure_pct >= max_exposure:
            print(f"  — Portfolio at {exposure_pct:.1%} exposure (max 40%) — skipping {ticker}")
            return None

        try:
            position        = api.get_position(ticker)
            has_position    = True
            current_qty     = int(float(position.qty))
            unrealized_pl   = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc)
            current_price   = float(position.current_price)
        except:
            has_position    = False
            current_qty     = 0
            unrealized_pl   = 0
            unrealized_plpc = 0
            current_price   = price

        # Take profit check before adding
        if has_position and unrealized_plpc >= TAKE_PROFIT_PCT:
            print(f"  💰 Take profit triggered for {ticker} — up {unrealized_plpc:.1%} — selling instead of adding")
            execute_sell(ticker, reason=f"Take profit at {unrealized_plpc:.1%}")
            return None

        qty, dollars, is_high_vol = get_position_size(confidence, current_price, portfolio_value, volatility)
        vol_tag = " ⚡high-vol" if is_high_vol else ""

        if not has_position:
            order = api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "buy",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ BUY {qty} share(s) of {ticker} (~${dollars:,.0f}){vol_tag} | Confidence: {confidence:.1%} | Exposure: {exposure_pct:.1%}")
            return order

        elif unrealized_pl > 0 or confidence >= 0.65:
            order = api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "buy",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ Adding {qty} share(s) to {ticker} (~${dollars:,.0f}){vol_tag} | Confidence: {confidence:.1%} | Exposure: {exposure_pct:.1%}")
            return order

        else:
            print(f"  — Position at loss, confidence below 65% — holding {ticker}")
            return None

    except Exception as e:
        print(f"  ✗ Buy failed for {ticker}: {e}")
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
        plpc           = float(p.unrealized_plpc) * 100
        total_pnl     += pnl
        total_invested += value
        flag = ""
        if plpc >= 8:
            flag = " 💰 NEAR TAKE PROFIT"
        elif plpc <= -5:
            flag = " 🛑 NEAR STOP LOSS"
        print(f"  {p.symbol:<6} {qty} shares | Value: ${value:,.2f} | P&L: ${pnl:+.2f} ({plpc:+.1f}%){flag}")
    exposure = (total_invested / portfolio_value) * 100
    print(f"\n  Total invested    : ${total_invested:,.2f} ({exposure:.1f}% of portfolio)")
    print(f"  Total P&L         : ${total_pnl:+.2f}")
    print(f"  Cash remaining    : ${float(account.cash):,.2f}")
    print(f"  Max exposure limit: ${portfolio_value * 0.40:,.2f} (40%)")
    print(f"  Take profit level : {TAKE_PROFIT_PCT:.0%} per position")
    print(f"  Stop loss level   : {STOP_LOSS_PCT:.0%} per position")
    print("\n── Recent Orders ──────────────────────────")
    orders = api.list_orders(status="all", limit=10)
    for o in orders:
        print(f"  {o.symbol} {o.side.upper()} {o.qty} shares — {o.status} @ {o.created_at}")

# ── Watchlist ──────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "V",    "JPM",   "ORCL", "COST",
    "ADBE", "CRM",  "AMD",   "NFLX", "PYPL",
    "MA",   "UNH",  "HD",    "BAC",  "QCOM",
]

# ── STEP 1 — Collect all signals ───────────────────────────────────────────
print("\n── Collecting Signals for All 20 Stocks ───")
sell_signals = []
buy_signals  = []

for ticker in WATCHLIST:
    lines       = [f"\nAnalyzing {ticker}..."]
    result, err = get_signal(ticker)

    if result is None:
        lines += err if err else [f"  — Skipping {ticker}"]
    else:
        vol_tag = " ⚡high-vol" if result["volatility"] >= HIGH_VOL_THRESHOLD else ""
        lines.append(f"  Signal    : {result['signal']}{vol_tag}")
        lines.append(f"  Confidence: {result['confidence']:.1%}")
        lines.append(f"  Price     : ${result['price']:.2f}")
        lines.append(f"  Volatility: {result['volatility']:.3f}")

        if result["signal"] == "SELL":
            sell_signals.append(result)
        else:
            # Volatility-adjusted confidence gate applied at collection too
            min_conf = HIGH_VOL_MIN_CONF if result["volatility"] >= HIGH_VOL_THRESHOLD else 0.60
            if result["confidence"] >= min_conf:
                buy_signals.append(result)
            else:
                lines.append(f"  — Below {min_conf:.0%} confidence threshold — not queued for buying")

    print("\n".join(lines))

# ── STEP 2 — Take profit / stop loss check on all held positions ──────────
print("\n── Take Profit / Stop Loss Check ──────────")
positions    = api.list_positions()
held_tickers = [p.symbol for p in positions]
exited       = []

for ticker in held_tickers:
    if check_take_profit(ticker):
        exited.append(ticker)
        continue
    if check_stop_loss(ticker):
        exited.append(ticker)

if not exited:
    print("  No positions at take profit or stop loss threshold")

buy_signals = [b for b in buy_signals if b["ticker"] not in exited]

# ── STEP 3 — Execute all sells ─────────────────────────────────────────────
print(f"\n── Pass 1: Executing {len(sell_signals)} Sell Signal(s) ───")
if not sell_signals:
    print("  No sell signals today")
else:
    for s in sell_signals:
        ticker = s["ticker"]
        if ticker in exited:
            print(f"\n  — {ticker} already exited via take profit/stop loss — skipping")
            continue
        print(f"\n  Processing SELL for {ticker}...")
        try:
            api.get_position(ticker)
            execute_sell(ticker)
        except:
            print(f"  — No position in {ticker}, nothing to sell")

if sell_signals or exited:
    print("\n  Waiting 10 seconds for sell orders to settle...")
    time.sleep(10)

# ── STEP 4 — Rank buys by confidence, execute highest first ───────────────
buy_signals_sorted = sorted(buy_signals, key=lambda x: x["confidence"], reverse=True)

print(f"\n── Pass 2: Executing {len(buy_signals_sorted)} Buy Signal(s) — Ranked by Confidence ───")
if not buy_signals_sorted:
    print("  No qualifying buy signals today")
else:
    print("\n  Buy signal rankings:")
    for i, b in enumerate(buy_signals_sorted, 1):
        vol_tag = " ⚡" if b["volatility"] >= HIGH_VOL_THRESHOLD else ""
        print(f"  #{i} {b['ticker']:<6} Confidence: {b['confidence']:.1%}{vol_tag} | Price: ${b['price']:.2f}")

    print("\n  Executing buys...")
    for b in buy_signals_sorted:
        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        if exposure_pct >= 0.40:
            remaining = [x["ticker"] for x in buy_signals_sorted[buy_signals_sorted.index(b):]]
            print(f"\n  — Portfolio at {exposure_pct:.1%} exposure — cap reached, stopping buys")
            print(f"  — Skipped: {remaining}")
            break
        print(f"\n  Processing BUY for {b['ticker']}...")
        execute_buy(b["ticker"], b["confidence"], b["price"], portfolio_value, b["volatility"])

print_portfolio()
print("\nDone. Run this script daily to execute your strategy.")