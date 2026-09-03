# position_monitor.py
#
# Checks currently held positions every 30 minutes during market hours
# and sells any position that has hit take-profit or stop-loss. Does NOT
# generate new BUY signals — entries remain once-daily via Paper_Trader.py,
# since the model's features are computed from daily-resolution data.
#
# Deploy as a separate, always-on Railway worker (not the same service
# as the 9:30 AM scheduled Paper_Trader.py job).

import alpaca_trade_api as tradeapi
import time
import pytz
from datetime import datetime, time as dtime
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
if not os.getenv("ALPACA_API_KEY"):
    load_dotenv(r"C:\Users\ryanc\OneDrive\Desktop\algo-trading-project\.env")

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Risk parameters — must match Paper_Trader.py ────────────────────────────
TAKE_PROFIT_PCT = 0.10
STOP_LOSS_PCT   = -0.07

# Check every 30 minutes, fixed
CHECK_INTERVAL_SECONDS = 30 * 60

EST = pytz.timezone("US/Eastern")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# ── Market hours check ──────────────────────────────────────────────────────
def is_market_open_now():
    now_est = datetime.now(EST)
    if now_est.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    current_time = now_est.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE

# ── Execute sell ─────────────────────────────────────────────────────────────
def execute_sell(ticker, reason):
    try:
        position      = api.get_position(ticker)
        current_qty   = int(float(position.qty))
        unrealized_pl = float(position.unrealized_pl)
        api.submit_order(
            symbol        = ticker,
            qty           = current_qty,
            side          = "sell",
            type          = "market",
            time_in_force = "day"
        )
        print(f"  ✓ SELL {current_qty} shares of {ticker} | Reason: {reason} | P&L: ${unrealized_pl:+.2f}")
    except Exception as e:
        print(f"  ✗ Sell failed for {ticker}: {e}")

# ── Single monitoring pass ───────────────────────────────────────────────────
def check_positions():
    timestamp = datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S")
    try:
        positions = api.list_positions()
    except Exception as e:
        print(f"  ✗ [{timestamp} EST] Could not fetch positions: {e}")
        return

    if not positions:
        print(f"  [{timestamp} EST] No open positions to monitor")
        return

    print(f"\n── Position Check [{timestamp} EST] ──────────")
    for p in positions:
        ticker          = p.symbol
        unrealized_plpc = float(p.unrealized_plpc)
        unrealized_pl   = float(p.unrealized_pl)

        if unrealized_plpc >= TAKE_PROFIT_PCT:
            print(f"  💰 Take profit — {ticker} up {unrealized_plpc:.1%} (${unrealized_pl:+.2f})")
            execute_sell(ticker, reason=f"Take profit at {unrealized_plpc:.1%}")

        elif unrealized_plpc <= STOP_LOSS_PCT:
            print(f"  🛑 Stop loss — {ticker} down {unrealized_plpc:.1%} (${unrealized_pl:+.2f})")
            execute_sell(ticker, reason=f"Stop loss at {unrealized_plpc:.1%}")

        else:
            print(f"  — {ticker} at {unrealized_plpc:+.1%}, within range")

# ── Main loop ─────────────────────────────────────────────────────────────
print("Intraday position monitor starting...")
print(f"Checking every 30 minutes during market hours")
print(f"Take profit: {TAKE_PROFIT_PCT:.0%} | Stop loss: {STOP_LOSS_PCT:.0%}")

while True:
    if is_market_open_now():
        check_positions()
    else:
        now_est = datetime.now(EST)
        print(f"  [{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST] Market closed — sleeping")

    time.sleep(CHECK_INTERVAL_SECONDS)