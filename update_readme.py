import alpaca_trade_api as tradeapi
import os
import json
import pytz
from datetime import datetime

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

api     = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
account = api.get_account()

# ── Pull live stats ────────────────────────────────────────────────────────
portfolio_value = float(account.portfolio_value)
starting_value  = 100000.00
total_pnl       = portfolio_value - starting_value
pnl_pct         = (total_pnl / starting_value) * 100
cash            = float(account.cash)

# ── Daily P&L tracking ─────────────────────────────────────────────────────
TRACKER_FILE = "portfolio_tracker.json"

def load_tracker():
    try:
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    except:
        return {"yesterday_value": portfolio_value, "date": ""}

def save_tracker(value, date):
    with open(TRACKER_FILE, "w") as f:
        json.dump({"yesterday_value": value, "date": date}, f)

est          = pytz.timezone("US/Eastern")
today        = datetime.now(est).strftime("%Y-%m-%d")
tracker      = load_tracker()

# Calculate daily P&L
if tracker["date"] != today:
    # New day — yesterday's value is what we saved last run
    yesterday_value = tracker["yesterday_value"]
    daily_pnl       = portfolio_value - yesterday_value
    daily_pnl_pct   = (daily_pnl / yesterday_value) * 100
    # Save today's value for tomorrow
    save_tracker(portfolio_value, today)
else:
    # Same day — use same yesterday value
    yesterday_value = tracker["yesterday_value"]
    daily_pnl       = portfolio_value - yesterday_value
    daily_pnl_pct   = (daily_pnl / yesterday_value) * 100

daily_emoji = "📈" if daily_pnl >= 0 else "📉"

# ── Get open positions ─────────────────────────────────────────────────────
positions      = api.list_positions()
open_positions = len(positions)
positions_text = ""
for p in positions:
    pnl   = float(p.unrealized_pl)
    qty   = int(float(p.qty))
    value = float(p.market_value)
    positions_text += f"| {p.symbol} | {qty} | ${value:,.2f} | ${pnl:+.2f} |\n"

# ── Get recent orders ──────────────────────────────────────────────────────
orders       = api.list_orders(status="all", limit=50)
total_trades = len(orders)

# ── Build dynamic section ──────────────────────────────────────────────────
now        = datetime.now(est).strftime("%B %d, %Y %I:%M %p EST")
pnl_emoji  = "📈" if total_pnl >= 0 else "📉"

status_block = f"""## Live Performance — Auto-updated {now}

| Metric | Value |
|--------|-------|
| Portfolio Value | ${portfolio_value:,.2f} |
| Total P&L | ${total_pnl:+,.2f} ({pnl_pct:+.2f}%) |
| Daily P&L | {daily_emoji} ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%) |
| Cash Available | ${cash:,.2f} |
| Open Positions | {open_positions} |
| Total Trades Executed | {total_trades} |

{pnl_emoji} Bot has been live since March 27, 2026

### Current Open Positions
| Ticker | Shares | Value | Unrealized P&L |
|--------|--------|-------|----------------|
{positions_text if positions_text else "| — | No open positions | — | — |"}
"""

# ── Update README ──────────────────────────────────────────────────────────
with open("README.md", "r") as f:
    content = f.read()

start_marker = "## Live Performance"
end_marker   = "## Results"

if start_marker in content:
    start_idx   = content.index(start_marker)
    end_idx     = content.index(end_marker)
    new_content = content[:start_idx] + status_block + "\n---\n\n" + content[end_idx:]
else:
    end_idx     = content.index(end_marker)
    new_content = content[:end_idx] + status_block + "\n---\n\n" + content[end_idx:]

with open("README.md", "w") as f:
    f.write(new_content)

print(f"README updated — Portfolio: ${portfolio_value:,.2f} | Daily P&L: ${daily_pnl:+,.2f} | Total P&L: ${total_pnl:+,.2f}")