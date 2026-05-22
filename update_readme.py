import alpaca_trade_api as tradeapi
import os
from datetime import datetime

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

api     = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
account = api.get_account()

# Pull live stats
portfolio_value = float(account.portfolio_value)
starting_value  = 100000.00
total_pnl       = portfolio_value - starting_value
pnl_pct         = (total_pnl / starting_value) * 100
cash            = float(account.cash)

# Get open positions
positions      = api.list_positions()
open_positions = len(positions)
positions_text = ""
for p in positions:
    pnl   = float(p.unrealized_pl)
    qty   = int(float(p.qty))
    value = float(p.market_value)
    positions_text += f"| {p.symbol} | {qty} | ${value:,.2f} | ${pnl:+.2f} |\n"

# Get recent orders
orders       = api.list_orders(status="all", limit=50)
total_trades = len(orders)

# Build the dynamic section
from datetime import datetime
import pytz
est = pytz.timezone("US/Eastern")
now = datetime.now(est).strftime("%B %d, %Y %I:%M %p EST")
pnl_emoji    = "📈" if total_pnl >= 0 else "📉"
status_block = f"""## Live Performance — Auto-updated {now}

| Metric | Value |
|--------|-------|
| Portfolio Value | ${portfolio_value:,.2f} |
| Total P&L | ${total_pnl:+,.2f} ({pnl_pct:+.2f}%) |
| Cash Available | ${cash:,.2f} |
| Open Positions | {open_positions} |
| Total Trades Executed | {total_trades} |

{pnl_emoji} Bot has been live since March 27, 2026

### Current Open Positions
| Ticker | Shares | Value | Unrealized P&L |
|--------|--------|-------|----------------|
{positions_text if positions_text else "| — | No open positions | — | — |"}
"""

# Read current README
with open("README.md", "r") as f:
    content = f.read()

# Replace the live performance section
start_marker = "## Live Performance"
end_marker   = "## Results"

if start_marker in content:
    start_idx    = content.index(start_marker)
    end_idx      = content.index(end_marker)
    new_content  = content[:start_idx] + status_block + "\n---\n\n" + content[end_idx:]
else:
    end_idx     = content.index(end_marker)
    new_content = content[:end_idx] + status_block + "\n---\n\n" + content[end_idx:]

with open("README.md", "w") as f:
    f.write(new_content)

print(f"README updated — Portfolio: ${portfolio_value:,.2f} | P&L: ${total_pnl:+,.2f}")