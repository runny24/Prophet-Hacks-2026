"""Print current portfolio P&L from the Prophet Arena API."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge_trader_bot.config import BotConfig, load_env_file

load_env_file(Path(__file__).parent.parent / ".env")

from ai_prophet_core import DEFAULT_API_URL, ServerAPIClient

config = BotConfig.from_env()

api = ServerAPIClient(
    base_url=os.getenv("PA_SERVER_URL", DEFAULT_API_URL),
    api_key=os.getenv("PA_SERVER_API_KEY"),
    timeout=30,
)

EXPERIMENT_ID = "28522a55-366d-43e4-b950-44d91a80abc4"

portfolio = api.get_portfolio(EXPERIMENT_ID, 0)

if portfolio is None:
    print("No portfolio found.")
    sys.exit(1)

cash = float(portfolio.cash)
equity = float(portfolio.equity)
total_pnl = float(portfolio.total_pnl)
starting_cash = config.starting_cash
positions = portfolio.positions or []

print(f"{'='*55}")
print(f"  PORTFOLIO SUMMARY")
print(f"{'='*55}")
print(f"  Starting cash : ${starting_cash:>10,.2f}")
print(f"  Cash          : ${cash:>10,.2f}")
print(f"  Equity        : ${equity:>10,.2f}")
print(f"  Total P&L     : ${total_pnl:>+10,.2f}  ({total_pnl/starting_cash*100:+.2f}%)")
print(f"  Total fills   : {portfolio.total_fills}")
print(f"  Open positions: {len(positions)}")

if positions:
    print(f"\n{'='*55}")
    print(f"  OPEN POSITIONS")
    print(f"{'='*55}")
    print(f"  {'Market':<40} {'Side':<5} {'Shares':>6} {'Entry':>7} {'Now':>7} {'UPnL':>8}")
    print(f"  {'-'*40} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")

    total_unrealized = 0.0
    total_realized = 0.0
    for p in sorted(positions, key=lambda x: float(x.unrealized_pnl), reverse=True):
        upnl = float(p.unrealized_pnl)
        rpnl = float(p.realized_pnl)
        total_unrealized += upnl
        total_realized += rpnl
        entry = float(p.avg_entry_price)
        current = float(p.current_price)
        shares = float(p.shares)
        mkt = p.market_id.replace("kalshi:", "")
        print(f"  {mkt:<40} {p.side:<5} {shares:>6.0f} {entry:>7.3f} {current:>7.3f} {upnl:>+8.2f}")

    print(f"  {'-'*55}")
    print(f"  {'Unrealized P&L':>50}: ${total_unrealized:>+.2f}")
    print(f"  {'Realized P&L':>50}: ${total_realized:>+.2f}")
else:
    print("\n  No open positions.")
