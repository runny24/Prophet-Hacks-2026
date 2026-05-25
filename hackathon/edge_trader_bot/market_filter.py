"""Deterministic market filters used before expensive forecasting."""

from __future__ import annotations

from ai_prophet_core.client_models import PortfolioResponse

from .config import BotConfig
from .portfolio_risk import open_positions_count, positions_by_market
from .schemas import MarketView


def deterministic_filter(
    market: MarketView,
    portfolio: PortfolioResponse | None,
    config: BotConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if market.yes_bid < 0 or market.yes_ask > 1 or market.yes_ask < market.yes_bid:
        reasons.append("invalid_quote")
        return False, reasons

    if market.spread > config.max_spread:
        reasons.append("spread_too_wide")

    if market.volume_24h < config.min_volume_24h:
        reasons.append("volume_too_low")

    if open_positions_count(portfolio) >= config.max_open_positions_target:
        position = positions_by_market(portfolio).get(market.market_id)
        if position is None:
            reasons.append("open_position_budget_low")

    return len(reasons) == 0, reasons
