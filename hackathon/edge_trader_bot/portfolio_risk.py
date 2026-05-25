"""Portfolio and exposure helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ai_prophet_core.client_models import PortfolioResponse, PositionData


@dataclass(frozen=True)
class PositionView:
    market_id: str
    side: str
    shares: float
    avg_entry_price: float
    current_price: float
    notional: float


def positions_by_market(portfolio: PortfolioResponse | None) -> dict[str, PositionView]:
    if portfolio is None:
        return {}
    return {p.market_id: position_view(p) for p in portfolio.positions}


def position_view(position: PositionData) -> PositionView:
    shares = float(position.shares)
    avg_entry = float(position.avg_entry_price)
    current_price = float(position.current_price or avg_entry)
    return PositionView(
        market_id=position.market_id,
        side=position.side,
        shares=shares,
        avg_entry_price=avg_entry,
        current_price=current_price,
        notional=shares * current_price,
    )


def open_positions_count(portfolio: PortfolioResponse | None) -> int:
    return len(portfolio.positions) if portfolio else 0


def cash_available(portfolio: PortfolioResponse | None, fallback: float) -> float:
    if portfolio is None:
        return fallback
    return float(portfolio.cash)


def gross_exposure(portfolio: PortfolioResponse | None) -> float:
    if portfolio is None:
        return 0.0
    return sum(position_view(position).notional for position in portfolio.positions)


def total_fills(portfolio: PortfolioResponse | None) -> int:
    return int(portfolio.total_fills) if portfolio else 0
