"""Deterministic trading policy."""

from __future__ import annotations

from ai_prophet_core.client_models import PortfolioResponse

from .config import BotConfig
from .portfolio_risk import cash_available, positions_by_market
from .schemas import ForecastSignals, MarketView, TradeDecision
from .sizer import size_trade


def decide_trade(
    market: MarketView,
    signals: ForecastSignals,
    portfolio: PortfolioResponse | None,
    config: BotConfig,
) -> TradeDecision | None:
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    threshold = max(config.min_edge, 1.5 * signals.uncertainty)
    if signals.confidence == "low":
        threshold = max(threshold, config.low_confidence_min_edge)

    positions = positions_by_market(portfolio)
    position = positions.get(market.market_id)

    if position is not None:
        exit_decision = maybe_exit_position(market, signals, position, config)
        if exit_decision is not None:
            return exit_decision

    resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
    if (
        config.block_trade_on_high_resolution_risk
        and resolution_check.get("trade_blocker") is True
    ):
        return None
    if (
        resolution_check.get("official_source_required") is True
        and resolution_check.get("official_source_found") is False
    ):
        threshold *= config.official_source_missing_edge_multiplier

    edge_yes = p_final - market.yes_ask
    edge_no = (1.0 - p_final) - market.no_ask

    if edge_yes <= threshold and edge_no <= threshold:
        return None

    if position is not None:
        # Avoid accidental opposite-side netting in MVP. Same-side adds are
        # allowed if the edge still clears threshold.
        if edge_yes > edge_no and position.side != "YES":
            return None
        if edge_no > edge_yes and position.side != "NO":
            return None

    if edge_yes >= edge_no:
        side = "YES"
        price = market.yes_ask
        edge = edge_yes
    else:
        side = "NO"
        price = market.no_ask
        edge = edge_no

    shares = size_trade(edge=edge, price=price, confidence=signals.confidence, config=config)
    if shares <= 0:
        return None

    cost = shares * price
    if cost > max(0.0, cash_available(portfolio, config.starting_cash) - config.reserve_cash):
        return None

    if position is not None and position.notional + cost > config.max_notional_per_market_target:
        return None

    return TradeDecision(
        market_id=market.market_id,
        action="BUY",
        side=side,
        shares=shares,
        price=price,
        edge=edge,
        expected_value=edge * shares,
        confidence=signals.confidence,
        reason=signals.reason or "edge clears deterministic threshold",
        p_final=p_final,
    )


def maybe_exit_position(
    market: MarketView,
    signals: ForecastSignals,
    position,
    config: BotConfig,
) -> TradeDecision | None:
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    if position.side == "YES":
        exit_price = market.yes_bid
        hold_edge = p_final - exit_price
    else:
        exit_price = market.no_bid
        hold_edge = (1.0 - p_final) - exit_price

    if hold_edge >= config.exit_edge:
        return None

    shares = int(position.shares)
    if shares <= 0:
        return None

    return TradeDecision(
        market_id=market.market_id,
        action="SELL",
        side=position.side,
        shares=shares,
        price=exit_price,
        edge=hold_edge,
        expected_value=hold_edge * shares,
        confidence=signals.confidence,
        reason="exit: hold edge compressed below exit threshold",
        p_final=p_final,
    )


def rank_decisions(decisions: list[TradeDecision], limit: int) -> list[TradeDecision]:
    ranked = sorted(decisions, key=lambda d: d.expected_value, reverse=True)
    return ranked[:limit]
