"""Conversion helpers for Prophet Arena market wire models."""

from __future__ import annotations

from ai_prophet_core.client_models import MarketData

from .schemas import MarketView


def market_view(market: MarketData) -> MarketView:
    yes_bid = float(market.quote.best_bid)
    yes_ask = float(market.quote.best_ask)
    yes_mid = (yes_bid + yes_ask) / 2.0
    return MarketView(
        market_id=market.market_id,
        question=market.question,
        description=market.description,
        resolution_time=market.resolution_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=yes_mid,
        no_bid=1.0 - yes_ask,
        no_ask=1.0 - yes_bid,
        no_mid=1.0 - yes_mid,
        spread=yes_ask - yes_bid,
        volume_24h=float(market.quote.volume_24h or 0.0),
        topic=market.topic,
        family=market.family,
        source_url=market.source_url,
    )
