"""Cheap statistical and market-implied priors."""

from __future__ import annotations

from .math_utils import clamp_probability, logit, sigmoid
from .schemas import ForecastSignals, MarketView


def compute_statistical_prior(market: MarketView) -> float:
    """Return a conservative statistical prior.

    MVP behavior intentionally anchors to market midpoint. Later versions can
    add category base rates, price movement, and time-to-resolution effects.
    """
    return clamp_probability(market.yes_mid)


def initial_signals(market: MarketView) -> ForecastSignals:
    p_market = clamp_probability(market.yes_mid)
    p_stat = compute_statistical_prior(market)
    p_raw = sigmoid(0.75 * logit(p_market) + 0.25 * logit(p_stat))
    return ForecastSignals(
        market_id=market.market_id,
        p_market=p_market,
        p_stat=p_stat,
        p_final=p_raw,
        confidence="low",
        uncertainty=0.08,
        reason="market/statistical prior only",
    )
