"""Position sizing."""

from __future__ import annotations

from .config import BotConfig


def size_trade(edge: float, price: float, confidence: str, config: BotConfig) -> int:
    if edge < config.min_edge:
        return 0
    if confidence == "low" and edge < config.low_confidence_min_edge:
        return 0

    if edge < 0.08:
        dollars = 50.0
    elif edge < 0.12:
        dollars = 100.0
    elif edge < 0.20:
        dollars = 150.0
    else:
        dollars = config.max_new_notional_per_trade if confidence == "high" else 150.0

    dollars = min(dollars, config.max_new_notional_per_trade)
    return max(int(dollars / max(price, 0.01)), 0)
