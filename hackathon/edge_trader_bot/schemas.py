"""Internal strategy data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class MarketView:
    market_id: str
    question: str
    description: str | None
    resolution_time: datetime
    yes_bid: float
    yes_ask: float
    yes_mid: float
    no_bid: float
    no_ask: float
    no_mid: float
    spread: float
    volume_24h: float
    topic: str | None = None
    family: str | None = None
    source_url: str | None = None


@dataclass
class ForecastSignals:
    market_id: str
    p_market: float
    p_stat: float
    p_2402: float | None = None
    p_blf: float | None = None
    p_final: float | None = None
    confidence: str = "low"
    uncertainty: float = 0.08
    evidence_quality: int = 0
    risk_flags: list[str] = field(default_factory=list)
    reason: str = ""
    evidence_package: dict[str, Any] | None = None
    blf_package: dict[str, Any] | None = None
    p_final_before_blf: float | None = None
    p_final_after_blf: float | None = None
    blf_adjustment: float | None = None
    aggregation_reason: str = ""


@dataclass
class TradeDecision:
    market_id: str
    action: str
    side: str
    shares: int
    price: float
    edge: float
    expected_value: float
    confidence: str
    reason: str
    p_final: float

    def to_intent_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "action": self.action,
            "side": self.side,
            "shares": str(self.shares),
            "rationale": self.reason,
        }

    def to_plan_dict(self) -> dict[str, Any]:
        return asdict(self)
