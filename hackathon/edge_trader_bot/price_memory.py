"""Local cross-tick market state for short-horizon price trading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json_utils import json_safe
from .schemas import MarketView


@dataclass(frozen=True)
class PriceFeatures:
    mid_now: float
    mid_1_tick_ago: float | None
    mid_2_ticks_ago: float | None
    delta_1_tick: float | None
    delta_2_ticks: float | None
    spread_now: float
    spread_percent: float
    repeated_seen_count: int
    volatility: float
    forecast_market_gap: float | None
    forecast_stability: float | None
    price_moved_forecast_stable: bool
    liquidity_proxy: float
    has_history: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PriceMemory:
    def __init__(self, path: Path = Path(".edge_trader/market_history.jsonl")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_market_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(snapshot), sort_keys=True) + "\n")

    def load_recent_market_history(self, market_id: str, lookback_ticks: int = 8) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("market_id") == market_id:
                    rows.append(row)
        return rows[-lookback_ticks:]


def market_snapshot(
    *,
    tick_id: str,
    candidate_set_id: str,
    generated_at: str,
    market: MarketView,
    signals,
    position: Any = None,
    traded: bool = False,
) -> dict[str, Any]:
    return {
        "timestamp": generated_at,
        "tick_id": tick_id,
        "candidate_set_id": candidate_set_id,
        "market_id": market.market_id,
        "question": market.question,
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "no_bid": market.no_bid,
        "no_ask": market.no_ask,
        "midpoint": market.yes_mid,
        "market_probability": market.yes_mid,
        "spread": market.spread,
        "volume_24h": market.volume_24h,
        "current_position": position_to_dict(position),
        "p_final_after_blf": getattr(signals, "p_final_after_blf", None),
        "p_2402": getattr(signals, "p_2402", None),
        "p_blf": getattr(signals, "p_blf", None),
        "risk_flags": list(getattr(signals, "risk_flags", []) or [])[:12],
        "market_type": ((getattr(signals, "evidence_package", None) or {}).get("market_type")),
        "traded": traded,
    }


def compute_price_features(history: list[dict[str, Any]], market: MarketView, signals) -> PriceFeatures:
    mids = [float(row["midpoint"]) for row in history if isinstance(row.get("midpoint"), (int, float))]
    forecasts = [
        float(row["p_final_after_blf"])
        for row in history
        if isinstance(row.get("p_final_after_blf"), (int, float))
    ]
    mid_1 = mids[-1] if len(mids) >= 1 else None
    mid_2 = mids[-2] if len(mids) >= 2 else None
    delta_1 = market.yes_mid - mid_1 if mid_1 is not None else None
    delta_2 = market.yes_mid - mid_2 if mid_2 is not None else None
    recent = [*mids[-4:], market.yes_mid]
    changes = [recent[idx] - recent[idx - 1] for idx in range(1, len(recent))]
    volatility = sum(abs(change) for change in changes) / len(changes) if changes else 0.0
    p_final = getattr(signals, "p_final_after_blf", None) or getattr(signals, "p_final", None)
    gap = p_final - market.yes_mid if isinstance(p_final, (int, float)) else None
    forecast_stability = None
    if forecasts and isinstance(p_final, (int, float)):
        forecast_stability = abs(p_final - forecasts[-1])
    return PriceFeatures(
        mid_now=market.yes_mid,
        mid_1_tick_ago=mid_1,
        mid_2_ticks_ago=mid_2,
        delta_1_tick=delta_1,
        delta_2_ticks=delta_2,
        spread_now=market.spread,
        spread_percent=market.spread / max(market.yes_mid, 0.01),
        repeated_seen_count=len(history),
        volatility=volatility,
        forecast_market_gap=gap,
        forecast_stability=forecast_stability,
        price_moved_forecast_stable=bool(
            delta_1 is not None
            and abs(delta_1) >= 0.003
            and (forecast_stability is None or forecast_stability <= 0.003)
        ),
        liquidity_proxy=market.volume_24h,
        has_history=bool(history),
    )


def position_to_dict(position: Any) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "market_id": getattr(position, "market_id", None),
        "side": getattr(position, "side", None),
        "shares": getattr(position, "shares", None),
        "avg_entry_price": getattr(position, "avg_entry_price", None),
        "current_price": getattr(position, "current_price", None),
        "notional": getattr(position, "notional", None),
    }
