"""Cheap deterministic candidate pre-ranking for live price trading."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .schemas import MarketView


CATALYST_TERMS = {
    "today": 25,
    "tomorrow": 25,
    "this week": 20,
    "this month": 12,
    "before": 12,
    " by ": 10,
    "release": 18,
    "announce": 20,
    "announcement": 20,
    "approval": 18,
    "vote": 18,
    "report": 14,
    "earnings": 25,
    "cpi": 25,
    "fed": 22,
    "court": 20,
    "trial": 18,
    "deadline": 18,
    "meeting": 16,
    "launch": 18,
    "update": 12,
    "resign": 20,
    "confirm": 18,
    "testify": 18,
}


@dataclass(frozen=True)
class CandidateRankResult:
    market_id: str
    score: float
    score_components: dict[str, float]
    priority_bucket: str
    reject_or_penalty_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_candidate_for_processing(
    market: MarketView,
    memory_features: dict[str, Any] | None = None,
    portfolio_position: Any = None,
    now: datetime | None = None,
) -> CandidateRankResult:
    now = now or datetime.now(UTC)
    memory_features = memory_features or {}
    components: dict[str, float] = {}
    penalties: list[str] = []

    if portfolio_position is not None:
        components["existing_position"] = 200

    repeated = int(memory_features.get("repeated_market_count") or 0)
    delta_1 = as_float(memory_features.get("mid_change_1tick"))
    delta_2 = as_float(memory_features.get("mid_change_2tick"))
    moving = (delta_1 is not None and abs(delta_1) >= 0.003) or (
        delta_2 is not None and abs(delta_2) >= 0.005
    )

    if repeated >= 2:
        components["repeated_market_count"] = 20
    if delta_1 is not None and abs(delta_1) >= 0.003:
        components["mid_change_1tick"] = 25
    if delta_2 is not None and abs(delta_2) >= 0.005:
        components["mid_change_2tick"] = 20
    quote_valid = valid_quote(market)
    if quote_valid and market.spread <= 0.005:
        components["spread_le_0_005"] = 25
    elif quote_valid and market.spread <= 0.01:
        components["spread_le_0_01"] = 15

    if memory_features.get("previous_price_signal"):
        components["previous_price_signal"] = 15
    if memory_features.get("previous_trade_candidate") or memory_features.get("traded"):
        components["previous_trade_candidate"] = 20

    catalyst_score = catalyst_word_score(market)
    if catalyst_score:
        components["near_term_catalyst_words"] = catalyst_score

    if not quote_valid:
        components["invalid_quote_penalty"] = -100
        penalties.append("invalid_quote")
    if market.spread >= 0.10:
        components["spread_ge_0_10_penalty"] = -100
        penalties.append("spread_ge_0_10")
    elif market.spread >= 0.05:
        components["spread_ge_0_05_penalty"] = -60
        penalties.append("spread_ge_0_05")

    text = market_text(market)
    long_horizon = is_long_horizon_market(market, text, now)
    if long_horizon and not moving:
        components["long_horizon_no_movement_penalty"] = -25
        penalties.append("long_horizon_no_movement")
    if is_alien_disclosure(text) and not moving:
        components["alien_disclosure_no_movement_penalty"] = -40
        penalties.append("alien_disclosure_no_movement")
    if is_long_horizon_sports(text) and not (moving and market.spread <= 0.01):
        components["long_horizon_sports_penalty"] = -25
        penalties.append("long_horizon_sports")
    if repeated == 0 and not catalyst_score:
        components["no_history_no_catalyst_penalty"] = -15
        penalties.append("no_history_no_catalyst")

    score = sum(components.values())
    bucket = priority_bucket(
        score=score,
        existing=portfolio_position is not None,
        repeated=repeated,
        moving=moving,
        tight_spread=market.spread <= 0.01,
        catalyst=bool(catalyst_score),
        penalties=penalties,
    )
    return CandidateRankResult(
        market_id=market.market_id,
        score=score,
        score_components=components,
        priority_bucket=bucket,
        reject_or_penalty_reasons=penalties,
    )


def rank_candidates(
    markets: list[MarketView],
    *,
    memory_by_market: dict[str, dict[str, Any]] | None = None,
    positions_by_market: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[tuple[MarketView, CandidateRankResult]]:
    memory_by_market = memory_by_market or {}
    positions_by_market = positions_by_market or {}
    ranked = [
        (
            market,
            score_candidate_for_processing(
                market,
                memory_by_market.get(market.market_id, {}),
                positions_by_market.get(market.market_id),
                now,
            ),
        )
        for market in markets
    ]
    ranked.sort(key=lambda item: (-item[1].score, item[0].market_id))
    return ranked


def memory_features_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    mids = [as_float(row.get("midpoint")) for row in history if as_float(row.get("midpoint")) is not None]
    latest = history[-1] if history else {}
    return {
        "repeated_market_count": len(history),
        "mid_change_1tick": mids[-1] - mids[-2] if len(mids) >= 2 else None,
        "mid_change_2tick": mids[-1] - mids[-3] if len(mids) >= 3 else None,
        "previous_price_signal": bool(latest.get("previous_price_signal") or latest.get("price_signal_type")),
        "previous_trade_candidate": bool(latest.get("previous_trade_candidate") or latest.get("traded")),
        "traded": bool(latest.get("traded")),
    }


def candidate_selection_summary(
    *,
    ranked: list[tuple[MarketView, CandidateRankResult]],
    selected: list[MarketView],
) -> dict[str, Any]:
    selected_ids = {market.market_id for market in selected}
    selected_rank = [item for item in ranked if item[0].market_id in selected_ids]
    deprioritized = [item for item in reversed(ranked) if item[0].market_id not in selected_ids]
    spreads = [market.spread for market, _ in selected_rank]
    return {
        "candidate_rerank_enabled": True,
        "candidates_total": len(ranked),
        "candidates_selected": len(selected),
        "selected_bucket_counts": bucket_counts(result for _, result in selected_rank),
        "top_selected_candidates": [summary_row(market, result) for market, result in selected_rank[:15]],
        "top_deprioritized_candidates": [summary_row(market, result) for market, result in deprioritized[:15]],
        "avg_selected_spread": sum(spreads) / len(spreads) if spreads else 0.0,
        "selected_repeated_count": sum(
            1 for _, result in selected_rank if result.score_components.get("repeated_market_count", 0) > 0
        ),
        "selected_recent_movers_count": sum(
            1
            for _, result in selected_rank
            if result.score_components.get("mid_change_1tick", 0) > 0
            or result.score_components.get("mid_change_2tick", 0) > 0
        ),
        "selected_near_term_catalyst_count": sum(
            1 for _, result in selected_rank if result.score_components.get("near_term_catalyst_words", 0) > 0
        ),
    }


def disabled_candidate_selection_summary(total: int, selected: int) -> dict[str, Any]:
    return {
        "candidate_rerank_enabled": False,
        "candidates_total": total,
        "candidates_selected": selected,
        "selected_bucket_counts": {},
        "top_selected_candidates": [],
        "top_deprioritized_candidates": [],
        "avg_selected_spread": 0.0,
        "selected_repeated_count": 0,
        "selected_recent_movers_count": 0,
        "selected_near_term_catalyst_count": 0,
    }


def summary_row(market: MarketView, result: CandidateRankResult) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "question": market.question,
        "score": result.score,
        "priority_bucket": result.priority_bucket,
        "spread": market.spread,
        "components": result.score_components,
        "penalty_reasons": result.reject_or_penalty_reasons,
    }


def bucket_counts(results) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.priority_bucket] = counts.get(result.priority_bucket, 0) + 1
    return counts


def priority_bucket(
    *,
    score: float,
    existing: bool,
    repeated: int,
    moving: bool,
    tight_spread: bool,
    catalyst: bool,
    penalties: list[str],
) -> str:
    if existing:
        return "existing_position"
    if repeated >= 2 and moving:
        return "repeated_mover"
    if tight_spread and score > 0:
        return "tight_spread_active"
    if catalyst:
        return "near_term_catalyst"
    if score <= -40 or "invalid_quote" in penalties:
        return "deprioritized"
    return "exploratory"


def valid_quote(market: MarketView) -> bool:
    return 0 <= market.yes_bid <= market.yes_ask <= 1 and market.spread >= 0


def catalyst_word_score(market: MarketView) -> int:
    text = market_text(market)
    return min(25, sum(score for term, score in CATALYST_TERMS.items() if term in text))


def is_long_horizon_market(market: MarketView, text: str, now: datetime) -> bool:
    days = (market.resolution_time - now).days
    phrases = ("2028", "2029", "2030", "presidency", "nomination", "election")
    return days > 180 or any(phrase in text for phrase in phrases)


def is_alien_disclosure(text: str) -> bool:
    return bool(re.search(r"\b(aliens?|ufo|uap|disclosure)\b", text))


def is_long_horizon_sports(text: str) -> bool:
    return (
        ("world cup" in text and "2026" in text)
        or ("championship" in text and "2026" in text)
        or ("win the 2026" in text and any(term in text for term in ("cup", "championship", "nba", "mlb")))
    )


def market_text(market: MarketView) -> str:
    return " ".join(str(part or "").lower() for part in (market.question, market.description, market.topic, market.family))


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
