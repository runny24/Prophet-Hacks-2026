"""Lightweight market type and horizon diagnostics."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .schemas import MarketView


FUTURE_EVENT_TYPES = {"future_announcement", "future_candidacy"}


def classify_market(market: MarketView, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    text = f"{market.question} {market.description or ''} {market.topic or ''} {market.family or ''}".lower()
    days = max((market.resolution_time - now).days, 0)

    market_type = "other"
    if contains_any(text, ("run for", "presidential nomination", "announce a presidential campaign", "candidacy")):
        market_type = "future_candidacy"
    elif contains_any(text, ("announce", "announces", "announcement", "launched", "launches")) and days >= 90:
        market_type = "future_announcement"
    elif contains_any(text, ("senate", "house", "democratic party", "republican party", "election", "control")):
        market_type = "election_control"
    elif contains_any(text, ("actor", "actress", "cast", "casting", "white lotus", "movie", "film", "album")):
        market_type = "entertainment_casting"
    elif contains_any(text, ("world cup", "nba", "mlb", "nfl", "nhl", "championship", "sports")):
        market_type = "sports_outcome"
    elif contains_any(text, ("fusion", "mars", "moon", "scientific", "fda", "approval")):
        market_type = "scientific_milestone"
    elif contains_any(text, ("co2", "carbon", "temperature", "climate", "emissions")):
        market_type = "climate_metric"

    if market_type in {"future_announcement", "future_candidacy"} and contains_any(
        text, ("2028", "2029", "2030", "2035", "2040", "2045")
    ):
        long_horizon_political = contains_any(text, ("president", "nomination", "senate", "house", "republican", "democrat"))
    else:
        long_horizon_political = False

    return {
        "market_type": "long_horizon_political" if long_horizon_political and market_type == "other" else market_type,
        "base_market_type": market_type,
        "long_horizon_days_to_resolution": days,
        "is_long_horizon": days >= 180,
        "is_future_event": market_type in FUTURE_EVENT_TYPES,
        "future_event_should_anchor_to_market": market_type in FUTURE_EVENT_TYPES and days >= 180,
    }


def future_event_prompt_guidance(market_metadata: dict[str, Any]) -> list[str]:
    if not market_metadata.get("future_event_should_anchor_to_market"):
        return []
    return [
        "For future-announcement or future-candidacy markets, do not interpret lack of a current announcement as strong evidence of NO.",
        "If the resolution date is far away, absence of direct evidence should usually keep probability near the market prior unless there is strong contrary evidence.",
        "Distinguish: no announcement yet; explicit decline/ruled out/ineligible; official announcement happened; credible reporting says preparing to announce.",
        "Only assign very low raw probability if there is strong contrary evidence, not merely missing evidence.",
        "Treat the market midpoint as a strong prior for unresolved future action.",
    ]


def detect_absence_of_evidence_penalty(
    *,
    market_metadata: dict[str, Any],
    p_raw: float | None,
    market_mid: float,
    reasoning_summary: str = "",
    risk_flags: list[str] | None = None,
    evidence_for_yes: list[Any] | None = None,
    evidence_for_no: list[Any] | None = None,
    missing_information: list[Any] | None = None,
) -> dict[str, Any]:
    risk_flags = risk_flags or []
    evidence_for_yes = evidence_for_yes or []
    evidence_for_no = evidence_for_no or []
    missing_information = missing_information or []
    text = " ".join(
        [
            reasoning_summary,
            " ".join(str(flag) for flag in risk_flags),
            " ".join(str(item) for item in evidence_for_yes),
            " ".join(str(item) for item in evidence_for_no),
            " ".join(str(item) for item in missing_information),
        ]
    ).lower()
    no_direct = contains_any(
        text,
        (
            "no direct evidence",
            "no evidence",
            "no announcement",
            "has not announced",
            "not announced",
            "missing information",
            "no external evidence",
            "absence of evidence",
        ),
    )
    contrary = contains_any(
        text,
        (
            "declined",
            "ruled out",
            "not running",
            "ineligible",
            "withdrew",
            "withdrawn",
            "not part of",
            "will not",
            "confirmed not",
            "disqualified",
        ),
    )
    detected = (
        bool(market_metadata.get("future_event_should_anchor_to_market"))
        and p_raw is not None
        and p_raw < 0.10
        and no_direct
        and not contrary
    )
    return {
        "absence_of_evidence_penalty_detected": detected,
        "no_direct_evidence_reasoning": no_direct,
        "future_event_should_anchor_to_market": bool(market_metadata.get("future_event_should_anchor_to_market")),
        "strong_contrary_evidence_detected": contrary,
        "guarded_probability": guarded_probability(p_raw, market_mid) if detected and p_raw is not None else p_raw,
    }


def guarded_probability(p_raw: float, market_mid: float) -> float:
    return max(p_raw, min(max(market_mid - 0.05, 0.02), 0.98))


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def compact_type_text(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))
