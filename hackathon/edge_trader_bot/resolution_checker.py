"""Resolution-risk checks for prediction-market evidence."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .schemas import MarketView


OFFICIAL_SOURCE_HINTS = (
    "official",
    "government",
    ".gov",
    "sec.gov",
    "federalreserve.gov",
    "court",
    "commission",
    "agency",
    "press release",
    "filing",
)

OFFICIAL_REQUIREMENT_TERMS = (
    "official",
    "officially",
    "certified",
    "published",
    "filed",
    "reported by",
    "according to",
    "announced by",
    "confirmed by",
)

RESOLUTION_ACTION_TERMS = (
    "announced",
    "launched",
    "approved",
    "implemented",
    "reported",
    "confirmed",
    "published",
    "certified",
    "effective",
    "signed",
)

WEAK_EVIDENCE_TERMS = (
    "expected",
    "expects",
    "plans to",
    "could",
    "may",
    "might",
    "rumor",
    "rumour",
    "reportedly",
    "proposal",
    "proposed",
    "likely",
    "set to",
)

HEADLINE_ONLY_TERMS = (
    "headline",
    "reported",
    "reportedly",
    "expected",
    "plans",
    "proposal",
)


def check_resolution_risk(
    market: MarketView,
    evidence_items: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    market_text = f"{market.question}\n{market.description or ''}".lower()
    evidence_text = " ".join(
        f"{item.get('summary', '')} {item.get('source', '')} {item.get('url', '')}"
        for item in evidence_items
    ).lower()

    risk_flags: list[str] = []
    notes: list[str] = []

    official_source_required = contains_any(market_text, OFFICIAL_REQUIREMENT_TERMS)
    official_source_found = contains_any(evidence_text, OFFICIAL_SOURCE_HINTS) if evidence_items else False

    resolution_terms = [term for term in RESOLUTION_ACTION_TERMS if term in market_text]
    evidence_has_resolution_support = evidence_supports_resolution(evidence_items)
    headline_support = contains_any(evidence_text, HEADLINE_ONLY_TERMS) or contains_any(
        evidence_text, WEAK_EVIDENCE_TERMS
    )

    if evidence_items:
        headline_matches_resolution: bool | None = evidence_has_resolution_support
    else:
        headline_matches_resolution = None

    if resolution_terms and headline_support and not evidence_has_resolution_support:
        risk_flags.append("headline_resolution_mismatch")
        notes.append("Evidence appears to support the headline but not the exact resolution condition.")

    if official_source_required and not official_source_found:
        risk_flags.append("official_source_not_confirmed")
        notes.append("Market appears to require official confirmation, but no official source was found.")

    deadline_satisfied = infer_deadline_satisfied(market, evidence_items, now)
    if deadline_satisfied is False:
        risk_flags.append("deadline_not_satisfied")
        notes.append("Evidence appears stale or outside the market deadline.")
    elif deadline_satisfied is None and is_deadline_sensitive(market_text):
        risk_flags.append("deadline_unclear")
        notes.append("The market is deadline-sensitive and evidence timing is unclear.")

    if stale_evidence_found(market, evidence_items):
        risk_flags.append("stale_evidence")
        notes.append("Some evidence appears stale relative to the resolution window.")

    if not evidence_items:
        risk_flags.append("no_external_evidence")
        notes.append("No external evidence is available for resolution checking.")

    ambiguity_level = "low"
    trade_blocker = False
    if any(
        flag in risk_flags
        for flag in ("headline_resolution_mismatch", "official_source_not_confirmed", "deadline_not_satisfied")
    ):
        ambiguity_level = "high"
        trade_blocker = True
    elif risk_flags:
        ambiguity_level = "medium"

    # Backward-compatible aliases are kept for existing planner/tests.
    return {
        "headline_matches_resolution": headline_matches_resolution,
        "official_source_required": official_source_required,
        "official_source_found": official_source_found if official_source_required or evidence_items else None,
        "deadline_satisfied": deadline_satisfied,
        "ambiguity_level": ambiguity_level,
        "trade_blocker": trade_blocker,
        "notes": notes,
        "risk_flags": dedupe(risk_flags),
        "risk_level": ambiguity_level,
        "flags": dedupe(risk_flags),
        "resolution_terms_found": resolution_terms,
        "checked_at": now.isoformat(),
    }


def evidence_supports_resolution(evidence_items: list[dict[str, Any]]) -> bool:
    for item in evidence_items:
        if item.get("supports_resolution_condition") is True:
            return True
        text = f"{item.get('summary', '')} {item.get('source', '')} {item.get('url', '')}".lower()
        if contains_any(text, ("officially", "confirmed", "certified", "published", "approved", "implemented")):
            if not contains_any(text, WEAK_EVIDENCE_TERMS):
                return True
    return False


def infer_deadline_satisfied(
    market: MarketView,
    evidence_items: list[dict[str, Any]],
    now: datetime,
) -> bool | None:
    if market.resolution_time <= now:
        return False
    dated_items = [parse_evidence_datetime(item.get("timestamp")) for item in evidence_items]
    dated_items = [dt for dt in dated_items if dt is not None]
    if not dated_items:
        return None if is_deadline_sensitive(f"{market.question} {market.description or ''}".lower()) else True
    return any(dt <= market.resolution_time for dt in dated_items)


def stale_evidence_found(market: MarketView, evidence_items: list[dict[str, Any]]) -> bool:
    stale_cutoff = market.resolution_time - timedelta(days=30)
    for item in evidence_items:
        dt = parse_evidence_datetime(item.get("timestamp"))
        if dt is not None and dt < stale_cutoff:
            return True
        text = str(item.get("timestamp") or "").lower()
        if "year" in text or "month" in text:
            return True
    return False


def parse_evidence_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        pass
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=UTC)
    return None


def is_deadline_sensitive(text: str) -> bool:
    return any(term in text for term in ("before", " by ", "deadline", "on or before", "no later than"))


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
