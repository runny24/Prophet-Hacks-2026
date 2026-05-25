"""Deterministic live-readiness diagnostics.

This module deliberately does not submit orders or alter trading policy. It
only explains whether the current dry-run evidence would clear a stricter
manual live-submit gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STRONG_SPORTS_SUPPORT_TYPES = {
    "odds_market",
    "sportsbook_odds",
    "model_forecast",
    "elo_rating",
    "simulation",
    "quantitative_ranking",
}


@dataclass(frozen=True)
class LiveReadinessConfig:
    taker_edge: float = 0.03
    mid_edge: float = 0.02
    min_evidence_quality: int = 4
    max_spread: float = 0.02
    require_blf: bool = True


def config_from_bot_config(config: Any) -> LiveReadinessConfig:
    return LiveReadinessConfig(
        taker_edge=float(getattr(config, "live_ready_taker_edge", 0.03)),
        mid_edge=float(getattr(config, "live_ready_mid_edge", 0.02)),
        min_evidence_quality=int(getattr(config, "live_ready_min_evidence_quality", 4)),
        max_spread=float(getattr(config, "live_ready_max_spread", 0.02)),
        require_blf=bool(getattr(config, "live_ready_require_blf", True)),
    )


def evaluate_market_live_readiness(
    row: dict[str, Any],
    *,
    config: LiveReadinessConfig | None = None,
    blf_enabled: bool = True,
) -> dict[str, Any]:
    cfg = config or LiveReadinessConfig()
    reasons: list[str] = []
    blockers: list[str] = []

    best_taker_edge = numeric(row.get("best_taker_edge"))
    best_mid_edge = numeric(row.get("best_mid_edge"))
    best_spread_adjusted_maker_edge = numeric(row.get("maker_edge_after_half_spread"))
    best_spread = numeric(row.get("best_spread"))
    evidence_quality = int(row.get("evidence_quality") or 0)
    confidence = row.get("confidence")
    market_type = row.get("market_type")
    sports_support_type = row.get("sports_support_source_type") or "none"
    p_blf = row.get("p_blf_final_after_shrinkage")
    p_2402 = row.get("p_2402_final_after_shrinkage") or row.get("p_2402")

    if cfg.require_blf and not blf_enabled:
        blockers.append("blf_not_enabled")
    if cfg.require_blf and p_blf is None:
        blockers.append("blf_missing")
    if evidence_quality < cfg.min_evidence_quality:
        blockers.append("evidence_quality_below_live_ready_min")
    if confidence not in {"medium", "high"}:
        blockers.append("confidence_below_live_ready_min")
    if row.get("resolution_trade_blocker"):
        blockers.append("resolution_trade_blocker")
    if best_spread is not None and best_spread > cfg.max_spread:
        blockers.append("spread_above_live_ready_max")
    if row.get("sports_llm_overconfidence_detected"):
        blockers.append("sports_llm_overconfidence")
    if market_type == "sports_outcome":
        if not row.get("sports_quantitative_support"):
            blockers.append("sports_quantitative_support_missing")
        if sports_support_type not in STRONG_SPORTS_SUPPORT_TYPES:
            blockers.append("sports_support_source_type_not_actionable")
    if blf_strongly_opposes_rag(p_blf, p_2402, row):
        blockers.append("blf_strongly_opposes_rag")
    if not evidence_matches_resolution(row):
        blockers.append("evidence_not_resolution_aligned")
    blockers.extend(safety_hold_blockers(row.get("hold_reasons", [])))

    taker_ready = best_taker_edge is not None and best_taker_edge >= cfg.taker_edge
    passive_ready = (
        best_mid_edge is not None
        and best_mid_edge >= cfg.mid_edge
        and row.get("passive_opportunity_quality") == "strong"
    )
    if taker_ready:
        reasons.append(f"taker_edge>={cfg.taker_edge:.3f}")
    if passive_ready:
        reasons.append(f"mid_edge>={cfg.mid_edge:.3f}_with_strong_passive_quality")
    if not taker_ready and not passive_ready:
        blockers.append("edge_below_live_ready_threshold")

    live_ready = not blockers and (taker_ready or passive_ready)
    return {
        "market_id": row.get("market_id"),
        "question": row.get("question"),
        "market_type": market_type,
        "live_ready": live_ready,
        "live_ready_side": row.get("best_taker_side") if taker_ready else row.get("best_mid_side") if passive_ready else None,
        "live_ready_reason": reasons,
        "live_blockers": list(dict.fromkeys(blockers)),
        "best_taker_edge": best_taker_edge,
        "best_mid_edge": best_mid_edge,
        "best_spread_adjusted_maker_edge": best_spread_adjusted_maker_edge,
        "evidence_quality": evidence_quality,
        "confidence": confidence,
    }


def evaluate_tick_live_readiness(
    rows: list[dict[str, Any]],
    *,
    config: LiveReadinessConfig | None = None,
    blf_enabled: bool = True,
    allow_live_submit: bool = False,
) -> dict[str, Any]:
    cfg = config or LiveReadinessConfig()
    per_market = [
        evaluate_market_live_readiness(row, config=cfg, blf_enabled=blf_enabled)
        for row in rows
    ]
    ready_markets = [row for row in per_market if row["live_ready"]]
    blockers = aggregate_blockers(per_market)
    reasons: list[str] = []
    if not ready_markets:
        reasons.extend(readiness_summary_reasons(rows, cfg))
    if not allow_live_submit:
        blockers.append("live_submit_disabled")
        reasons.append("live submit remains disabled")
    if cfg.require_blf and not blf_enabled:
        blockers.append("blf_not_enabled")
        reasons.append("BLF-disabled diagnostic scenarios are not actionable")
    return {
        "LIVE_SUBMIT_READY": bool(ready_markets) and allow_live_submit and not blockers,
        "candidate_count": len(ready_markets),
        "max_blf_enabled_taker_edge": max_numeric(rows, "best_taker_edge"),
        "max_blf_enabled_mid_edge": max_numeric(rows, "best_mid_edge"),
        "max_blf_enabled_spread_adjusted_maker_edge": max_numeric(rows, "maker_edge_after_half_spread"),
        "reasons": list(dict.fromkeys(reasons)),
        "blockers": list(dict.fromkeys(blockers)),
        "live_ready_markets": ready_markets,
        "per_market": per_market,
    }


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def max_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return max(values) if values else None


def blf_strongly_opposes_rag(p_blf: Any, p_2402: Any, row: dict[str, Any]) -> bool:
    if not isinstance(p_blf, (int, float)) or not isinstance(p_2402, (int, float)):
        return False
    p_market = row.get("p_market")
    if abs(float(p_blf) - float(p_2402)) >= 0.12:
        return True
    if isinstance(p_market, (int, float)):
        rag_move = abs(float(p_2402) - float(p_market))
        final_move = abs(float(row.get("p_final_after_blf") or p_market) - float(p_market))
        return rag_move >= 0.08 and final_move <= rag_move * 0.5
    return False


def evidence_matches_resolution(row: dict[str, Any]) -> bool:
    flags = set(row.get("normalized_risk_flags") or [])
    if "resolution_mismatch" in flags or "missing_official_source" in flags:
        return False
    if "high_ambiguity" in flags:
        return False
    return True


def safety_hold_blockers(hold_reasons: list[str]) -> list[str]:
    safety_terms = {
        "resolution_trade_blocker",
        "missing_official_source_requires_larger_edge",
        "wide_spread",
        "low_liquidity",
        "low_evidence_quality",
    }
    return [reason for reason in hold_reasons if reason in safety_terms]


def aggregate_blockers(per_market: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in per_market:
        blockers.extend(item.get("live_blockers", []))
    return sorted(set(blockers))


def readiness_summary_reasons(rows: list[dict[str, Any]], cfg: LiveReadinessConfig) -> list[str]:
    reasons: list[str] = []
    max_taker = max_numeric(rows, "best_taker_edge")
    max_mid = max_numeric(rows, "best_mid_edge")
    qualities = [row.get("passive_opportunity_quality") for row in rows]
    if max_taker is None or max_taker < cfg.taker_edge:
        reasons.append(f"max BLF-enabled taker edge below {cfg.taker_edge:.2f}")
    if max_mid is None or max_mid < cfg.mid_edge:
        reasons.append(f"no mid_edge >= {cfg.mid_edge:.2f}")
    if "strong" not in qualities:
        reasons.append("no passive opportunity stronger than weak")
    return reasons
