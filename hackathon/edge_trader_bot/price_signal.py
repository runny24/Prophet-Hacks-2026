"""Short-horizon price-trading signals and exits."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from typing import Any

from ai_prophet_core.client_models import PortfolioResponse

from .config import BotConfig
from .portfolio_risk import open_positions_count, positions_by_market
from .schemas import ForecastSignals, MarketView, TradeDecision

FALLBACK_RISK_MARKERS = {
    "llm_rate_limit",
    "blf_rate_limit",
    "llm_rag_failed",
    "stale_evidence",
    "malformed_response",
    "invalid_json",
    "timeout",
    "fallback",
    "empty_response",
    "low_evidence_quality",
    "search_failure",
    "resolution_risk",
    "ambiguous_resolution",
}
FORECAST_DEPENDENT_SIGNALS = {"forecast_edge", "spread_capture"}


@dataclass
class ShortHorizonSignal:
    market_id: str
    side: str
    signal_type: str
    expected_tick_edge: float
    confidence: str
    reason: str
    blockers: list[str]
    suggested_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FreshEventGateResult:
    allowed: bool
    reasons: list[str]
    channel: str
    max_size: int
    side: str
    edge: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastMispricingGateResult:
    allowed: bool
    reasons: list[str]
    channel: str
    max_size: int
    side: str
    edge: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_short_horizon_trade_signal(
    market: MarketView,
    signals: ForecastSignals,
    features,
    portfolio: PortfolioResponse | None,
    config: BotConfig,
) -> ShortHorizonSignal:
    blockers: list[str] = []
    positions = positions_by_market(portfolio)
    position = positions.get(market.market_id)

    if not getattr(features, "has_history", False):
        blockers.append("no_history")
    if market.spread > config.max_spread:
        blockers.append("spread_too_wide")
    if market.spread > 0.12:
        blockers.append("spread_absurd")
    if open_positions_count(portfolio) >= config.price_max_total_open_positions and position is None:
        blockers.append("position_limit_total")
    if position is not None and float(position.shares) >= config.price_max_position_per_market:
        blockers.append("position_limit_market")

    side = "HOLD"
    signal_type = "none"
    edge = 0.0
    confidence = "low"
    reason = "no short-horizon price signal"

    delta_1 = features.delta_1_tick
    delta_2 = features.delta_2_ticks
    forecast_gap = features.forecast_market_gap
    stable_forecast = features.forecast_stability is None or features.forecast_stability <= 0.004

    if delta_1 is not None and delta_2 is not None and same_direction(delta_1, delta_2) and abs(delta_1) >= 0.001:
        side = "YES" if delta_1 > 0 else "NO"
        signal_type = "momentum"
        edge = min(0.01, (abs(delta_1) + abs(delta_2)) / 2.0)
        confidence = "medium" if abs(delta_1) >= 0.003 else "low"
        reason = "recent midpoint moved consistently in one direction"
    elif delta_1 is not None and abs(delta_1) >= 0.008 and stable_forecast:
        side = "NO" if delta_1 > 0 else "YES"
        signal_type = "mean_reversion"
        edge = min(0.008, abs(delta_1) / 2.0)
        confidence = "medium"
        reason = "sharp price move without matching forecast movement"
    elif forecast_gap is not None and abs(forecast_gap) >= config.forecast_edge_min_edge:
        side = "YES" if forecast_gap > 0 else "NO"
        signal_type = "forecast_edge"
        edge = min(0.008, abs(forecast_gap))
        confidence = signals.confidence if signals.confidence in {"medium", "high"} else "low"
        reason = "small event-forecast sanity edge"
    elif forecast_gap is not None and market.spread <= 0.08:
        # Passive diagnostic: not directly executable as a taker order, but
        # useful for tiny guarded tests if server semantics allow posted orders
        # later. Current TradeIntentRequest is still taker-like, so this remains
        # conservative and generally blocked by edge.
        mid_edge = abs(forecast_gap)
        spread_adjusted = mid_edge - market.spread / 2.0
        if spread_adjusted > config.price_signal_min_edge:
            side = "YES" if forecast_gap > 0 else "NO"
            signal_type = "spread_capture"
            edge = spread_adjusted
            confidence = "low"
            reason = "spread-adjusted midpoint edge is positive"

    if side == "HOLD":
        blockers.append("no_signal")
    if edge < config.price_signal_min_edge:
        blockers.append("edge_below_price_threshold")
    if confidence == "low" and signal_type != "forecast_edge":
        blockers.append("low_confidence_price_signal")
    if would_contradict_strong_forecast(side, signals):
        blockers.append("forecast_contradiction")

    suggested_size = config.price_max_trade_size if not blockers else 0
    if config.live_guard_mode:
        suggested_size = min(suggested_size, 1)
    return ShortHorizonSignal(
        market_id=market.market_id,
        side=side,
        signal_type=signal_type,
        expected_tick_edge=edge,
        confidence=confidence,
        reason=reason,
        blockers=list(dict.fromkeys(blockers)),
        suggested_size=suggested_size,
    )


def price_signal_to_decision(
    market: MarketView,
    signal: ShortHorizonSignal,
    signals: ForecastSignals,
) -> TradeDecision | None:
    if signal.side == "HOLD" or signal.blockers or signal.suggested_size <= 0:
        return None
    price = market.yes_ask if signal.side == "YES" else market.no_ask
    return TradeDecision(
        market_id=market.market_id,
        action="BUY",
        side=signal.side,
        shares=signal.suggested_size,
        price=price,
        edge=signal.expected_tick_edge,
        expected_value=signal.expected_tick_edge * signal.suggested_size,
        confidence=signal.confidence,
        reason=f"price_signal:{signal.signal_type}: {signal.reason}",
        p_final=signals.p_final_after_blf or signals.p_final or signals.p_market,
    )


def maybe_price_exit(
    market: MarketView,
    signals: ForecastSignals,
    position,
    history: list[dict[str, Any]],
    config: BotConfig,
    current_signal: ShortHorizonSignal | None = None,
) -> tuple[TradeDecision | None, dict[str, Any]]:
    if position is None:
        return None, {}
    side = position.side
    current_price = market.yes_bid if side == "YES" else market.no_bid
    entry = float(position.avg_entry_price)
    movement = current_price - entry
    held_ticks = count_position_seen_ticks(history, side)
    reasons = []
    if movement >= config.price_take_profit_ticks:
        reasons.append("take_profit")
    if movement <= config.price_stop_loss_ticks and held_ticks >= config.price_stop_loss_min_ticks:
        reasons.append("stop_loss")
    if current_signal is not None and current_signal.side not in {"HOLD", side} and not current_signal.blockers:
        if current_signal.confidence in {"medium", "high"} or held_ticks >= 5:
            reasons.append("signal_reversal")
    if market.spread > config.max_spread:
        reasons.append("stale_or_wide_quote")
    diag = {
        "market_id": market.market_id,
        "side": side,
        "shares": float(position.shares),
        "avg_entry_price": entry,
        "current_exit_price": current_price,
        "unrealized_movement": movement,
        "held_ticks_estimate": held_ticks,
        "exit_reasons": reasons,
    }
    if not reasons:
        return None, diag
    shares = min(int(float(position.shares)), config.price_max_position_per_market)
    if config.live_guard_mode:
        shares = min(shares, 1)
    if shares <= 0:
        return None, diag
    decision = TradeDecision(
        market_id=market.market_id,
        action="SELL",
        side=side,
        shares=shares,
        price=current_price,
        edge=movement,
        expected_value=movement * shares,
        confidence=signals.confidence,
        reason=";".join(f"price_exit:{reason}" for reason in reasons),
        p_final=signals.p_final_after_blf or signals.p_final or signals.p_market,
    )
    return decision, diag


def same_direction(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def would_contradict_strong_forecast(side: str, signals: ForecastSignals) -> bool:
    p_final = signals.p_final_after_blf or signals.p_final or signals.p_market
    if side == "YES" and p_final < signals.p_market - 0.04:
        return True
    if side == "NO" and p_final > signals.p_market + 0.04:
        return True
    return False


def is_live_tradable_signal(
    *,
    market: MarketView,
    signals: ForecastSignals,
    features,
    signal: ShortHorizonSignal,
    action: str = "BUY",
) -> tuple[bool, list[str]]:
    """Strict final gate for guarded live entries.

    This gate is deliberately narrower than dry-run diagnostics. It prevents
    rate-limit/fallback forecast artifacts from becoming live trades and allows
    only price-action entries with repeated quote history.
    """
    reasons: list[str] = []
    if action == "SELL":
        return True, []
    if signal.side == "HOLD" or signal.signal_type == "none":
        reasons.append("no_live_entry_signal")

    raw_flags = all_risk_flags(signals)
    normalized = normalized_live_flags(raw_flags, market)
    fallback_hits = sorted(flag for flag in raw_flags if any(marker in flag for marker in FALLBACK_RISK_MARKERS))
    if fallback_hits:
        reasons.extend(f"fallback_or_rate_limit_risk:{flag}" for flag in fallback_hits)

    if {"stale_evidence", "llm_uncertainty"} & set(normalized) and not is_pure_momentum_with_history(signal, features):
        reasons.append("normalized_risk_requires_price_only_momentum")

    if signal.signal_type == "forecast_edge":
        reasons.append("forecast_edge_not_live_entry")
    if signal.signal_type not in {"momentum", "mean_reversion"}:
        reasons.append(f"signal_type_not_live_allowed:{signal.signal_type}")

    if repeated_count(features) < 3:
        reasons.append("insufficient_repeated_history")
    if getattr(features, "mid_1_tick_ago", None) is None or getattr(features, "delta_1_tick", None) is None:
        reasons.append("missing_previous_mid")
    if market.spread > 0.005:
        reasons.append("spread_above_live_hard_cap")

    required_edge = max(0.003, 1.5 * market.spread)
    if signal.signal_type == "momentum" and signal.expected_tick_edge < required_edge:
        reasons.append("momentum_edge_below_spread_adjusted_minimum")

    if signal.signal_type == "mean_reversion":
        if repeated_count(features) < 4:
            reasons.append("mean_reversion_needs_four_repeats")
        if abs(getattr(features, "delta_1_tick", 0.0) or 0.0) < 0.006:
            reasons.append("mean_reversion_move_too_small")
        if market.spread > 0.003:
            reasons.append("mean_reversion_spread_too_wide")
        if signal.suggested_size > 1:
            reasons.append("mean_reversion_size_above_one")

    if signal.signal_type in FORECAST_DEPENDENT_SIGNALS:
        if signals.p_2402 is None:
            reasons.append("forecast_dependent_missing_p_2402")
        if signals.evidence_quality < 4:
            reasons.append("forecast_dependent_low_evidence_quality")
        if market_type(signals) == "sports_outcome" and not sports_quantitative_support(signals):
            reasons.append("sports_forecast_signal_without_quant_support")
        blf_package = signals.blf_package or {}
        if blf_package.get("attempted") and not blf_package.get("succeeded"):
            reasons.append("forecast_dependent_blf_selected_but_failed")

    return not reasons, list(dict.fromkeys(reasons))


def is_live_tradable_fresh_event_signal(
    *,
    market: MarketView,
    signals: ForecastSignals,
    features,
    portfolio: PortfolioResponse | None,
    config: BotConfig,
) -> FreshEventGateResult:
    reasons: list[str] = []
    if not_evaluated_by_forecast_budget(signals):
        return fresh_result(False, ["not_evaluated_by_forecast_budget"])
    p_final = signals.p_final_after_blf if signals.p_final_after_blf is not None else signals.p_final
    if p_final is None:
        reasons.append("fresh_event_blocked_missing_p_final")
        return fresh_result(False, reasons)

    raw_flags = all_risk_flags(signals)
    fallback_hits = sorted(flag for flag in raw_flags if any(marker in flag for marker in FALLBACK_RISK_MARKERS))
    if fallback_hits:
        reasons.append("fresh_event_blocked_fallback_or_rate_limit")
    if signals.evidence_quality < 4:
        reasons.append("fresh_event_blocked_low_evidence")
    if signals.confidence not in {"medium", "high"}:
        reasons.append("fresh_event_blocked_confidence")
    if market.spread > config.fresh_event_max_spread:
        reasons.append("fresh_event_blocked_spread_too_wide")

    edge_yes = p_final - market.yes_ask
    edge_no = (1.0 - p_final) - market.no_ask
    side = "YES" if edge_yes >= edge_no else "NO"
    edge = max(edge_yes, edge_no)
    threshold = 0.010
    if edge < threshold:
        reasons.append("fresh_event_blocked_edge_below_threshold")

    mtype = market_type(signals)
    if mtype == "sports_outcome":
        if not explicit_near_term_catalyst(market, signals) or not sports_quantitative_support(signals):
            reasons.append("fresh_event_blocked_sports_long_horizon")
        threshold = max(threshold, 0.015)
        if edge < threshold and "fresh_event_blocked_edge_below_threshold" not in reasons:
            reasons.append("fresh_event_blocked_edge_below_threshold")

    if not has_fresh_event_catalyst(market, signals):
        reasons.append("fresh_event_blocked_no_catalyst")

    positions = positions_by_market(portfolio)
    existing = positions.get(market.market_id)
    if existing is not None:
        if existing.side != side:
            reasons.append("fresh_event_blocked_position_risk")
        try:
            if float(existing.shares) >= config.price_max_position_per_market:
                reasons.append("fresh_event_blocked_position_risk")
        except (TypeError, ValueError):
            reasons.append("fresh_event_blocked_position_risk")
    if open_positions_count(portfolio) >= config.price_max_total_open_positions and existing is None:
        reasons.append("fresh_event_blocked_position_risk")

    allowed = not reasons
    return FreshEventGateResult(
        allowed=allowed,
        reasons=["fresh_event_allowed"] if allowed else list(dict.fromkeys(reasons)),
        channel="fresh_event",
        max_size=1,
        side=side if allowed else "HOLD",
        edge=edge,
        threshold=threshold,
    )


def is_live_tradable_forecast_mispricing_signal(
    *,
    market: MarketView,
    signals: ForecastSignals,
    features,
    portfolio: PortfolioResponse | None,
    config: BotConfig,
) -> ForecastMispricingGateResult:
    """Guarded micro-entry for clean post-shrinkage forecast mispricings.

    This gate intentionally ignores raw RAG probability and only uses the final
    post-aggregation probability. It is narrower than diagnostics and exists so
    clean, non-fallback forecast edges can trade one guarded share.
    """
    reasons: list[str] = []
    if not_evaluated_by_forecast_budget(signals):
        return forecast_mispricing_result(False, ["not_evaluated_by_forecast_budget"])
    p_final = signals.p_final_after_blf if signals.p_final_after_blf is not None else signals.p_final
    if p_final is None:
        reasons.append("forecast_mispricing_blocked_missing_p_final")
        return forecast_mispricing_result(False, reasons)

    raw_flags = all_risk_flags(signals)
    normalized = set(normalized_live_flags(raw_flags, market))
    metadata_only = bool((signals.evidence_package or {}).get("metadata_only_evidence"))
    cached_used = bool((signals.evidence_package or {}).get("cached_evidence_used"))
    if metadata_only:
        reasons.append("forecast_mispricing_blocked_metadata_only")
    blocking_raw_flags = raw_flags_block_live_forecast(raw_flags)
    if cached_used:
        blocking_raw_flags = raw_flags_block_live_forecast(
            {
                flag for flag in raw_flags
                if not flag.startswith("brave_") and flag not in {"cached_evidence_used"}
            }
        )
    if blocking_raw_flags or normalized & {
        "stale_evidence",
        "llm_uncertainty",
        "search_failure",
        "resolution_risk",
        "ambiguous_resolution",
    }:
        reasons.append("forecast_mispricing_blocked_fallback_or_rate_limit")
    min_quality = 3 if cached_used else 4
    if signals.evidence_quality < min_quality:
        reasons.append("forecast_mispricing_blocked_low_evidence")
    if signals.confidence not in {"medium", "high"}:
        reasons.append("forecast_mispricing_blocked_confidence")
    if market.spread > config.live_forecast_max_spread:
        reasons.append("forecast_mispricing_blocked_spread_too_wide")
    if resolution_is_risky(signals):
        reasons.append("forecast_mispricing_blocked_resolution_risk")
    if blf_selected_but_failed(signals):
        reasons.append("forecast_mispricing_blocked_blf_failed")

    edge_yes = p_final - market.yes_ask
    edge_no = (1.0 - p_final) - market.no_ask
    side = "YES" if edge_yes >= edge_no else "NO"
    edge = max(edge_yes, edge_no)

    mtype = market_type(signals)
    threshold = 0.008
    if mtype == "sports_outcome":
        if not sports_quantitative_support(signals):
            reasons.append("forecast_mispricing_blocked_sports_without_quant_support")
        threshold = 0.012
    if not blf_was_selected(signals):
        threshold += 0.004
    if edge < threshold:
        reasons.append("forecast_mispricing_blocked_edge_below_threshold")

    positions = positions_by_market(portfolio)
    existing = positions.get(market.market_id)
    if existing is not None:
        if existing.side != side:
            reasons.append("forecast_mispricing_blocked_position_risk")
        try:
            if float(existing.shares) >= config.price_max_position_per_market:
                reasons.append("forecast_mispricing_blocked_position_risk")
        except (TypeError, ValueError):
            reasons.append("forecast_mispricing_blocked_position_risk")
    if open_positions_count(portfolio) >= config.price_max_total_open_positions and existing is None:
        reasons.append("forecast_mispricing_blocked_position_risk")

    allowed = not reasons
    return ForecastMispricingGateResult(
        allowed=allowed,
        reasons=[
            "clean_forecast_mispricing_cached_evidence" if cached_used else "forecast_mispricing_allowed"
        ] if allowed else list(dict.fromkeys(reasons)),
        channel="clean_forecast_mispricing",
        max_size=1,
        side=side if allowed else "HOLD",
        edge=edge,
        threshold=threshold,
    )


def fresh_result(allowed: bool, reasons: list[str]) -> FreshEventGateResult:
    return FreshEventGateResult(
        allowed=allowed,
        reasons=list(dict.fromkeys(reasons)),
        channel="fresh_event",
        max_size=1,
        side="HOLD",
        edge=0.0,
        threshold=0.0,
    )


def forecast_mispricing_result(allowed: bool, reasons: list[str]) -> ForecastMispricingGateResult:
    return ForecastMispricingGateResult(
        allowed=allowed,
        reasons=list(dict.fromkeys(reasons)),
        channel="clean_forecast_mispricing",
        max_size=1,
        side="HOLD",
        edge=0.0,
        threshold=0.0,
    )


def is_pure_momentum_with_history(signal: ShortHorizonSignal, features) -> bool:
    return signal.signal_type == "momentum" and repeated_count(features) >= 3


def repeated_count(features) -> int:
    return int(getattr(features, "repeated_seen_count", 0) or 0)


def all_risk_flags(signals: ForecastSignals) -> set[str]:
    flags = set(str(flag) for flag in (signals.risk_flags or []))
    for package in (signals.evidence_package or {}, signals.blf_package or {}):
        for flag in package.get("risk_flags", []) or []:
            flags.add(str(flag))
        for key in ("fallback_reason", "llm_error_category", "scanner_error", "error"):
            value = package.get(key)
            if value:
                flags.add(str(value))
    return flags


def raw_flags_block_live_forecast(raw_flags: set[str]) -> bool:
    return any(any(marker in flag for marker in FALLBACK_RISK_MARKERS) for flag in raw_flags)


def not_evaluated_by_forecast_budget(signals: ForecastSignals) -> bool:
    flags = all_risk_flags(signals)
    return (
        "rag_skipped_budget" in flags
        and signals.p_2402 is None
        and signals.evidence_quality <= 0
        and not (signals.evidence_package or {}).get("llm_attempted")
    )


def normalized_live_flags(raw_flags: set[str], market: MarketView) -> list[str]:
    normalized: set[str] = set()
    for flag in raw_flags:
        if "stale" in flag:
            normalized.add("stale_evidence")
        if "rate_limit" in flag or "fallback" in flag or "llm" in flag:
            normalized.add("llm_uncertainty")
        if "timeout" in flag:
            normalized.add("search_failure")
        if "search_failure" in flag:
            normalized.add("search_failure")
        if "resolution_risk" in flag or "ambiguous_resolution" in flag:
            normalized.add(flag)
    if market.spread > 0.005:
        normalized.add("wide_spread")
    return sorted(normalized)


def market_type(signals: ForecastSignals) -> str:
    return str((signals.evidence_package or {}).get("market_type") or (signals.blf_package or {}).get("market_type") or "other")


def sports_quantitative_support(signals: ForecastSignals) -> bool:
    package = signals.evidence_package or {}
    if package.get("sports_quantitative_support", False):
        return True
    return package.get("sports_support_source_type") in {"sportsbook_odds", "odds_market"}


def resolution_is_risky(signals: ForecastSignals) -> bool:
    resolution_check = (signals.evidence_package or {}).get("resolution_check", {}) or {}
    if resolution_check.get("trade_blocker") is True:
        return True
    flags = set(str(flag) for flag in resolution_check.get("risk_flags", []) or [])
    flags.update(all_risk_flags(signals))
    return bool(flags & {"resolution_risk", "ambiguous_resolution"})


def blf_was_selected(signals: ForecastSignals) -> bool:
    package = signals.blf_package or {}
    return bool(package.get("attempted") or package.get("selected") or package.get("blf_selected"))


def blf_selected_but_failed(signals: ForecastSignals) -> bool:
    package = signals.blf_package or {}
    return blf_was_selected(signals) and not bool(package.get("succeeded"))


def has_fresh_event_catalyst(market: MarketView, signals: ForecastSignals) -> bool:
    package = signals.evidence_package or {}
    if any(package.get(key) for key in ("fresh_evidence", "catalyst", "near_term_catalyst")):
        return True
    if explicit_near_term_catalyst(market, signals):
        return True
    if signals.evidence_quality >= 4 and signals.confidence == "high" and not all_risk_flags(signals):
        return True
    text = " ".join(
        str(part or "").lower()
        for part in (market.question, market.description, (package.get("reasoning_summary") or ""))
    )
    catalyst_terms = (
        "announce",
        "announcement",
        "earnings",
        "court",
        "ruling",
        "regulatory",
        "approval",
        "product",
        "launch",
        "election",
        "macro",
        "cpi",
        "fed",
        "rate decision",
        "deadline",
    )
    return any(term in text for term in catalyst_terms)


def explicit_near_term_catalyst(market: MarketView, signals: ForecastSignals) -> bool:
    package = signals.evidence_package or {}
    if any(package.get(key) for key in ("fresh_evidence", "catalyst", "near_term_catalyst")):
        return True
    now = datetime.now(UTC)
    days = (market.resolution_time - now).days
    mtype = market_type(signals)
    return days <= 14 and mtype in {
        "future_announcement",
        "government_action",
        "appointment_nomination",
        "geopolitical_action",
        "award_outcome",
        "election_result",
        "election_control",
        "scientific_milestone",
        "climate_metric",
        "other",
    }


def count_position_seen_ticks(history: list[dict[str, Any]], side: str) -> int:
    count = 0
    for row in reversed(history):
        position = row.get("current_position") or {}
        if position.get("side") == side and float(position.get("shares") or 0) > 0:
            count += 1
        else:
            break
    return count
