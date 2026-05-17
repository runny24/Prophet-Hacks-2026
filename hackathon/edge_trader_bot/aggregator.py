"""Forecast aggregation and shrinkage."""

from __future__ import annotations

from .math_utils import clamp_probability, logit, logit_mean, shrink_toward, sigmoid
from .schemas import ForecastSignals


def combine_signals(signals: ForecastSignals) -> ForecastSignals:
    # TODO: Replace manual weights with learned calibration once enough
    # resolved trades accumulate.
    base_logits: list[tuple[float, float]] = [
        (0.65, logit(signals.p_market)),
        (0.20, logit(signals.p_stat)),
    ]
    if signals.p_2402 is not None:
        base_logits.append((0.10, logit(signals.p_2402)))

    p_without_blf = weighted_logit_probability(base_logits)
    signals.p_final_before_blf = p_without_blf

    weighted_logits = list(base_logits)
    if signals.p_blf is not None:
        resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
        rag_agrees = signals.p_2402 is not None and abs(signals.p_blf - signals.p_2402) <= 0.08
        low_resolution_risk = not resolution_check.get("trade_blocker") and not resolution_check.get("risk_flags")
        if rag_agrees and low_resolution_risk:
            blf_weight = 0.20
            signals.aggregation_reason = "BLF agrees with RAG; modest verifier weight applied"
        else:
            blf_weight = 0.08
            signals.aggregation_reason = "BLF disagreement or resolution risk; shrunken verifier weight applied"
        weighted_logits.append((blf_weight, logit(signals.p_blf)))
    else:
        signals.aggregation_reason = "RAG/stat/market aggregation without BLF"

    p_raw = weighted_logit_probability(weighted_logits)

    if signals.confidence == "high":
        shrink_weight = 0.55
    elif signals.confidence == "medium":
        shrink_weight = 0.35
    else:
        shrink_weight = 0.20

    signals.p_final = shrink_toward(p_raw, signals.p_market, shrink_weight)
    signals.p_final_after_blf = signals.p_final
    signals.blf_adjustment = signals.p_final - p_without_blf
    return signals


def weighted_logit_probability(weighted_logits: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for weight, _ in weighted_logits)
    return sigmoid(sum(weight * value for weight, value in weighted_logits) / total_weight)


def aggregate_trials(probabilities: list[float], anchor: float) -> tuple[float, float]:
    if not probabilities:
        return clamp_probability(anchor), 0.10
    p = logit_mean(probabilities)
    if len(probabilities) == 1:
        return p, 0.10
    mean = sum(probabilities) / len(probabilities)
    variance = sum((x - mean) ** 2 for x in probabilities) / (len(probabilities) - 1)
    return p, variance ** 0.5
