"""Forecast aggregation and shrinkage."""

from __future__ import annotations

from .math_utils import clamp_probability, logit, logit_mean, shrink_toward, sigmoid
from .schemas import ForecastSignals


def combine_signals(signals: ForecastSignals) -> ForecastSignals:
    # TODO: Replace manual weights with learned calibration once enough
    # resolved trades accumulate.
    weighted_logits: list[tuple[float, float]] = [
        (0.65, logit(signals.p_market)),
        (0.20, logit(signals.p_stat)),
    ]
    if signals.p_2402 is not None:
        weighted_logits.append((0.10, logit(signals.p_2402)))
    if signals.p_blf is not None:
        weighted_logits.append((0.25, logit(signals.p_blf)))

    total_weight = sum(weight for weight, _ in weighted_logits)
    p_raw = sigmoid(sum(weight * value for weight, value in weighted_logits) / total_weight)

    if signals.confidence == "high":
        shrink_weight = 0.55
    elif signals.confidence == "medium":
        shrink_weight = 0.35
    else:
        shrink_weight = 0.20

    signals.p_final = shrink_toward(p_raw, signals.p_market, shrink_weight)
    return signals


def aggregate_trials(probabilities: list[float], anchor: float) -> tuple[float, float]:
    if not probabilities:
        return clamp_probability(anchor), 0.10
    p = logit_mean(probabilities)
    if len(probabilities) == 1:
        return p, 0.10
    mean = sum(probabilities) / len(probabilities)
    variance = sum((x - mean) ** 2 for x in probabilities) / (len(probabilities) - 1)
    return p, variance ** 0.5
