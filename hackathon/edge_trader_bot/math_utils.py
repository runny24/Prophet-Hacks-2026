"""Probability math helpers."""

from __future__ import annotations

import math


def clamp_probability(p: float, eps: float = 1e-4) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def logit(p: float) -> float:
    p = clamp_probability(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit_mean(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.5
    return sigmoid(sum(logit(p) for p in probabilities) / len(probabilities))


def shrink_toward(p: float, anchor: float, weight: float) -> float:
    weight = min(max(weight, 0.0), 1.0)
    return clamp_probability(weight * p + (1.0 - weight) * anchor)
