"""Evaluation module for the forecasting track."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import Prediction, Submission


def load_submission(path: str | Path) -> Submission:
    """Load and validate a submission file."""
    data = json.loads(Path(path).read_text())
    return Submission.model_validate(data)


def load_actuals(path: str | Path) -> dict[str, Any]:
    """Load actual outcomes.

    Expected format: {"market_ticker": resolved_value, ...}. Binary forecasts
    accept 1.0/0.0 or Yes/No-style labels. Probability-distribution forecasts
    accept labels, lists, or resolved_outcome-style {"value": [...]} payloads.
    """
    data = json.loads(Path(path).read_text())
    return {str(k): v for k, v in data.items()}


def score(predictions: list[Prediction], actuals: dict[str, Any]) -> dict[str, Any]:
    """Score predictions against actual outcomes using Brier score.

    Binary forecasts use (p_yes - actual)^2. Probability distributions use the
    multiclass Brier score: sum((p_i - outcome_i)^2) across submitted markets.
    """
    matched = [p for p in predictions if p.market_ticker in actuals]
    if not matched:
        return {
            "n_predictions": len(predictions),
            "n_matched": 0,
            "brier_score": None,
        }
    brier = sum(_prediction_brier(p, actuals[p.market_ticker]) for p in matched) / len(matched)
    return {
        "n_predictions": len(predictions),
        "n_matched": len(matched),
        "brier_score": round(brier, 6),
    }


def _prediction_brier(prediction: Prediction, actual: Any) -> float:
    if prediction.probabilities:
        actual_market = _actual_market(actual)
        probabilities = {p.market: p.probability for p in prediction.probabilities}
        if actual_market not in probabilities:
            raise ValueError(
                f"Actual outcome {actual_market!r} missing from "
                f"{prediction.market_ticker} probabilities"
            )
        return sum(
            (probability - (1.0 if market == actual_market else 0.0)) ** 2
            for market, probability in probabilities.items()
        )

    if prediction.p_yes is None:
        raise ValueError(f"Prediction {prediction.market_ticker} has no probability")
    return (prediction.p_yes - _actual_binary(actual)) ** 2


def _actual_market(actual: Any) -> str:
    if isinstance(actual, dict) and "value" in actual:
        return _actual_market(actual["value"])
    if isinstance(actual, list):
        if not actual:
            raise ValueError("Actual outcome list is empty")
        return str(actual[0])
    return str(actual)


def _actual_binary(actual: Any) -> float:
    if isinstance(actual, dict) and "value" in actual:
        return _actual_binary(actual["value"])
    if isinstance(actual, list):
        if not actual:
            raise ValueError("Actual outcome list is empty")
        return _actual_binary(actual[0])
    if isinstance(actual, bool):
        return 1.0 if actual else 0.0
    if isinstance(actual, (int, float)):
        return float(actual)
    normalized = str(actual).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return 1.0
    if normalized in {"0", "false", "no", "n"}:
        return 0.0
    raise ValueError(f"Cannot convert actual outcome {actual!r} to binary")
