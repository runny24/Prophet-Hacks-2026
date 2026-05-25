from __future__ import annotations

from ai_prophet_core.forecast.evaluate import score
from ai_prophet_core.forecast.schemas import Prediction


def test_score_binary_prediction():
    result = score(
        [Prediction(market_ticker="binary-task", p_yes=0.7)],
        {"binary-task": 1.0},
    )

    assert result["n_predictions"] == 1
    assert result["n_matched"] == 1
    assert result["brier_score"] == 0.09


def test_score_probability_distribution_prediction():
    result = score(
        [
            Prediction(
                market_ticker="multi-task",
                probabilities=[
                    {"market": "Pittsburgh", "probability": 0.68},
                    {"market": "Atlanta", "probability": 0.32},
                ],
            )
        ],
        {"multi-task": {"value": ["Pittsburgh"]}},
    )

    assert result["n_predictions"] == 1
    assert result["n_matched"] == 1
    assert result["brier_score"] == 0.2048
