"""Pydantic models for the forecasting track."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Event(BaseModel):
    """An event selected for forecasting."""

    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: datetime
    outcomes: list[str] | None = None
    resolved_outcome: dict[str, Any] | None = None


class MarketProbability(BaseModel):
    """A probability assigned to one outcome in a forecast event."""

    market: str
    probability: float = Field(ge=0.0, le=1.0)


class Prediction(BaseModel):
    """A single forecast for a market or event."""

    market_ticker: str
    p_yes: float | None = Field(default=None, ge=0.01, le=0.99)
    probabilities: list[MarketProbability] | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def _require_forecast(self) -> Prediction:
        if self.p_yes is None and not self.probabilities:
            raise ValueError("Prediction requires p_yes or probabilities")
        return self


class Submission(BaseModel):
    """A set of predictions for a day."""

    timestamp: datetime
    predictions: list[Prediction] = Field(min_length=1)
