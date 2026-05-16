"""Example forecast agent server.

A minimal FastAPI agent that receives events from ``prophet forecast predict``
and returns calibrated outcome probability estimates using Claude.

Usage:
    # Install deps (if not already):  pip install fastapi uvicorn anthropic
    # Start the server:
    python -m ai_prophet.forecast.example_agent

    # In another terminal:
    prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Example Forecast Agent")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] | None = None


class MarketProbability(BaseModel):
    market: str
    probability: float


class PredictionResponse(BaseModel):
    probabilities: list[MarketProbability]


# ---------------------------------------------------------------------------
# Claude-based forecasting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert forecaster specialized in calibrated probability estimation.

Your task is to estimate a probability distribution over the possible markets
or outcomes for the given event.

CALIBRATION GUIDELINES:
- Consider base rates for similar events.
- Weight evidence by reliability and recency.
- Account for uncertainty — don't be overconfident.
- Extremes (p < 0.10 or p > 0.90) require very strong evidence.
- Probabilities must be decimals between 0 and 1 and should sum to 1.
- Use the exact market/outcome labels provided in the event.

Respond with ONLY valid JSON in this shape:
{"probabilities": [{"market": "<label>", "probability": <float>}]}
Do not include any other text outside the JSON object."""


def _build_user_prompt(event: EventRequest) -> str:
    parts = [f"Event: {event.title}"]
    if event.subtitle:
        parts.append(f"Subtitle: {event.subtitle}")
    if event.description:
        parts.append(f"Description: {event.description}")
    if event.rules:
        parts.append(f"Rules: {event.rules}")
    if event.outcomes:
        parts.append(f"Possible markets/outcomes: {', '.join(event.outcomes)}")
    parts.append(f"Category: {event.category}")
    parts.append(f"Close time: {event.close_time}")
    parts.append(
        "\nBased on your knowledge, what probability should be assigned to each "
        "possible market/outcome?"
    )
    return "\n".join(parts)


def _event_markets(event: EventRequest) -> list[str]:
    if event.outcomes:
        return event.outcomes
    return [event.market_ticker]


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env and source it, "
                "or export it directly: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def forecast_with_claude(event: EventRequest) -> PredictionResponse:
    """Call Claude to produce outcome probabilities for the event."""
    import json

    client = _get_client()
    response = client.messages.create(
        model=os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(event)}],
    )

    text = "\n".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    try:
        data = _parse_json_object(text)
        raw_probabilities = _coerce_probabilities(data)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Claude returned invalid probability JSON for %s; using uniform fallback: %s",
            event.market_ticker,
            exc,
        )
        markets = _event_markets(event)
        probability = 1.0 / len(markets)
        raw_probabilities = [
            {"market": market, "probability": probability}
            for market in markets
        ]

    expected_markets = set(_event_markets(event))
    probabilities = _normalize_probabilities(raw_probabilities, expected_markets)
    if not probabilities:
        raise ValueError("No probabilities returned for the event markets")

    total = sum(item.probability for item in probabilities)
    if total <= 0:
        raise ValueError("Claude returned probabilities that sum to zero")

    return PredictionResponse(
        probabilities=[
            MarketProbability(
                market=item.market,
                probability=item.probability / total,
            )
            for item in probabilities
        ]
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    import json

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in Claude response: {text[:200]!r}")

    return json.loads(text[start:end + 1])


def _coerce_probabilities(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the requested response shape and a couple of common JSON variants."""
    raw = data["probabilities"]
    if isinstance(raw, dict):
        return [
            {"market": market, "probability": probability}
            for market, probability in raw.items()
        ]
    if not isinstance(raw, list):
        raise TypeError("probabilities must be a list or object")
    return raw


def _normalize_probabilities(
    raw_probabilities: list[dict[str, Any]],
    expected_markets: set[str],
) -> list[MarketProbability]:
    values = [
        (str(item["market"]), float(item["probability"]))
        for item in raw_probabilities
        if not expected_markets or str(item["market"]) in expected_markets
    ]
    if any(probability > 1.0 for _market, probability in values):
        values = [(market, probability / 100) for market, probability in values]
    return [
        MarketProbability(
            market=market,
            probability=max(0.0, min(1.0, probability)),
        )
        for market, probability in values
    ]


# ---------------------------------------------------------------------------
# Local predict function (used by: prophet forecast predict --local)
# ---------------------------------------------------------------------------

def predict(event: dict) -> dict:
    """Predict function for --local mode.

    Args:
        event: Event dict with keys like market_ticker, title, category, etc.

    Returns:
        Dict with probabilities: [{"market": str, "probability": float}, ...].
    """
    event_req = EventRequest(**event)
    resp = forecast_with_claude(event_req)
    return resp.model_dump()


# ---------------------------------------------------------------------------
# Server endpoint (used by: prophet forecast predict --agent-url)
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    """Receive an event and return outcome probability forecasts."""
    logger.info("Forecasting %s: %s", event.market_ticker, event.title)
    return forecast_with_claude(event)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    uvicorn.run(
        "ai_prophet.forecast.example_agent:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
