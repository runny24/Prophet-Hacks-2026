"""2604-inspired BLF verifier.

Runs K independent trials. Each trial makes one LLM call (Perplexity/sonar
by default, which searches the web automatically) and asks for a structured
belief state update. Aggregates trial probabilities in logit space.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field

import requests

from .aggregator import aggregate_trials
from .math_utils import clamp_probability, logit_mean
from .schemas import ForecastSignals, MarketView

logger = logging.getLogger(__name__)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are an expert calibrated forecaster. "
    "Given a prediction market question, search for recent evidence and return "
    "a structured JSON belief state. "
    "Return strict JSON only — no markdown, no explanation outside the JSON."
)

_USER_TEMPLATE = """\
Prediction market question: {question}

Current market probability: {p_market:.1%}

Estimate the true probability this resolves YES. Search for recent news and \
evidence. Return JSON with exactly these fields:
{{
  "p_yes": <float 0-1>,
  "confidence": "<low|medium|high>",
  "evidence_for_yes": ["<1-3 key facts supporting YES>"],
  "evidence_for_no": ["<1-3 key facts supporting NO>"],
  "reasoning": "<1-2 sentence summary>"
}}"""


class BlfVerifier:
    def __init__(
        self,
        enabled: bool = False,
        max_steps: int = 4,
        trials: int = 3,
        timeout_seconds: int = 25,
    ) -> None:
        self.enabled = enabled
        self.max_steps = max_steps
        self.trials = trials
        self.timeout_seconds = timeout_seconds

    def verify(self, market: MarketView, signals: ForecastSignals) -> ForecastSignals:
        if not self.enabled:
            signals.risk_flags.append("blf_disabled")
            return signals

        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("EDGE_TRADER_BLF_MODEL", "perplexity/sonar")

        if not api_key:
            signals.risk_flags.append("blf_no_api_key")
            return signals

        trial_probs: list[float] = []
        trial_confidences: list[str] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self.trials) as pool:
            future_map = {
                pool.submit(self._run_trial, market, signals.p_market, api_key, model): i
                for i in range(self.trials)
            }
            try:
                for future in as_completed(future_map, timeout=self.timeout_seconds + 10):
                    trial_idx = future_map[future]
                    try:
                        result = future.result()
                        trial_probs.append(result["p_yes"])
                        trial_confidences.append(result.get("confidence", "low"))
                        logger.debug(
                            "BLF trial %d/%d market=%s p_yes=%.3f confidence=%s",
                            trial_idx + 1, self.trials, market.market_id,
                            result["p_yes"], result.get("confidence"),
                        )
                    except Exception as exc:
                        errors.append(f"trial_{trial_idx}: {type(exc).__name__}: {exc}")
                        logger.warning("BLF trial %d failed for %s: %s", trial_idx + 1, market.market_id, exc)
            except FuturesTimeoutError:
                errors.append("blf_trials_timeout")
                logger.warning("BLF trials timed out for %s after %ds", market.market_id, self.timeout_seconds + 10)

        if not trial_probs:
            signals.risk_flags.append("blf_all_trials_failed")
            signals.risk_flags.extend(errors[:3])
            return signals

        p_blf, uncertainty = aggregate_trials(trial_probs, signals.p_market)
        signals.p_blf = p_blf
        signals.uncertainty = max(0.02, min(uncertainty, 0.15))

        high_count = trial_confidences.count("high")
        medium_count = trial_confidences.count("medium")
        if high_count >= 2:
            signals.confidence = "high"
        elif high_count + medium_count >= 2:
            signals.confidence = "medium"

        if errors:
            signals.risk_flags.append(f"blf_partial_{len(errors)}_failed")

        logger.info(
            "BLF verified market=%s trials=%d/%d p_blf=%.3f confidence=%s",
            market.market_id, len(trial_probs), self.trials, p_blf, signals.confidence,
        )
        return signals

    def _run_trial(self, market: MarketView, p_market: float, api_key: str, model: str) -> dict:
        prompt = _USER_TEMPLATE.format(question=market.question, p_market=p_market)
        response = requests.post(
            OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/runny24/Prophet-Hacks-2026",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON object — handles markdown fences and trailing text
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]

        parsed = json.loads(content)
        p_yes = float(parsed["p_yes"])
        if not (0.0 <= p_yes <= 1.0):
            raise ValueError(f"p_yes out of range: {p_yes}")
        parsed["p_yes"] = clamp_probability(p_yes)
        return parsed
