"""Minimal 2604-inspired BLF verifier.

This first slice keeps BLF advisory: it produces a structured belief-state
package and a shrunken probability, but deterministic policy remains the only
component that can choose trades or sizes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import requests

from .math_utils import clamp_probability, shrink_toward
from .market_classifier import (
    classify_market,
    detect_absence_of_evidence_penalty,
    future_event_prompt_guidance,
)
from .rag_scanner import (
    LlmProviderConfig,
    OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    OPENROUTER_CHAT_COMPLETIONS_ENDPOINT,
    build_llm_headers,
)
from .schemas import ForecastSignals, MarketView

logger = logging.getLogger(__name__)


class BlfSummarizer(Protocol):
    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict[str, Any]:
        ...


class BlfError(RuntimeError):
    def __init__(self, category: str, message: str, *, json_parse_error: str | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.json_parse_error = json_parse_error


@dataclass(frozen=True)
class BlfRuntimeConfig:
    provider: str = "openrouter"
    model: str = "deepseek/deepseek-chat"
    timeout_seconds: int = 20
    json_retries: int = 1
    max_steps: int = 2
    enable_extra_search: bool = False


class ChatCompletionsBlfSummarizer:
    def __init__(self, provider_config: LlmProviderConfig, runtime: BlfRuntimeConfig) -> None:
        self.provider_config = provider_config
        self.runtime = runtime

    def verify(self, market: MarketView, signals: ForecastSignals, *, max_steps: int) -> dict[str, Any]:
        prompt = build_blf_prompt(market, signals, max_steps=max_steps)
        last_error: BlfError | None = None
        for _ in range(max(1, self.runtime.json_retries)):
            try:
                response = requests.post(
                    self.provider_config.endpoint,
                    headers=build_llm_headers(self.provider_config),
                    json={
                        "model": self.provider_config.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a conservative prediction-market belief-state verifier. "
                                    "Return strict JSON only. Do not include hidden reasoning or chain-of-thought."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.runtime.timeout_seconds,
                )
                if response.status_code == 429:
                    raise BlfError("rate_limit", "BLF provider rate limited request")
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise BlfError("malformed_response", str(exc)) from exc
                try:
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise BlfError("empty_response", "BLF response missing message content") from exc
                if not content:
                    raise BlfError("empty_response", "BLF response content is empty")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise BlfError("invalid_json", str(exc), json_parse_error=str(exc)) from exc
                return validate_blf_output(parsed)
            except requests.Timeout as exc:
                last_error = BlfError("timeout", str(exc))
            except requests.HTTPError as exc:
                last_error = BlfError("http_error", str(exc))
            except BlfError as exc:
                last_error = exc
            except Exception as exc:
                last_error = BlfError("unexpected_exception", str(exc))
            logger.warning("BLF summarization attempt failed: %s", last_error)
        raise last_error or BlfError("unexpected_exception", "BLF summarization failed")


class BlfVerifier:
    def __init__(
        self,
        enabled: bool = False,
        *,
        max_steps: int = 2,
        provider: str = "openrouter",
        model: str = "deepseek/deepseek-chat",
        timeout_seconds: int = 20,
        json_retries: int = 1,
        enable_extra_search: bool = False,
        summarizer: BlfSummarizer | None = None,
    ) -> None:
        self.enabled = enabled
        self.runtime = BlfRuntimeConfig(
            provider=provider,
            model=model,
            timeout_seconds=timeout_seconds,
            json_retries=json_retries,
            max_steps=max_steps,
            enable_extra_search=enable_extra_search,
        )
        self.summarizer = summarizer or build_default_blf_summarizer(self.runtime)

    def verify(self, market: MarketView, signals: ForecastSignals) -> ForecastSignals:
        started = time.monotonic()
        package = fallback_blf_package(market, signals, enabled=self.enabled)
        market_metadata = classify_market(market)
        package.update(market_metadata)
        if not self.enabled:
            signals.blf_package = package
            signals.risk_flags.append("blf_disabled")
            return signals

        package["attempted"] = True
        if self.summarizer is None:
            package.update(
                {
                    "fallback_reason": "missing_blf_provider_config",
                    "error": "missing_api_key",
                    "elapsed_ms": elapsed_ms(started),
                }
            )
            package["risk_flags"].append("blf_missing_api_key")
            signals.blf_package = package
            signals.risk_flags.append("blf_missing_api_key")
            return signals

        try:
            raw = self.summarizer.verify(market, signals, max_steps=self.runtime.max_steps)
            result = validate_blf_output(raw)
            p_raw = result["current_probability"]
            absence_diag = detect_absence_of_evidence_penalty(
                market_metadata=market_metadata,
                p_raw=p_raw,
                market_mid=market.yes_mid,
                reasoning_summary=result.get("reasoning_summary", ""),
                risk_flags=result.get("risk_flags", []),
                evidence_for_yes=result.get("evidence_for_yes", []),
                evidence_for_no=result.get("evidence_for_no", []),
                missing_information=result.get("missing_information", []),
            )
            if absence_diag["absence_of_evidence_penalty_detected"]:
                result["current_probability_original"] = p_raw
                result["current_probability"] = clamp_probability(float(absence_diag["guarded_probability"]), eps=0.02)
                result["risk_flags"] = merge_flags(result["risk_flags"], ["absence_as_negative_guardrail"])
            final_p = shrink_blf_probability(
                result["current_probability"],
                market.yes_mid,
                signals,
                result,
            )
            package.update(
                {
                    "succeeded": True,
                    "p_blf_raw": result["current_probability"],
                    "p_blf_raw_original": result.get("current_probability_original"),
                    "p_blf_final_after_shrinkage": final_p,
                    "confidence": result["confidence"],
                    "uncertainty": result["uncertainty"],
                    "belief_state": {
                        "prior_probability": signals.p_market,
                        "current_probability": final_p,
                        "evidence_for_yes": result["evidence_for_yes"],
                        "evidence_for_no": result["evidence_for_no"],
                        "key_uncertainties": result["key_uncertainties"],
                        "resolution_risks": result["resolution_risks"],
                        "missing_information": result["missing_information"],
                    },
                    "update_steps": build_update_steps(signals, result, self.runtime.max_steps),
                    "risk_flags": merge_flags(package["risk_flags"], result["risk_flags"]),
                    **absence_diag,
                    "elapsed_ms": elapsed_ms(started),
                }
            )
            signals.p_blf = final_p
            signals.confidence = combine_confidence(signals.confidence, result["confidence"])
            signals.uncertainty = max(signals.uncertainty, result["uncertainty"])
            signals.blf_package = package
            for flag in package["risk_flags"]:
                if flag not in signals.risk_flags:
                    signals.risk_flags.append(flag)
            return signals
        except BlfError as exc:
            logger.warning("Falling back to RAG forecast for BLF market %s: %s", market.market_id, exc)
            package.update(
                {
                    "fallback_reason": f"blf_{exc.category}",
                    "error": str(exc),
                    "json_parse_error": exc.json_parse_error,
                    "elapsed_ms": elapsed_ms(started),
                }
            )
            package["risk_flags"].append(f"blf_{exc.category}")
        except Exception as exc:
            logger.warning("BLF failed for %s: %s", market.market_id, exc)
            package.update(
                {
                    "fallback_reason": "blf_invalid_output",
                    "error": str(exc),
                    "elapsed_ms": elapsed_ms(started),
                }
            )
            package["risk_flags"].append("blf_invalid_output")

        signals.blf_package = package
        for flag in package["risk_flags"]:
            if flag not in signals.risk_flags:
                signals.risk_flags.append(flag)
        return signals


def build_default_blf_summarizer(runtime: BlfRuntimeConfig) -> BlfSummarizer | None:
    provider_config = build_blf_provider_config_from_env(runtime.provider, runtime.model)
    if provider_config is None:
        return None
    return ChatCompletionsBlfSummarizer(provider_config, runtime)


def build_blf_provider_config_from_env(provider: str, model: str) -> LlmProviderConfig | None:
    provider = provider.strip().lower()
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return LlmProviderConfig(provider=provider, api_key=api_key, model=model, endpoint=OPENROUTER_CHAT_COMPLETIONS_ENDPOINT)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return LlmProviderConfig(provider=provider, api_key=api_key, model=model, endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT)
    logger.warning("Unsupported BLF provider %s; BLF disabled for this tick", provider)
    return None


def fallback_blf_package(market: MarketView, signals: ForecastSignals, *, enabled: bool) -> dict[str, Any]:
    fallback_p = signals.p_2402 if signals.p_2402 is not None else signals.p_market
    market_metadata = classify_market(market)
    return {
        "market_id": market.market_id,
        **market_metadata,
        "enabled": enabled,
        "attempted": False,
        "succeeded": False,
        "p_blf_raw": None,
        "p_blf_raw_original": None,
        "p_blf_final_after_shrinkage": fallback_p,
        "confidence": "low",
        "uncertainty": signals.uncertainty,
        "belief_state": {
            "prior_probability": signals.p_market,
            "current_probability": fallback_p,
            "evidence_for_yes": [],
            "evidence_for_no": [],
            "key_uncertainties": [],
            "resolution_risks": [],
            "missing_information": [],
        },
        "update_steps": [],
        "risk_flags": [],
        "fallback_reason": None,
        "error": None,
        "json_parse_error": None,
        "absence_of_evidence_penalty_detected": False,
        "no_direct_evidence_reasoning": False,
        "future_event_should_anchor_to_market": market_metadata["future_event_should_anchor_to_market"],
        "strong_contrary_evidence_detected": False,
        "elapsed_ms": 0,
    }


def build_blf_prompt(market: MarketView, signals: ForecastSignals, *, max_steps: int) -> str:
    evidence_package = signals.evidence_package or {}
    market_metadata = classify_market(market)
    return json.dumps(
        {
            "task": "Maintain a conservative belief state for a binary prediction market.",
            "rules": [
                "Return strict JSON only.",
                "Do not provide chain-of-thought; use concise reasoning_summary only.",
                "Start from market midpoint unless it is missing.",
                "Treat RAG as evidence, not truth.",
                "Be skeptical of headline-only evidence.",
                "Explicitly consider exact resolution criteria.",
                "Avoid large probability moves unless evidence is strong and resolution-relevant.",
                "Do not choose trading action or share size.",
                *future_event_prompt_guidance(market_metadata),
                "For future action markets, ask whether evidence is about the current state or about future action probability.",
                "Use missing_information rather than strong negative evidence when evidence is absent.",
            ],
            "max_steps": max_steps,
            "market": {
                "market_id": market.market_id,
                "question": market.question,
                "description_or_resolution_criteria": market.description,
                "resolution_time": market.resolution_time.isoformat(),
                "market_midpoint": market.yes_mid,
                **market_metadata,
            },
            "priors": {
                "p_market": signals.p_market,
                "p_stat": signals.p_stat,
                "p_2402": signals.p_2402,
                "evidence_quality": signals.evidence_quality,
                "confidence": signals.confidence,
            },
            "rag_evidence": {
                "evidence_for_yes": evidence_package.get("evidence_for_yes", [])[:5],
                "evidence_for_no": evidence_package.get("evidence_for_no", [])[:5],
                "open_questions": evidence_package.get("open_questions", [])[:5],
                "reasoning_summary": evidence_package.get("reasoning_summary", ""),
            },
            "resolution_check": evidence_package.get("resolution_check", {}),
            "known_risk_flags": signals.risk_flags,
            "output_schema": {
                "current_probability": "number 0.0-1.0",
                "confidence": "low|medium|high",
                "uncertainty": "number 0.0-1.0",
                "evidence_for_yes": ["string"],
                "evidence_for_no": ["string"],
                "key_uncertainties": ["string"],
                "missing_information": ["string"],
                "resolution_risks": ["string"],
                "reasoning_summary": "short explanation, no chain-of-thought",
                "risk_flags": ["string"],
            },
        },
        default=str,
    )


def validate_blf_output(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BlfError("invalid_json", "BLF output must be a JSON object")
    required = {
        "current_probability",
        "confidence",
        "uncertainty",
        "evidence_for_yes",
        "evidence_for_no",
        "key_uncertainties",
        "missing_information",
        "resolution_risks",
        "reasoning_summary",
        "risk_flags",
    }
    missing = required - set(data)
    if missing:
        raise BlfError("missing_required_fields", f"BLF output missing fields: {sorted(missing)}")
    if data["confidence"] not in {"low", "medium", "high"}:
        raise BlfError("missing_required_fields", "BLF confidence must be low|medium|high")
    try:
        p = float(data["current_probability"])
        uncertainty = float(data["uncertainty"])
    except (TypeError, ValueError) as exc:
        raise BlfError("invalid_probability", "BLF probability and uncertainty must be numeric") from exc
    data["current_probability"] = clamp_probability(p, eps=0.02)
    data["uncertainty"] = max(0.0, min(1.0, uncertainty))
    for key in ("evidence_for_yes", "evidence_for_no", "key_uncertainties", "missing_information", "resolution_risks", "risk_flags"):
        if not isinstance(data[key], list):
            raise BlfError("missing_required_fields", f"BLF {key} must be an array")
        data[key] = [str(item)[:300] for item in data[key][:12]]
    data["reasoning_summary"] = str(data["reasoning_summary"])[:700]
    return data


def shrink_blf_probability(
    p_raw: float,
    market_mid: float,
    signals: ForecastSignals,
    result: dict[str, Any],
) -> float:
    confidence = result["confidence"]
    uncertainty = float(result["uncertainty"])
    quality = signals.evidence_quality
    resolution_check = (signals.evidence_package or {}).get("resolution_check", {})

    if confidence == "high":
        weight = 0.45
    elif confidence == "medium":
        weight = 0.30
    else:
        weight = 0.15
    if uncertainty >= 0.20:
        weight = min(weight, 0.15)
    if quality <= 2:
        weight = min(weight, 0.18)
    if signals.p_2402 is not None and abs(p_raw - signals.p_2402) >= 0.20:
        weight = min(weight, 0.12)
    if resolution_check.get("trade_blocker") or resolution_check.get("risk_flags"):
        weight = min(weight, 0.10)
    return shrink_toward(p_raw, clamp_probability(market_mid), weight)


def build_update_steps(signals: ForecastSignals, result: dict[str, Any], max_steps: int) -> list[dict[str, Any]]:
    before = signals.p_2402 if signals.p_2402 is not None else signals.p_market
    after = result["current_probability"]
    return [
        {
            "step": 1,
            "input_summary": "RAG evidence package and resolution-risk check",
            "probability_before": before,
            "probability_after": after,
            "delta": after - before,
            "reasoning_summary": result["reasoning_summary"],
            "new_search_query": None,
            "risk_flags": result["risk_flags"],
        }
    ][:max_steps]


def combine_confidence(existing: str, blf_confidence: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return existing if order.get(existing, 0) <= order.get(blf_confidence, 0) else blf_confidence


def merge_flags(*flag_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for flags in flag_lists:
        for flag in flags:
            if flag not in merged:
                merged.append(flag)
    return merged


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
