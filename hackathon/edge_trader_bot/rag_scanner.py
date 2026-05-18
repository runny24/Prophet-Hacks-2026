"""2402-lite RAG scanner.

This module implements the first live-safe slice of the 2402-style forecasting
pipeline: deterministic query generation, optional web search, evidence
normalization, resolution-risk checks, and a conservative rough probability.

If search is unavailable or fails, the scanner returns a structured fallback and
leaves ``p_2402`` unset so the trading policy naturally stays conservative.
"""

from __future__ import annotations

import logging
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Protocol

import requests

from .math_utils import clamp_probability, shrink_toward
from .market_classifier import (
    classify_market,
    detect_absence_of_evidence_penalty,
    future_event_prompt_guidance,
)
from .resolution_checker import check_resolution_risk
from .schemas import ForecastSignals, MarketView

logger = logging.getLogger(__name__)

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MarketInput = MarketView | Mapping[str, Any]

YES_TERMS = (
    "confirmed",
    "approved",
    "passes",
    "passed",
    "signed",
    "implemented",
    "wins",
    "won",
    "launches",
    "launched",
    "reaches",
    "exceeds",
    "above",
    "surpasses",
)

NO_TERMS = (
    "denied",
    "rejected",
    "fails",
    "failed",
    "blocked",
    "delayed",
    "cancelled",
    "canceled",
    "withdrawn",
    "below",
    "misses",
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""
    timestamp: str | None = None
    rank: int = 0


class SearchAdapter(Protocol):
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        ...


class LlmRagSummarizer(Protocol):
    def summarize(self, market: MarketView, evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
        ...


class NullSearchAdapter:
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return []


class BraveSearchError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class LlmRagError(RuntimeError):
    def __init__(self, category: str, message: str, *, json_parse_error: str | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.json_parse_error = json_parse_error


class BraveSearchAdapter:
    def __init__(self, api_key: str, timeout: float = 8.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params: dict[str, str | int] = {
            "q": query,
            "count": max_results,
            "country": "US",
            "search_lang": "en",
            "result_filter": "web",
        }
        try:
            response = requests.get(
                BRAVE_API_ENDPOINT,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                raise BraveSearchError("rate_limit", "Brave Search rate limited request")
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout as exc:
            raise BraveSearchError("timeout", str(exc)) from exc
        except requests.HTTPError as exc:
            raise BraveSearchError("http_error", str(exc)) from exc
        except ValueError as exc:
            raise BraveSearchError("malformed_response", str(exc)) from exc
        except BraveSearchError:
            raise
        except Exception as exc:
            raise BraveSearchError("unexpected_exception", str(exc)) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("web", {}), dict):
            raise BraveSearchError("malformed_response", "Brave response missing web object")
        results: list[SearchResult] = []
        for idx, item in enumerate(payload.get("web", {}).get("results", []), start=1):
            url = item.get("url") or ""
            if not url:
                continue
            results.append(
                SearchResult(
                    title=strip_html(item.get("title") or ""),
                    url=url,
                    snippet=strip_html(item.get("description") or ""),
                    source=domain_from_url(url),
                    timestamp=item.get("age"),
                    rank=idx,
                )
            )
        return results


OPENROUTER_CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"


@dataclass(frozen=True)
class LlmProviderConfig:
    provider: str
    api_key: str
    model: str
    endpoint: str


class ChatCompletionsJsonSummarizer:
    """Small optional JSON-only chat-completions summarizer.

    This is intentionally isolated from trading policy. It can summarize and
    score evidence, but it cannot choose trades or sizes.
    """

    def __init__(
        self,
        provider_config: LlmProviderConfig,
        *,
        timeout_seconds: int,
        json_retries: int,
        max_items: int,
    ) -> None:
        self.provider_config = provider_config
        self.timeout_seconds = timeout_seconds
        self.json_retries = max(1, json_retries)
        self.max_items = max_items

    def summarize(self, market: MarketView, evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_llm_rag_prompt(market, evidence_items[: self.max_items])
        last_error: LlmRagError | None = None
        for _ in range(self.json_retries):
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
                                    "You are a prediction-market evidence analyst. "
                                    "Return strict JSON only. Do not include hidden reasoning or chain-of-thought."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429:
                    raise LlmRagError("rate_limit", "LLM provider rate limited request")
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise LlmRagError("malformed_response", str(exc)) from exc
                try:
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise LlmRagError("empty_response", "LLM response missing message content") from exc
                if not content:
                    raise LlmRagError("empty_response", "LLM response content is empty")
                logger.info(
                    "LLM RAG raw content preview market=%s len=%d: %.400s",
                    market.market_id,
                    len(content),
                    content[:400].replace("\n", " "),
                )
                parsed = extract_json_from_llm_content(content, market_id=market.market_id)
                return validate_llm_rag_output(parsed)
            except requests.Timeout as exc:
                last_error = LlmRagError("timeout", str(exc))
            except requests.HTTPError as exc:
                last_error = LlmRagError("http_error", f"{exc} body={exc.response.text[:300] if exc.response is not None else ''}")
            except LlmRagError as exc:
                last_error = exc
            except Exception as exc:
                last_error = LlmRagError("unexpected_exception", str(exc))
                logger.warning("LLM RAG unexpected exception for %s", market.market_id, exc_info=True)
            logger.warning("LLM RAG summarization attempt failed market=%s: %s", market.market_id, last_error)
        raise last_error or LlmRagError("unexpected_exception", "LLM RAG summarization failed")


class RagScanner:
    def __init__(
        self,
        enabled: bool = False,
        search_adapter: SearchAdapter | None = None,
        llm_summarizer: LlmRagSummarizer | None = None,
        max_queries: int = 3,
        max_results_per_query: int = 5,
        enable_llm_summary: bool = False,
    ) -> None:
        self.enabled = enabled
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query
        self.search_adapter = search_adapter or build_default_search_adapter()
        self.enable_llm_summary = enable_llm_summary
        self.llm_summarizer = llm_summarizer or build_default_llm_summarizer()

    def scan(self, market: MarketInput, signals: ForecastSignals) -> ForecastSignals:
        market = normalize_market(market)
        package = self.scan_package(market, market_mid=signals.p_market)
        signals.evidence_package = package
        signals.risk_flags.extend(
            flag for flag in package["risk_flags"] if flag not in signals.risk_flags
        )

        if package["p_2402"] is not None:
            signals.p_2402 = package["p_2402"]
            signals.evidence_quality = int(package["evidence_quality"])
            signals.confidence = package["confidence"]
            signals.uncertainty = uncertainty_from_quality(signals.evidence_quality, signals.confidence)
            signals.reason = "2402-lite scanner evidence package"
        return signals

    def scan_package(self, market: MarketInput, *, market_mid: float | None = None) -> dict[str, Any]:
        started = time.monotonic()
        market = normalize_market(market)
        base = fallback_package(market)
        market_metadata = classify_market(market)
        base.update(market_metadata)
        if not self.enabled:
            base["risk_flags"].append("rag_disabled")
            base["scanner_elapsed_ms"] = elapsed_ms(started)
            return base
        if isinstance(self.search_adapter, NullSearchAdapter):
            base["scanner_error"] = "missing_key: BRAVE_SEARCH_API_KEY"
            base["risk_flags"].append("missing_brave_api_key")
            base["scanner_elapsed_ms"] = elapsed_ms(started)
            return base

        try:
            queries = generate_queries(market)[: self.max_queries]
            raw_results, results = collect_results(
                self.search_adapter,
                queries,
                max_results_per_query=self.max_results_per_query,
            )
            evidence_items = build_evidence_items(results, market)
            resolution_check = check_resolution_risk(market, evidence_items)
            risk_flags = list(dict.fromkeys([*resolution_check["flags"]]))

            if not evidence_items:
                base["risk_flags"].extend(["no_search_results", *risk_flags])
                base["resolution_check"] = resolution_check
                base["queries"] = queries
                base["search_queries"] = queries
                base["raw_search_result_count"] = len(raw_results)
                base["deduped_search_result_count"] = len(results)
                base["result_count"] = len(results)
                base["scanner_elapsed_ms"] = elapsed_ms(started)
                return base

            llm_package = None
            llm_diagnostics = llm_default_diagnostics()
            llm_diagnostics.update(market_metadata)
            if self.enable_llm_summary and self.llm_summarizer is not None:
                llm_started = time.monotonic()
                llm_diagnostics.update(
                    {
                        "llm_attempted": True,
                        "llm_prompt_metadata": llm_prompt_metadata(market, evidence_items, self.max_results_per_query),
                    }
                )
                try:
                    raw_llm_package = self.llm_summarizer.summarize(market, evidence_items)
                    llm_package = validate_llm_rag_output(raw_llm_package)
                    llm_diagnostics.update({"llm_succeeded": True, "llm_failed": False})
                    evidence_items = evidence_items_from_llm(llm_package)
                    resolution_check = check_resolution_risk(market, evidence_items)
                    risk_flags = merge_flags(resolution_check.get("risk_flags", []), llm_package.get("risk_flags", []))
                except LlmRagError as exc:
                    logger.warning("Falling back to heuristic RAG package for %s: %s", market.market_id, exc)
                    llm_diagnostics.update(
                        {
                            "llm_failed": True,
                            "llm_fallback": True,
                            "llm_error_category": exc.category,
                            "json_parse_error": exc.json_parse_error,
                            "fallback_reason": f"llm_{exc.category}",
                        }
                    )
                    risk_flags = merge_flags(risk_flags, ["llm_rag_failed", f"llm_{exc.category}"])
                except Exception as exc:
                    logger.warning("Falling back to heuristic RAG package for %s: %s", market.market_id, exc)
                    llm_diagnostics.update(
                        {
                            "llm_failed": True,
                            "llm_fallback": True,
                            "llm_error_category": "invalid_output",
                            "fallback_reason": "llm_invalid_output",
                        }
                    )
                    risk_flags = merge_flags(risk_flags, ["llm_rag_failed", "llm_invalid_output"])
                finally:
                    llm_diagnostics["llm_elapsed_ms"] = elapsed_ms(llm_started)
            elif self.enable_llm_summary:
                llm_diagnostics.update(
                    {
                        "llm_fallback": True,
                        "llm_error_category": "missing_api_key",
                        "fallback_reason": "missing_llm_provider_config",
                    }
                )
                risk_flags = merge_flags(risk_flags, ["llm_rag_unavailable", "llm_missing_api_key"])

            p_rough = estimate_probability(market_mid or market.yes_mid, evidence_items, resolution_check)
            evidence_quality = evidence_quality_score(evidence_items, resolution_check)
            confidence = confidence_from_quality(evidence_quality, resolution_check)
            split = split_evidence(evidence_items)

            if llm_package is not None:
                llm_p = clamp_probability(float(llm_package["p_2402_raw"]), eps=0.02)
                absence_diag = detect_absence_of_evidence_penalty(
                    market_metadata=market_metadata,
                    p_raw=llm_p,
                    market_mid=market_mid or market.yes_mid,
                    reasoning_summary=str(llm_package.get("reasoning_summary", "")),
                    risk_flags=llm_package.get("risk_flags", []),
                    evidence_for_yes=llm_package.get("evidence_for_yes", []),
                    evidence_for_no=llm_package.get("evidence_for_no", []),
                    missing_information=llm_package.get("open_questions", []),
                )
                llm_diagnostics.update(absence_diag)
                if absence_diag["absence_of_evidence_penalty_detected"]:
                    llm_diagnostics["p_2402_raw_original"] = llm_p
                    llm_p = clamp_probability(float(absence_diag["guarded_probability"]), eps=0.02)
                    llm_package["p_2402_raw"] = llm_p
                    risk_flags = merge_flags(risk_flags, ["absence_as_negative_guardrail"])
                sports_diag = sports_overconfidence_diagnostics(
                    market_metadata,
                    p_raw=llm_p,
                    market_mid=market_mid or market.yes_mid,
                    llm_package=llm_package,
                )
                llm_diagnostics.update(sports_diag)
                if sports_diag["sports_llm_overconfidence_detected"]:
                    risk_flags = merge_flags(risk_flags, ["sports_llm_overconfidence"])
                llm_quality = int(llm_package["evidence_quality"])
                llm_confidence = str(llm_package["confidence"])
                p_rough = shrink_llm_probability(
                    llm_p,
                    market_mid or market.yes_mid,
                    llm_quality,
                    llm_confidence,
                    resolution_check,
                    market_metadata=market_metadata,
                    quantitative_support=sports_diag["sports_quantitative_support"],
                )
                llm_diagnostics["p_2402_raw"] = llm_p
                llm_diagnostics["p_2402_final_after_shrinkage"] = p_rough
                evidence_quality = min(llm_quality, evidence_quality_score(evidence_items, resolution_check))
                confidence = confidence_from_quality(evidence_quality, resolution_check)
                split = {
                    "yes": sanitize_llm_evidence(llm_package.get("evidence_for_yes", [])),
                    "no": sanitize_llm_evidence(llm_package.get("evidence_for_no", [])),
                }

            return {
                "market_id": market.market_id,
                "question": market.question,
                **market_metadata,
                "p_2402": p_rough,
                "evidence_quality": evidence_quality,
                "confidence": confidence,
                "evidence_for_yes": split["yes"],
                "evidence_for_no": split["no"],
                "open_questions": infer_open_questions(market, resolution_check),
                "resolution_check": resolution_check,
                "risk_flags": risk_flags,
                "scanner_error": None,
                "queries": queries,
                "search_queries": queries,
                "result_count": len(results),
                "raw_search_result_count": len(raw_results),
                "deduped_search_result_count": len(results),
                "llm_summary_used": llm_package is not None,
                "reasoning_summary": (llm_package or {}).get("reasoning_summary", ""),
                **llm_diagnostics,
                "scanner_elapsed_ms": elapsed_ms(started),
            }
        except BraveSearchError as exc:
            logger.warning("RAG scanner Brave failure for %s: %s", market.market_id, exc)
            base["scanner_error"] = f"{exc.category}: {exc}"
            base["risk_flags"].append(f"brave_{exc.category}")
            base["scanner_elapsed_ms"] = elapsed_ms(started)
            return base
        except Exception as exc:
            logger.warning("RAG scanner failed for %s: %s", market.market_id, exc)
            base["scanner_error"] = f"{type(exc).__name__}: {exc}"
            base["risk_flags"].append("rag_scanner_error")
            base["scanner_elapsed_ms"] = elapsed_ms(started)
            return base


def build_default_search_adapter() -> SearchAdapter:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if api_key:
        return BraveSearchAdapter(api_key)
    return NullSearchAdapter()


def build_default_llm_summarizer() -> LlmRagSummarizer | None:
    provider_config = build_llm_provider_config_from_env()
    if provider_config is None:
        return None
    return ChatCompletionsJsonSummarizer(
        provider_config,
        timeout_seconds=int(os.getenv("EDGE_TRADER_LLM_TIMEOUT_SECONDS", "20")),
        json_retries=int(os.getenv("EDGE_TRADER_LLM_JSON_RETRIES", "1")),
        max_items=int(os.getenv("EDGE_TRADER_MAX_LLM_EVIDENCE_ITEMS", "5")),
    )


def build_llm_provider_config_from_env() -> LlmProviderConfig | None:
    provider = os.getenv("EDGE_TRADER_LLM_PROVIDER", "openrouter").strip().lower()
    model = os.getenv("EDGE_TRADER_LLM_MODEL", "deepseek/deepseek-chat")

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return LlmProviderConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            endpoint=OPENROUTER_CHAT_COMPLETIONS_ENDPOINT,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return LlmProviderConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
        )

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        return LlmProviderConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            endpoint=DEEPSEEK_CHAT_COMPLETIONS_ENDPOINT,
        )

    logger.warning("Unsupported LLM provider %s; LLM RAG summarizer disabled", provider)
    return None


def build_llm_headers(config: LlmProviderConfig) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if config.provider == "openrouter":
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "edge_trader_bot")
    return headers


def normalize_market(market: MarketInput) -> MarketView:
    if isinstance(market, MarketView):
        return market

    resolution_time = market["resolution_time"]
    if isinstance(resolution_time, str):
        resolution_time = datetime.fromisoformat(resolution_time)
    if resolution_time.tzinfo is None:
        resolution_time = resolution_time.replace(tzinfo=UTC)

    yes_bid = float(market["yes_bid"])
    yes_ask = float(market["yes_ask"])
    yes_mid = float(market.get("yes_mid", (yes_bid + yes_ask) / 2.0))
    no_bid = float(market.get("no_bid", 1.0 - yes_ask))
    no_ask = float(market.get("no_ask", 1.0 - yes_bid))

    return MarketView(
        market_id=str(market["market_id"]),
        question=str(market["question"]),
        description=market.get("description"),
        resolution_time=resolution_time,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=yes_mid,
        no_bid=no_bid,
        no_ask=no_ask,
        no_mid=float(market.get("no_mid", 1.0 - yes_mid)),
        spread=float(market.get("spread", yes_ask - yes_bid)),
        volume_24h=float(market.get("volume_24h", 0.0)),
        topic=market.get("topic"),
        family=market.get("family"),
        source_url=market.get("source_url"),
    )


def fallback_package(market: MarketView) -> dict[str, Any]:
    market_metadata = classify_market(market)
    return {
        "market_id": market.market_id,
        "question": market.question,
        **market_metadata,
        "p_2402": None,
        "evidence_quality": 0,
        "confidence": "low",
        "evidence_for_yes": [],
        "evidence_for_no": [],
        "open_questions": ["No external RAG evidence available."],
        "resolution_check": {},
        "risk_flags": [],
        "scanner_error": None,
        "queries": [],
        "search_queries": [],
        "result_count": 0,
        "raw_search_result_count": 0,
        "deduped_search_result_count": 0,
        "llm_summary_used": False,
        "reasoning_summary": "",
        **llm_default_diagnostics(),
        "scanner_elapsed_ms": 0,
    }


def llm_default_diagnostics() -> dict[str, Any]:
    return {
        "llm_attempted": False,
        "llm_succeeded": False,
        "llm_failed": False,
        "llm_fallback": False,
        "llm_error_category": None,
        "llm_elapsed_ms": 0,
        "p_2402_raw": None,
        "p_2402_raw_original": None,
        "p_2402_final_after_shrinkage": None,
        "absence_of_evidence_penalty_detected": False,
        "no_direct_evidence_reasoning": False,
        "future_event_should_anchor_to_market": False,
        "strong_contrary_evidence_detected": False,
        "sports_llm_overconfidence_detected": False,
        "sports_quantitative_support": False,
        "sports_support_source_type": "none",
        "sports_risk_explanation": None,
        "json_parse_error": None,
        "fallback_reason": None,
        "llm_prompt_metadata": {},
    }


def llm_prompt_metadata(
    market: MarketView,
    evidence_items: list[dict[str, Any]],
    max_items: int,
) -> dict[str, Any]:
    return {
        "market_id": market.market_id,
        "evidence_items_available": len(evidence_items),
        "evidence_items_sent": min(len(evidence_items), max_items),
        "has_description": bool(market.description),
        "resolution_time": market.resolution_time.isoformat(),
    }


def generate_queries(market: MarketView) -> list[str]:
    question = compact_query_text(market.question)
    label = compact_query_text(market.description or "")
    topic = compact_query_text(" ".join(x for x in [market.topic or "", market.family or ""] if x))

    queries = [
        question,
        f"{question} latest official",
    ]
    if label:
        queries.append(f"{question} {label[:80]}")
    if topic:
        queries.append(f"{question} {topic}")

    deduped: list[str] = []
    for query in queries:
        query = query.strip()
        if query and query.lower() not in {q.lower() for q in deduped}:
            deduped.append(query[:220])
    return deduped


def collect_results(
    search_adapter: SearchAdapter,
    queries: list[str],
    *,
    max_results_per_query: int,
) -> tuple[list[SearchResult], list[SearchResult]]:
    seen: set[str] = set()
    raw_results: list[SearchResult] = []
    results: list[SearchResult] = []
    for query in queries:
        for result in search_adapter.search(query, max_results=max_results_per_query):
            raw_results.append(result)
            normalized_url = result.url.rstrip("/")
            if normalized_url in seen:
                continue
            seen.add(normalized_url)
            results.append(result)
    return raw_results, results


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def build_evidence_items(results: list[SearchResult], market: MarketView) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    question_terms = token_set(market.question)
    for result in results:
        text = f"{result.title}. {result.snippet}".strip()
        relevance = relevance_score(question_terms, text, result.rank)
        if relevance <= 0:
            continue
        items.append(
            {
                "summary": text[:700],
                "source": result.source or domain_from_url(result.url),
                "url": result.url,
                "timestamp": result.timestamp,
                "relevance": relevance,
                "stance": evidence_stance(text),
            }
        )
    return sorted(items, key=lambda item: item["relevance"], reverse=True)


def split_evidence(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    yes_items = [{k: v for k, v in item.items() if k != "stance"} for item in items if item["stance"] == "yes"]
    no_items = [{k: v for k, v in item.items() if k != "stance"} for item in items if item["stance"] == "no"]
    neutral_items = [{k: v for k, v in item.items() if k != "stance"} for item in items if item["stance"] == "neutral"]
    return {
        "yes": (yes_items + neutral_items)[:8],
        "no": no_items[:8],
    }


def estimate_probability(
    market_mid: float,
    evidence_items: list[dict[str, Any]],
    resolution_check: dict[str, Any],
) -> float:
    yes_score = sum(item["relevance"] for item in evidence_items if item["stance"] == "yes")
    no_score = sum(item["relevance"] for item in evidence_items if item["stance"] == "no")
    total = yes_score + no_score
    if total <= 0:
        evidence_p = 0.5
    else:
        evidence_p = yes_score / total

    # Keep the scanner a ranking signal, not a trading oracle.
    weight = 0.20
    if resolution_check.get("risk_level") == "high":
        weight = 0.08
    elif resolution_check.get("risk_level") == "medium":
        weight = 0.12
    return shrink_toward(evidence_p, clamp_probability(market_mid), weight)


def shrink_llm_probability(
    p_raw: float,
    market_mid: float,
    evidence_quality: int,
    confidence: str,
    resolution_check: dict[str, Any],
    market_metadata: dict[str, Any] | None = None,
    quantitative_support: bool = False,
) -> float:
    if evidence_quality <= 2:
        weight = 0.15
    elif confidence == "high":
        weight = 0.45
    elif confidence == "medium":
        weight = 0.30
    else:
        weight = 0.18

    if resolution_check.get("trade_blocker") or resolution_check.get("ambiguity_level") == "high":
        weight = min(weight, 0.10)
    elif resolution_check.get("ambiguity_level") == "medium":
        weight = min(weight, 0.20)
    if (market_metadata or {}).get("market_type") == "sports_outcome":
        if not quantitative_support:
            weight = min(weight, 0.08)
        elif abs(p_raw - market_mid) >= 0.12:
            weight = min(weight, 0.18)
    return shrink_toward(p_raw, clamp_probability(market_mid), weight)


def sports_overconfidence_diagnostics(
    market_metadata: dict[str, Any],
    *,
    p_raw: float,
    market_mid: float,
    llm_package: dict[str, Any],
) -> dict[str, Any]:
    if market_metadata.get("market_type") != "sports_outcome":
        return {
            "sports_llm_overconfidence_detected": False,
            "sports_quantitative_support": False,
            "sports_support_source_type": "none",
            "sports_risk_explanation": None,
        }
    support_type = sports_support_source_type(llm_package)
    quantitative = support_type in {
        "odds_market",
        "sportsbook_odds",
        "model_forecast",
        "elo_rating",
        "simulation",
        "quantitative_ranking",
    }
    disagreement = abs(p_raw - market_mid)
    overconfident = disagreement >= 0.08 and not quantitative
    if disagreement >= 0.12 and support_type not in {
        "odds_market",
        "sportsbook_odds",
        "model_forecast",
        "elo_rating",
        "simulation",
    }:
        overconfident = True
    return {
        "sports_llm_overconfidence_detected": overconfident,
        "sports_quantitative_support": quantitative,
        "sports_support_source_type": support_type,
        "sports_risk_explanation": (
            "RAG sports probability moved far from market without strong quantitative odds/model support."
            if overconfident
            else None
        ),
    }


def has_quantitative_sports_support(llm_package: dict[str, Any]) -> bool:
    return sports_support_source_type(llm_package) in {
        "odds_market",
        "sportsbook_odds",
        "model_forecast",
        "elo_rating",
        "simulation",
        "quantitative_ranking",
    }


def sports_support_source_type(llm_package: dict[str, Any]) -> str:
    support_items = [
        *(llm_package.get("evidence_for_yes") or []),
        *(llm_package.get("evidence_for_no") or []),
    ]
    text = json.dumps(support_items, default=str).lower()
    if not text or text == "[]":
        return "none"
    if any(term in text for term in ("prediction market", "market odds", "market-implied", "market implied")):
        return "odds_market"
    if any(term in text for term in ("sportsbook", "bookmaker", "betting odds", "implied odds")):
        return "sportsbook_odds"
    if any(term in text for term in ("model forecast", "forecast model", "statistical projection", "projected probability")):
        return "model_forecast"
    if any(term in text for term in ("elo", "team rating", "rating system")):
        return "elo_rating"
    if "simulation" in text or "monte carlo" in text:
        return "simulation"
    if any(term in text for term in ("quantitative ranking", "power rating", "ranked by", "statistical ranking")):
        return "quantitative_ranking"
    if any(term in text for term in ("favorite", "contender", "roster", "injury", "coach", "player", "qualitative")):
        return "qualitative_news"
    return "unclear"


def evidence_quality_score(
    evidence_items: list[dict[str, Any]],
    resolution_check: dict[str, Any],
) -> int:
    if not evidence_items:
        return 0
    top_relevance = max(int(item["relevance"]) for item in evidence_items)
    source_count = len({item["source"] for item in evidence_items if item.get("source")})
    score = min(5, max(top_relevance, min(5, source_count)))
    if resolution_check.get("risk_level") == "high":
        score = min(score, 2)
    elif resolution_check.get("risk_level") == "medium":
        score = min(score, 3)
    return int(score)


def confidence_from_quality(score: int, resolution_check: dict[str, Any]) -> str:
    if resolution_check.get("risk_level") == "high" or score <= 2:
        return "low"
    if score >= 5:
        return "high"
    return "medium"


def uncertainty_from_quality(score: int, confidence: str) -> float:
    if confidence == "high":
        return 0.04
    if confidence == "medium":
        return 0.06
    if score > 0:
        return 0.08
    return 0.10


def infer_open_questions(market: MarketView, resolution_check: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    flags = set(resolution_check.get("risk_flags", resolution_check.get("flags", [])))
    if "official_source_not_confirmed" in flags:
        questions.append("Is there an official source satisfying the resolution criteria?")
    if "headline_resolution_mismatch" in flags:
        questions.append("Does the evidence satisfy the exact contract, not just the headline?")
    if "deadline_sensitive" in flags:
        questions.append("Did the qualifying event occur before the market deadline?")
    if not questions:
        questions.append("Would deeper BLF verification find contrary or stricter resolution evidence?")
    return questions


def relevance_score(question_terms: set[str], text: str, rank: int) -> int:
    text_terms = token_set(text)
    overlap = len(question_terms & text_terms)
    score = min(5, overlap)
    if rank <= 2:
        score += 1
    return min(5, score)


def evidence_stance(text: str) -> str:
    lowered = text.lower()
    yes_hits = sum(1 for term in YES_TERMS if term in lowered)
    no_hits = sum(1 for term in NO_TERMS if term in lowered)
    if yes_hits > no_hits:
        return "yes"
    if no_hits > yes_hits:
        return "no"
    return "neutral"


def token_set(text: str) -> set[str]:
    stopwords = {
        "will",
        "the",
        "and",
        "for",
        "with",
        "before",
        "after",
        "this",
        "that",
        "from",
        "have",
        "has",
        "does",
        "into",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        if token not in stopwords
    }


def compact_query_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9$% .'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def domain_from_url(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1).lower() if match else ""


def build_llm_rag_prompt(market: MarketView, evidence_items: list[dict[str, Any]]) -> str:
    market_metadata = classify_market(market)
    normalized = [
        {
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "timestamp": item.get("timestamp"),
            "relevance": item.get("relevance", 0),
        }
        for item in evidence_items
    ]
    return json.dumps(
        {
            "task": "Summarize and judge relevance of search evidence for a binary prediction market.",
            "rules": [
                "Return strict JSON only.",
                "Do not provide long chain-of-thought.",
                "Use concise reasoning_summary only.",
                "Judge whether evidence supports the exact resolution condition, not just the headline.",
                "For sports_outcome markets, include base rate, field size, and current betting/odds context. Do not let narrative alone create a large probability move.",
                "For sports_outcome markets, treat the market midpoint as a strong prior unless evidence is quantitative and odds-aware.",
                *future_event_prompt_guidance(market_metadata),
            ],
            "market": {
                "market_id": market.market_id,
                "question": market.question,
                "description_or_resolution_criteria": market.description,
                "resolution_time": market.resolution_time.isoformat(),
                **market_metadata,
            },
            "search_results": normalized,
            "output_schema": {
                "evidence_for_yes": [
                    {
                        "summary": "string",
                        "source": "string",
                        "url": "string",
                        "relevance": "integer 0-5",
                        "supports_resolution_condition": "boolean",
                    }
                ],
                "evidence_for_no": [],
                "open_questions": ["string"],
                "p_2402_raw": "number 0.0-1.0",
                "confidence": "low|medium|high",
                "evidence_quality": "integer 0-5",
                "risk_flags": ["string"],
                "reasoning_summary": "short explanation, no chain-of-thought",
            },
        },
        default=str,
    )


def extract_json_from_llm_content(content: str, *, market_id: str = "") -> dict[str, Any]:
    """Extract a JSON object from LLM output that may include markdown fences or prose."""
    stripped = content.strip()
    # Direct parse (happy path — model obeyed response_format)
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` or ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stripped)
    if fence:
        try:
            result = json.loads(fence.group(1))
            if isinstance(result, dict):
                logger.info("LLM RAG JSON extracted from markdown fence market=%s", market_id)
                return result
        except json.JSONDecodeError:
            pass
    # Last resort: find outermost { ... }
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(stripped[start: end + 1])
            if isinstance(result, dict):
                logger.info("LLM RAG JSON extracted from brace scan market=%s", market_id)
                return result
        except json.JSONDecodeError as exc:
            raise LlmRagError(
                "invalid_json",
                f"Brace-scan JSON parse failed market={market_id}: {exc}",
                json_parse_error=str(exc),
            ) from exc
    raise LlmRagError(
        "invalid_json",
        f"No JSON object found in LLM content market={market_id} len={len(content)}: {stripped[:120]!r}",
    )


_LLM_RAG_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "p_2402_raw": ("p_estimate", "probability", "p_yes", "p_raw", "estimated_probability", "p_final"),
    "evidence_quality": ("quality", "evidence_score", "score"),
    "reasoning_summary": ("summary", "reasoning", "explanation", "rationale"),
    "evidence_for_yes": ("yes_evidence", "supporting_evidence", "evidence_yes", "for_yes"),
    "evidence_for_no": ("no_evidence", "opposing_evidence", "evidence_no", "counter_evidence", "for_no"),
    "open_questions": ("questions", "uncertainties", "open_issues", "unanswered_questions"),
    "risk_flags": ("flags", "risks", "risk_factors", "warnings"),
    "confidence": ("confidence_level", "confidence_score"),
}


def normalize_llm_rag_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Map common field-name variants to canonical names before validation."""
    for canonical, aliases in _LLM_RAG_FIELD_ALIASES.items():
        if canonical not in data:
            for alias in aliases:
                if alias in data:
                    data[canonical] = data[alias]
                    logger.debug("LLM RAG field alias %r → %r", alias, canonical)
                    break
    return data


def validate_llm_rag_output(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LlmRagError("invalid_json", "LLM output must be a JSON object")
    data = normalize_llm_rag_fields(data)
    required = {
        "evidence_for_yes",
        "evidence_for_no",
        "open_questions",
        "p_2402_raw",
        "confidence",
        "evidence_quality",
        "risk_flags",
        "reasoning_summary",
    }
    missing = required - set(data)
    if missing:
        raise LlmRagError("missing_required_fields", f"LLM output missing fields: {sorted(missing)}")
    if data["confidence"] not in {"low", "medium", "high"}:
        raise LlmRagError("missing_required_fields", "LLM confidence must be low|medium|high")
    try:
        raw_probability = float(data["p_2402_raw"])
    except (TypeError, ValueError) as exc:
        raise LlmRagError("invalid_probability", "LLM p_2402_raw must be numeric") from exc
    data["p_2402_raw"] = clamp_probability(raw_probability, eps=0.02)
    try:
        data["evidence_quality"] = int(max(0, min(5, int(data["evidence_quality"]))))
    except (TypeError, ValueError) as exc:
        raise LlmRagError("missing_required_fields", "LLM evidence_quality must be integer-like") from exc
    data["evidence_for_yes"] = sanitize_llm_evidence(data["evidence_for_yes"])
    data["evidence_for_no"] = sanitize_llm_evidence(data["evidence_for_no"])
    if not isinstance(data["open_questions"], list) or not isinstance(data["risk_flags"], list):
        raise LlmRagError("missing_required_fields", "LLM open_questions and risk_flags must be arrays")
    data["open_questions"] = [str(q)[:300] for q in data["open_questions"][:8]]
    data["risk_flags"] = [str(flag)[:80] for flag in data["risk_flags"][:12]]
    data["reasoning_summary"] = str(data["reasoning_summary"])[:700]
    return data


def sanitize_llm_evidence(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise LlmRagError("missing_required_fields", "LLM evidence fields must be arrays")
    sanitized: list[dict[str, Any]] = []
    for item in items[:12]:
        if not isinstance(item, dict):
            raise LlmRagError("missing_required_fields", "LLM evidence item must be object")
        try:
            relevance = int(max(0, min(5, int(item.get("relevance", 0)))))
        except (TypeError, ValueError) as exc:
            raise LlmRagError("missing_required_fields", "LLM evidence relevance must be integer-like") from exc
        sanitized.append(
            {
                "summary": str(item.get("summary", ""))[:700],
                "source": str(item.get("source", ""))[:160],
                "url": str(item.get("url", ""))[:500],
                "timestamp": item.get("timestamp"),
                "relevance": relevance,
                "supports_resolution_condition": bool(item.get("supports_resolution_condition", False)),
            }
        )
    return sanitized


def evidence_items_from_llm(package: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in package["evidence_for_yes"]:
        items.append({**item, "stance": "yes"})
    for item in package["evidence_for_no"]:
        items.append({**item, "stance": "no"})
    return items


def merge_flags(*flag_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for flags in flag_lists:
        for flag in flags:
            if flag not in merged:
                merged.append(flag)
    return merged
