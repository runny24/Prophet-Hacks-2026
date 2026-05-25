from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import requests

from edge_trader_bot.rag_scanner import (
    ChatCompletionsJsonSummarizer,
    BraveSearchError,
    LlmProviderConfig,
    LlmRagError,
    RagScanner,
    SearchResult,
)
from edge_trader_bot.resolution_checker import check_resolution_risk
from edge_trader_bot.schemas import MarketView
from edge_trader_bot.stat_priors import initial_signals


class FakeSearchAdapter:
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                title="Regulator approved the merger",
                url="https://example.com/approved",
                snippet="Officials confirmed the merger was approved before the deadline.",
                source="example.com",
                timestamp="2 days ago",
                rank=1,
            ),
            SearchResult(
                title="Opposition says deal could be delayed",
                url="https://example.org/delay",
                snippet="Some analysts say the proposal may still face delays.",
                source="example.org",
                timestamp="1 day ago",
                rank=2,
            ),
        ][:max_results]


class SportsSearchAdapter:
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                title="France World Cup power ranking discussion",
                url="https://example.com/france-world-cup",
                snippet="Analysts discuss France as a strong World Cup contender, but without betting odds context.",
                source="example.com",
                timestamp="1 day ago",
                rank=1,
            )
        ][:max_results]


class FakeLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "Official source confirmed approval before the deadline.",
                    "source": "agency.gov",
                    "url": "https://agency.gov/approval",
                    "relevance": 5,
                    "supports_resolution_condition": True,
                }
            ],
            "evidence_for_no": [],
            "open_questions": ["Was the publication timestamp before the exact deadline?"],
            "p_2402_raw": 0.9,
            "confidence": "high",
            "evidence_quality": 5,
            "risk_flags": [],
            "reasoning_summary": "Official confirmation appears to satisfy the contract.",
        }


class BrokenLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        raise ValueError("bad json")


class InvalidJsonLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        raise LlmRagError("invalid_json", "bad json", json_parse_error="line 1")


class OverconfidentLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "Reports suggest approval is expected soon.",
                    "source": "example.com",
                    "url": "https://example.com/report",
                    "relevance": 5,
                    "supports_resolution_condition": False,
                }
            ],
            "evidence_for_no": [],
            "open_questions": [],
            "p_2402_raw": 1.5,
            "confidence": "high",
            "evidence_quality": 5,
            "risk_flags": [],
            "reasoning_summary": "Optimistic but not official.",
        }


class LowQualityLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "One weak mention says this could happen.",
                    "source": "example.com",
                    "url": "https://example.com/weak",
                    "relevance": 1,
                    "supports_resolution_condition": False,
                }
            ],
            "evidence_for_no": [],
            "open_questions": ["Need official confirmation."],
            "p_2402_raw": 0.9,
            "confidence": "low",
            "evidence_quality": 1,
            "risk_flags": [],
            "reasoning_summary": "Weak evidence only.",
        }


class NoDirectFutureLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [],
            "evidence_for_no": [],
            "open_questions": ["No direct announcement evidence yet."],
            "p_2402_raw": 0.02,
            "confidence": "low",
            "evidence_quality": 1,
            "risk_flags": ["No direct evidence that an announcement has happened yet."],
            "reasoning_summary": "No direct evidence of an announcement yet.",
        }


class ExplicitDeclineFutureLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [],
            "evidence_for_no": [
                {
                    "summary": "The person explicitly declined and ruled out running.",
                    "source": "example.com",
                    "url": "https://example.com/decline",
                    "relevance": 5,
                    "supports_resolution_condition": True,
                }
            ],
            "open_questions": [],
            "p_2402_raw": 0.02,
            "confidence": "medium",
            "evidence_quality": 4,
            "risk_flags": ["explicit_decline"],
            "reasoning_summary": "The person declined and ruled out the future announcement.",
        }


class SportsNarrativeLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "France has elite players and a strong recent tournament narrative.",
                    "source": "example.com",
                    "url": "https://example.com/france-world-cup",
                    "relevance": 5,
                    "supports_resolution_condition": False,
                }
            ],
            "evidence_for_no": [],
            "open_questions": ["Need base rates, field size, and current odds."],
            "p_2402_raw": 0.46,
            "confidence": "high",
            "evidence_quality": 5,
            "risk_flags": [],
            "reasoning_summary": "Narrative support, but no odds-aware quantitative support.",
        }


class SportsQuantitativeLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "Betting odds imply a 20% probability after accounting for field size.",
                    "source": "sportsbook.example",
                    "url": "https://sportsbook.example/world-cup",
                    "relevance": 5,
                    "supports_resolution_condition": True,
                }
            ],
            "evidence_for_no": [],
            "open_questions": [],
            "p_2402_raw": 0.20,
            "confidence": "medium",
            "evidence_quality": 5,
            "risk_flags": [],
            "reasoning_summary": "Odds-aware estimate uses betting odds, base rate, and field size.",
        }


class SportsSimulationLlmSummarizer:
    def summarize(self, market: MarketView, evidence_items: list[dict]) -> dict:
        return {
            "evidence_for_yes": [
                {
                    "summary": "A Monte Carlo simulation gives the team a 22% title probability.",
                    "source": "model.example",
                    "url": "https://model.example/world-cup",
                    "relevance": 5,
                    "supports_resolution_condition": True,
                }
            ],
            "evidence_for_no": [],
            "open_questions": [],
            "p_2402_raw": 0.22,
            "confidence": "medium",
            "evidence_quality": 5,
            "risk_flags": [],
            "reasoning_summary": "Simulation-based model forecast.",
        }


class EmptySearchAdapter:
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return []


class TimeoutSearchAdapter:
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        raise BraveSearchError("timeout", "timed out")


class FakeHttpResponse:
    def __init__(self, content: str, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def make_market(**overrides) -> MarketView:
    base = {
        "market_id": "test-market",
        "question": "Will the regulator approve the merger before June 1?",
        "description": "Resolves YES if official approval is published before the deadline.",
        "resolution_time": datetime.now(UTC) + timedelta(days=10),
        "yes_bid": 0.40,
        "yes_ask": 0.44,
        "yes_mid": 0.42,
        "no_bid": 0.56,
        "no_ask": 0.60,
        "no_mid": 0.58,
        "spread": 0.04,
        "volume_24h": 1000.0,
    }
    base.update(overrides)
    return MarketView(**base)


class RagScannerTests(unittest.TestCase):
    def test_disabled_scanner_returns_structured_fallback(self) -> None:
        market = make_market()
        package = RagScanner(enabled=False).scan_package(market)

        self.assertEqual(package["market_id"], market.market_id)
        self.assertIsNone(package["p_2402"])
        self.assertEqual(package["confidence"], "low")
        self.assertIn("rag_disabled", package["risk_flags"])
        self.assertIsNone(package["scanner_error"])

    def test_scanner_builds_evidence_package_with_fake_search(self) -> None:
        market = make_market()
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertEqual(package["market_id"], market.market_id)
        self.assertIsNotNone(package["p_2402"])
        self.assertGreaterEqual(package["evidence_quality"], 1)
        self.assertIn(package["confidence"], {"low", "medium", "high"})
        self.assertGreater(len(package["evidence_for_yes"]), 0)
        self.assertIn("resolution_check", package)
        self.assertIn("search_queries", package)
        self.assertEqual(package["raw_search_result_count"], 2)
        self.assertEqual(package["deduped_search_result_count"], 2)
        self.assertIn("scanner_elapsed_ms", package)

    def test_empty_search_results_return_structured_fallback(self) -> None:
        market = make_market()
        scanner = RagScanner(enabled=True, search_adapter=EmptySearchAdapter(), max_queries=1)

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertIsNone(package["p_2402"])
        self.assertIn("no_search_results", package["risk_flags"])
        self.assertIsNone(package["scanner_error"])
        self.assertEqual(package["raw_search_result_count"], 0)

    def test_brave_timeout_returns_structured_scanner_error(self) -> None:
        market = make_market()
        scanner = RagScanner(enabled=True, search_adapter=TimeoutSearchAdapter(), max_queries=1)

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertIsNone(package["p_2402"])
        self.assertIn("brave_timeout", package["risk_flags"])
        self.assertIn("timeout", package["scanner_error"])

    def test_llm_summary_is_optional_and_structured(self) -> None:
        market = make_market()
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=FakeLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertTrue(package["llm_summary_used"])
        self.assertIsNotNone(package["p_2402"])
        self.assertGreaterEqual(package["evidence_quality"], 1)
        self.assertIn("reasoning_summary", package)

    def test_llm_summary_failure_falls_back_to_heuristic_package(self) -> None:
        market = make_market()
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=BrokenLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertFalse(package["llm_summary_used"])
        self.assertIn("llm_rag_failed", package["risk_flags"])
        self.assertIsNotNone(package["p_2402"])
        self.assertTrue(package["llm_failed"])
        self.assertTrue(package["llm_fallback"])
        self.assertEqual(package["fallback_reason"], "llm_invalid_output")

    def test_openrouter_invalid_json_falls_back_to_heuristic_package(self) -> None:
        market = make_market()
        summarizer = ChatCompletionsJsonSummarizer(
            LlmProviderConfig(
                provider="openrouter",
                api_key="test-key",
                model="deepseek/deepseek-chat",
                endpoint="https://openrouter.test/chat/completions",
            ),
            timeout_seconds=1,
            json_retries=1,
            max_items=2,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=summarizer,
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        with patch("edge_trader_bot.rag_scanner.requests.post", return_value=FakeHttpResponse("{bad json")):
            package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertFalse(package["llm_summary_used"])
        self.assertTrue(package["llm_failed"])
        self.assertEqual(package["llm_error_category"], "invalid_json")
        self.assertEqual(package["fallback_reason"], "llm_invalid_json")
        self.assertIsNotNone(package["json_parse_error"])
        self.assertIsNotNone(package["p_2402"])

    def test_missing_openrouter_key_falls_back_to_heuristic_package(self) -> None:
        market = make_market()
        with patch.dict("os.environ", {"EDGE_TRADER_LLM_PROVIDER": "openrouter"}, clear=True):
            scanner = RagScanner(
                enabled=True,
                search_adapter=FakeSearchAdapter(),
                enable_llm_summary=True,
                max_queries=1,
                max_results_per_query=2,
            )
            package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertFalse(package["llm_attempted"])
        self.assertTrue(package["llm_fallback"])
        self.assertEqual(package["llm_error_category"], "missing_api_key")
        self.assertIn("llm_missing_api_key", package["risk_flags"])
        self.assertIsNotNone(package["p_2402"])

    def test_llm_probability_is_clamped_before_shrinkage(self) -> None:
        market = make_market()
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=OverconfidentLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertEqual(package["p_2402_raw"], 0.98)
        self.assertLessEqual(package["p_2402_final_after_shrinkage"], 0.98)

    def test_low_evidence_quality_shrinks_toward_market_mid(self) -> None:
        market = make_market(yes_mid=0.42)
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=LowQualityLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertEqual(package["confidence"], "low")
        self.assertLess(package["p_2402_final_after_shrinkage"], 0.5)

    def test_resolution_trade_blocker_limits_llm_optimism(self) -> None:
        market = make_market()
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=OverconfidentLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertTrue(package["resolution_check"]["trade_blocker"])
        self.assertIn("official_source_not_confirmed", package["risk_flags"])
        self.assertLess(package["p_2402_final_after_shrinkage"], 0.5)

    def test_future_candidacy_no_direct_evidence_anchors_near_market_prior(self) -> None:
        market = make_market(
            question="Who will run for the Republican presidential nomination in 2028?",
            description="Resolves YES if Marco Rubio announces a presidential campaign before 2028.",
            resolution_time=datetime.now(UTC) + timedelta(days=600),
            yes_mid=0.64,
            yes_bid=0.62,
            yes_ask=0.66,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=NoDirectFutureLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertEqual(package["market_type"], "future_candidacy")
        self.assertTrue(package["absence_of_evidence_penalty_detected"])
        self.assertTrue(package["future_event_should_anchor_to_market"])
        self.assertGreater(package["p_2402_raw"], 0.50)
        self.assertLess(abs(package["p_2402"] - market.yes_mid), 0.08)

    def test_explicit_decline_can_still_lower_future_candidacy_probability(self) -> None:
        market = make_market(
            question="Who will run for the Republican presidential nomination in 2028?",
            description="Resolves YES if Marco Rubio announces a presidential campaign before 2028.",
            resolution_time=datetime.now(UTC) + timedelta(days=600),
            yes_mid=0.64,
            yes_bid=0.62,
            yes_ask=0.66,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=FakeSearchAdapter(),
            llm_summarizer=ExplicitDeclineFutureLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=2,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertFalse(package["absence_of_evidence_penalty_detected"])
        self.assertTrue(package["strong_contrary_evidence_detected"])
        self.assertEqual(package["p_2402_raw"], 0.02)
        self.assertLess(package["p_2402"], market.yes_mid)

    def test_sports_narrative_overconfidence_gets_flag_and_stronger_shrinkage(self) -> None:
        market = make_market(
            question="Will France win the 2026 FIFA World Cup?",
            description="Resolves YES if France wins the 2026 FIFA World Cup.",
            resolution_time=datetime.now(UTC) + timedelta(days=60),
            yes_mid=0.179,
            yes_bid=0.17,
            yes_ask=0.188,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=SportsSearchAdapter(),
            llm_summarizer=SportsNarrativeLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=1,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertEqual(package["market_type"], "sports_outcome")
        self.assertTrue(package["sports_llm_overconfidence_detected"])
        self.assertIn("sports_llm_overconfidence", package["risk_flags"])
        self.assertFalse(package["sports_quantitative_support"])
        self.assertEqual(package["sports_support_source_type"], "qualitative_news")
        self.assertIn("without strong quantitative", package["sports_risk_explanation"])
        self.assertLess(package["p_2402_final_after_shrinkage"], market.yes_mid + 0.04)

    def test_sports_quantitative_support_avoids_overconfidence_flag(self) -> None:
        market = make_market(
            question="Will France win the 2026 FIFA World Cup?",
            description="Resolves YES if France wins the 2026 FIFA World Cup.",
            resolution_time=datetime.now(UTC) + timedelta(days=60),
            yes_mid=0.179,
            yes_bid=0.17,
            yes_ask=0.188,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=SportsSearchAdapter(),
            llm_summarizer=SportsQuantitativeLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=1,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertTrue(package["sports_quantitative_support"])
        self.assertEqual(package["sports_support_source_type"], "sportsbook_odds")
        self.assertFalse(package["sports_llm_overconfidence_detected"])
        self.assertNotIn("sports_llm_overconfidence", package["risk_flags"])

    def test_sports_simulation_support_type_is_quantitative(self) -> None:
        market = make_market(
            question="Will France win the 2026 FIFA World Cup?",
            description="Resolves YES if France wins the 2026 FIFA World Cup.",
            resolution_time=datetime.now(UTC) + timedelta(days=60),
            yes_mid=0.179,
            yes_bid=0.17,
            yes_ask=0.188,
        )
        scanner = RagScanner(
            enabled=True,
            search_adapter=SportsSearchAdapter(),
            llm_summarizer=SportsSimulationLlmSummarizer(),
            enable_llm_summary=True,
            max_queries=1,
            max_results_per_query=1,
        )

        package = scanner.scan_package(market, market_mid=market.yes_mid)

        self.assertTrue(package["sports_quantitative_support"])
        self.assertEqual(package["sports_support_source_type"], "simulation")

    def test_scan_updates_forecast_signals_without_sizing(self) -> None:
        market = make_market()
        signals = initial_signals(market)
        scanner = RagScanner(enabled=True, search_adapter=FakeSearchAdapter(), max_queries=1)

        updated = scanner.scan(market, signals)

        self.assertIsNotNone(updated.evidence_package)
        self.assertIsNotNone(updated.p_2402)
        self.assertGreaterEqual(updated.evidence_quality, 1)
        self.assertNotIn("rag_disabled", updated.risk_flags)

    def test_resolution_checker_flags_strict_contract_with_weak_evidence(self) -> None:
        market = make_market()
        evidence = [
            {
                "summary": "Media reports say the regulator is expected to approve the proposal.",
                "source": "example.com",
                "url": "https://example.com",
                "timestamp": None,
                "relevance": 4,
            }
        ]

        result = check_resolution_risk(market, evidence)

        self.assertEqual(result["risk_level"], "high")
        self.assertEqual(result["ambiguity_level"], "high")
        self.assertTrue(result["trade_blocker"])
        self.assertIn("headline_resolution_mismatch", result["flags"])
        self.assertIn("official_source_not_confirmed", result["flags"])

    def test_scanner_accepts_normalized_dict(self) -> None:
        market = make_market()
        scanner = RagScanner(enabled=False)
        package = scanner.scan_package(
            {
                "market_id": market.market_id,
                "question": market.question,
                "description": market.description,
                "resolution_time": market.resolution_time.isoformat(),
                "yes_bid": market.yes_bid,
                "yes_ask": market.yes_ask,
                "volume_24h": market.volume_24h,
            }
        )

        self.assertEqual(package["market_id"], market.market_id)
        self.assertIn("rag_disabled", package["risk_flags"])


if __name__ == "__main__":
    unittest.main()
