from __future__ import annotations

import json
import os
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from unittest.mock import patch

from pydantic import BaseModel

from ai_prophet_core.arena import TickLease

from edge_trader_bot.config import BotConfig
from edge_trader_bot.json_utils import json_safe
from edge_trader_bot.runner import (
    EdgeTraderBot,
    get_env_status,
    normalize_risk_flags,
    opportunity_report,
    threshold_debug_report,
)
from edge_trader_bot.schemas import ForecastSignals, MarketView, TradeDecision


class FakeSession:
    def __init__(self) -> None:
        self.submit_called = False

    def submit_intents(self, lease, participant_idx, intents):
        self.submit_called = True
        raise AssertionError("submit_intents should not be called in dry-run")


class WeirdEnum(Enum):
    VALUE = "value"


class WeirdModel(BaseModel):
    at: datetime
    amount: Decimal


@dataclass
class WeirdDataclass:
    created_at: datetime
    model: WeirdModel


class RunnerSafetyTests(unittest.TestCase):
    def test_dry_run_submission_plan_does_not_call_submit(self) -> None:
        bot = EdgeTraderBot(BotConfig(dry_run=True))
        session = FakeSession()
        decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.5,
            edge=0.1,
            expected_value=0.1,
            confidence="high",
            reason="test",
            p_final=0.6,
        )

        submission = bot.build_submission_plan(
            session=session,
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            participant_idx=0,
            decisions=[decision],
        )

        self.assertFalse(session.submit_called)
        self.assertFalse(submission["submit_called"])
        self.assertEqual(submission["reason"], "dry_run")

    def test_live_submit_requires_explicit_allow_flag(self) -> None:
        bot = EdgeTraderBot(BotConfig(dry_run=False, allow_live_submit=False))
        session = FakeSession()
        decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.5,
            edge=0.1,
            expected_value=0.1,
            confidence="high",
            reason="test",
            p_final=0.6,
        )

        submission = bot.build_submission_plan(
            session=session,
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            participant_idx=0,
            decisions=[decision],
        )

        self.assertFalse(session.submit_called)
        self.assertFalse(submission["submit_called"])
        self.assertEqual(submission["reason"], "live_submit_disabled")

    def test_max_markets_caps_processed_candidates(self) -> None:
        bot = EdgeTraderBot(BotConfig(max_markets_to_consider=2))
        markets = [
            MarketView(
                market_id=f"m{i}",
                question="Will X happen?",
                description=None,
                resolution_time=datetime.now(UTC) + timedelta(days=1),
                yes_bid=0.4,
                yes_ask=0.42,
                yes_mid=0.41,
                no_bid=0.58,
                no_ask=0.60,
                no_mid=0.59,
                spread=0.02,
                volume_24h=100,
            )
            for i in range(5)
        ]

        selected, skipped = bot.filter_markets(markets, None)

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(skipped), 3)

    def test_rag_budget_diagnostics_in_plan(self) -> None:
        bot = EdgeTraderBot(
            BotConfig(
                dry_run=True,
                enable_rag=True,
                enable_llm_rag_summary=True,
                rag_max_markets_per_tick=1,
            )
        )
        lease = TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs")
        markets = [
            MarketView(
                market_id=f"m{i}",
                question="Will X happen?",
                description=None,
                resolution_time=datetime.now(UTC) + timedelta(days=1),
                yes_bid=0.4,
                yes_ask=0.42,
                yes_mid=0.41,
                no_bid=0.58,
                no_ask=0.60,
                no_mid=0.59,
                spread=0.02,
                volume_24h=100,
            )
            for i in range(2)
        ]
        signals = {
            "m0": ForecastSignals(
                market_id="m0",
                p_market=0.41,
                p_stat=0.41,
                p_2402=0.45,
                p_final=0.42,
                evidence_package={
                    "scanner_error": None,
                    "risk_flags": [],
                    "search_queries": ["query"],
                    "raw_search_result_count": 2,
                    "deduped_search_result_count": 1,
                    "scanner_elapsed_ms": 12,
                    "llm_attempted": True,
                    "llm_succeeded": False,
                    "llm_failed": True,
                    "llm_fallback": True,
                    "llm_error_category": "invalid_json",
                    "llm_elapsed_ms": 34,
                    "p_2402_raw": None,
                    "p_2402_final_after_shrinkage": None,
                    "reasoning_summary": "",
                    "json_parse_error": "line 1",
                    "fallback_reason": "llm_invalid_json",
                    "llm_prompt_metadata": {"evidence_items_sent": 2},
                },
            ),
            "m1": ForecastSignals(
                market_id="m1",
                p_market=0.41,
                p_stat=0.41,
                p_final=0.41,
                risk_flags=["rag_skipped_budget"],
            ),
        }
        signals["m0"].blf_package = {
            "attempted": True,
            "succeeded": False,
            "fallback_reason": "blf_invalid_json",
            "error": "bad json",
            "elapsed_ms": 9,
            "market_type": "future_candidacy",
            "long_horizon_days_to_resolution": 500,
            "absence_of_evidence_penalty_detected": True,
            "no_direct_evidence_reasoning": True,
            "future_event_should_anchor_to_market": True,
        }
        signals["m0"].p_final_before_blf = 0.42
        signals["m0"].p_final_after_blf = 0.43
        signals["m0"].blf_adjustment = 0.01
        signals["m0"].aggregation_reason = "test aggregation"

        plan = bot.build_plan(
            lease=lease,
            candidates_count=3,
            selected=markets,
            skipped=[{"market_id": "m2", "reasons": ["consideration_limit"]}],
            signals_by_market=signals,
            decisions=[],
        )

        self.assertTrue(plan["rag_enabled"])
        self.assertTrue(plan["llm_rag_enabled"])
        self.assertEqual(plan["llm_provider"], "openrouter")
        self.assertEqual(plan["llm_model"], "deepseek/deepseek-chat")
        self.assertEqual(plan["llm_attempted"], 1)
        self.assertEqual(plan["llm_succeeded"], 0)
        self.assertEqual(plan["llm_failed"], 1)
        self.assertEqual(plan["llm_fallback_count"], 1)
        self.assertEqual(plan["llm_error_categories"], ["invalid_json"])
        self.assertEqual(plan["total_llm_elapsed_ms"], 34)
        self.assertFalse(plan["blf_enabled"])
        self.assertEqual(plan["blf_markets_attempted"], 1)
        self.assertEqual(plan["blf_failed"], 1)
        self.assertEqual(plan["blf_fallback_count"], 1)
        self.assertEqual(plan["total_blf_elapsed_ms"], 9)
        self.assertEqual(plan["signals"]["m0"]["blf_package"]["fallback_reason"], "blf_invalid_json")
        self.assertEqual(plan["signals"]["m0"]["p_final_before_blf"], 0.42)
        self.assertEqual(plan["signals"]["m0"]["p_final_after_blf"], 0.43)
        self.assertEqual(plan["signals"]["m0"]["market_type"], "future_candidacy")
        self.assertTrue(plan["signals"]["m0"]["absence_of_evidence_penalty_detected"])
        self.assertIn("market_type", plan["pipeline_debug"][0])
        self.assertEqual(plan["max_rag_markets_per_tick"], 1)
        self.assertEqual(plan["rag_markets_attempted"], 1)
        self.assertEqual(plan["rag_markets_scanned"], 1)
        self.assertEqual(plan["rag_budget_skipped_markets"], ["m1"])
        self.assertEqual(plan["markets_skipped_before_rag"], 1)
        self.assertEqual(plan["total_rag_elapsed_ms"], 12)
        self.assertEqual(plan["signals"]["m0"]["llm_error_category"], "invalid_json")
        self.assertEqual(plan["signals"]["m0"]["fallback_reason"], "llm_invalid_json")
        self.assertIn("top_apparent_edges", plan)

    def test_rag_and_llm_enabled_dry_run_submission_still_does_not_call_submit(self) -> None:
        bot = EdgeTraderBot(BotConfig(dry_run=True, enable_rag=True, enable_llm_rag_summary=True))
        session = FakeSession()
        decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.5,
            edge=0.1,
            expected_value=0.1,
            confidence="high",
            reason="test",
            p_final=0.6,
        )

        submission = bot.build_submission_plan(
            session=session,
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            participant_idx=0,
            decisions=[decision],
        )

        self.assertFalse(session.submit_called)
        self.assertFalse(submission["submit_called"])
        self.assertEqual(submission["reason"], "dry_run")

    def test_blf_enabled_selects_top_n_markets(self) -> None:
        bot = EdgeTraderBot(BotConfig(enable_blf=True, blf_max_markets_per_tick=1))
        markets = [
            MarketView(
                market_id="m_low",
                question="Will X happen?",
                description=None,
                resolution_time=datetime.now(UTC) + timedelta(days=1),
                yes_bid=0.4,
                yes_ask=0.42,
                yes_mid=0.41,
                no_bid=0.58,
                no_ask=0.60,
                no_mid=0.59,
                spread=0.02,
                volume_24h=100,
            ),
            MarketView(
                market_id="m_high",
                question="Will Y happen?",
                description=None,
                resolution_time=datetime.now(UTC) + timedelta(days=1),
                yes_bid=0.4,
                yes_ask=0.42,
                yes_mid=0.41,
                no_bid=0.58,
                no_ask=0.60,
                no_mid=0.59,
                spread=0.02,
                volume_24h=100,
            ),
        ]
        signals = {
            "m_low": ForecastSignals(market_id="m_low", p_market=0.41, p_stat=0.41, p_2402=0.42, p_final=0.41),
            "m_high": ForecastSignals(
                market_id="m_high",
                p_market=0.41,
                p_stat=0.41,
                p_2402=0.70,
                p_final=0.50,
                evidence_quality=5,
            ),
        }

        selected = bot.select_blf_markets(markets, signals)

        self.assertEqual(selected, ["m_high"])

    def test_blf_enabled_dry_run_submission_still_does_not_call_submit(self) -> None:
        bot = EdgeTraderBot(BotConfig(dry_run=True, enable_blf=True))
        session = FakeSession()
        decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.5,
            edge=0.1,
            expected_value=0.1,
            confidence="high",
            reason="test",
            p_final=0.6,
        )

        submission = bot.build_submission_plan(
            session=session,
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            participant_idx=0,
            decisions=[decision],
        )

        self.assertFalse(session.submit_called)
        self.assertFalse(submission["submit_called"])
        self.assertEqual(submission["reason"], "dry_run")

    def test_threshold_debug_submission_still_does_not_call_submit(self) -> None:
        bot = EdgeTraderBot(BotConfig(dry_run=False, threshold_debug=True, allow_live_submit=True))
        session = FakeSession()
        decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.5,
            edge=0.1,
            expected_value=0.1,
            confidence="high",
            reason="test",
            p_final=0.6,
        )

        submission = bot.build_submission_plan(
            session=session,
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            participant_idx=0,
            decisions=[decision],
        )

        self.assertFalse(session.submit_called)
        self.assertFalse(submission["submit_called"])
        self.assertEqual(submission["reason"], "threshold_debug_no_submit")

    def test_threshold_debug_report_is_diagnostic_only(self) -> None:
        config = BotConfig(threshold_debug=True)
        market = MarketView(
            market_id="m1",
            question="Will X happen?",
            description=None,
            resolution_time=datetime.now(UTC) + timedelta(days=1),
            yes_bid=0.40,
            yes_ask=0.42,
            yes_mid=0.41,
            no_bid=0.58,
            no_ask=0.60,
            no_mid=0.59,
            spread=0.02,
            volume_24h=100,
        )
        signals = ForecastSignals(
            market_id="m1",
            p_market=0.41,
            p_stat=0.41,
            p_2402=0.80,
            p_final=0.55,
            p_final_before_blf=0.50,
            p_final_after_blf=0.55,
            confidence="medium",
            uncertainty=0.06,
            evidence_quality=4,
            evidence_package={"resolution_check": {"trade_blocker": False, "risk_flags": []}},
        )

        real_decision = TradeDecision(
            market_id="m1",
            action="BUY",
            side="YES",
            shares=1,
            price=0.42,
            edge=0.13,
            expected_value=0.13,
            confidence="medium",
            reason="real policy",
            p_final=0.55,
        )
        bot = EdgeTraderBot(config)
        plan = bot.build_plan(
            lease=TickLease(available=True, tick_id=datetime.now(UTC).isoformat(), candidate_set_id="cs"),
            candidates_count=1,
            selected=[market],
            skipped=[],
            signals_by_market={"m1": signals},
            decisions=[real_decision],
        )

        self.assertTrue(plan["threshold_debug"])
        self.assertEqual(plan["decisions"][0]["market_id"], "m1")
        self.assertTrue(plan["threshold_debug_report"]["diagnostic_only"])
        self.assertTrue(plan["threshold_debug_report"]["real_policy_unchanged"])
        self.assertGreater(len(plan["threshold_debug_report"]["scenarios"]), 0)

    def test_resolution_blocker_overrides_relaxed_scenarios(self) -> None:
        config = BotConfig(threshold_debug=True)
        market = MarketView(
            market_id="m1",
            question="Will X happen?",
            description=None,
            resolution_time=datetime.now(UTC) + timedelta(days=1),
            yes_bid=0.40,
            yes_ask=0.42,
            yes_mid=0.41,
            no_bid=0.58,
            no_ask=0.60,
            no_mid=0.59,
            spread=0.02,
            volume_24h=100,
        )
        signals = ForecastSignals(
            market_id="m1",
            p_market=0.41,
            p_stat=0.41,
            p_2402=0.95,
            p_final=0.85,
            p_final_before_blf=0.85,
            p_final_after_blf=0.85,
            confidence="high",
            uncertainty=0.02,
            evidence_quality=5,
            evidence_package={
                "resolution_check": {
                    "trade_blocker": True,
                    "risk_flags": ["official_source_not_confirmed"],
                }
            },
        )

        report = threshold_debug_report({"m1": market}, {"m1": signals}, config)

        self.assertTrue(all(scenario["hypothetical_trade_count"] == 0 for scenario in report["scenarios"]))
        self.assertTrue(all(scenario["blocked_by_resolution_risk"] == 1 for scenario in report["scenarios"]))

    def test_opportunity_report_includes_blf_fields(self) -> None:
        market = MarketView(
            market_id="m1",
            question="Will X happen?",
            description=None,
            resolution_time=datetime.now(UTC) + timedelta(days=1),
            yes_bid=0.40,
            yes_ask=0.42,
            yes_mid=0.41,
            no_bid=0.58,
            no_ask=0.60,
            no_mid=0.59,
            spread=0.02,
            volume_24h=100,
        )
        signals = ForecastSignals(
            market_id="m1",
            p_market=0.41,
            p_stat=0.41,
            p_2402=0.70,
            p_blf=0.65,
            p_final=0.50,
            p_final_before_blf=0.46,
            p_final_after_blf=0.50,
            confidence="medium",
            evidence_quality=5,
            blf_package={"succeeded": True, "p_blf_final_after_shrinkage": 0.65},
            evidence_package={"resolution_check": {"trade_blocker": False, "risk_flags": []}},
        )

        report = opportunity_report({"m1": market}, {"m1": signals})

        self.assertEqual(report["blf_changed_probability_most"][0]["market_id"], "m1")
        self.assertIn("p_blf_final_after_shrinkage", report["top_by_best_edge"][0])
        self.assertIn("rag_disagrees_with_market_most_excluding_absence_penalty", report)
        self.assertIn("markets_with_no_positive_edge_all_relaxed_thresholds", report)

    def test_normalized_risk_flags_are_populated(self) -> None:
        flags = normalize_risk_flags(
            ["deadline_unclear", "LLM returned no direct evidence", "blf_invalid_json"],
            {"risk_flags": ["official_source_not_confirmed"], "ambiguity_level": "high"},
        )

        self.assertIn("missing_official_source", flags)
        self.assertIn("deadline_unclear", flags)
        self.assertIn("llm_uncertainty", flags)
        self.assertIn("json_fallback", flags)
        self.assertIn("high_ambiguity", flags)

    def test_json_safe_serializes_weird_server_like_objects(self) -> None:
        payload = {
            "decimal": Decimal("1.23"),
            "enum": WeirdEnum.VALUE,
            "exception": RuntimeError("boom"),
            "dataclass": WeirdDataclass(
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                model=WeirdModel(at=datetime(2026, 1, 2, tzinfo=UTC), amount=Decimal("4.56")),
            ),
        }

        safe = json_safe(payload)
        encoded = json.dumps(safe, sort_keys=True)

        self.assertIn("1.23", encoded)
        self.assertIn("RuntimeError", encoded)
        self.assertIn("2026-01-01T00:00:00+00:00", encoded)

    def test_missing_pa_key_fails_before_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PA_SERVER_API_KEY"):
                EdgeTraderBot(BotConfig()).run(once=True)

    def test_env_status_reports_presence_without_secret_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PA_SERVER_API_KEY": "secret-pa",
                "BRAVE_SEARCH_API_KEY": "secret-brave",
                "OPENROUTER_API_KEY": "secret-openrouter",
                "EDGE_TRADER_LLM_PROVIDER": "openrouter",
                "EDGE_TRADER_LLM_MODEL": "deepseek/deepseek-chat",
                "EDGE_TRADER_ENABLE_RAG": "1",
                "EDGE_TRADER_ENABLE_LLM_RAG": "0",
                "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK": "3",
            },
            clear=True,
        ):
            status = get_env_status()

        self.assertTrue(status["PA_SERVER_API_KEY"]["present"])
        self.assertTrue(status["BRAVE_SEARCH_API_KEY"]["present"])
        self.assertTrue(status["OPENROUTER_API_KEY"]["present"])
        self.assertNotIn("secret-pa", str(status))
        self.assertEqual(status["EDGE_TRADER_LLM_MODEL"]["value"], "deepseek/deepseek-chat")


if __name__ == "__main__":
    unittest.main()
