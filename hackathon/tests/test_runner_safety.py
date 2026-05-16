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
from edge_trader_bot.runner import EdgeTraderBot, get_env_status
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
