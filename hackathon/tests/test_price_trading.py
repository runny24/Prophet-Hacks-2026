from __future__ import annotations

import tempfile
import unittest
import os
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_prophet_core.client_models import PortfolioResponse, PositionData

from edge_trader_bot.config import BotConfig
from edge_trader_bot.aggregator import combine_signals
from edge_trader_bot.candidate_ranker import (
    memory_features_from_history,
    rank_candidates,
    score_candidate_for_processing,
)
from edge_trader_bot.price_memory import PriceMemory, compute_price_features, market_snapshot
from edge_trader_bot.price_signal import (
    compute_short_horizon_trade_signal,
    is_live_tradable_forecast_mispricing_signal,
    is_live_tradable_fresh_event_signal,
    is_live_tradable_signal,
    maybe_price_exit,
    price_signal_to_decision,
)
from edge_trader_bot.runner import EdgeTraderBot
from edge_trader_bot.schemas import ForecastSignals, MarketView, TradeDecision
from edge_trader_bot.trading_policy import rank_decisions


def make_market(**overrides) -> MarketView:
    base = {
        "market_id": "m1",
        "question": "Will ACME close above $100?",
        "description": "Resolves YES if ACME closes above $100.",
        "resolution_time": datetime.now(UTC) + timedelta(days=7),
        "yes_bid": 0.40,
        "yes_ask": 0.42,
        "yes_mid": 0.41,
        "no_bid": 0.58,
        "no_ask": 0.60,
        "no_mid": 0.59,
        "spread": 0.02,
        "volume_24h": 1000.0,
    }
    base.update(overrides)
    return MarketView(**base)


def make_signals(**overrides) -> ForecastSignals:
    base = {
        "market_id": "m1",
        "p_market": 0.41,
        "p_stat": 0.41,
        "p_final": 0.415,
        "p_final_after_blf": 0.415,
        "confidence": "medium",
        "uncertainty": 0.04,
    }
    base.update(overrides)
    return ForecastSignals(**base)


def make_portfolio(positions=None) -> PortfolioResponse:
    return PortfolioResponse(
        experiment_id="exp",
        participant_idx=0,
        cash="10000",
        equity="10000",
        total_pnl="0",
        positions=positions or [],
        total_fills=0,
    )


class PriceTradingTests(unittest.TestCase):
    def test_price_memory_appends_and_loads_recent_history(self) -> None:
        market = make_market()
        signals = make_signals()
        with tempfile.TemporaryDirectory() as tmp:
            memory = PriceMemory(Path(tmp) / "history.jsonl")
            memory.append_market_snapshot(
                market_snapshot(
                    tick_id="t1",
                    candidate_set_id="cs1",
                    generated_at=datetime.now(UTC).isoformat(),
                    market=market,
                    signals=signals,
                    traded=False,
                )
            )

            rows = memory.load_recent_market_history("m1", lookback_ticks=4)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_id"], "m1")
        self.assertEqual(rows[0]["midpoint"], 0.41)

    def test_momentum_signal_generates_tiny_buy_when_history_exists(self) -> None:
        market = make_market(yes_bid=0.415, yes_ask=0.425, yes_mid=0.42, spread=0.01)
        signals = make_signals(p_market=0.42, p_final=0.421, p_final_after_blf=0.421)
        history = [
            {"midpoint": 0.410, "p_final_after_blf": 0.421},
            {"midpoint": 0.415, "p_final_after_blf": 0.421},
        ]
        features = compute_price_features(history, market, signals)

        signal = compute_short_horizon_trade_signal(
            market,
            signals,
            features,
            make_portfolio(),
            BotConfig(enable_price_trading=True, price_signal_min_edge=0.001, price_max_trade_size=1),
        )
        decision = price_signal_to_decision(market, signal, signals)

        self.assertEqual(signal.signal_type, "momentum")
        self.assertEqual(signal.side, "YES")
        self.assertEqual(signal.suggested_size, 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "BUY")

    def test_no_history_blocks_price_entry(self) -> None:
        market = make_market()
        signals = make_signals(p_final=0.46, p_final_after_blf=0.46)
        features = compute_price_features([], market, signals)

        signal = compute_short_horizon_trade_signal(
            market,
            signals,
            features,
            make_portfolio(),
            BotConfig(enable_price_trading=True, forecast_edge_min_edge=0.003),
        )

        self.assertIn("no_history", signal.blockers)
        self.assertEqual(signal.suggested_size, 0)

    def test_no_history_clean_fresh_event_can_create_one_share_candidate(self) -> None:
        bot = EdgeTraderBot(BotConfig(enable_price_trading=True, live_guard_mode=True, price_max_trade_size=5))
        market = make_market(
            market_id="fresh",
            question="Will ACME announce a merger this week?",
            yes_bid=0.40,
            yes_ask=0.42,
            yes_mid=0.41,
            no_bid=0.58,
            no_ask=0.60,
            no_mid=0.59,
            spread=0.02,
        )
        market = make_market(
            market_id="fresh",
            question="Will ACME announce a merger this week?",
            yes_bid=0.40,
            yes_ask=0.425,
            yes_mid=0.4125,
            no_bid=0.575,
            no_ask=0.60,
            no_mid=0.5875,
            spread=0.012,
        )
        signals = {
            "fresh": make_signals(
                market_id="fresh",
                p_market=0.4125,
                p_final=0.46,
                p_final_after_blf=0.46,
                p_2402=0.47,
                confidence="high",
                evidence_quality=5,
                evidence_package={"market_type": "future_announcement", "fresh_evidence": True, "risk_flags": []},
                blf_package={"attempted": True, "succeeded": True, "risk_flags": []},
            )
        }
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: []

        decisions, diag = bot.build_price_trading_decisions([market], signals, make_portfolio(), lease=None)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].shares, 1)
        self.assertIn("fresh_event", decisions[0].reason)
        self.assertEqual(diag["live_fresh_event_candidate_count"], 1)
        self.assertEqual(diag["fresh_event_candidates"][0]["channel"], "fresh_event")

    def test_no_history_weak_fresh_event_is_blocked(self) -> None:
        market = make_market(yes_bid=0.40, yes_ask=0.41, yes_mid=0.405, spread=0.01)
        signals = make_signals(
            p_market=0.405,
            p_final=0.43,
            p_final_after_blf=0.43,
            confidence="medium",
            evidence_quality=2,
            evidence_package={"market_type": "future_announcement"},
        )
        features = compute_price_features([], market, signals)

        gate = is_live_tradable_fresh_event_signal(
            market=market,
            signals=signals,
            features=features,
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("fresh_event_blocked_low_evidence", gate.reasons)

    def test_no_history_sports_long_horizon_fresh_event_is_blocked(self) -> None:
        market = make_market(
            question="Will England win the 2026 World Cup?",
            resolution_time=datetime.now(UTC) + timedelta(days=400),
            yes_bid=0.10,
            yes_ask=0.11,
            yes_mid=0.105,
            spread=0.01,
        )
        signals = make_signals(
            p_market=0.105,
            p_final=0.20,
            p_final_after_blf=0.20,
            p_2402=0.21,
            confidence="high",
            evidence_quality=5,
            evidence_package={"market_type": "sports_outcome", "sports_quantitative_support": False},
        )
        features = compute_price_features([], market, signals)

        gate = is_live_tradable_fresh_event_signal(
            market=market,
            signals=signals,
            features=features,
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("fresh_event_blocked_sports_long_horizon", gate.reasons)

    def test_rate_limit_blocks_fresh_event_entry(self) -> None:
        market = make_market(question="Will ACME announce a merger this week?", yes_ask=0.42, yes_mid=0.41, spread=0.01)
        signals = make_signals(
            p_market=0.41,
            p_final=0.47,
            p_final_after_blf=0.47,
            p_2402=0.48,
            confidence="high",
            evidence_quality=5,
            risk_flags=["llm_rate_limit"],
            evidence_package={"market_type": "future_announcement", "fresh_evidence": True},
        )
        features = compute_price_features([], market, signals)

        gate = is_live_tradable_fresh_event_signal(
            market=market,
            signals=signals,
            features=features,
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("fresh_event_blocked_fallback_or_rate_limit", gate.reasons)

    def test_fresh_event_does_not_flip_opposite_existing_position(self) -> None:
        market = make_market(question="Will ACME announce a merger this week?", yes_ask=0.42, yes_mid=0.41, spread=0.01)
        signals = make_signals(
            p_market=0.41,
            p_final=0.47,
            p_final_after_blf=0.47,
            p_2402=0.48,
            confidence="high",
            evidence_quality=5,
            evidence_package={"market_type": "future_announcement", "fresh_evidence": True},
        )
        position = PositionData(
            market_id="m1",
            side="NO",
            shares="1",
            avg_entry_price="0.60",
            current_price="0.59",
            updated_at=datetime.now(UTC),
        )
        features = compute_price_features([], market, signals)

        gate = is_live_tradable_fresh_event_signal(
            market=market,
            signals=signals,
            features=features,
            portfolio=make_portfolio([position]),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("fresh_event_blocked_position_risk", gate.reasons)

    def test_clean_forecast_mispricing_non_sports_can_trade_one_share(self) -> None:
        bot = EdgeTraderBot(BotConfig(enable_price_trading=True, live_guard_mode=True, price_max_trade_size=5))
        market = make_market(
            market_id="forecast",
            yes_bid=0.411,
            yes_ask=0.421,
            yes_mid=0.416,
            no_bid=0.579,
            no_ask=0.589,
            no_mid=0.584,
            spread=0.010,
        )
        signals = {
            "forecast": make_signals(
                market_id="forecast",
                p_market=0.416,
                p_final=0.430,
                p_final_after_blf=0.430,
                p_2402=0.435,
                confidence="medium",
                evidence_quality=4,
                evidence_package={"market_type": "government_action", "risk_flags": [], "resolution_check": {}},
                blf_package={"attempted": True, "succeeded": True, "risk_flags": []},
            )
        }
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: []

        decisions, diag = bot.build_price_trading_decisions([market], signals, make_portfolio(), lease=None)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].shares, 1)
        self.assertIn("clean_forecast_mispricing", decisions[0].reason)
        self.assertEqual(diag["live_forecast_mispricing_candidate_count"], 1)

    def test_live_reranked_candidate_is_forced_into_rag_selection(self) -> None:
        bot = EdgeTraderBot(
            BotConfig(
                enable_price_trading=True,
                live_guard_mode=True,
                enable_rag=True,
                enable_candidate_rerank=True,
                rag_max_markets_per_tick=1,
            )
        )
        active = make_market(
            market_id="active",
            question="Will ACME announce earnings today?",
            yes_bid=0.40,
            yes_ask=0.405,
            yes_mid=0.4025,
            spread=0.005,
        )
        stale = make_market(market_id="stale", question="Will aliens be disclosed by 2030?", spread=0.01)
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: (
            [{"midpoint": 0.39}, {"midpoint": 0.397}, {"midpoint": 0.402}] if market_id == "active" else []
        )
        selected, _ = bot.filter_markets([stale, active], make_portfolio())

        selection = bot.rag_selection_metadata(selected)

        self.assertTrue(selection["active"]["rag_selected"])
        self.assertTrue(selection["active"]["rag_forced_by_live_candidate_selection"])
        self.assertEqual(bot.live_candidate_selection_summary["live_candidates_sent_to_rag_count"], 1)

    def test_unforecasted_candidate_is_marked_not_evaluated_by_budget(self) -> None:
        market = make_market()
        signals = make_signals(
            p_2402=None,
            confidence="low",
            evidence_quality=0,
            risk_flags=["rag_skipped_budget", "blf_skipped_budget"],
            evidence_package={},
        )
        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.reasons, ["not_evaluated_by_forecast_budget"])
        self.assertNotIn("forecast_mispricing_blocked_confidence", gate.reasons)
        self.assertNotIn("forecast_mispricing_blocked_low_evidence", gate.reasons)

    def test_sports_clean_forecast_with_quant_support_can_trade(self) -> None:
        market = make_market(
            question="Will Team A win tonight?",
            yes_bid=0.095,
            yes_ask=0.100,
            yes_mid=0.0975,
            no_bid=0.900,
            no_ask=0.905,
            no_mid=0.9025,
            spread=0.005,
        )
        signals = make_signals(
            p_market=0.0975,
            p_final=0.113,
            p_final_after_blf=0.113,
            p_2402=0.115,
            confidence="high",
            evidence_quality=4,
            evidence_package={
                "market_type": "sports_outcome",
                "sports_quantitative_support": True,
                "sports_support_source_type": "sportsbook_odds",
                "resolution_check": {},
            },
            blf_package={"attempted": True, "succeeded": True},
        )
        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertTrue(gate.allowed, gate.reasons)
        self.assertEqual(gate.max_size, 1)

    def test_sports_clean_forecast_without_quant_support_is_blocked(self) -> None:
        market = make_market(question="Will Team A win tonight?", yes_ask=0.10, yes_mid=0.095, spread=0.005)
        signals = make_signals(
            p_market=0.095,
            p_final=0.13,
            p_final_after_blf=0.13,
            confidence="high",
            evidence_quality=5,
            evidence_package={"market_type": "sports_outcome", "sports_quantitative_support": False},
            blf_package={"attempted": True, "succeeded": True},
        )
        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_sports_without_quant_support", gate.reasons)

    def test_forecast_mispricing_blocks_rate_limit_and_stale_evidence(self) -> None:
        market = make_market(yes_ask=0.421, yes_mid=0.416, spread=0.010)
        for flag in ("llm_rate_limit", "stale_evidence"):
            signals = make_signals(
                p_market=0.416,
                p_final=0.440,
                p_final_after_blf=0.440,
                confidence="high",
                evidence_quality=5,
                risk_flags=[flag],
                evidence_package={"market_type": "government_action"},
                blf_package={"attempted": True, "succeeded": True},
            )
            gate = is_live_tradable_forecast_mispricing_signal(
                market=market,
                signals=signals,
                features=compute_price_features([], market, signals),
                portfolio=make_portfolio(),
                config=BotConfig(),
            )
            self.assertFalse(gate.allowed)
            self.assertIn("forecast_mispricing_blocked_fallback_or_rate_limit", gate.reasons)

    def test_forecast_mispricing_blocks_selected_failed_blf(self) -> None:
        market = make_market(yes_ask=0.421, yes_mid=0.416, spread=0.010)
        signals = make_signals(
            p_market=0.416,
            p_final=0.440,
            p_final_after_blf=0.440,
            confidence="high",
            evidence_quality=5,
            evidence_package={"market_type": "government_action"},
            blf_package={"attempted": True, "succeeded": False},
        )
        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_blf_failed", gate.reasons)

    def test_forecast_mispricing_blocks_edge_below_threshold_and_wide_spread(self) -> None:
        market = make_market(yes_bid=0.410, yes_ask=0.421, yes_mid=0.4155, spread=0.011)
        signals = make_signals(
            p_market=0.4155,
            p_final=0.426,
            p_final_after_blf=0.426,
            confidence="high",
            evidence_quality=5,
            evidence_package={"market_type": "government_action"},
            blf_package={"attempted": True, "succeeded": True},
        )
        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )
        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_edge_below_threshold", gate.reasons)

        wide = make_market(yes_bid=0.400, yes_ask=0.420, yes_mid=0.410, spread=0.020)
        gate = is_live_tradable_forecast_mispricing_signal(
            market=wide,
            signals=make_signals(
                p_market=0.410,
                p_final=0.450,
                p_final_after_blf=0.450,
                confidence="high",
                evidence_quality=5,
                evidence_package={"market_type": "government_action"},
                blf_package={"attempted": True, "succeeded": True},
            ),
            features=compute_price_features([], wide, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )
        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_spread_too_wide", gate.reasons)

    def test_brave_failure_without_cache_uses_metadata_only_and_blocks_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot = EdgeTraderBot(BotConfig(enable_cached_evidence_fallback=True), memory_path=Path(tmp) / "memory.jsonl")
            market = make_market()
            signals = make_signals(
                evidence_package={
                    "scanner_error": "http_error: 500",
                    "risk_flags": ["brave_http_error"],
                    "p_2402": None,
                    "evidence_quality": 0,
                    "confidence": "low",
                },
                risk_flags=["brave_http_error"],
            )

            recovered = bot.apply_rag_evidence_recovery(market, signals)
            gate = is_live_tradable_forecast_mispricing_signal(
                market=market,
                signals=recovered,
                features=compute_price_features([], market, recovered),
                portfolio=make_portfolio(),
                config=BotConfig(),
            )

        self.assertTrue(recovered.evidence_package["metadata_only_evidence"])
        self.assertEqual(recovered.confidence, "low")
        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_metadata_only", gate.reasons)

    def test_brave_failure_with_fresh_cache_caps_quality_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            cached_package = {
                "p_2402": 0.70,
                "evidence_quality": 5,
                "confidence": "high",
                "scanner_error": None,
                "risk_flags": [],
                "resolution_check": {},
            }
            memory_path.write_text(
                json.dumps(
                    {
                        "tick_id": "old-tick",
                        "signals": {
                            "m1": {
                                "evidence_package": cached_package,
                                "p_final_after_blf": 0.70,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            bot = EdgeTraderBot(BotConfig(enable_cached_evidence_fallback=True), memory_path=memory_path)
            market = make_market()
            signals = make_signals(
                evidence_package={"scanner_error": "http_error: 500", "risk_flags": ["brave_http_error"]},
                risk_flags=["brave_http_error"],
            )

            recovered = bot.apply_rag_evidence_recovery(market, signals)

        self.assertTrue(recovered.evidence_package["cached_evidence_used"])
        self.assertEqual(recovered.evidence_package["cached_evidence_age_ticks"], 1)
        self.assertEqual(recovered.evidence_quality, 3)
        self.assertEqual(recovered.confidence, "medium")
        self.assertIn("cached_evidence_used", recovered.risk_flags)

    def test_cached_evidence_too_old_is_not_live_tradable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            lines = []
            lines.append(
                json.dumps(
                    {
                        "tick_id": "old",
                        "signals": {
                            "m1": {
                                "evidence_package": {
                                    "p_2402": 0.70,
                                    "evidence_quality": 5,
                                    "confidence": "high",
                                    "scanner_error": None,
                                    "risk_flags": [],
                                }
                            }
                        },
                    }
                )
            )
            for idx in range(3):
                lines.append(json.dumps({"tick_id": f"new-{idx}", "signals": {}}))
            memory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            bot = EdgeTraderBot(
                BotConfig(enable_cached_evidence_fallback=True, max_cached_evidence_age_ticks=1),
                memory_path=memory_path,
            )
            market = make_market()
            signals = make_signals(
                evidence_package={"scanner_error": "http_error: 500", "risk_flags": ["brave_http_error"]},
                risk_flags=["brave_http_error"],
            )

            recovered = bot.apply_rag_evidence_recovery(market, signals)

        self.assertFalse(recovered.evidence_package.get("cached_evidence_used", False))
        self.assertTrue(recovered.evidence_package["metadata_only_evidence"])
        self.assertTrue(recovered.evidence_package["cached_evidence_too_old"])

    def test_cached_evidence_can_pass_forecast_mispricing_when_other_gates_clean(self) -> None:
        market = make_market(yes_bid=0.411, yes_ask=0.421, yes_mid=0.416, spread=0.010)
        signals = make_signals(
            p_market=0.416,
            p_2402=0.95,
            confidence="medium",
            evidence_quality=3,
            evidence_package={
                "p_2402": 0.95,
                "cached_evidence_used": True,
                "cached_evidence_age_ticks": 1,
                "current_rag_error": "http_error: 500",
                "evidence_quality": 3,
                "confidence": "medium",
                "risk_flags": ["cached_evidence_used"],
                "resolution_check": {},
            },
            blf_package={"attempted": True, "succeeded": True},
            risk_flags=["cached_evidence_used"],
        )
        signals = combine_signals(signals)

        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertTrue(gate.allowed, gate.reasons)
        self.assertEqual(gate.reasons, ["clean_forecast_mispricing_cached_evidence"])

    def test_rate_limit_plus_cached_evidence_still_blocks(self) -> None:
        market = make_market(yes_bid=0.411, yes_ask=0.421, yes_mid=0.416, spread=0.010)
        signals = make_signals(
            p_market=0.416,
            p_final=0.44,
            p_final_after_blf=0.44,
            confidence="medium",
            evidence_quality=3,
            evidence_package={"cached_evidence_used": True, "current_rag_error": "http_error: 500"},
            risk_flags=["llm_rate_limit", "cached_evidence_used"],
        )

        gate = is_live_tradable_forecast_mispricing_signal(
            market=market,
            signals=signals,
            features=compute_price_features([], market, signals),
            portfolio=make_portfolio(),
            config=BotConfig(),
        )

        self.assertFalse(gate.allowed)
        self.assertIn("forecast_mispricing_blocked_fallback_or_rate_limit", gate.reasons)

    def test_metadata_only_does_not_block_clean_price_action_momentum(self) -> None:
        market = make_market(yes_bid=0.426, yes_ask=0.429, yes_mid=0.4275, spread=0.003)
        signals = make_signals(
            p_market=0.4275,
            p_final=0.428,
            p_final_after_blf=0.428,
            evidence_package={"metadata_only_evidence": True, "risk_flags": ["metadata_only_evidence"]},
            risk_flags=["metadata_only_evidence"],
        )
        features = compute_price_features(
            [{"midpoint": 0.410}, {"midpoint": 0.417}, {"midpoint": 0.423}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(market=market, signals=signals, features=features, signal=signal)

        self.assertEqual(signal.signal_type, "momentum")
        self.assertTrue(ok, reasons)

    def test_candidate_ranker_prefers_tight_spread_to_wide_spread(self) -> None:
        tight = make_market(market_id="tight", yes_bid=0.40, yes_ask=0.405, yes_mid=0.4025, spread=0.005)
        wide = make_market(market_id="wide", yes_bid=0.30, yes_ask=0.42, yes_mid=0.36, spread=0.12)

        self.assertGreater(
            score_candidate_for_processing(tight).score,
            score_candidate_for_processing(wide).score,
        )

    def test_candidate_ranker_repeated_mover_beats_long_horizon_no_history(self) -> None:
        mover = make_market(market_id="mover", yes_bid=0.40, yes_ask=0.405, yes_mid=0.4025, spread=0.005)
        stale = make_market(
            market_id="stale",
            question="Who will win the 2028 presidential election?",
            resolution_time=datetime.now(UTC) + timedelta(days=900),
            spread=0.01,
        )
        features = memory_features_from_history([
            {"midpoint": 0.38},
            {"midpoint": 0.386},
            {"midpoint": 0.392},
        ])

        self.assertGreater(
            score_candidate_for_processing(mover, features).score,
            score_candidate_for_processing(stale, {}).score,
        )

    def test_candidate_ranker_existing_position_ranks_high(self) -> None:
        market = make_market(market_id="held", spread=0.05)

        result = score_candidate_for_processing(market, {}, portfolio_position=object())

        self.assertEqual(result.priority_bucket, "existing_position")
        self.assertGreaterEqual(result.score, 100)

    def test_candidate_ranker_penalizes_alien_and_world_cup_without_movement(self) -> None:
        alien = make_market(market_id="alien", question="Will aliens be disclosed by 2030?", spread=0.01)
        world_cup = make_market(market_id="wc", question="Will England win the World Cup 2026?", spread=0.01)

        self.assertIn("alien_disclosure_no_movement", score_candidate_for_processing(alien, {}).reject_or_penalty_reasons)
        self.assertIn("long_horizon_sports", score_candidate_for_processing(world_cup, {}).reject_or_penalty_reasons)

    def test_candidate_ranker_boosts_near_term_catalyst(self) -> None:
        catalyst = make_market(market_id="cat", question="Will ACME announce earnings today?", spread=0.01)
        plain = make_market(market_id="plain", question="Will ACME do something someday?", spread=0.01)

        self.assertGreater(
            score_candidate_for_processing(catalyst, {}).score,
            score_candidate_for_processing(plain, {}).score,
        )

    def test_candidate_ranker_invalid_quote_is_heavily_penalized(self) -> None:
        invalid = make_market(market_id="bad", yes_bid=0.60, yes_ask=0.50, spread=-0.10)

        result = score_candidate_for_processing(invalid, {})

        self.assertLessEqual(result.score, -100)
        self.assertIn("invalid_quote", result.reject_or_penalty_reasons)

    def test_candidate_ranker_stable_deterministic_ordering(self) -> None:
        markets = [
            make_market(market_id="b", spread=0.01),
            make_market(market_id="a", spread=0.01),
        ]

        ranked = rank_candidates(markets)

        self.assertEqual([market.market_id for market, _ in ranked], ["a", "b"])

    def test_candidate_rerank_selects_active_late_markets(self) -> None:
        markets = [
            make_market(market_id=f"dead-{idx}", question="Will aliens be disclosed by 2030?", spread=0.01)
            for idx in range(10)
        ]
        markets.append(make_market(market_id="active", question="Will ACME announce earnings today?", spread=0.003))
        bot = EdgeTraderBot(BotConfig(max_markets_to_consider=3, enable_candidate_rerank=True))
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: (
            [{"midpoint": 0.40}, {"midpoint": 0.405}, {"midpoint": 0.412}] if market_id == "active" else []
        )

        selected, _ = bot.filter_markets(markets, make_portfolio())

        self.assertIn("active", [market.market_id for market in selected])
        self.assertTrue(bot.candidate_selection_summary["candidate_rerank_enabled"])

    def test_take_profit_exit_generates_sell(self) -> None:
        market = make_market(yes_bid=0.43, yes_ask=0.44, yes_mid=0.435)
        signals = make_signals(p_market=0.435, p_final=0.436, p_final_after_blf=0.436)
        position = PositionData(
            market_id="m1",
            side="YES",
            shares="2",
            avg_entry_price="0.42",
            current_price="0.43",
            updated_at=datetime.now(UTC),
        )
        decision, diag = maybe_price_exit(
            market,
            signals,
            position,
            history=[{"current_position": {"side": "YES", "shares": 2}}],
            config=BotConfig(price_take_profit_ticks=0.005, price_max_position_per_market=3),
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "SELL")
        self.assertIn("take_profit", diag["exit_reasons"])

    def test_rank_decisions_prioritizes_exits_before_entries(self) -> None:
        entry = TradeDecision("m1", "BUY", "YES", 1, 0.5, 0.02, 0.02, "medium", "entry", 0.52)
        exit_decision = TradeDecision("m2", "SELL", "YES", 1, 0.5, -0.01, -0.01, "medium", "exit", 0.49)

        ranked = rank_decisions([entry, exit_decision], limit=1)

        self.assertEqual(ranked[0].action, "SELL")

    def test_rate_limit_signal_is_not_live_tradable(self) -> None:
        market = make_market(yes_bid=0.418, yes_ask=0.421, yes_mid=0.4195, spread=0.003)
        signals = make_signals(risk_flags=["llm_rate_limit"], p_market=0.4195)
        features = compute_price_features(
            [{"midpoint": 0.410}, {"midpoint": 0.414}, {"midpoint": 0.417}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(market=market, signals=signals, features=features, signal=signal)

        self.assertFalse(ok)
        self.assertTrue(any("llm_rate_limit" in reason for reason in reasons))

    def test_forecast_edge_only_signal_is_not_live_tradable(self) -> None:
        market = make_market(yes_bid=0.419, yes_ask=0.421, yes_mid=0.420, spread=0.002)
        signals = make_signals(p_market=0.42, p_final=0.47, p_final_after_blf=0.47, p_2402=0.47, evidence_quality=5)
        features = compute_price_features(
            [{"midpoint": 0.420}, {"midpoint": 0.420}, {"midpoint": 0.420}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(market=market, signals=signals, features=features, signal=signal)

        self.assertEqual(signal.signal_type, "forecast_edge")
        self.assertFalse(ok)
        self.assertIn("forecast_edge_not_live_entry", reasons)

    def test_clean_momentum_with_history_can_be_live_tradable(self) -> None:
        market = make_market(yes_bid=0.426, yes_ask=0.429, yes_mid=0.4275, spread=0.003)
        signals = make_signals(p_market=0.4275, p_final=0.428, p_final_after_blf=0.428)
        features = compute_price_features(
            [{"midpoint": 0.410}, {"midpoint": 0.417}, {"midpoint": 0.423}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(market=market, signals=signals, features=features, signal=signal)

        self.assertEqual(signal.signal_type, "momentum")
        self.assertTrue(ok, reasons)

    def test_wide_spread_blocks_live_tradable_signal(self) -> None:
        market = make_market(yes_bid=0.420, yes_ask=0.430, yes_mid=0.425, spread=0.010)
        signals = make_signals(p_market=0.425)
        features = compute_price_features(
            [{"midpoint": 0.410}, {"midpoint": 0.416}, {"midpoint": 0.421}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(market=market, signals=signals, features=features, signal=signal)

        self.assertFalse(ok)
        self.assertIn("spread_above_live_hard_cap", reasons)

    def test_guard_mode_clamps_entry_and_exit_size_to_one(self) -> None:
        market = make_market(yes_bid=0.426, yes_ask=0.429, yes_mid=0.4275, spread=0.003)
        signals = make_signals(p_market=0.4275)
        features = compute_price_features(
            [{"midpoint": 0.410}, {"midpoint": 0.417}, {"midpoint": 0.423}],
            market,
            signals,
        )
        signal = compute_short_horizon_trade_signal(
            market,
            signals,
            features,
            make_portfolio(),
            BotConfig(live_guard_mode=True, price_max_trade_size=5),
        )
        position = PositionData(
            market_id="m1",
            side="YES",
            shares="3",
            avg_entry_price="0.42",
            current_price="0.43",
            updated_at=datetime.now(UTC),
        )
        exit_decision, _ = maybe_price_exit(
            market,
            signals,
            position,
            history=[{"current_position": {"side": "YES", "shares": 3}}],
            config=BotConfig(live_guard_mode=True, price_take_profit_ticks=0.001),
        )

        self.assertEqual(signal.suggested_size, 1)
        self.assertEqual(exit_decision.shares, 1)

    def test_existing_sell_exit_can_pass_despite_fallback_flags(self) -> None:
        market = make_market()
        signals = make_signals(risk_flags=["llm_rate_limit", "blf_rate_limit"])
        features = compute_price_features([], market, signals)
        signal = compute_short_horizon_trade_signal(market, signals, features, make_portfolio(), BotConfig())

        ok, reasons = is_live_tradable_signal(
            market=market,
            signals=signals,
            features=features,
            signal=signal,
            action="SELL",
        )

        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_forecast_only_freeze_blocks_forecast_edge_not_clean_momentum(self) -> None:
        bot = EdgeTraderBot(BotConfig(enable_price_trading=True, live_guard_mode=True, price_max_trade_size=1))
        markets = [
            make_market(market_id="mom", yes_bid=0.428, yes_ask=0.431, yes_mid=0.4295, spread=0.003),
            make_market(market_id="fc", yes_bid=0.419, yes_ask=0.421, yes_mid=0.420, spread=0.002),
        ]
        signals = {
            "mom": make_signals(market_id="mom", p_market=0.4295, p_final=0.430, p_final_after_blf=0.430),
            "fc": make_signals(
                market_id="fc",
                p_market=0.420,
                p_final=0.470,
                p_final_after_blf=0.470,
                p_2402=0.470,
                evidence_quality=5,
            ),
        }
        history = {
            "mom": [{"midpoint": 0.400}, {"midpoint": 0.410}, {"midpoint": 0.420}],
            "fc": [{"midpoint": 0.420}, {"midpoint": 0.420}, {"midpoint": 0.420}],
        }
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: history[market_id]
        old_scope = os.environ.get("EDGE_TRADER_LIVE_FREEZE_SCOPE")
        try:
            os.environ["EDGE_TRADER_LIVE_FREEZE_SCOPE"] = "forecast_only"
            decisions, diag = bot.build_price_trading_decisions(markets, signals, make_portfolio(), lease=None)
        finally:
            if old_scope is None:
                os.environ.pop("EDGE_TRADER_LIVE_FREEZE_SCOPE", None)
            else:
                os.environ["EDGE_TRADER_LIVE_FREEZE_SCOPE"] = old_scope

        self.assertEqual([decision.market_id for decision in decisions], ["mom"])
        rows = {row["market_id"]: row for row in diag["price_signal_rows"]}
        self.assertTrue(rows["mom"]["live_tradable"])
        self.assertFalse(rows["fc"]["live_tradable"])
        self.assertIn("live_entries_frozen_forecast_only", rows["fc"]["live_block_reasons"])
        self.assertEqual(diag["blocked_by_freeze_forecast_only_count"], 1)

    def test_all_entry_freeze_blocks_clean_momentum(self) -> None:
        bot = EdgeTraderBot(BotConfig(enable_price_trading=True, live_guard_mode=True, price_max_trade_size=1))
        market = make_market(market_id="mom", yes_bid=0.428, yes_ask=0.431, yes_mid=0.4295, spread=0.003)
        signals = {"mom": make_signals(market_id="mom", p_market=0.4295, p_final=0.430, p_final_after_blf=0.430)}
        bot.price_memory.load_recent_market_history = lambda market_id, lookback_ticks=8: [
            {"midpoint": 0.400},
            {"midpoint": 0.410},
            {"midpoint": 0.420},
        ]
        old_scope = os.environ.get("EDGE_TRADER_LIVE_FREEZE_SCOPE")
        try:
            os.environ["EDGE_TRADER_LIVE_FREEZE_SCOPE"] = "all_entries"
            decisions, diag = bot.build_price_trading_decisions([market], signals, make_portfolio(), lease=None)
        finally:
            if old_scope is None:
                os.environ.pop("EDGE_TRADER_LIVE_FREEZE_SCOPE", None)
            else:
                os.environ["EDGE_TRADER_LIVE_FREEZE_SCOPE"] = old_scope

        self.assertEqual(decisions, [])
        self.assertEqual(diag["blocked_by_freeze_all_count"], 1)


if __name__ == "__main__":
    unittest.main()
