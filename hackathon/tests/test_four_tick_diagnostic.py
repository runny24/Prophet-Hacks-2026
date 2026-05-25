from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_four_tick_diagnostic import (
    aggregate_tick_reports,
    assert_safe_env,
    compute_sleep_seconds,
    diagnostic_profile_defaults,
    render_markdown,
    repeated_market_tracking,
    run_diagnostic,
    safe_diagnostic_env,
    summarize_plan,
)
from datetime import UTC, datetime


def make_plan(tick_id: str, *, submit_called: bool = False, edge: float = 0.01) -> dict:
    return {
        "tick_id": tick_id,
        "candidate_set_id": f"cs-{tick_id}",
        "generated_at": "2026-05-16T00:00:00+00:00",
        "candidates_loaded": 256,
        "candidates_processed": 2,
        "markets_filtered": 254,
        "rag_markets_attempted": 1,
        "rag_markets_scanned": 1,
        "rag_scanner_errors": [],
        "llm_attempted": 1,
        "llm_succeeded": 1,
        "llm_failed": 0,
        "llm_fallback_count": 0,
        "blf_markets_attempted": 1,
        "blf_succeeded": 1,
        "blf_failed": 0,
        "blf_fallback_count": 0,
        "hypothetical_intents_count": 0,
        "trade_count": 0,
        "submission": {"submit_called": submit_called},
        "top_apparent_edges": [{"market_id": "m1", "apparent_edge": edge}],
        "threshold_debug_report": {
            "zero_trade_relaxed_threshold_count": 1,
            "diagnostic_edge_thresholds": [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
            "scenarios": [
                {
                    "min_edge_threshold": 0.005,
                    "hypothetical_trade_count": 1 if edge > 0.04 else 0,
                    "max_edge": edge,
                    "blocked_by_resolution_risk": 0,
                    "blf_enabled": True,
                },
                {
                    "min_edge_threshold": 0.005,
                    "hypothetical_trade_count": 1 if edge > 0.004 else 0,
                    "max_edge": edge,
                    "blocked_by_resolution_risk": 0,
                    "blf_enabled": False,
                }
            ],
        },
        "opportunity_report": {
            "blocked_by_evidence_quality": [{"market_id": "m2"}],
            "blocked_only_by_edge_threshold": [{"market_id": "m1"}],
            "count_positive_taker_edge": 1 if edge > 0 else 0,
            "count_positive_maker_edge": 1,
            "count_positive_mid_edge": 1,
            "count_mid_edge_ge_0_005": 1,
            "count_mid_edge_ge_0_01": 1,
            "count_mid_edge_ge_0_02": 1 if edge + 0.01 >= 0.02 else 0,
            "count_taker_edge_ge_0_005": 1 if edge >= 0.005 else 0,
            "count_taker_edge_ge_0_01": 1 if edge >= 0.01 else 0,
            "count_taker_edge_ge_0_02": 1 if edge >= 0.02 else 0,
            "top_by_maker_edge": [{"market_id": "m1", "best_maker_edge": edge + 0.02}],
            "top_by_mid_edge": [{"market_id": "m1", "best_mid_edge": edge + 0.01}],
            "top_by_spread_adjusted_maker_edge": [
                {"market_id": "m1", "maker_edge_after_half_spread": edge + 0.005}
            ],
            "manual_review_candidates": [
                {
                    "market_id": "m1",
                    "market_type": "election_control",
                    "best_taker_edge": edge,
                    "best_mid_edge": edge + 0.01,
                    "evidence_quality": 4,
                    "confidence": "medium",
                }
            ],
            "blf_changed_probability_most": [
                {"market_id": "m1", "market_type": "election_control", "absolute_change": 0.02}
            ],
            "rag_disagrees_with_market_most": [
                {
                    "market_id": "m1",
                    "market_type": "election_control",
                    "absolute_disagreement": 0.03,
                    "p_market": 0.5,
                    "p_2402": 0.53,
                }
            ],
        },
        "pipeline_debug": [
            {
                "market_id": "m1",
                "question": "Will X happen?",
                "market_type": "election_control",
                "best_side": "YES",
                "best_edge": edge,
                "best_taker_edge": edge,
                "best_maker_edge": edge + 0.02,
                "best_mid_edge": edge + 0.01,
                "maker_edge_after_half_spread": edge + 0.005,
                "maker_edge_after_full_spread": edge - 0.01,
                "passive_opportunity_quality": "strong",
                "p_market": 0.50,
                "p_2402_final_after_shrinkage": 0.53,
                "p_blf_final_after_shrinkage": 0.52,
                "p_final_after_blf": 0.51 + edge,
                "sports_quantitative_support": False,
                "sports_support_source_type": "none",
                "evidence_quality": 4,
                "confidence": "medium",
                "hold_reasons": ["edge_below_policy_threshold"],
                "decision": None,
                "llm_elapsed_ms": 12,
                "scanner_elapsed_ms": 20,
                "blf_elapsed_ms": 7,
                "blf_package": {"elapsed_ms": 7},
                "absence_of_evidence_penalty_detected": False,
            }
        ],
        "total_rag_elapsed_ms": 20,
        "total_llm_elapsed_ms": 12,
        "total_blf_elapsed_ms": 7,
        "tick_time_budget_seconds": 420,
        "early_stop_triggered": False,
        "skipped_due_to_deadline_count": 0,
        "stage_wall_times": {"rag_wall_time_ms": 20, "llm_rag_wall_time_ms": 12, "blf_wall_time_ms": 7},
        "concurrency_settings": {"rag_concurrency": 4, "llm_rag_concurrency": 3, "blf_concurrency": 2},
        "live_blocked_reason_counts": {"fallback_or_rate_limit_risk:llm_rate_limit": 1},
        "live_tradable_candidate_count": 1,
        "skipped_by_live_safety_count": 1,
        "live_trade_candidates": [{"market_id": "m1", "signal_type": "momentum"}],
        "rejected_trade_candidates_top10": [{"market_id": "m2", "signal_type": "forecast_edge"}],
        "signal_type_counts": {"momentum": 1, "forecast_edge": 1},
        "repeated_history_candidate_count": 1,
        "fallback_risk_candidate_count": 1,
        "price_only_candidate_count": 1,
        "forecast_only_candidate_count": 1,
    }


class FourTickDiagnosticTests(unittest.TestCase):
    def test_multi_tick_runner_runs_four_iterations(self) -> None:
        calls = []
        plans = [make_plan(f"t{i}", edge=0.01 * i) for i in range(1, 5)]

        def runner(cmd, env, check):
            calls.append((cmd, env, check))
            return SimpleNamespace(returncode=0)

        def reader(path):
            return plans[len(calls) - 1]

        with tempfile.TemporaryDirectory() as tmp:
            report = run_diagnostic(
                ticks=4,
                sleep_seconds=0,
                max_markets=100,
                continue_on_error=False,
                memory_path=Path("unused"),
                output_dir=Path(tmp),
                python_executable="python",
                runner=runner,
                sleeper=lambda seconds: None,
                plan_reader=reader,
            )

        self.assertEqual(len(calls), 4)
        self.assertIn("100", calls[0][0])
        self.assertEqual(calls[0][1]["EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK"], "8")
        self.assertEqual(calls[0][1]["EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK"], "2")
        self.assertEqual(calls[0][1]["EDGE_TRADER_RAG_CONCURRENCY"], "4")
        self.assertEqual(calls[0][1]["EDGE_TRADER_TICK_TIME_BUDGET_SECONDS"], "420")
        self.assertEqual(calls[0][1]["EDGE_TRADER_ALLOW_LIVE_SUBMIT"], "0")
        self.assertEqual(calls[0][1]["EDGE_TRADER_RAG_SELECTION_MODE"], "alpha_diagnostic")
        self.assertEqual(report["total_ticks_attempted"], 4)
        self.assertEqual(report["total_ticks_succeeded"], 4)
        self.assertEqual(report["total_live_submit_calls"], 0)

    def test_continues_on_one_tick_failure_when_enabled(self) -> None:
        calls = []

        def runner(cmd, env, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

        def reader(path):
            return make_plan(f"t{len(calls)}")

        with tempfile.TemporaryDirectory() as tmp:
            report = run_diagnostic(
                ticks=4,
                sleep_seconds=0,
                max_markets=50,
                continue_on_error=True,
                memory_path=Path("unused"),
                output_dir=Path(tmp),
                python_executable="python",
                runner=runner,
                sleeper=lambda seconds: None,
                plan_reader=reader,
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(report["total_ticks_failed"], 1)
        self.assertEqual(report["total_ticks_succeeded"], 3)

    def test_aborts_if_live_submit_enabled(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "live submit"):
            assert_safe_env(
                {
                    "EDGE_TRADER_DRY_RUN": "1",
                    "EDGE_TRADER_ALLOW_LIVE_SUBMIT": "1",
                    "EDGE_TRADER_THRESHOLD_DEBUG": "1",
                }
            )

    def test_default_safe_diagnostic_env(self) -> None:
        env = safe_diagnostic_env({})

        self.assertEqual(env["EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK"], "15")
        self.assertEqual(env["EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK"], "5")
        self.assertEqual(env["EDGE_TRADER_BLF_MAX_STEPS"], "2")
        self.assertEqual(env["EDGE_TRADER_ALLOW_LIVE_SUBMIT"], "0")
        self.assertEqual(env["EDGE_TRADER_MAX_TICK_SECONDS"], "600")
        self.assertEqual(env["EDGE_TRADER_LATENCY_WARNING_SECONDS"], "450")
        self.assertEqual(env["EDGE_TRADER_RAG_CONCURRENCY"], "4")
        self.assertEqual(env["EDGE_TRADER_LLM_RAG_CONCURRENCY"], "3")
        self.assertEqual(env["EDGE_TRADER_BLF_CONCURRENCY"], "2")
        self.assertEqual(env["EDGE_TRADER_ENABLE_CANDIDATE_RERANK"], "1")

    def test_diagnostic_profile_defaults(self) -> None:
        self.assertEqual(diagnostic_profile_defaults("fast")["rag_budget"], 8)
        self.assertEqual(diagnostic_profile_defaults("fast")["blf_budget"], 2)
        self.assertEqual(diagnostic_profile_defaults("fast")["tick_time_budget_seconds"], 420)
        self.assertEqual(diagnostic_profile_defaults("normal")["rag_budget"], 15)
        self.assertEqual(diagnostic_profile_defaults("normal")["blf_budget"], 5)
        self.assertEqual(diagnostic_profile_defaults("wide")["rag_budget"], 25)
        self.assertEqual(diagnostic_profile_defaults("wide")["blf_budget"], 5)
        self.assertEqual(diagnostic_profile_defaults("live_guarded")["rag_budget"], 8)
        self.assertEqual(diagnostic_profile_defaults("live_guarded")["blf_budget"], 2)

    def test_cli_overrides_profile_values(self) -> None:
        calls = []

        def runner(cmd, env, check):
            calls.append((cmd, env))
            return SimpleNamespace(returncode=0)

        def reader(path):
            return make_plan("t1")

        with tempfile.TemporaryDirectory() as tmp:
            run_diagnostic(
                ticks=1,
                sleep_seconds=0,
                diagnostic_profile="fast",
                max_markets=77,
                rag_budget=11,
                blf_budget=3,
                rag_concurrency=6,
                llm_rag_concurrency=4,
                blf_concurrency=3,
                tick_time_budget_seconds=500,
                stop_new_work_before_deadline_seconds=80,
                continue_on_error=False,
                memory_path=Path("unused"),
                output_dir=Path(tmp),
                python_executable="python",
                runner=runner,
                sleeper=lambda seconds: None,
                plan_reader=reader,
            )

        self.assertIn("77", calls[0][0])
        self.assertEqual(calls[0][1]["EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK"], "11")
        self.assertEqual(calls[0][1]["EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK"], "3")
        self.assertEqual(calls[0][1]["EDGE_TRADER_RAG_CONCURRENCY"], "6")
        self.assertEqual(calls[0][1]["EDGE_TRADER_LLM_RAG_CONCURRENCY"], "4")
        self.assertEqual(calls[0][1]["EDGE_TRADER_BLF_CONCURRENCY"], "3")
        self.assertEqual(calls[0][1]["EDGE_TRADER_TICK_TIME_BUDGET_SECONDS"], "500")
        self.assertEqual(calls[0][1]["EDGE_TRADER_STOP_NEW_WORK_BEFORE_DEADLINE_SECONDS"], "80")

    def test_next_tick_buffer_sleep_avoids_full_900_seconds(self) -> None:
        sleep_for, target = compute_sleep_seconds(
            now=datetime(2026, 5, 17, 1, 7, 0, tzinfo=UTC),
            mode="next-tick-buffer",
            fixed_sleep_seconds=720,
            tick_buffer_seconds=60,
            min_sleep_seconds=30,
        )

        self.assertEqual(target.minute, 15)
        self.assertEqual(sleep_for, 420)

    def test_aggregate_report_sums_attempts_and_failures(self) -> None:
        tick1 = summarize_plan(make_plan("t1", edge=0.05), tick_elapsed_ms=100, tick_index=1)
        tick2 = summarize_plan(make_plan("t2", edge=0.02), tick_elapsed_ms=200, tick_index=2)
        tick2["rag_failed"] = 1
        tick2["llm_failed"] = 1
        tick2["blf_failed"] = 1

        report = aggregate_tick_reports([tick1, tick2])

        self.assertEqual(report["total_rag_attempts"], 2)
        self.assertEqual(report["total_rag_failures"], 1)
        self.assertEqual(report["total_llm_failures"], 1)
        self.assertEqual(report["total_blf_failures"], 1)
        self.assertEqual(report["total_candidates_processed"], 4)
        self.assertEqual(report["count_positive_maker_edge"], 2)
        self.assertIn("top_10_maker_edges", report)
        self.assertIn("top_10_spread_adjusted_maker_edges", report)
        self.assertEqual(report["count_mid_edge_ge_0_005"], 2)
        self.assertGreater(report["cost_latency_summary"]["average_llm_time_per_attempt_ms"], 0)
        self.assertIn("trade_readiness_verdict", report)
        self.assertIn("relaxed_trades_with_blf_enabled", report)
        self.assertIn("relaxed_trades_without_blf", report)
        self.assertEqual(report["price_trading_summary"]["live_tradable_candidate_count"], 2)
        self.assertIn("live_blocked_reason_counts", report["price_trading_summary"])
        self.assertEqual(report["price_trading_summary"]["signal_type_counts"]["momentum"], 2)

    def test_aggregate_report_detects_submit_called_unsafe(self) -> None:
        tick = summarize_plan(make_plan("t1", submit_called=True), tick_elapsed_ms=100, tick_index=1)

        report = aggregate_tick_reports([tick])

        self.assertEqual(report["total_live_submit_calls"], 1)
        self.assertTrue(report["unsafe_submit_detected"])

    def test_repeated_market_tracking(self) -> None:
        tick1 = summarize_plan(make_plan("t1", edge=0.01), tick_elapsed_ms=100, tick_index=1)
        tick2 = summarize_plan(make_plan("t2", edge=0.04), tick_elapsed_ms=100, tick_index=2)

        repeated = repeated_market_tracking([tick1, tick2])

        self.assertEqual(repeated[0]["market_id"], "m1")
        self.assertGreater(repeated[0]["p_final_movement"], 0)

    def test_markdown_report_includes_required_sections(self) -> None:
        report = aggregate_tick_reports([summarize_plan(make_plan("t1", edge=0.05), tick_elapsed_ms=100, tick_index=1)])
        md = render_markdown(report)

        self.assertIn("Top Apparent Edges", md)
        self.assertIn("Top Maker Edges", md)
        self.assertIn("Top Mid Edges", md)
        self.assertIn("Top Spread-Adjusted Maker Edges", md)
        self.assertIn("Manual Review Candidates", md)
        self.assertIn("Top RAG-Market Disagreements", md)
        self.assertIn("Top BLF Adjustments", md)
        self.assertIn("Market-Type Distribution", md)
        self.assertIn("Relaxed-threshold scenarios producing trades", md)
        self.assertIn("Trade Readiness Verdict", md)
        self.assertIn("Relaxed Threshold Split", md)
        self.assertIn("4-Tick Reliability Verdict", md)

    def test_latency_warning_and_slowest_lists_populated(self) -> None:
        tick = summarize_plan(make_plan("t1", edge=0.01), tick_elapsed_ms=728_000, tick_index=1)

        report = aggregate_tick_reports([tick])

        self.assertTrue(tick["latency_warnings"])
        self.assertTrue(report["cost_latency_summary"]["latency_warnings"])
        self.assertEqual(report["cost_latency_summary"]["latency_status"], "critical")
        self.assertEqual(report["cost_latency_summary"]["slowest_markets_by_llm_elapsed_ms"][0]["market_id"], "m1")
        self.assertEqual(report["cost_latency_summary"]["slowest_markets_by_rag_elapsed_ms"][0]["market_id"], "m1")
        self.assertEqual(report["cost_latency_summary"]["slowest_markets_by_blf_elapsed_ms"][0]["market_id"], "m1")

    def test_latency_warning_triggers_above_450_seconds(self) -> None:
        tick = summarize_plan(make_plan("t1", edge=0.01), tick_elapsed_ms=491_048, tick_index=1)
        report = aggregate_tick_reports([tick])

        self.assertEqual(report["cost_latency_summary"]["latency_status"], "warning")
        self.assertTrue(tick["latency_warnings"])

    def test_no_tick_retry_does_not_count_completed_tick(self) -> None:
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"

            def runner(cmd, env, check):
                calls.append(cmd)
                if len(calls) == 2:
                    memory_path.write_text(__import__("json").dumps(make_plan("t1")) + "\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            report = run_diagnostic(
                ticks=1,
                sleep_seconds=0,
                continue_on_error=False,
                memory_path=memory_path,
                output_dir=Path(tmp),
                python_executable="python",
                runner=runner,
                sleeper=lambda seconds: None,
                wait_for_tick=True,
                max_wait_for_tick_seconds=60,
                no_tick_retry_seconds=1,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(report["total_ticks_succeeded"], 1)
        self.assertEqual(report["total_ticks_attempted"], 1)


if __name__ == "__main__":
    unittest.main()
