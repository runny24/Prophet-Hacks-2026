from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_four_tick_diagnostic import (
    aggregate_tick_reports,
    assert_safe_env,
    render_markdown,
    repeated_market_tracking,
    run_diagnostic,
    summarize_plan,
)


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
            "scenarios": [
                {
                    "min_edge_threshold": 0.04,
                    "hypothetical_trade_count": 1 if edge > 0.04 else 0,
                    "max_edge": edge,
                    "blocked_by_resolution_risk": 0,
                }
            ],
        },
        "opportunity_report": {
            "blocked_by_evidence_quality": [{"market_id": "m2"}],
            "blocked_only_by_edge_threshold": [{"market_id": "m1"}],
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
                "p_market": 0.50,
                "p_2402_final_after_shrinkage": 0.53,
                "p_blf_final_after_shrinkage": 0.52,
                "p_final_after_blf": 0.51 + edge,
                "evidence_quality": 4,
                "confidence": "medium",
                "hold_reasons": ["edge_below_policy_threshold"],
                "decision": None,
                "llm_elapsed_ms": 12,
                "scanner_elapsed_ms": 20,
                "blf_package": {"elapsed_ms": 7},
                "absence_of_evidence_penalty_detected": False,
            }
        ],
        "total_rag_elapsed_ms": 20,
        "total_llm_elapsed_ms": 12,
        "total_blf_elapsed_ms": 7,
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
                max_markets=50,
                continue_on_error=False,
                memory_path=Path("unused"),
                output_dir=Path(tmp),
                python_executable="python",
                runner=runner,
                sleeper=lambda seconds: None,
                plan_reader=reader,
            )

        self.assertEqual(len(calls), 4)
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
        self.assertIn("Top RAG-Market Disagreements", md)
        self.assertIn("Top BLF Adjustments", md)
        self.assertIn("Market-Type Distribution", md)
        self.assertIn("Relaxed-threshold scenarios producing trades", md)


if __name__ == "__main__":
    unittest.main()
