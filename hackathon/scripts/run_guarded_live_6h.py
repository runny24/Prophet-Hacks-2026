"""Run guarded live price-trading for a bounded wall-clock session.

This script intentionally keeps the live path narrow:
- live submit requires EDGE_TRADER_ALLOW_LIVE_SUBMIT=1 and guard mode on
- threshold-debug is disabled because it blocks submit
- price trading is enabled with tiny position limits
- every tick plan is copied into rolling JSONL logs for inspection
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_four_tick_diagnostic import (
    compute_sleep_seconds,
    diagnostic_profile_defaults,
    latest_plan_marker,
    read_latest_plan,
    retry_seconds_from_runner_result,
)


DEFAULT_MEMORY_PATH = Path(".edge_trader/memory.jsonl")
DEFAULT_OUTPUT_ROOT = Path("outputs/live_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guarded live Edge Trader for a bounded session.")
    parser.add_argument("--diagnostic-profile", choices=("fast", "normal", "wide", "live_guarded"), default="live_guarded")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--ticks", type=int, help="Stop after this many completed claimed ticks.")
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--rag-budget", type=int)
    parser.add_argument("--blf-budget", type=int)
    parser.add_argument("--blf-max-steps", type=int, default=2)
    parser.add_argument("--max-trade-size", type=int, default=1)
    parser.add_argument("--max-position-per-market", type=int, default=3)
    parser.add_argument("--max-total-open-positions", type=int, default=12)
    parser.add_argument("--max-session-loss", type=float, default=100.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait-for-tick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-wait-for-tick-seconds", type=int, default=900)
    parser.add_argument("--no-tick-retry-seconds", type=int, default=30)
    parser.add_argument("--sleep-mode", choices=("fixed", "next-tick-buffer"), default="next-tick-buffer")
    parser.add_argument("--sleep-seconds", type=int, default=720)
    parser.add_argument("--tick-buffer-seconds", type=int, default=60)
    parser.add_argument("--min-sleep-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = guarded_live_env(args)
    print_live_header(env, args)
    report = run_guarded_live(args=args, env=env, output_dir=output_dir)
    write_summary(output_dir / "summary.md", report)
    print(f"wrote live_run_dir={output_dir}")


def guarded_live_env(args: argparse.Namespace) -> dict[str, str]:
    profile = diagnostic_profile_defaults(args.diagnostic_profile)
    max_markets = int(args.max_markets if args.max_markets is not None else profile["max_markets"])
    rag_budget = int(args.rag_budget if args.rag_budget is not None else profile["rag_budget"])
    blf_budget = int(args.blf_budget if args.blf_budget is not None else profile["blf_budget"])
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH", "")
    local_paths = "ai-prophet/packages/core:."
    env["PYTHONPATH"] = f"{local_paths}:{current_pythonpath}" if current_pythonpath else local_paths
    env["EDGE_TRADER_SLUG"] = f"rag-blf-edge-trader-live-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    env.update(
        {
            "EDGE_TRADER_ENABLE_RAG": "1",
            "EDGE_TRADER_ENABLE_LLM_RAG": "1",
            "EDGE_TRADER_ENABLE_BLF": "1",
            "EDGE_TRADER_ENABLE_PRICE_TRADING": "1",
            "EDGE_TRADER_ENABLE_CANDIDATE_RERANK": "1",
            "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK": str(rag_budget),
            "EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK": str(blf_budget),
            "EDGE_TRADER_BLF_MAX_STEPS": str(args.blf_max_steps),
            "EDGE_TRADER_RAG_SELECTION_MODE": "alpha_diagnostic",
            "EDGE_TRADER_RAG_CONCURRENCY": str(profile["rag_concurrency"]),
            "EDGE_TRADER_LLM_RAG_CONCURRENCY": str(profile["llm_rag_concurrency"]),
            "EDGE_TRADER_BLF_CONCURRENCY": str(profile["blf_concurrency"]),
            "EDGE_TRADER_TICK_TIME_BUDGET_SECONDS": str(profile["tick_time_budget_seconds"]),
            "EDGE_TRADER_STOP_NEW_WORK_BEFORE_DEADLINE_SECONDS": str(
                profile["stop_new_work_before_deadline_seconds"]
            ),
            "EDGE_TRADER_DRY_RUN": "0",
            "EDGE_TRADER_THRESHOLD_DEBUG": "0",
            "EDGE_TRADER_LIVE_GUARD_MODE": "1",
            "EDGE_TRADER_MAX_TRADE_SIZE": str(args.max_trade_size),
            "EDGE_TRADER_MAX_POSITION_PER_MARKET": str(args.max_position_per_market),
            "EDGE_TRADER_MAX_TOTAL_OPEN_POSITIONS": str(args.max_total_open_positions),
            "EDGE_TRADER_MAX_SESSION_LOSS": str(args.max_session_loss),
            "EDGE_TRADER_LIVE_MAX_MARKETS": str(max_markets),
        }
    )
    return env


def print_live_header(env: dict[str, str], args: argparse.Namespace) -> None:
    live_enabled = env.get("EDGE_TRADER_ALLOW_LIVE_SUBMIT", "0").lower() in {"1", "true", "yes"}
    print(f"LIVE_SUBMIT_ENABLED={live_enabled}")
    print(f"GUARD_MODE={env.get('EDGE_TRADER_LIVE_GUARD_MODE')}")
    print(f"max_trade_size={args.max_trade_size}")
    print(f"max_position_per_market={args.max_position_per_market}")
    print(f"max_session_loss={args.max_session_loss}")
    if not live_enabled:
        print("EDGE_TRADER_ALLOW_LIVE_SUBMIT is not enabled; runner live guard will block submissions.")


def run_guarded_live(*, args: argparse.Namespace, env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + args.duration_hours * 3600
    tick_count = 0
    failed_count = 0
    submit_count = 0
    unsafe_stop = False
    freeze_scope = "none"
    freeze_reason = ""
    consecutive_high_rate_limit_ticks = 0
    ticks_path = output_dir / "ticks.jsonl"
    trades_path = output_dir / "trades.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    max_markets = int(env["EDGE_TRADER_LIVE_MAX_MARKETS"])

    while time.monotonic() < deadline and (args.ticks is None or tick_count < args.ticks):
        started = time.monotonic()
        previous_marker = latest_plan_marker(args.memory_path)
        cmd = build_live_tick_command(args.python, max_markets=max_markets)
        waited = 0
        try:
            env["EDGE_TRADER_FREEZE_NEW_ENTRIES"] = "0"
            env["EDGE_TRADER_LIVE_FREEZE_SCOPE"] = freeze_scope
            env["EDGE_TRADER_LIVE_FREEZE_REASON"] = freeze_reason
            while True:
                result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout or f"runner exited {result.returncode}")[-1000:])
                current_marker = latest_plan_marker(args.memory_path)
                if current_marker != previous_marker:
                    break
                if not args.wait_for_tick or waited >= args.max_wait_for_tick_seconds:
                    raise RuntimeError("runner exited without a new plan; likely no tick was available")
                retry = retry_seconds_from_runner_result(result) or args.no_tick_retry_seconds
                retry = min(retry, args.max_wait_for_tick_seconds - waited)
                if retry <= 0:
                    raise RuntimeError("runner exited without a new plan; likely no tick was available")
                print(f"no tick available; retrying in {retry}s")
                time.sleep(retry)
                waited += retry

            plan = read_latest_plan(args.memory_path)
            summary = live_tick_summary(plan, elapsed_ms(started), waited)
            append_jsonl(ticks_path, summary)
            for decision in plan.get("decisions", []):
                append_jsonl(decisions_path, {"tick_id": plan.get("tick_id"), **decision})
            for fill in (plan.get("submission") or {}).get("fills", []):
                append_jsonl(trades_path, {"tick_id": plan.get("tick_id"), **fill})
            tick_count += 1
            submit_count += int(bool((plan.get("submission") or {}).get("submit_called")))
            if rate_limit_failure_ratio(plan) > 0.50:
                consecutive_high_rate_limit_ticks += 1
            else:
                consecutive_high_rate_limit_ticks = 0
            fallback_blocks = fallback_block_count(plan)
            freeze_scope, freeze_reason = determine_next_freeze_scope(
                rate_limit_ratio=rate_limit_failure_ratio(plan),
                fallback_blocks=fallback_blocks,
                consecutive_high_rate_limit_ticks=consecutive_high_rate_limit_ticks,
            )
            print_one_line_summary(summary)
            if unsafe_trade_size(plan, args.max_trade_size):
                unsafe_stop = True
                break
        except Exception as exc:
            failed_count += 1
            append_jsonl(ticks_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"tick failed: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                break

        if time.monotonic() >= deadline:
            break
        if args.ticks is not None and tick_count >= args.ticks:
            break
        sleep_for, target = compute_sleep_seconds(
            now=datetime.now(UTC),
            mode=args.sleep_mode,
            fixed_sleep_seconds=args.sleep_seconds,
            tick_buffer_seconds=args.tick_buffer_seconds,
            min_sleep_seconds=args.min_sleep_seconds,
        )
        print(f"sleeping {sleep_for}s until target_boundary={target.isoformat() if target else None}")
        time.sleep(min(sleep_for, max(0, int(deadline - time.monotonic()))))

    live_aggregate = aggregate_live_tick_summaries(ticks_path)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "duration_hours_requested": args.duration_hours,
        "ticks_completed": tick_count,
        "ticks_failed": failed_count,
        "submit_called_count": submit_count,
        "unsafe_stop": unsafe_stop,
        "new_entries_frozen": freeze_scope != "none",
        "freeze_scope": freeze_scope,
        "freeze_reason": freeze_reason,
        "output_dir": str(output_dir),
        **live_aggregate,
    }


def build_live_tick_command(python_executable: str, *, max_markets: int) -> list[str]:
    return [
        python_executable,
        "-m",
        "edge_trader_bot.runner",
        "--once",
        "--max-markets",
        str(max_markets),
        "--enable-blf",
        "--allow-live-submit",
    ]


def live_tick_summary(plan: dict[str, Any], tick_elapsed_ms: int, waited_seconds: int) -> dict[str, Any]:
    submission = plan.get("submission") or {}
    return {
        "status": "succeeded",
        "tick_id": plan.get("tick_id"),
        "candidate_set_id": plan.get("candidate_set_id"),
        "generated_at": plan.get("generated_at"),
        "candidates_processed": plan.get("candidates_processed", 0),
        "trade_count": plan.get("trade_count", 0),
        "count_price_signals": plan.get("count_price_signals", 0),
        "count_trade_intents": plan.get("count_trade_intents", 0),
        "count_exit_intents": plan.get("count_exit_intents", 0),
        "submit_called": bool(submission.get("submit_called", False)),
        "live_submit_enabled": (not plan.get("dry_run", True)) and bool(plan.get("allow_live_submit", False)),
        "guard_mode": True,
        "submission_reason": submission.get("reason"),
        "live_submission_reason": plan.get("live_submission_reason"),
        "accepted": submission.get("accepted", 0),
        "rejected": submission.get("rejected", 0),
        "submitted_trade_count": submission.get("accepted", 0),
        "skipped_by_live_safety_count": plan.get("skipped_by_live_safety_count", 0),
        "blocked_reasons_counts": plan.get("live_blocked_reason_counts", {}),
        "live_tradable_candidate_count": plan.get("live_tradable_candidate_count", 0),
        "live_price_action_candidate_count": plan.get("live_price_action_candidate_count", 0),
        "live_fresh_event_candidate_count": plan.get("live_fresh_event_candidate_count", 0),
        "live_forecast_mispricing_candidate_count": plan.get("live_forecast_mispricing_candidate_count", 0),
        "live_candidate_pool_count": plan.get("live_candidate_pool_count", 0),
        "live_candidate_pool_top10": plan.get("live_candidate_pool_top10", [])[:5],
        "live_candidates_sent_to_rag_count": plan.get("live_candidates_sent_to_rag_count", 0),
        "live_candidates_sent_to_rag_top10": plan.get("live_candidates_sent_to_rag_top10", [])[:5],
        "live_candidates_sent_to_blf_count": plan.get("live_candidates_sent_to_blf_count", 0),
        "live_candidates_sent_to_blf_top10": plan.get("live_candidates_sent_to_blf_top10", [])[:5],
        "not_evaluated_by_forecast_budget_count": plan.get("not_evaluated_by_forecast_budget_count", 0),
        "analyzed_live_candidate_count": plan.get("analyzed_live_candidate_count", 0),
        "analyzed_live_candidate_top10": plan.get("analyzed_live_candidate_top10", [])[:5],
        "rag_failure_count_by_reason": plan.get("rag_failure_count_by_reason", {}),
        "cached_evidence_fallback_count": plan.get("cached_evidence_fallback_count", 0),
        "cached_evidence_used_count": plan.get("cached_evidence_used_count", 0),
        "cached_evidence_too_old_count": plan.get("cached_evidence_too_old_count", 0),
        "metadata_only_fallback_count": plan.get("metadata_only_fallback_count", 0),
        "previous_p_final_used_count": plan.get("previous_p_final_used_count", 0),
        "live_forecast_mispricing_cached_candidate_count": plan.get(
            "live_forecast_mispricing_cached_candidate_count", 0
        ),
        "rejected_cached_evidence_top10": plan.get("rejected_cached_evidence_top10", [])[:5],
        "metadata_only_rejected_top10": plan.get("metadata_only_rejected_top10", [])[:5],
        "recovered_rag_candidates_top10": plan.get("recovered_rag_candidates_top10", [])[:5],
        "skipped_by_price_action_history_count": plan.get("skipped_by_price_action_history_count", 0),
        "skipped_by_fresh_event_safety_count": plan.get("skipped_by_fresh_event_safety_count", 0),
        "fresh_event_candidates": plan.get("fresh_event_candidates", [])[:5],
        "fresh_event_rejected_top10": plan.get("fresh_event_rejected_top10", [])[:5],
        "rejected_forecast_mispricing_top10": plan.get("rejected_forecast_mispricing_top10", [])[:5],
        "forecast_gate_rejection_top10": plan.get("forecast_gate_rejection_top10", [])[:5],
        "live_freeze_active": plan.get("live_freeze_active", False),
        "live_freeze_reason": plan.get("live_freeze_reason", ""),
        "freeze_scope": plan.get("freeze_scope", "none"),
        "clean_price_action_candidates_count": plan.get("clean_price_action_candidates_count", 0),
        "blocked_by_freeze_forecast_only_count": plan.get("blocked_by_freeze_forecast_only_count", 0),
        "blocked_by_freeze_all_count": plan.get("blocked_by_freeze_all_count", 0),
        "live_trade_candidates": plan.get("live_trade_candidates", [])[:5],
        "current_portfolio_summary": {
            "positions_held": plan.get("positions_held", 0),
        },
        "candidate_rerank_enabled": (plan.get("candidate_selection_summary") or {}).get(
            "candidate_rerank_enabled",
            False,
        ),
        "selected_bucket_counts": (plan.get("candidate_selection_summary") or {}).get("selected_bucket_counts", {}),
        "top_selected_candidates": (plan.get("candidate_selection_summary") or {}).get("top_selected_candidates", [])[:5],
        "tick_elapsed_ms": tick_elapsed_ms,
        "no_tick_wait_seconds": waited_seconds,
        "top_trade_candidates": plan.get("top_trade_candidates", [])[:5],
        "exit_candidates": plan.get("exit_candidates", [])[:5],
        "warnings": plan.get("warnings", []),
    }


def rate_limit_failure_ratio(plan: dict[str, Any]) -> float:
    attempts = int(plan.get("llm_attempted", 0) or 0) + int(plan.get("blf_markets_attempted", 0) or 0)
    failures = int(plan.get("llm_failed", 0) or 0) + int(plan.get("blf_failed", 0) or 0)
    return failures / attempts if attempts else 0.0


def fallback_block_count(plan: dict[str, Any]) -> int:
    counts = plan.get("live_blocked_reason_counts", {}) or {}
    return sum(
        int(value or 0)
        for reason, value in counts.items()
        if "fallback" in reason or "rate_limit" in reason
    )


def determine_next_freeze_scope(
    *,
    rate_limit_ratio: float,
    fallback_blocks: int,
    consecutive_high_rate_limit_ticks: int,
) -> tuple[str, str]:
    if consecutive_high_rate_limit_ticks > 2:
        return "all_entries", "consecutive_high_rate_limit_ticks"
    if rate_limit_ratio > 0.50:
        return "forecast_dependent_entries_only", "high_rate_limit_ratio"
    if fallback_blocks > 3:
        return "forecast_dependent_entries_only", "fallback_or_rate_limit_blocks"
    return "none", ""


def unsafe_trade_size(plan: dict[str, Any], max_trade_size: int) -> bool:
    for decision in plan.get("decisions", []):
        try:
            shares = int(decision.get("shares", 0))
        except (TypeError, ValueError):
            return True
        if shares > max_trade_size:
            return True
    return False


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Guarded Live Run Summary",
                "",
                f"- generated_at: {report['generated_at']}",
                f"- ticks_completed: {report['ticks_completed']}",
                f"- ticks_failed: {report['ticks_failed']}",
                f"- submit_called_count: {report['submit_called_count']}",
                f"- unsafe_stop: {report['unsafe_stop']}",
                f"- new_entries_frozen: {report['new_entries_frozen']}",
                f"- freeze_scope: {report['freeze_scope']}",
                f"- freeze_reason: {report['freeze_reason']}",
                f"- live_tradable_candidate_count: {report.get('live_tradable_candidate_count', 0)}",
                f"- live_fresh_event_candidate_count: {report.get('live_fresh_event_candidate_count', 0)}",
                f"- live_forecast_mispricing_candidate_count: {report.get('live_forecast_mispricing_candidate_count', 0)}",
                f"- live_price_action_candidate_count: {report.get('live_price_action_candidate_count', 0)}",
                f"- live_candidate_pool_count: {report.get('live_candidate_pool_count', 0)}",
                f"- live_candidates_sent_to_rag_count: {report.get('live_candidates_sent_to_rag_count', 0)}",
                f"- live_candidates_sent_to_blf_count: {report.get('live_candidates_sent_to_blf_count', 0)}",
                f"- analyzed_live_candidate_count: {report.get('analyzed_live_candidate_count', 0)}",
                f"- not_evaluated_by_forecast_budget_count: {report.get('not_evaluated_by_forecast_budget_count', 0)}",
                f"- cached_evidence_used_count: {report.get('cached_evidence_used_count', 0)}",
                f"- metadata_only_fallback_count: {report.get('metadata_only_fallback_count', 0)}",
                f"- previous_p_final_used_count: {report.get('previous_p_final_used_count', 0)}",
                f"- skipped_by_live_safety_count: {report.get('skipped_by_live_safety_count', 0)}",
                f"- blocked_reasons_counts: {json.dumps(report.get('blocked_reasons_counts', {}), sort_keys=True)}",
                f"- output_dir: {report['output_dir']}",
                "",
                "Inspect `ticks.jsonl`, `decisions.jsonl`, and `trades.jsonl` for tick-level details.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def aggregate_live_tick_summaries(path: Path) -> dict[str, Any]:
    totals = Counter()
    blocked = Counter()
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "failed":
            continue
        for key in (
            "live_tradable_candidate_count",
            "live_fresh_event_candidate_count",
            "live_forecast_mispricing_candidate_count",
            "live_candidate_pool_count",
            "live_candidates_sent_to_rag_count",
            "live_candidates_sent_to_blf_count",
            "not_evaluated_by_forecast_budget_count",
            "analyzed_live_candidate_count",
            "cached_evidence_used_count",
            "cached_evidence_too_old_count",
            "metadata_only_fallback_count",
            "previous_p_final_used_count",
            "live_forecast_mispricing_cached_candidate_count",
            "live_price_action_candidate_count",
            "skipped_by_live_safety_count",
        ):
            totals[key] += int(row.get(key, 0) or 0)
        blocked.update(row.get("blocked_reasons_counts", {}) or {})
    return {**dict(totals), "blocked_reasons_counts": dict(blocked)}


def print_one_line_summary(summary: dict[str, Any]) -> None:
    print(
        "tick={tick_id} processed={processed} signals={signals} intents={intents} exits={exits} submit={submit} accepted={accepted} elapsed_ms={elapsed}".format(
            tick_id=summary.get("tick_id"),
            processed=summary.get("candidates_processed"),
            signals=summary.get("count_price_signals"),
            intents=summary.get("count_trade_intents"),
            exits=summary.get("count_exit_intents"),
            submit=summary.get("submit_called"),
            accepted=summary.get("accepted"),
            elapsed=summary.get("tick_elapsed_ms"),
        )
    )


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    main()
