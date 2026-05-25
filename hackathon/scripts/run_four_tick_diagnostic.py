"""Run a safe four-tick diagnostic dry-run session.

This script wraps the existing single-tick runner. It never enables live submit:
each tick is forced through dry-run + threshold-debug and the resulting plan is
read from the local JSONL memory for per-tick and aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_trader_bot.live_readiness import LiveReadinessConfig, evaluate_tick_live_readiness


DEFAULT_MEMORY_PATH = Path(".edge_trader/memory.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four safe diagnostic dry-run ticks.")
    parser.add_argument("--ticks", type=int, default=4, help="Number of diagnostic ticks to attempt.")
    parser.add_argument("--sleep-seconds", type=int, default=720, help="Fallback/fixed sleep. 900s plus processing can skip a 15-minute tick.")
    parser.add_argument("--sleep-mode", choices=("fixed", "next-tick-buffer"), default="next-tick-buffer")
    parser.add_argument("--tick-buffer-seconds", type=int, default=60)
    parser.add_argument("--min-sleep-seconds", type=int, default=30)
    parser.add_argument("--diagnostic-profile", choices=("fast", "normal", "wide", "live_guarded"), default="fast")
    parser.add_argument("--max-markets", type=int, default=None, help="Max markets passed to the single-tick runner.")
    parser.add_argument("--rag-budget", type=int, default=None, help="Normal=15, wide=25 unless overridden.")
    parser.add_argument("--blf-budget", type=int, default=None)
    parser.add_argument("--blf-max-steps", type=int, default=2)
    parser.add_argument("--rag-concurrency", type=int, default=None)
    parser.add_argument("--llm-rag-concurrency", type=int, default=None)
    parser.add_argument("--blf-concurrency", type=int, default=None)
    parser.add_argument("--tick-time-budget-seconds", type=int, default=None)
    parser.add_argument("--stop-new-work-before-deadline-seconds", type=int, default=None)
    parser.add_argument("--rag-selection-mode", default="alpha_diagnostic", choices=("current", "alpha_diagnostic"))
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after one tick fails.")
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-sleep", action="store_true", help="Skip sleeping between ticks; useful for tests.")
    parser.add_argument("--wait-for-tick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-wait-for-tick-seconds", type=int, default=900)
    parser.add_argument("--no-tick-retry-seconds", type=int, default=30)
    parser.add_argument("--duration-hours", type=float, help="Run dry-run diagnostics until this wall-clock duration expires.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_diagnostic(
        ticks=max(args.ticks, int(args.duration_hours * 4) if args.duration_hours else args.ticks),
        sleep_seconds=0 if args.no_sleep else args.sleep_seconds,
        sleep_mode="fixed" if args.no_sleep else args.sleep_mode,
        tick_buffer_seconds=args.tick_buffer_seconds,
        min_sleep_seconds=args.min_sleep_seconds,
        diagnostic_profile=args.diagnostic_profile,
        max_markets=args.max_markets,
        rag_budget=args.rag_budget,
        blf_budget=args.blf_budget,
        blf_max_steps=args.blf_max_steps,
        rag_concurrency=args.rag_concurrency,
        llm_rag_concurrency=args.llm_rag_concurrency,
        blf_concurrency=args.blf_concurrency,
        tick_time_budget_seconds=args.tick_time_budget_seconds,
        stop_new_work_before_deadline_seconds=args.stop_new_work_before_deadline_seconds,
        rag_selection_mode=args.rag_selection_mode,
        continue_on_error=args.continue_on_error,
        memory_path=args.memory_path,
        output_dir=args.output_dir,
        python_executable=args.python,
        wait_for_tick=args.wait_for_tick,
        max_wait_for_tick_seconds=args.max_wait_for_tick_seconds,
        no_tick_retry_seconds=args.no_tick_retry_seconds,
    )
    print(f"wrote aggregate_json={report['json_path']}")
    print(f"wrote aggregate_markdown={report['markdown_path']}")


def run_diagnostic(
    *,
    ticks: int = 4,
    sleep_seconds: int = 720,
    sleep_mode: str = "next-tick-buffer",
    tick_buffer_seconds: int = 60,
    min_sleep_seconds: int = 30,
    diagnostic_profile: str = "fast",
    max_markets: int | None = None,
    rag_budget: int | None = None,
    blf_budget: int | None = None,
    blf_max_steps: int = 2,
    rag_concurrency: int | None = None,
    llm_rag_concurrency: int | None = None,
    blf_concurrency: int | None = None,
    tick_time_budget_seconds: int | None = None,
    stop_new_work_before_deadline_seconds: int | None = None,
    rag_selection_mode: str = "alpha_diagnostic",
    continue_on_error: bool = False,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[int], None] = time.sleep,
    plan_reader: Callable[[Path], dict[str, Any]] | None = None,
    wait_for_tick: bool = True,
    max_wait_for_tick_seconds: int = 600,
    no_tick_retry_seconds: int = 60,
) -> dict[str, Any]:
    profile = diagnostic_profile_defaults(diagnostic_profile)
    max_markets = int(max_markets if max_markets is not None else profile["max_markets"])
    rag_budget = int(rag_budget if rag_budget is not None else profile["rag_budget"])
    blf_budget = int(blf_budget if blf_budget is not None else profile["blf_budget"])
    rag_concurrency = int(rag_concurrency if rag_concurrency is not None else profile["rag_concurrency"])
    llm_rag_concurrency = int(
        llm_rag_concurrency if llm_rag_concurrency is not None else profile["llm_rag_concurrency"]
    )
    blf_concurrency = int(blf_concurrency if blf_concurrency is not None else profile["blf_concurrency"])
    tick_time_budget_seconds = int(
        tick_time_budget_seconds if tick_time_budget_seconds is not None else profile["tick_time_budget_seconds"]
    )
    stop_new_work_before_deadline_seconds = int(
        stop_new_work_before_deadline_seconds
        if stop_new_work_before_deadline_seconds is not None
        else profile["stop_new_work_before_deadline_seconds"]
    )
    env = safe_diagnostic_env(
        os.environ,
        rag_budget=rag_budget,
        blf_budget=blf_budget,
        blf_max_steps=blf_max_steps,
        rag_selection_mode=rag_selection_mode,
        rag_concurrency=rag_concurrency,
        llm_rag_concurrency=llm_rag_concurrency,
        blf_concurrency=blf_concurrency,
        tick_time_budget_seconds=tick_time_budget_seconds,
        stop_new_work_before_deadline_seconds=stop_new_work_before_deadline_seconds,
    )
    assert_safe_env(env)
    enforce_new_plan_marker = plan_reader is None
    plan_reader = plan_reader or read_latest_plan
    tick_reports: list[dict[str, Any]] = []
    unsafe = False

    index = 0
    attempts = 0
    while index < ticks:
        started = time.monotonic()
        cmd = build_tick_command(python_executable, max_markets=max_markets)
        try:
            previous_marker = latest_plan_marker(memory_path) if enforce_new_plan_marker else None
            current_marker = previous_marker
            waited_seconds = 0
            result = None
            while True:
                attempts += 1
                result = runner(cmd, env=env, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"runner exited with code {result.returncode}")
                current_marker = latest_plan_marker(memory_path) if enforce_new_plan_marker else None
                if not enforce_new_plan_marker or current_marker != previous_marker:
                    break
                if not wait_for_tick or waited_seconds >= max_wait_for_tick_seconds:
                    raise RuntimeError("runner exited without producing a new plan; likely no tick was available")
                retry_seconds = retry_seconds_from_runner_result(result) or no_tick_retry_seconds
                retry_seconds = min(retry_seconds, max_wait_for_tick_seconds - waited_seconds)
                if retry_seconds <= 0:
                    raise RuntimeError("runner exited without producing a new plan; likely no tick was available")
                print(f"no tick available; retrying in {retry_seconds}s")
                sleeper(retry_seconds)
                waited_seconds += retry_seconds
            plan = plan_reader(memory_path)
            summary = summarize_plan(plan, tick_elapsed_ms=elapsed_ms(started), tick_index=index + 1)
            summary["no_tick_wait_seconds"] = waited_seconds
            tick_reports.append(summary)
            print_tick_summary(summary)
            if summary["submit_called"]:
                unsafe = True
                break
            index += 1
        except Exception as exc:
            summary = failed_tick_summary(index + 1, exc, elapsed_ms(started))
            tick_reports.append(summary)
            print_tick_summary(summary)
            if not continue_on_error:
                break
            index += 1
        if index < ticks and sleep_seconds > 0:
            sleep_for, target = compute_sleep_seconds(
                now=datetime.now(UTC),
                mode=sleep_mode,
                fixed_sleep_seconds=sleep_seconds,
                tick_buffer_seconds=tick_buffer_seconds,
                min_sleep_seconds=min_sleep_seconds,
            )
            print(
                "tick completed_at={now} sleep_mode={mode} next_target_boundary={target} computed_sleep_seconds={sleep}".format(
                    now=datetime.now(UTC).isoformat(),
                    mode=sleep_mode,
                    target=target.isoformat() if target else None,
                    sleep=sleep_for,
                )
            )
            sleeper(sleep_for)

    aggregate = aggregate_tick_reports(tick_reports)
    aggregate["unsafe_submit_detected"] = unsafe or aggregate["total_live_submit_calls"] > 0
    paths = write_reports(aggregate, output_dir)
    return {**aggregate, **paths}


def diagnostic_profile_defaults(profile: str) -> dict[str, int]:
    if profile == "wide":
        return {
            "max_markets": 100,
            "rag_budget": 25,
            "blf_budget": 5,
            "rag_concurrency": 5,
            "llm_rag_concurrency": 3,
            "blf_concurrency": 2,
            "tick_time_budget_seconds": 600,
            "stop_new_work_before_deadline_seconds": 90,
        }
    if profile == "normal":
        return {
            "max_markets": 100,
            "rag_budget": 15,
            "blf_budget": 5,
            "rag_concurrency": 4,
            "llm_rag_concurrency": 3,
            "blf_concurrency": 2,
            "tick_time_budget_seconds": 600,
            "stop_new_work_before_deadline_seconds": 90,
        }
    if profile == "live_guarded":
        return {
            "max_markets": 100,
            "rag_budget": 8,
            "blf_budget": 2,
            "rag_concurrency": 4,
            "llm_rag_concurrency": 3,
            "blf_concurrency": 2,
            "tick_time_budget_seconds": 420,
            "stop_new_work_before_deadline_seconds": 90,
        }
    return {
        "max_markets": 80,
        "rag_budget": 8,
        "blf_budget": 2,
        "rag_concurrency": 4,
        "llm_rag_concurrency": 3,
        "blf_concurrency": 2,
        "tick_time_budget_seconds": 420,
        "stop_new_work_before_deadline_seconds": 90,
    }


def safe_diagnostic_env(
    source: os._Environ[str] | dict[str, str],
    *,
    rag_budget: int = 15,
    blf_budget: int = 5,
    blf_max_steps: int = 2,
    rag_selection_mode: str = "alpha_diagnostic",
    rag_concurrency: int = 4,
    llm_rag_concurrency: int = 3,
    blf_concurrency: int = 2,
    tick_time_budget_seconds: int = 600,
    stop_new_work_before_deadline_seconds: int = 90,
) -> dict[str, str]:
    env = dict(source)
    current_pythonpath = env.get("PYTHONPATH", "")
    local_paths = "ai-prophet/packages/core:."
    env["PYTHONPATH"] = f"{local_paths}:{current_pythonpath}" if current_pythonpath else local_paths
    env.update(
        {
            "EDGE_TRADER_ENABLE_RAG": "1",
            "EDGE_TRADER_ENABLE_LLM_RAG": "1",
            "EDGE_TRADER_ENABLE_BLF": "1",
            "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK": str(rag_budget),
            "EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK": str(blf_budget),
            "EDGE_TRADER_BLF_MAX_STEPS": str(blf_max_steps),
            "EDGE_TRADER_THRESHOLD_DEBUG": "1",
            "EDGE_TRADER_DRY_RUN": "1",
            "EDGE_TRADER_ALLOW_LIVE_SUBMIT": "0",
            "EDGE_TRADER_RAG_SELECTION_MODE": rag_selection_mode,
            "EDGE_TRADER_MAX_TICK_SECONDS": env.get("EDGE_TRADER_MAX_TICK_SECONDS", "600"),
            "EDGE_TRADER_LATENCY_WARNING_SECONDS": env.get("EDGE_TRADER_LATENCY_WARNING_SECONDS", "450"),
            "EDGE_TRADER_RAG_CONCURRENCY": str(rag_concurrency),
            "EDGE_TRADER_LLM_RAG_CONCURRENCY": str(llm_rag_concurrency),
            "EDGE_TRADER_BLF_CONCURRENCY": str(blf_concurrency),
            "EDGE_TRADER_TICK_TIME_BUDGET_SECONDS": str(tick_time_budget_seconds),
            "EDGE_TRADER_STOP_NEW_WORK_BEFORE_DEADLINE_SECONDS": str(stop_new_work_before_deadline_seconds),
            "EDGE_TRADER_ENABLE_PRICE_TRADING": "1",
            "EDGE_TRADER_ENABLE_CANDIDATE_RERANK": "1",
            "EDGE_TRADER_MAX_TRADE_SIZE": env.get("EDGE_TRADER_MAX_TRADE_SIZE", "1"),
            "EDGE_TRADER_MAX_POSITION_PER_MARKET": env.get("EDGE_TRADER_MAX_POSITION_PER_MARKET", "3"),
            "EDGE_TRADER_LIVE_GUARD_MODE": env.get("EDGE_TRADER_LIVE_GUARD_MODE", "1"),
        }
    )
    return env


def compute_sleep_seconds(
    *,
    now: datetime,
    mode: str,
    fixed_sleep_seconds: int,
    tick_buffer_seconds: int,
    min_sleep_seconds: int,
) -> tuple[int, datetime | None]:
    if mode == "fixed":
        return max(min_sleep_seconds, fixed_sleep_seconds), None
    next_boundary = next_quarter_hour(now)
    target = next_boundary - timedelta(seconds=tick_buffer_seconds)
    sleep_for = int((target - now).total_seconds())
    if sleep_for < min_sleep_seconds:
        sleep_for = min_sleep_seconds
    return sleep_for, next_boundary


def next_quarter_hour(now: datetime) -> datetime:
    minute = ((now.minute // 15) + 1) * 15
    if minute >= 60:
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return now.replace(minute=minute, second=0, microsecond=0)


def assert_safe_env(env: dict[str, str]) -> None:
    if env.get("EDGE_TRADER_DRY_RUN") not in {"1", "true", "yes"}:
        raise RuntimeError("Unsafe diagnostic config: dry-run must be true")
    if env.get("EDGE_TRADER_ALLOW_LIVE_SUBMIT", "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Unsafe diagnostic config: live submit must be disabled")
    if env.get("EDGE_TRADER_THRESHOLD_DEBUG") not in {"1", "true", "yes"}:
        raise RuntimeError("Unsafe diagnostic config: threshold-debug must be enabled")


def build_tick_command(python_executable: str, *, max_markets: int) -> list[str]:
    return [
        python_executable,
        "-m",
        "edge_trader_bot.runner",
        "--once",
        "--dry-run",
        "--max-markets",
        str(max_markets),
        "--threshold-debug",
    ]


def read_latest_plan(memory_path: Path) -> dict[str, Any]:
    if not memory_path.exists():
        raise FileNotFoundError(f"memory file not found: {memory_path}")
    last = ""
    with memory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        raise RuntimeError(f"memory file is empty: {memory_path}")
    return json.loads(last)


def latest_plan_marker(memory_path: Path) -> tuple[int, int] | None:
    if not memory_path.exists():
        return None
    stat = memory_path.stat()
    return stat.st_size, stat.st_mtime_ns


def retry_seconds_from_runner_result(result: subprocess.CompletedProcess) -> int | None:
    text = " ".join(
        part.decode("utf-8", errors="ignore") if isinstance(part, bytes) else str(part or "")
        for part in (getattr(result, "stdout", ""), getattr(result, "stderr", ""))
    )
    for marker in ("retry=", "retry_after=", "retry_after_sec="):
        if marker in text:
            tail = text.split(marker, 1)[1].split()[0].strip("s,")
            try:
                return int(float(tail))
            except ValueError:
                continue
    return None


def summarize_plan(plan: dict[str, Any], *, tick_elapsed_ms: int, tick_index: int) -> dict[str, Any]:
    submission = plan.get("submission", {})
    debug_rows = plan.get("pipeline_debug", [])
    threshold = plan.get("threshold_debug_report", {})
    scenarios = threshold.get("scenarios", [])
    scenario_edges = [
        row.get("max_edge")
        for row in scenarios
        if isinstance(row.get("max_edge"), (int, float))
    ]
    split = split_relaxed_threshold_scenarios(scenarios)
    blocked_resolution = max((row.get("blocked_by_resolution_risk", 0) for row in scenarios), default=0)
    max_tick_seconds = int(os.getenv("EDGE_TRADER_MAX_TICK_SECONDS", "600"))
    warning_seconds = int(os.getenv("EDGE_TRADER_LATENCY_WARNING_SECONDS", "450"))
    tick_warnings = latency_warnings(
        tick_elapsed_ms,
        max_tick_seconds=max_tick_seconds,
        warning_seconds=warning_seconds,
    )
    rag_attempted = plan.get("rag_markets_attempted", 0)
    llm_attempted = plan.get("llm_attempted", 0)
    blf_attempted = plan.get("blf_markets_attempted", 0)
    opportunity = plan.get("opportunity_report", {})
    return {
        "tick_index": tick_index,
        "status": "succeeded",
        "tick_id": plan.get("tick_id"),
        "candidate_set_id": plan.get("candidate_set_id"),
        "generated_at": plan.get("generated_at"),
        "candidates_loaded": plan.get("candidates_loaded", 0),
        "candidates_processed": plan.get("candidates_processed", 0),
        "markets_filtered": plan.get("markets_filtered", 0),
        "candidate_selection_summary": plan.get("candidate_selection_summary", {}),
        "rag_attempted": rag_attempted,
        "rag_succeeded": plan.get("rag_markets_scanned", 0),
        "rag_failed": len(plan.get("rag_scanner_errors", [])),
        "llm_attempted": llm_attempted,
        "llm_succeeded": plan.get("llm_succeeded", 0),
        "llm_failed": plan.get("llm_failed", 0),
        "llm_fallback": plan.get("llm_fallback_count", 0),
        "blf_attempted": blf_attempted,
        "blf_succeeded": plan.get("blf_succeeded", 0),
        "blf_failed": plan.get("blf_failed", 0),
        "blf_fallback": plan.get("blf_fallback_count", 0),
        "hypothetical_intents_count": plan.get("hypothetical_intents_count", 0),
        "trade_count": plan.get("trade_count", 0),
        "submit_called": bool(submission.get("submit_called", False)),
        "top_apparent_edges": plan.get("top_apparent_edges", [])[:5],
        "zero_trade_relaxed_threshold_count": threshold.get("zero_trade_relaxed_threshold_count", 0),
        "threshold_debug_scenarios": scenarios,
        "scenarios_with_hypothetical_trades": sum(
            1 for row in scenarios if row.get("hypothetical_trade_count", 0) > 0
        ),
        "relaxed_trades_with_blf_enabled": split["relaxed_trades_with_blf_enabled"],
        "relaxed_trades_without_blf": split["relaxed_trades_without_blf"],
        "relaxed_trade_count_by_threshold_and_blf_mode": split["relaxed_trade_count_by_threshold_and_blf_mode"],
        "would_trade_by_threshold_with_blf": split["would_trade_by_threshold_with_blf"],
        "would_trade_by_threshold_without_blf": split["would_trade_by_threshold_without_blf"],
        "max_edge_across_relaxed_scenarios": max(scenario_edges) if scenario_edges else None,
        "blocked_by_resolution_risk_count": blocked_resolution,
        "blocked_by_evidence_quality_count": len(
            plan.get("opportunity_report", {}).get("blocked_by_evidence_quality", [])
        ),
        "blocked_only_by_edge_threshold_count": len(
            plan.get("opportunity_report", {}).get("blocked_only_by_edge_threshold", [])
        ),
        "count_positive_taker_edge": opportunity.get("count_positive_taker_edge", 0),
        "count_positive_maker_edge": opportunity.get("count_positive_maker_edge", 0),
        "count_positive_mid_edge": opportunity.get("count_positive_mid_edge", 0),
        "count_mid_edge_ge_0_005": opportunity.get("count_mid_edge_ge_0_005", 0),
        "count_mid_edge_ge_0_01": opportunity.get("count_mid_edge_ge_0_01", 0),
        "count_mid_edge_ge_0_02": opportunity.get("count_mid_edge_ge_0_02", 0),
        "count_taker_edge_ge_0_005": opportunity.get("count_taker_edge_ge_0_005", 0),
        "count_taker_edge_ge_0_01": opportunity.get("count_taker_edge_ge_0_01", 0),
        "count_taker_edge_ge_0_02": opportunity.get("count_taker_edge_ge_0_02", 0),
        "top_by_maker_edge": opportunity.get("top_by_maker_edge", [])[:5],
        "top_by_mid_edge": opportunity.get("top_by_mid_edge", [])[:5],
        "top_by_spread_adjusted_maker_edge": opportunity.get("top_by_spread_adjusted_maker_edge", [])[:5],
        "manual_review_candidates": opportunity.get("manual_review_candidates", [])[:10],
        "manual_review_actionable_count": opportunity.get("manual_review_actionable_count", 0),
        "manual_review_watchlist_count": opportunity.get("manual_review_watchlist_count", 0),
        "manual_review_weak_count": opportunity.get("manual_review_weak_count", 0),
        "positive_maker_negative_taker": opportunity.get(
            "markets_where_maker_edge_positive_but_taker_edge_negative", []
        )[:5],
        "positive_mid_negative_taker": opportunity.get(
            "markets_where_mid_edge_positive_but_taker_edge_negative", []
        )[:5],
        "top_blf_adjustments": opportunity.get("blf_changed_probability_most", [])[:5],
        "top_rag_market_disagreements": opportunity.get(
            "rag_disagrees_with_market_most", []
        )[:5],
        "absence_of_evidence_penalty_detected_count": sum(
            1 for row in debug_rows if row.get("absence_of_evidence_penalty_detected")
        ),
        "market_type_distribution": dict(Counter(row.get("market_type", "unknown") for row in debug_rows)),
        "total_rag_elapsed_ms": plan.get("total_rag_elapsed_ms", 0),
        "total_llm_elapsed_ms": plan.get("total_llm_elapsed_ms", 0),
        "total_blf_elapsed_ms": plan.get("total_blf_elapsed_ms", 0),
        "stage_wall_times": plan.get("stage_wall_times", {}),
        "stage_summed_market_times": plan.get("stage_summed_market_times", {}),
        "concurrency_settings": plan.get("concurrency_settings", {}),
        "tick_time_budget_seconds": plan.get("tick_time_budget_seconds"),
        "early_stop_triggered": plan.get("early_stop_triggered", False),
        "early_stop_stage": plan.get("early_stop_stage", "none"),
        "skipped_due_to_deadline_count": plan.get("skipped_due_to_deadline_count", 0),
        "rag_skipped_due_to_deadline_count": plan.get("rag_skipped_due_to_deadline_count", 0),
        "blf_skipped_due_to_deadline_count": plan.get("blf_skipped_due_to_deadline_count", 0),
        "average_rag_elapsed_ms": average_elapsed(plan.get("total_rag_elapsed_ms", 0), rag_attempted),
        "average_llm_elapsed_ms": average_elapsed(plan.get("total_llm_elapsed_ms", 0), llm_attempted),
        "average_blf_elapsed_ms": average_elapsed(plan.get("total_blf_elapsed_ms", 0), blf_attempted),
        "latency_warnings": tick_warnings,
        "warnings": [*plan.get("warnings", []), *tick_warnings],
        "tick_elapsed_ms": tick_elapsed_ms,
        "trade_readiness_verdict": plan.get("trade_readiness_verdict")
        or evaluate_tick_live_readiness(debug_rows, blf_enabled=bool(plan.get("blf_enabled", True)), allow_live_submit=False),
        "price_trading_enabled": plan.get("price_trading_enabled", False),
        "count_price_signals": plan.get("count_price_signals", 0),
        "count_momentum_signals": plan.get("count_momentum_signals", 0),
        "count_mean_reversion_signals": plan.get("count_mean_reversion_signals", 0),
        "count_spread_capture_signals": plan.get("count_spread_capture_signals", 0),
        "count_forecast_edge_signals": plan.get("count_forecast_edge_signals", 0),
        "count_trade_intents": plan.get("count_trade_intents", 0),
        "count_exit_intents": plan.get("count_exit_intents", 0),
        "top_price_signals": plan.get("top_price_signals", [])[:10],
        "top_trade_candidates": plan.get("top_trade_candidates", [])[:10],
        "blocked_by_no_history": plan.get("blocked_by_no_history", 0),
        "blocked_by_spread": plan.get("blocked_by_spread", 0),
        "blocked_by_position_limit": plan.get("blocked_by_position_limit", 0),
        "blocked_by_live_guard": plan.get("blocked_by_live_guard", 0),
        "blocked_by_edge_threshold": plan.get("blocked_by_edge_threshold", 0),
        "positions_table": plan.get("positions_table", []),
        "exit_candidates": plan.get("exit_candidates", []),
        "live_readiness_verdict": plan.get("live_readiness_verdict", {}),
        "live_blocked_reason_counts": plan.get("live_blocked_reason_counts", {}),
        "live_tradable_candidate_count": plan.get("live_tradable_candidate_count", 0),
        "skipped_by_live_safety_count": plan.get("skipped_by_live_safety_count", 0),
        "live_trade_candidates": plan.get("live_trade_candidates", [])[:10],
        "rejected_trade_candidates_top10": plan.get("rejected_trade_candidates_top10", [])[:10],
        "signal_type_counts": plan.get("signal_type_counts", {}),
        "repeated_history_candidate_count": plan.get("repeated_history_candidate_count", 0),
        "fallback_risk_candidate_count": plan.get("fallback_risk_candidate_count", 0),
        "price_only_candidate_count": plan.get("price_only_candidate_count", 0),
        "forecast_only_candidate_count": plan.get("forecast_only_candidate_count", 0),
        "live_price_action_candidate_count": plan.get("live_price_action_candidate_count", 0),
        "live_fresh_event_candidate_count": plan.get("live_fresh_event_candidate_count", 0),
        "live_forecast_mispricing_candidate_count": plan.get("live_forecast_mispricing_candidate_count", 0),
        "live_candidate_pool_count": plan.get("live_candidate_pool_count", 0),
        "live_candidate_pool_top10": plan.get("live_candidate_pool_top10", [])[:10],
        "live_candidates_sent_to_rag_count": plan.get("live_candidates_sent_to_rag_count", 0),
        "live_candidates_sent_to_rag_top10": plan.get("live_candidates_sent_to_rag_top10", [])[:10],
        "live_candidates_sent_to_blf_count": plan.get("live_candidates_sent_to_blf_count", 0),
        "live_candidates_sent_to_blf_top10": plan.get("live_candidates_sent_to_blf_top10", [])[:10],
        "not_evaluated_by_forecast_budget_count": plan.get("not_evaluated_by_forecast_budget_count", 0),
        "analyzed_live_candidate_count": plan.get("analyzed_live_candidate_count", 0),
        "analyzed_live_candidate_top10": plan.get("analyzed_live_candidate_top10", [])[:10],
        "rag_failure_count_by_reason": plan.get("rag_failure_count_by_reason", {}),
        "cached_evidence_fallback_count": plan.get("cached_evidence_fallback_count", 0),
        "cached_evidence_used_count": plan.get("cached_evidence_used_count", 0),
        "cached_evidence_too_old_count": plan.get("cached_evidence_too_old_count", 0),
        "metadata_only_fallback_count": plan.get("metadata_only_fallback_count", 0),
        "previous_p_final_used_count": plan.get("previous_p_final_used_count", 0),
        "live_forecast_mispricing_cached_candidate_count": plan.get(
            "live_forecast_mispricing_cached_candidate_count", 0
        ),
        "rejected_cached_evidence_top10": plan.get("rejected_cached_evidence_top10", [])[:10],
        "metadata_only_rejected_top10": plan.get("metadata_only_rejected_top10", [])[:10],
        "recovered_rag_candidates_top10": plan.get("recovered_rag_candidates_top10", [])[:10],
        "skipped_by_price_action_history_count": plan.get("skipped_by_price_action_history_count", 0),
        "skipped_by_fresh_event_safety_count": plan.get("skipped_by_fresh_event_safety_count", 0),
        "fresh_event_candidates": plan.get("fresh_event_candidates", [])[:10],
        "fresh_event_rejected_top10": plan.get("fresh_event_rejected_top10", [])[:10],
        "rejected_forecast_mispricing_top10": plan.get("rejected_forecast_mispricing_top10", [])[:10],
        "forecast_gate_rejection_top10": plan.get("forecast_gate_rejection_top10", [])[:10],
        "live_submission_reason": plan.get("live_submission_reason"),
        "live_freeze_active": plan.get("live_freeze_active", False),
        "live_freeze_reason": plan.get("live_freeze_reason", ""),
        "freeze_scope": plan.get("freeze_scope", "none"),
        "clean_price_action_candidates_count": plan.get("clean_price_action_candidates_count", 0),
        "blocked_by_freeze_forecast_only_count": plan.get("blocked_by_freeze_forecast_only_count", 0),
        "blocked_by_freeze_all_count": plan.get("blocked_by_freeze_all_count", 0),
        "debug_rows": debug_rows,
    }


def split_relaxed_threshold_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {
        "with_blf": {"0.005": 0, "0.01": 0, "0.02": 0, "0.04": 0, "0.06": 0, "0.08": 0, "0.10": 0},
        "without_blf": {"0.005": 0, "0.01": 0, "0.02": 0, "0.04": 0, "0.06": 0, "0.08": 0, "0.10": 0},
    }
    for scenario in scenarios:
        threshold = scenario.get("min_edge_threshold")
        if not isinstance(threshold, (int, float)):
            continue
        key = "0.005" if abs(float(threshold) - 0.005) < 1e-9 else f"{threshold:.2f}"
        mode = "with_blf" if scenario.get("blf_enabled") else "without_blf"
        if key in by_mode[mode]:
            by_mode[mode][key] += int(scenario.get("hypothetical_trade_count", 0) or 0)
    return {
        "relaxed_trades_with_blf_enabled": sum(by_mode["with_blf"].values()),
        "relaxed_trades_without_blf": sum(by_mode["without_blf"].values()),
        "relaxed_trade_count_by_threshold_and_blf_mode": by_mode,
        "would_trade_by_threshold_with_blf": {key: value > 0 for key, value in by_mode["with_blf"].items()},
        "would_trade_by_threshold_without_blf": {key: value > 0 for key, value in by_mode["without_blf"].items()},
    }


def average_elapsed(total_ms: int | float, attempts: int | float) -> float:
    return float(total_ms) / float(attempts) if attempts else 0.0


def latency_warnings(
    tick_elapsed_ms: int,
    *,
    max_tick_seconds: int = 600,
    warning_seconds: int = 450,
) -> list[str]:
    if tick_elapsed_ms > max_tick_seconds * 1000:
        return [f"tick_elapsed_time_exceeded_budget:{tick_elapsed_ms / 1000:.1f}s>{max_tick_seconds}s"]
    if tick_elapsed_ms > warning_seconds * 1000:
        return [f"tick_elapsed_time_warning:{tick_elapsed_ms / 1000:.1f}s>{warning_seconds}s"]
    return []


def failed_tick_summary(tick_index: int, exc: Exception, tick_elapsed_ms: int) -> dict[str, Any]:
    return {
        "tick_index": tick_index,
        "status": "failed",
        "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
        "submit_called": False,
        "tick_elapsed_ms": tick_elapsed_ms,
        "debug_rows": [],
    }


def aggregate_tick_reports(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [tick for tick in ticks if tick.get("status") == "succeeded"]
    all_debug_rows = [row for tick in succeeded for row in tick.get("debug_rows", [])]
    repeated = repeated_market_tracking(succeeded)
    threshold_trades = threshold_trade_summary(succeeded)
    split = aggregate_threshold_split(succeeded)
    latency = aggregate_latency_summary(ticks, succeeded, all_debug_rows)
    readiness = aggregate_trade_readiness(succeeded, all_debug_rows)
    reliability = four_tick_reliability_verdict(ticks, latency)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "ticks": ticks,
        "unsafe_submit_detected": sum(1 for tick in ticks if tick.get("submit_called")) > 0,
        "total_ticks_attempted": len(ticks),
        "total_ticks_succeeded": len(succeeded),
        "total_ticks_failed": len(ticks) - len(succeeded),
        "real_claimed_ticks": len(ticks),
        "no_tick_retries": sum(1 for tick in ticks for _ in range(int(tick.get("no_tick_wait_seconds", 0) > 0))),
        "total_wall_clock_ms": sum(tick.get("tick_elapsed_ms", 0) for tick in ticks),
        "max_tick_elapsed_ms": max((tick.get("tick_elapsed_ms", 0) for tick in ticks), default=0),
        "latency_status_by_tick": [
            {
                "tick_index": tick.get("tick_index"),
                "tick_elapsed_ms": tick.get("tick_elapsed_ms"),
                "latency_status": tick_latency_status(tick.get("tick_elapsed_ms", 0)),
                "early_stop_triggered": tick.get("early_stop_triggered", False),
                "skipped_due_to_deadline_count": tick.get("skipped_due_to_deadline_count", 0),
            }
            for tick in ticks
        ],
        "four_tick_reliability_verdict": reliability,
        "total_candidates_processed": sum(tick.get("candidates_processed", 0) for tick in succeeded),
        "total_rag_attempts": sum(tick.get("rag_attempted", 0) for tick in succeeded),
        "total_rag_failures": sum(tick.get("rag_failed", 0) for tick in succeeded),
        "total_llm_attempts": sum(tick.get("llm_attempted", 0) for tick in succeeded),
        "total_llm_failures": sum(tick.get("llm_failed", 0) for tick in succeeded),
        "total_blf_attempts": sum(tick.get("blf_attempted", 0) for tick in succeeded),
        "total_blf_failures": sum(tick.get("blf_failed", 0) for tick in succeeded),
        "total_hypothetical_intents": sum(tick.get("hypothetical_intents_count", 0) for tick in succeeded),
        "total_live_submit_calls": sum(1 for tick in ticks if tick.get("submit_called")),
        "max_best_edge_seen": max_best_edge(all_debug_rows),
        "top_10_apparent_edges": top_rows(all_debug_rows, "best_edge", 10),
        "top_10_maker_edges": top_rows(all_debug_rows, "best_maker_edge", 10),
        "top_10_mid_edges": top_rows(all_debug_rows, "best_mid_edge", 10),
        "top_10_spread_adjusted_maker_edges": top_rows(all_debug_rows, "maker_edge_after_half_spread", 10),
        "top_10_rag_market_disagreements": top_rag_disagreements(succeeded, 10),
        "top_10_blf_probability_adjustments": top_blf_adjustments(succeeded, 10),
        "relaxed_threshold_scenarios_producing_trades": sum(
            tick.get("scenarios_with_hypothetical_trades", 0) for tick in succeeded
        ),
        "relaxed_trades_with_blf_enabled": split["relaxed_trades_with_blf_enabled"],
        "relaxed_trades_without_blf": split["relaxed_trades_without_blf"],
        "relaxed_trade_count_by_threshold_and_blf_mode": split["relaxed_trade_count_by_threshold_and_blf_mode"],
        "would_trade_by_threshold_with_blf": split["would_trade_by_threshold_with_blf"],
        "would_trade_by_threshold_without_blf": split["would_trade_by_threshold_without_blf"],
        "would_trade_by_threshold": threshold_trades,
        "trade_readiness_verdict": readiness,
        "markets_blocked_only_by_edge": sum(tick.get("blocked_only_by_edge_threshold_count", 0) for tick in succeeded),
        "markets_blocked_by_resolution_risk": sum(tick.get("blocked_by_resolution_risk_count", 0) for tick in succeeded),
        "markets_blocked_by_evidence_quality": sum(tick.get("blocked_by_evidence_quality_count", 0) for tick in succeeded),
        "count_positive_taker_edge": sum(tick.get("count_positive_taker_edge", 0) for tick in succeeded),
        "count_positive_maker_edge": sum(tick.get("count_positive_maker_edge", 0) for tick in succeeded),
        "count_positive_mid_edge": sum(tick.get("count_positive_mid_edge", 0) for tick in succeeded),
        "count_mid_edge_ge_0_005": sum(tick.get("count_mid_edge_ge_0_005", 0) for tick in succeeded),
        "count_mid_edge_ge_0_01": sum(tick.get("count_mid_edge_ge_0_01", 0) for tick in succeeded),
        "count_mid_edge_ge_0_02": sum(tick.get("count_mid_edge_ge_0_02", 0) for tick in succeeded),
        "count_taker_edge_ge_0_005": sum(tick.get("count_taker_edge_ge_0_005", 0) for tick in succeeded),
        "count_taker_edge_ge_0_01": sum(tick.get("count_taker_edge_ge_0_01", 0) for tick in succeeded),
        "count_taker_edge_ge_0_02": sum(tick.get("count_taker_edge_ge_0_02", 0) for tick in succeeded),
        "manual_review_candidates": [
            row for tick in succeeded for row in tick.get("manual_review_candidates", [])
        ][:20],
        "manual_review_actionable_count": sum(tick.get("manual_review_actionable_count", 0) for tick in succeeded),
        "manual_review_watchlist_count": sum(tick.get("manual_review_watchlist_count", 0) for tick in succeeded),
        "manual_review_weak_count": sum(tick.get("manual_review_weak_count", 0) for tick in succeeded),
        "markets_where_maker_edge_positive_but_taker_edge_negative": [
            row for tick in succeeded for row in tick.get("positive_maker_negative_taker", [])
        ][:20],
        "markets_where_mid_edge_positive_but_taker_edge_negative": [
            row for tick in succeeded for row in tick.get("positive_mid_negative_taker", [])
        ][:20],
        "absence_of_evidence_penalty_detected_count": sum(
            tick.get("absence_of_evidence_penalty_detected_count", 0) for tick in succeeded
        ),
        "market_type_distribution": dict(Counter(row.get("market_type", "unknown") for row in all_debug_rows)),
        "early_stop_triggered_ticks": sum(1 for tick in succeeded if tick.get("early_stop_triggered")),
        "skipped_due_to_deadline_total": sum(tick.get("skipped_due_to_deadline_count", 0) for tick in succeeded),
        "concurrency_settings": succeeded[-1].get("concurrency_settings", {}) if succeeded else {},
        "stage_wall_times_by_tick": [
            {"tick_index": tick.get("tick_index"), **(tick.get("stage_wall_times") or {})}
            for tick in succeeded
        ],
        "stage_summed_market_times_by_tick": [
            {"tick_index": tick.get("tick_index"), **(tick.get("stage_summed_market_times") or {})}
            for tick in succeeded
        ],
        "repeated_markets": repeated,
        "repeated_markets_largest_p_final_movement": repeated[:10],
        "cost_latency_summary": latency,
        "price_trading_summary": {
            "count_price_signals": sum(tick.get("count_price_signals", 0) for tick in succeeded),
            "count_momentum_signals": sum(tick.get("count_momentum_signals", 0) for tick in succeeded),
            "count_mean_reversion_signals": sum(tick.get("count_mean_reversion_signals", 0) for tick in succeeded),
            "count_spread_capture_signals": sum(tick.get("count_spread_capture_signals", 0) for tick in succeeded),
            "count_forecast_edge_signals": sum(tick.get("count_forecast_edge_signals", 0) for tick in succeeded),
            "count_trade_intents": sum(tick.get("count_trade_intents", 0) for tick in succeeded),
            "count_exit_intents": sum(tick.get("count_exit_intents", 0) for tick in succeeded),
            "blocked_by_no_history": sum(tick.get("blocked_by_no_history", 0) for tick in succeeded),
            "blocked_by_spread": sum(tick.get("blocked_by_spread", 0) for tick in succeeded),
            "blocked_by_position_limit": sum(tick.get("blocked_by_position_limit", 0) for tick in succeeded),
            "blocked_by_live_guard": sum(tick.get("blocked_by_live_guard", 0) for tick in succeeded),
            "blocked_by_edge_threshold": sum(tick.get("blocked_by_edge_threshold", 0) for tick in succeeded),
            "top_price_signals": [row for tick in succeeded for row in tick.get("top_price_signals", [])][:20],
            "top_trade_candidates": [row for tick in succeeded for row in tick.get("top_trade_candidates", [])][:20],
            "positions_table": [row for tick in succeeded for row in tick.get("positions_table", [])][:50],
            "exit_candidates": [row for tick in succeeded for row in tick.get("exit_candidates", [])][:50],
            "candidate_selection_summary": succeeded[-1].get("candidate_selection_summary", {}) if succeeded else {},
            "live_blocked_reason_counts": dict(
                sum_counters(tick.get("live_blocked_reason_counts", {}) for tick in succeeded)
            ),
            "live_tradable_candidate_count": sum(tick.get("live_tradable_candidate_count", 0) for tick in succeeded),
            "skipped_by_live_safety_count": sum(tick.get("skipped_by_live_safety_count", 0) for tick in succeeded),
            "live_trade_candidates": [row for tick in succeeded for row in tick.get("live_trade_candidates", [])][:20],
            "rejected_trade_candidates_top10": [
                row for tick in succeeded for row in tick.get("rejected_trade_candidates_top10", [])
            ][:20],
            "signal_type_counts": dict(sum_counters(tick.get("signal_type_counts", {}) for tick in succeeded)),
            "repeated_history_candidate_count": sum(
                tick.get("repeated_history_candidate_count", 0) for tick in succeeded
            ),
            "fallback_risk_candidate_count": sum(tick.get("fallback_risk_candidate_count", 0) for tick in succeeded),
            "price_only_candidate_count": sum(tick.get("price_only_candidate_count", 0) for tick in succeeded),
            "forecast_only_candidate_count": sum(tick.get("forecast_only_candidate_count", 0) for tick in succeeded),
            "live_price_action_candidate_count": sum(
                tick.get("live_price_action_candidate_count", 0) for tick in succeeded
            ),
            "live_fresh_event_candidate_count": sum(
                tick.get("live_fresh_event_candidate_count", 0) for tick in succeeded
            ),
            "live_forecast_mispricing_candidate_count": sum(
                tick.get("live_forecast_mispricing_candidate_count", 0) for tick in succeeded
            ),
            "live_candidate_pool_count": sum(tick.get("live_candidate_pool_count", 0) for tick in succeeded),
            "live_candidates_sent_to_rag_count": sum(
                tick.get("live_candidates_sent_to_rag_count", 0) for tick in succeeded
            ),
            "live_candidates_sent_to_blf_count": sum(
                tick.get("live_candidates_sent_to_blf_count", 0) for tick in succeeded
            ),
            "not_evaluated_by_forecast_budget_count": sum(
                tick.get("not_evaluated_by_forecast_budget_count", 0) for tick in succeeded
            ),
            "analyzed_live_candidate_count": sum(
                tick.get("analyzed_live_candidate_count", 0) for tick in succeeded
            ),
            "cached_evidence_used_count": sum(tick.get("cached_evidence_used_count", 0) for tick in succeeded),
            "cached_evidence_too_old_count": sum(
                tick.get("cached_evidence_too_old_count", 0) for tick in succeeded
            ),
            "metadata_only_fallback_count": sum(
                tick.get("metadata_only_fallback_count", 0) for tick in succeeded
            ),
            "previous_p_final_used_count": sum(
                tick.get("previous_p_final_used_count", 0) for tick in succeeded
            ),
            "live_forecast_mispricing_cached_candidate_count": sum(
                tick.get("live_forecast_mispricing_cached_candidate_count", 0) for tick in succeeded
            ),
            "live_candidate_pool_top10": [
                row for tick in succeeded for row in tick.get("live_candidate_pool_top10", [])
            ][:20],
            "live_candidates_sent_to_rag_top10": [
                row for tick in succeeded for row in tick.get("live_candidates_sent_to_rag_top10", [])
            ][:20],
            "live_candidates_sent_to_blf_top10": [
                row for tick in succeeded for row in tick.get("live_candidates_sent_to_blf_top10", [])
            ][:20],
            "analyzed_live_candidate_top10": [
                row for tick in succeeded for row in tick.get("analyzed_live_candidate_top10", [])
            ][:20],
            "recovered_rag_candidates_top10": [
                row for tick in succeeded for row in tick.get("recovered_rag_candidates_top10", [])
            ][:20],
            "rejected_cached_evidence_top10": [
                row for tick in succeeded for row in tick.get("rejected_cached_evidence_top10", [])
            ][:20],
            "metadata_only_rejected_top10": [
                row for tick in succeeded for row in tick.get("metadata_only_rejected_top10", [])
            ][:20],
            "skipped_by_price_action_history_count": sum(
                tick.get("skipped_by_price_action_history_count", 0) for tick in succeeded
            ),
            "skipped_by_fresh_event_safety_count": sum(
                tick.get("skipped_by_fresh_event_safety_count", 0) for tick in succeeded
            ),
            "fresh_event_candidates": [row for tick in succeeded for row in tick.get("fresh_event_candidates", [])][:20],
            "fresh_event_rejected_top10": [
                row for tick in succeeded for row in tick.get("fresh_event_rejected_top10", [])
            ][:20],
            "rejected_forecast_mispricing_top10": [
                row for tick in succeeded for row in tick.get("rejected_forecast_mispricing_top10", [])
            ][:20],
            "forecast_gate_rejection_top10": [
                row for tick in succeeded for row in tick.get("forecast_gate_rejection_top10", [])
            ][:20],
            "live_freeze_active_ticks": sum(1 for tick in succeeded if tick.get("live_freeze_active")),
            "freeze_scopes": dict(Counter(tick.get("freeze_scope", "none") for tick in succeeded)),
            "clean_price_action_candidates_count": sum(
                tick.get("clean_price_action_candidates_count", 0) for tick in succeeded
            ),
            "blocked_by_freeze_forecast_only_count": sum(
                tick.get("blocked_by_freeze_forecast_only_count", 0) for tick in succeeded
            ),
            "blocked_by_freeze_all_count": sum(tick.get("blocked_by_freeze_all_count", 0) for tick in succeeded),
        },
    }


def sum_counters(counters) -> Counter:
    total: Counter = Counter()
    for item in counters:
        total.update(item or {})
    return total


def aggregate_threshold_split(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    base = {
        "relaxed_trades_with_blf_enabled": 0,
        "relaxed_trades_without_blf": 0,
        "relaxed_trade_count_by_threshold_and_blf_mode": {
            "with_blf": {"0.005": 0, "0.01": 0, "0.02": 0, "0.04": 0, "0.06": 0, "0.08": 0, "0.10": 0},
            "without_blf": {"0.005": 0, "0.01": 0, "0.02": 0, "0.04": 0, "0.06": 0, "0.08": 0, "0.10": 0},
        },
    }
    for tick in ticks:
        split = {
            "with_blf": tick.get("relaxed_trade_count_by_threshold_and_blf_mode", {}).get("with_blf", {}),
            "without_blf": tick.get("relaxed_trade_count_by_threshold_and_blf_mode", {}).get("without_blf", {}),
        }
        for mode in ("with_blf", "without_blf"):
            for key, value in split[mode].items():
                if key in base["relaxed_trade_count_by_threshold_and_blf_mode"][mode]:
                    base["relaxed_trade_count_by_threshold_and_blf_mode"][mode][key] += int(value or 0)
    base["relaxed_trades_with_blf_enabled"] = sum(base["relaxed_trade_count_by_threshold_and_blf_mode"]["with_blf"].values())
    base["relaxed_trades_without_blf"] = sum(base["relaxed_trade_count_by_threshold_and_blf_mode"]["without_blf"].values())
    base["would_trade_by_threshold_with_blf"] = {
        key: value > 0 for key, value in base["relaxed_trade_count_by_threshold_and_blf_mode"]["with_blf"].items()
    }
    base["would_trade_by_threshold_without_blf"] = {
        key: value > 0 for key, value in base["relaxed_trade_count_by_threshold_and_blf_mode"]["without_blf"].items()
    }
    return base


def aggregate_trade_readiness(ticks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict = evaluate_tick_live_readiness(rows, config=LiveReadinessConfig(), blf_enabled=True, allow_live_submit=False)
    if any(tick.get("relaxed_trades_without_blf", 0) > 0 for tick in ticks):
        verdict["reasons"].append("BLF-disabled relaxed trades are diagnostic only and not actionable")
    return verdict


def aggregate_latency_summary(
    ticks: list[dict[str, Any]],
    succeeded: list[dict[str, Any]],
    all_debug_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    max_elapsed = max((tick.get("tick_elapsed_ms", 0) for tick in ticks), default=0)
    status = "critical" if max_elapsed > 600_000 else "warning" if max_elapsed > 450_000 else "ok"
    reasons = []
    if status == "warning":
        reasons.append("tick_elapsed_ms exceeded 450000")
    elif status == "critical":
        reasons.append("tick_elapsed_ms exceeded 600000")
    return {
        "latency_status": status,
        "latency_warning_reasons": reasons,
        "suggested_settings": {
            "normal_diagnostic": "max_markets=100 rag_budget=15 blf_budget=5",
            "wide_scan": "max_markets=100 rag_budget=25 blf_budget=5",
            "if_warning_or_critical": "reduce RAG budget or enable early stop",
        },
        "total_rag_time_ms": sum(tick.get("total_rag_elapsed_ms", 0) for tick in succeeded),
        "total_llm_time_ms": sum(tick.get("total_llm_elapsed_ms", 0) for tick in succeeded),
        "total_blf_time_ms": sum(tick.get("total_blf_elapsed_ms", 0) for tick in succeeded),
        "average_rag_time_per_attempt_ms": average_elapsed(
            sum(tick.get("total_rag_elapsed_ms", 0) for tick in succeeded),
            sum(tick.get("rag_attempted", 0) for tick in succeeded),
        ),
        "average_llm_time_per_attempt_ms": average_elapsed(
            sum(tick.get("total_llm_elapsed_ms", 0) for tick in succeeded),
            sum(tick.get("llm_attempted", 0) for tick in succeeded),
        ),
        "average_blf_time_per_attempt_ms": average_elapsed(
            sum(tick.get("total_blf_elapsed_ms", 0) for tick in succeeded),
            sum(tick.get("blf_attempted", 0) for tick in succeeded),
        ),
        "average_tick_time_ms": (
            sum(tick.get("tick_elapsed_ms", 0) for tick in ticks) / len(ticks) if ticks else 0
        ),
        "max_tick_time_ms": max((tick.get("tick_elapsed_ms", 0) for tick in ticks), default=0),
        "latency_warnings": [warning for tick in ticks for warning in tick.get("latency_warnings", [])],
        "slowest_markets_by_llm_elapsed_ms": slowest_markets(all_debug_rows, "llm_elapsed_ms"),
        "slowest_markets_by_rag_elapsed_ms": slowest_markets(all_debug_rows, "scanner_elapsed_ms"),
        "slowest_markets_by_blf_elapsed_ms": slowest_markets_from_blf(all_debug_rows),
    }


def tick_latency_status(tick_elapsed_ms: int | float) -> str:
    if tick_elapsed_ms > 600_000:
        return "critical"
    if tick_elapsed_ms > 450_000:
        return "warning"
    return "ok"


def four_tick_reliability_verdict(ticks: list[dict[str, Any]], latency: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    reasons = []
    if not ticks:
        blockers.append("no_ticks_attempted")
    if any(tick.get("status") != "succeeded" for tick in ticks):
        blockers.append("one_or_more_claimed_ticks_failed")
    else:
        reasons.append("all claimed ticks completed")
    if any(tick.get("submit_called") for tick in ticks):
        blockers.append("submit_called_true")
    else:
        reasons.append("no live submit")
        reasons.append("no unsafe submit")
    if latency.get("latency_status") == "critical":
        blockers.append("critical_latency")
    else:
        reasons.append("no critical latency")
    over_budget = [
        tick for tick in ticks
        if tick.get("tick_time_budget_seconds") and tick.get("tick_elapsed_ms", 0) > tick["tick_time_budget_seconds"] * 1000
    ]
    if over_budget:
        blockers.append("max tick elapsed above budget")
    else:
        reasons.append("max tick elapsed below budget")
    if any(tick.get("early_stop_triggered") for tick in ticks):
        reasons.append("deadline early stop handled safely")
    else:
        reasons.append("no deadline early stop")
    return {
        "FOUR_TICK_RUN_RELIABLE": not blockers,
        "reasons": reasons,
        "blockers": blockers,
    }


def max_best_edge(rows: list[dict[str, Any]]) -> float | None:
    edges = [row.get("best_edge") for row in rows if isinstance(row.get("best_edge"), (int, float))]
    return max(edges) if edges else None


def top_rows(rows: list[dict[str, Any]], field: str, limit: int) -> list[dict[str, Any]]:
    return [
        compact_market_row(row)
        for row in sorted(
            [row for row in rows if isinstance(row.get(field), (int, float))],
            key=lambda item: item[field],
            reverse=True,
        )[:limit]
    ]


def compact_market_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row.get("market_id"),
        "question": row.get("question"),
        "market_type": row.get("market_type"),
        "best_side": row.get("best_side"),
        "best_edge": row.get("best_edge"),
        "best_taker_edge": row.get("best_taker_edge"),
        "best_maker_edge": row.get("best_maker_edge"),
        "best_mid_edge": row.get("best_mid_edge"),
        "maker_edge_after_half_spread": row.get("maker_edge_after_half_spread"),
        "maker_edge_after_full_spread": row.get("maker_edge_after_full_spread"),
        "passive_opportunity_quality": row.get("passive_opportunity_quality"),
        "p_market": row.get("p_market"),
        "p_2402_final_after_shrinkage": row.get("p_2402_final_after_shrinkage"),
        "p_blf_final_after_shrinkage": row.get("p_blf_final_after_shrinkage"),
        "p_final_after_blf": row.get("p_final_after_blf"),
        "evidence_quality": row.get("evidence_quality"),
        "confidence": row.get("confidence"),
        "hold_reasons": row.get("hold_reasons", []),
    }


def top_rag_disagreements(ticks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [row for tick in ticks for row in tick.get("top_rag_market_disagreements", [])]
    return sorted(rows, key=lambda row: row.get("absolute_disagreement", 0), reverse=True)[:limit]


def top_blf_adjustments(ticks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [row for tick in ticks for row in tick.get("top_blf_adjustments", [])]
    return sorted(rows, key=lambda row: row.get("absolute_change", 0), reverse=True)[:limit]


def threshold_trade_summary(ticks: list[dict[str, Any]]) -> dict[str, bool]:
    summary = {"0.005": False, "0.01": False, "0.02": False, "0.04": False, "0.06": False, "0.08": False, "0.10": False}
    for tick in ticks:
        for scenario in tick.get("threshold_debug_scenarios", []):
            threshold = scenario.get("min_edge_threshold")
            if not isinstance(threshold, (int, float)):
                continue
            key = "0.005" if abs(float(threshold) - 0.005) < 1e-9 else f"{threshold:.2f}"
            if key in summary and scenario.get("hypothetical_trade_count", 0) > 0:
                summary[key] = True
    return summary


def repeated_market_tracking(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tick in ticks:
        for row in tick.get("debug_rows", []):
            by_market[row.get("market_id")].append({**row, "tick_id": tick.get("tick_id")})
    repeated = []
    for market_id, rows in by_market.items():
        if len(rows) < 2:
            continue
        p_values = [row.get("p_final_after_blf") for row in rows if isinstance(row.get("p_final_after_blf"), (int, float))]
        movement = max(p_values) - min(p_values) if p_values else 0
        repeated.append(
            {
                "market_id": market_id,
                "question": rows[0].get("question"),
                "market_type": rows[0].get("market_type"),
                "p_final_movement": movement,
                "per_tick": [
                    {
                        "tick_id": row.get("tick_id"),
                        "p_market": row.get("p_market"),
                        "p_2402": row.get("p_2402_final_after_shrinkage"),
                        "p_blf": row.get("p_blf_final_after_shrinkage"),
                        "p_final": row.get("p_final_after_blf"),
                        "best_edge": row.get("best_edge"),
                        "evidence_quality": row.get("evidence_quality"),
                        "confidence": row.get("confidence"),
                        "decision": row.get("decision"),
                        "hold_reasons": row.get("hold_reasons"),
                    }
                    for row in rows
                ],
            }
        )
    return sorted(repeated, key=lambda row: row["p_final_movement"], reverse=True)


def slowest_markets(rows: list[dict[str, Any]], field: str, limit: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "market_id": row.get("market_id"),
            "question": row.get("question"),
            "market_type": row.get("market_type"),
            field: row.get(field),
        }
        for row in sorted(
            [row for row in rows if isinstance(row.get(field), (int, float))],
            key=lambda item: item[field],
            reverse=True,
        )[:limit]
    ]


def slowest_markets_from_blf(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        blf = row.get("blf_package") or {}
        if isinstance(row.get("blf_elapsed_ms"), (int, float)):
            normalized.append(row)
        elif isinstance(blf.get("elapsed_ms"), (int, float)):
            normalized.append({**row, "blf_elapsed_ms": blf["elapsed_ms"]})
    return slowest_markets(normalized, "blf_elapsed_ms", limit)


def write_reports(aggregate: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"four_tick_{stamp}.json"
    md_path = output_dir / f"four_tick_{stamp}.md"
    json_payload = strip_debug_rows(aggregate)
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(json_payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def strip_debug_rows(aggregate: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(aggregate, default=str))
    for tick in cleaned.get("ticks", []):
        tick.pop("debug_rows", None)
    return cleaned


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Four-Tick Diagnostic Report",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Summary",
        "",
        f"- Ticks attempted/succeeded/failed: {report.get('total_ticks_attempted')} / {report.get('total_ticks_succeeded')} / {report.get('total_ticks_failed')}",
        f"- Total candidates processed: {report.get('total_candidates_processed')}",
        f"- RAG attempts/failures: {report.get('total_rag_attempts')} / {report.get('total_rag_failures')}",
        f"- LLM attempts/failures: {report.get('total_llm_attempts')} / {report.get('total_llm_failures')}",
        f"- BLF attempts/failures: {report.get('total_blf_attempts')} / {report.get('total_blf_failures')}",
        f"- Live submit calls: {report.get('total_live_submit_calls')}",
        f"- Max best edge seen: {report.get('max_best_edge_seen')}",
        f"- Relaxed-threshold scenarios producing trades: {report.get('relaxed_threshold_scenarios_producing_trades')}",
        f"- With BLF enabled relaxed trades: {report.get('relaxed_trades_with_blf_enabled')}",
        f"- With BLF disabled relaxed trades: {report.get('relaxed_trades_without_blf')} (diagnostic only)",
        "",
        "## 4-Tick Reliability Verdict",
        "",
        f"- FOUR_TICK_RUN_RELIABLE: {(report.get('four_tick_reliability_verdict') or {}).get('FOUR_TICK_RUN_RELIABLE')}",
        "",
        "Reasons:",
        *[f"- {reason}" for reason in (report.get("four_tick_reliability_verdict") or {}).get("reasons", [])],
        "",
        "Blockers:",
        *[f"- {blocker}" for blocker in (report.get("four_tick_reliability_verdict") or {}).get("blockers", [])],
        "",
        "## Trade Readiness Verdict",
        "",
        f"- LIVE_SUBMIT_READY: {(report.get('trade_readiness_verdict') or {}).get('LIVE_SUBMIT_READY')}",
        f"- candidate_count: {(report.get('trade_readiness_verdict') or {}).get('candidate_count')}",
        f"- max_blf_enabled_taker_edge: {(report.get('trade_readiness_verdict') or {}).get('max_blf_enabled_taker_edge')}",
        f"- max_blf_enabled_mid_edge: {(report.get('trade_readiness_verdict') or {}).get('max_blf_enabled_mid_edge')}",
        f"- max_blf_enabled_spread_adjusted_maker_edge: {(report.get('trade_readiness_verdict') or {}).get('max_blf_enabled_spread_adjusted_maker_edge')}",
        "",
        "Reasons:",
        *[f"- {reason}" for reason in (report.get("trade_readiness_verdict") or {}).get("reasons", [])],
        "",
        "Blockers:",
        *[f"- {blocker}" for blocker in (report.get("trade_readiness_verdict") or {}).get("blockers", [])],
        "",
        "## Relaxed Threshold Split",
        "",
        "With BLF enabled: "
        + ("relaxed threshold trades present." if report.get("relaxed_trades_with_blf_enabled") else "no relaxed threshold trades."),
        "With BLF disabled: "
        + f"{report.get('relaxed_trades_without_blf')} diagnostic-only trades.",
        "",
        "## Top Apparent Edges",
        *markdown_rows(report.get("top_10_apparent_edges", []), ("market_id", "market_type", "best_edge", "p_final_after_blf")),
        "",
        "## Top Maker Edges",
        *markdown_rows(report.get("top_10_maker_edges", []), ("market_id", "market_type", "best_maker_edge", "p_final_after_blf")),
        "",
        "## Top Mid Edges",
        *markdown_rows(report.get("top_10_mid_edges", []), ("market_id", "market_type", "best_mid_edge", "p_final_after_blf")),
        "",
        "## Top Spread-Adjusted Maker Edges",
        *markdown_rows(
            report.get("top_10_spread_adjusted_maker_edges", []),
            ("market_id", "market_type", "maker_edge_after_half_spread", "passive_opportunity_quality"),
        ),
        "",
        "## Manual Review Candidates",
        *markdown_rows(
            report.get("manual_review_candidates", []),
            ("market_id", "market_type", "best_taker_edge", "best_mid_edge", "evidence_quality", "confidence"),
        ),
        "",
        "## Price Trading",
        "",
        json.dumps(report.get("price_trading_summary", {}), indent=2, sort_keys=True),
        "",
        "## Top RAG-Market Disagreements",
        *markdown_rows(report.get("top_10_rag_market_disagreements", []), ("market_id", "market_type", "absolute_disagreement", "p_market", "p_2402")),
        "",
        "## Top BLF Adjustments",
        *markdown_rows(report.get("top_10_blf_probability_adjustments", []), ("market_id", "market_type", "absolute_change", "p_final_before_blf", "p_final_after_blf")),
        "",
        "## Market-Type Distribution",
        "",
        json.dumps(report.get("market_type_distribution", {}), indent=2, sort_keys=True),
        "",
        "## Repeated Markets With Largest p_final Movement",
        *markdown_rows(
            report.get("repeated_markets_largest_p_final_movement", []),
            ("market_id", "market_type", "p_final_movement"),
        ),
        "",
        "## Latency",
        "",
        json.dumps(report.get("cost_latency_summary", {}), indent=2, sort_keys=True),
        "",
    ]
    return "\n".join(lines)


def markdown_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    if not rows:
        return ["", "_None_", ""]
    lines = ["", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines + [""]


def print_tick_summary(summary: dict[str, Any]) -> None:
    if summary.get("status") != "succeeded":
        print(f"tick {summary.get('tick_index')} failed: {summary.get('error')}")
        return
    print(
        "tick {idx} {tick_id}: processed={processed} rag={rag}/{rag_fail} "
        "llm={llm}/{llm_fail} blf={blf}/{blf_fail} trades={trades} submit={submit} max_edge={edge}".format(
            idx=summary.get("tick_index"),
            tick_id=summary.get("tick_id"),
            processed=summary.get("candidates_processed"),
            rag=summary.get("rag_attempted"),
            rag_fail=summary.get("rag_failed"),
            llm=summary.get("llm_attempted"),
            llm_fail=summary.get("llm_failed"),
            blf=summary.get("blf_attempted"),
            blf_fail=summary.get("blf_failed"),
            trades=summary.get("trade_count"),
            submit=summary.get("submit_called"),
            edge=(summary.get("top_apparent_edges") or [{}])[0].get("apparent_edge"),
        )
    )


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    main()
