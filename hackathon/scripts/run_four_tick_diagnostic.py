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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


DEFAULT_MEMORY_PATH = Path(".edge_trader/memory.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four safe diagnostic dry-run ticks.")
    parser.add_argument("--ticks", type=int, default=4, help="Number of diagnostic ticks to attempt.")
    parser.add_argument("--sleep-seconds", type=int, default=900, help="Seconds to wait between ticks.")
    parser.add_argument("--max-markets", type=int, default=50, help="Max markets passed to the single-tick runner.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after one tick fails.")
    parser.add_argument("--memory-path", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-sleep", action="store_true", help="Skip sleeping between ticks; useful for tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_diagnostic(
        ticks=args.ticks,
        sleep_seconds=0 if args.no_sleep else args.sleep_seconds,
        max_markets=args.max_markets,
        continue_on_error=args.continue_on_error,
        memory_path=args.memory_path,
        output_dir=args.output_dir,
        python_executable=args.python,
    )
    print(f"wrote aggregate_json={report['json_path']}")
    print(f"wrote aggregate_markdown={report['markdown_path']}")


def run_diagnostic(
    *,
    ticks: int = 4,
    sleep_seconds: int = 900,
    max_markets: int = 50,
    continue_on_error: bool = False,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleeper: Callable[[int], None] = time.sleep,
    plan_reader: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    env = safe_diagnostic_env(os.environ)
    assert_safe_env(env)
    plan_reader = plan_reader or read_latest_plan
    tick_reports: list[dict[str, Any]] = []
    unsafe = False

    for index in range(ticks):
        started = time.monotonic()
        cmd = build_tick_command(python_executable, max_markets=max_markets)
        try:
            result = runner(cmd, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"runner exited with code {result.returncode}")
            plan = plan_reader(memory_path)
            summary = summarize_plan(plan, tick_elapsed_ms=elapsed_ms(started), tick_index=index + 1)
            tick_reports.append(summary)
            print_tick_summary(summary)
            if summary["submit_called"]:
                unsafe = True
                break
        except Exception as exc:
            summary = failed_tick_summary(index + 1, exc, elapsed_ms(started))
            tick_reports.append(summary)
            print_tick_summary(summary)
            if not continue_on_error:
                break
        if index < ticks - 1 and sleep_seconds > 0:
            sleeper(sleep_seconds)

    aggregate = aggregate_tick_reports(tick_reports)
    aggregate["unsafe_submit_detected"] = unsafe or aggregate["total_live_submit_calls"] > 0
    paths = write_reports(aggregate, output_dir)
    return {**aggregate, **paths}


def safe_diagnostic_env(source: os._Environ[str] | dict[str, str]) -> dict[str, str]:
    env = dict(source)
    current_pythonpath = env.get("PYTHONPATH", "")
    local_paths = "ai-prophet/packages/core:."
    env["PYTHONPATH"] = f"{local_paths}:{current_pythonpath}" if current_pythonpath else local_paths
    env.update(
        {
            "EDGE_TRADER_ENABLE_RAG": "1",
            "EDGE_TRADER_ENABLE_LLM_RAG": "1",
            "EDGE_TRADER_ENABLE_BLF": "1",
            "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK": "10",
            "EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK": "2",
            "EDGE_TRADER_BLF_MAX_STEPS": "2",
            "EDGE_TRADER_THRESHOLD_DEBUG": "1",
            "EDGE_TRADER_DRY_RUN": "1",
            "EDGE_TRADER_ALLOW_LIVE_SUBMIT": "0",
        }
    )
    return env


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
    blocked_resolution = max((row.get("blocked_by_resolution_risk", 0) for row in scenarios), default=0)
    return {
        "tick_index": tick_index,
        "status": "succeeded",
        "tick_id": plan.get("tick_id"),
        "candidate_set_id": plan.get("candidate_set_id"),
        "generated_at": plan.get("generated_at"),
        "candidates_loaded": plan.get("candidates_loaded", 0),
        "candidates_processed": plan.get("candidates_processed", 0),
        "markets_filtered": plan.get("markets_filtered", 0),
        "rag_attempted": plan.get("rag_markets_attempted", 0),
        "rag_succeeded": plan.get("rag_markets_scanned", 0),
        "rag_failed": len(plan.get("rag_scanner_errors", [])),
        "llm_attempted": plan.get("llm_attempted", 0),
        "llm_succeeded": plan.get("llm_succeeded", 0),
        "llm_failed": plan.get("llm_failed", 0),
        "llm_fallback": plan.get("llm_fallback_count", 0),
        "blf_attempted": plan.get("blf_markets_attempted", 0),
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
        "max_edge_across_relaxed_scenarios": max(scenario_edges) if scenario_edges else None,
        "blocked_by_resolution_risk_count": blocked_resolution,
        "blocked_by_evidence_quality_count": len(
            plan.get("opportunity_report", {}).get("blocked_by_evidence_quality", [])
        ),
        "blocked_only_by_edge_threshold_count": len(
            plan.get("opportunity_report", {}).get("blocked_only_by_edge_threshold", [])
        ),
        "top_blf_adjustments": plan.get("opportunity_report", {}).get("blf_changed_probability_most", [])[:5],
        "top_rag_market_disagreements": plan.get("opportunity_report", {}).get(
            "rag_disagrees_with_market_most", []
        )[:5],
        "absence_of_evidence_penalty_detected_count": sum(
            1 for row in debug_rows if row.get("absence_of_evidence_penalty_detected")
        ),
        "market_type_distribution": dict(Counter(row.get("market_type", "unknown") for row in debug_rows)),
        "total_rag_elapsed_ms": plan.get("total_rag_elapsed_ms", 0),
        "total_llm_elapsed_ms": plan.get("total_llm_elapsed_ms", 0),
        "total_blf_elapsed_ms": plan.get("total_blf_elapsed_ms", 0),
        "tick_elapsed_ms": tick_elapsed_ms,
        "debug_rows": debug_rows,
    }


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
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "ticks": ticks,
        "unsafe_submit_detected": sum(1 for tick in ticks if tick.get("submit_called")) > 0,
        "total_ticks_attempted": len(ticks),
        "total_ticks_succeeded": len(succeeded),
        "total_ticks_failed": len(ticks) - len(succeeded),
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
        "top_10_rag_market_disagreements": top_rag_disagreements(succeeded, 10),
        "top_10_blf_probability_adjustments": top_blf_adjustments(succeeded, 10),
        "relaxed_threshold_scenarios_producing_trades": sum(
            tick.get("scenarios_with_hypothetical_trades", 0) for tick in succeeded
        ),
        "would_trade_by_threshold": threshold_trades,
        "markets_blocked_only_by_edge": sum(tick.get("blocked_only_by_edge_threshold_count", 0) for tick in succeeded),
        "markets_blocked_by_resolution_risk": sum(tick.get("blocked_by_resolution_risk_count", 0) for tick in succeeded),
        "markets_blocked_by_evidence_quality": sum(tick.get("blocked_by_evidence_quality_count", 0) for tick in succeeded),
        "absence_of_evidence_penalty_detected_count": sum(
            tick.get("absence_of_evidence_penalty_detected_count", 0) for tick in succeeded
        ),
        "market_type_distribution": dict(Counter(row.get("market_type", "unknown") for row in all_debug_rows)),
        "repeated_markets": repeated,
        "repeated_markets_largest_p_final_movement": repeated[:10],
        "cost_latency_summary": {
            "total_rag_time_ms": sum(tick.get("total_rag_elapsed_ms", 0) for tick in succeeded),
            "total_llm_time_ms": sum(tick.get("total_llm_elapsed_ms", 0) for tick in succeeded),
            "total_blf_time_ms": sum(tick.get("total_blf_elapsed_ms", 0) for tick in succeeded),
            "average_tick_time_ms": (
                sum(tick.get("tick_elapsed_ms", 0) for tick in ticks) / len(ticks) if ticks else 0
            ),
            "slowest_markets_by_llm_elapsed_ms": slowest_markets(all_debug_rows, "llm_elapsed_ms"),
            "slowest_markets_by_rag_elapsed_ms": slowest_markets(all_debug_rows, "scanner_elapsed_ms"),
            "slowest_markets_by_blf_elapsed_ms": slowest_markets_from_blf(all_debug_rows),
        },
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
    summary = {"0.10": False, "0.08": False, "0.06": False, "0.04": False}
    for tick in ticks:
        for scenario in tick.get("threshold_debug_scenarios", []):
            key = f"{scenario.get('min_edge_threshold'):.2f}"
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
        if isinstance(blf.get("elapsed_ms"), (int, float)):
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
        "",
        "## Top Apparent Edges",
        *markdown_rows(report.get("top_10_apparent_edges", []), ("market_id", "market_type", "best_edge", "p_final_after_blf")),
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
