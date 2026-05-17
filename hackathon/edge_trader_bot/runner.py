"""Main runner for the RAG-BLF Edge Trader."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from ai_prophet_core import DEFAULT_API_URL, APIClientError, ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession, TickLease

from .aggregator import combine_signals
from .blf_verifier import BlfVerifier
from .config import BotConfig, load_env_file
from .json_utils import json_safe
from .market_data import market_view
from .market_filter import deterministic_filter
from .memory import JsonlMemory
from .market_classifier import classify_market
from .rag_scanner import RagScanner
from .schemas import ForecastSignals, MarketView, TradeDecision
from .stat_priors import initial_signals
from .trading_policy import decide_trade, rank_decisions
from .sizer import size_trade

logger = logging.getLogger(__name__)


class EdgeTraderBot:
    def __init__(self, config: BotConfig, memory_path: Path | None = None) -> None:
        self.config = config
        self.rag = RagScanner(
            enabled=config.enable_rag,
            max_queries=config.rag_max_queries,
            max_results_per_query=config.rag_max_results_per_query,
            enable_llm_summary=config.enable_llm_rag_summary,
        )
        self.blf = BlfVerifier(
            enabled=config.enable_blf,
            max_steps=config.blf_max_steps,
            provider=config.blf_provider,
            model=config.blf_model,
            timeout_seconds=config.blf_timeout_seconds,
            json_retries=config.blf_json_retries,
            enable_extra_search=config.blf_enable_extra_search,
        )
        self.memory = JsonlMemory(memory_path or Path(".edge_trader/memory.jsonl"))

    def run(self, once: bool = False) -> None:
        env_status = get_env_status()
        log_env_status(env_status)
        if not env_status["PA_SERVER_API_KEY"]["present"]:
            raise RuntimeError("PA_SERVER_API_KEY is required for Prophet Arena server runs")

        api = ServerAPIClient(
            base_url=os.getenv("PA_SERVER_URL", DEFAULT_API_URL),
            api_key=os.getenv("PA_SERVER_API_KEY"),
            timeout=30,
        )
        with BenchmarkSession(api) as session:
            experiment = self.create_experiment_with_slug_retry(session)
            participant = session.upsert_participant(
                model=self.config.model_name,
                starting_cash=self.config.starting_cash,
            )
            logger.info(
                "Experiment %s slug=%s participant=%s dry_run=%s",
                experiment.experiment_id,
                self.config.slug,
                participant.participant_idx,
                self.config.dry_run,
            )

            while True:
                lease = session.claim_tick()
                if not lease.available:
                    if lease.reason == "experiment_completed":
                        logger.info("Experiment completed.")
                        return
                    retry = lease.retry_after_sec or 15
                    logger.info("No tick available reason=%s retry=%ss", lease.reason, retry)
                    if once:
                        return
                    time.sleep(retry)
                    continue

                try:
                    self.process_tick(session, participant.participant_idx, lease)
                    session.complete_tick(lease)
                except Exception as exc:
                    logger.exception("Tick %s failed", lease.tick_id)
                    session.finalize(
                        lease,
                        participant.participant_idx,
                        status="FAILED",
                        error_code="EDGE_TRADER_ERROR",
                        error_detail=str(exc)[:1024],
                    )
                    raise

                if once:
                    return

    def create_experiment_with_slug_retry(self, session: BenchmarkSession):
        slug = self.config.slug
        for _ in range(10):
            try:
                return session.create_experiment(
                    slug=slug,
                    config_hash=self.config.config_hash(),
                    config_json={**self.config.to_experiment_config(), "resolved_slug": slug},
                    n_ticks=self.config.n_ticks,
                )
            except APIClientError as exc:
                if exc.status_code == 409 and "different config_hash" in str(exc):
                    previous = slug
                    slug = bump_slug(slug)
                    logger.warning("Slug conflict for %s; retrying with %s", previous, slug)
                    continue
                raise
        raise RuntimeError("Unable to resolve experiment slug after repeated config hash conflicts")

    def process_tick(
        self,
        session: BenchmarkSession,
        participant_idx: int,
        lease: TickLease,
    ) -> None:
        tick = session.load_candidates(lease)
        bound_lease = tick.lease
        candidates = [market_view(market) for market in tick.candidates.markets]
        portfolio = session.get_portfolio(participant_idx)

        plan: dict | None = None
        try:
            selected, skipped = self.filter_markets(candidates, portfolio)
            signals_by_market: dict[str, ForecastSignals] = {}
            rag_attempted_count = 0

            for market in selected:
                signals = initial_signals(market)
                if self.should_run_rag(market, rag_attempted_count):
                    signals = self.rag.scan(market, signals)
                    if self.config.enable_rag:
                        rag_attempted_count += 1
                else:
                    signals.risk_flags.append("rag_skipped_budget")
                signals = combine_signals(signals)
                signals_by_market[market.market_id] = signals

            if self.config.enable_blf:
                blf_market_ids = set(self.select_blf_markets(selected, signals_by_market))
                for market in selected:
                    signals = signals_by_market[market.market_id]
                    if market.market_id in blf_market_ids:
                        signals = self.blf.verify(market, signals)
                    else:
                        signals.risk_flags.append("blf_skipped_budget")
                    signals_by_market[market.market_id] = combine_signals(signals)
            else:
                for market in selected:
                    signals = self.blf.verify(market, signals_by_market[market.market_id])
                    signals_by_market[market.market_id] = combine_signals(signals)

            decisions: list[TradeDecision] = []
            for market in selected:
                signals = signals_by_market[market.market_id]
                decision = decide_trade(market, signals, portfolio, self.config)
                if decision is not None:
                    decisions.append(decision)

            ranked_decisions = rank_decisions(decisions, self.config.max_trades_per_tick_target)
            plan = self.build_plan(
                lease=bound_lease,
                candidates_count=len(candidates),
                selected=selected,
                skipped=skipped,
                signals_by_market=signals_by_market,
                decisions=ranked_decisions,
            )
            plan["submission"] = self.build_submission_plan(
                session=session,
                lease=bound_lease,
                participant_idx=participant_idx,
                decisions=ranked_decisions,
            )
            safe_plan = json_safe(plan)
            session.put_plan(bound_lease, participant_idx, safe_plan)
            self.memory.append(safe_plan)
            self.log_plan_summary(safe_plan)
            session.finalize(bound_lease, participant_idx)
        except Exception as exc:
            if plan is None:
                plan = self.build_error_plan(bound_lease, len(candidates), exc)
            else:
                plan.setdefault("errors", []).append(json_safe(exc))
            safe_plan = json_safe(plan)
            try:
                session.put_plan(bound_lease, participant_idx, safe_plan)
            except Exception:
                logger.exception("Failed to persist error plan for tick %s", bound_lease.tick_id)
            self.memory.append(safe_plan)
            raise

    def filter_markets(self, markets, portfolio):
        selected: list[MarketView] = []
        skipped: list[dict] = []
        for market in markets:
            ok, reasons = deterministic_filter(market, portfolio, self.config)
            if ok and len(selected) < self.config.max_markets_to_consider:
                selected.append(market)
            else:
                skipped.append({
                    "market_id": market.market_id,
                    "reasons": reasons or ["consideration_limit"],
                    "skipped_before_rag": True,
                })
        return selected, skipped

    def should_run_rag(self, market: MarketView, scanned_count: int) -> bool:
        if not self.config.enable_rag:
            return True
        if scanned_count >= self.config.rag_max_markets_per_tick:
            return False
        # Prioritize markets where the spread is executable and the mid price is
        # not already near-certain. Near-certain markets rarely justify live RAG
        # budget unless we are already holding a position, which exit logic can
        # handle via market/stat priors for now.
        return 0.05 < market.yes_mid < 0.95

    def select_blf_markets(
        self,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        by_id = {market.market_id: market for market in selected}
        for market_id, signals in signals_by_market.items():
            market = by_id[market_id]
            resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
            if resolution_check.get("trade_blocker"):
                risk_penalty = 0.10
            elif resolution_check.get("risk_flags"):
                risk_penalty = 0.04
            else:
                risk_penalty = 0.0
            rag_market_disagreement = abs((signals.p_2402 or signals.p_market) - signals.p_market)
            edge = abs(apparent_edge(market, signals) or 0.0)
            score = edge + 0.01 * signals.evidence_quality + 0.25 * rag_market_disagreement - risk_penalty
            scored.append((score, market_id))
        scored.sort(reverse=True)
        return [market_id for _, market_id in scored[: self.config.blf_max_markets_per_tick]]

    def build_plan(
        self,
        *,
        lease: TickLease,
        candidates_count: int,
        selected: list[MarketView],
        skipped: list[dict],
        signals_by_market: dict[str, ForecastSignals],
        decisions: list[TradeDecision],
    ) -> dict:
        selected_by_id = {market.market_id: market for market in selected}
        rag_attempted = sum(1 for signals in signals_by_market.values() if signals.evidence_package is not None)
        rag_scanned = sum(1 for signals in signals_by_market.values() if signals.p_2402 is not None)
        rag_scanner_errors = [
            {
                "market_id": market_id,
                "scanner_error": (signals.evidence_package or {}).get("scanner_error"),
                "risk_flags": (signals.evidence_package or {}).get("risk_flags", []),
            }
            for market_id, signals in signals_by_market.items()
            if (signals.evidence_package or {}).get("scanner_error")
        ]
        rag_budget_skipped = [
            market_id
            for market_id, signals in signals_by_market.items()
            if "rag_skipped_budget" in signals.risk_flags
        ]
        total_rag_elapsed_ms = sum(
            int((signals.evidence_package or {}).get("scanner_elapsed_ms", 0) or 0)
            for signals in signals_by_market.values()
        )
        llm_attempted = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("llm_attempted")
        )
        llm_succeeded = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("llm_succeeded")
        )
        llm_failed = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("llm_failed")
        )
        llm_fallback_count = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("llm_fallback")
        )
        llm_error_categories = sorted(
            {
                str((signals.evidence_package or {}).get("llm_error_category"))
                for signals in signals_by_market.values()
                if (signals.evidence_package or {}).get("llm_error_category")
            }
        )
        total_llm_elapsed_ms = sum(
            int((signals.evidence_package or {}).get("llm_elapsed_ms", 0) or 0)
            for signals in signals_by_market.values()
        )
        blf_attempted = sum(1 for signals in signals_by_market.values() if (signals.blf_package or {}).get("attempted"))
        blf_succeeded = sum(1 for signals in signals_by_market.values() if (signals.blf_package or {}).get("succeeded"))
        blf_failed = sum(
            1
            for signals in signals_by_market.values()
            if (signals.blf_package or {}).get("attempted") and not (signals.blf_package or {}).get("succeeded")
        )
        blf_fallback_count = sum(
            1 for signals in signals_by_market.values() if (signals.blf_package or {}).get("fallback_reason")
        )
        total_blf_elapsed_ms = sum(
            int((signals.blf_package or {}).get("elapsed_ms", 0) or 0)
            for signals in signals_by_market.values()
        )
        blf_errors = [
            {
                "market_id": market_id,
                "error": (signals.blf_package or {}).get("error"),
                "fallback_reason": (signals.blf_package or {}).get("fallback_reason"),
            }
            for market_id, signals in signals_by_market.items()
            if (signals.blf_package or {}).get("error")
        ]
        return {
            "strategy": self.config.strategy_name,
            "version": self.config.version,
            "run_mode": "dry_run" if self.config.dry_run else "live",
            "generated_at": datetime.now(UTC).isoformat(),
            "tick_id": lease.tick_id,
            "candidate_set_id": lease.candidate_set_id,
            "dry_run": self.config.dry_run,
            "allow_live_submit": self.config.allow_live_submit,
            "threshold_debug": self.config.threshold_debug,
            "rag_enabled": self.config.enable_rag,
            "llm_rag_enabled": self.config.enable_llm_rag_summary,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "llm_attempted": llm_attempted,
            "llm_succeeded": llm_succeeded,
            "llm_failed": llm_failed,
            "llm_fallback_count": llm_fallback_count,
            "llm_error_categories": llm_error_categories,
            "total_llm_elapsed_ms": total_llm_elapsed_ms,
            "brave_api_present": bool(os.getenv("BRAVE_SEARCH_API_KEY")),
            "max_rag_markets_per_tick": self.config.rag_max_markets_per_tick,
            "candidates_loaded": candidates_count,
            "candidates_processed": len(selected),
            "markets_considered": candidates_count,
            "markets_after_filter": len(selected),
            "markets_filtered": len(skipped),
            "rag_markets_attempted": rag_attempted,
            "rag_markets_scanned": rag_scanned,
            "markets_scanned_2402": rag_scanned,
            "rag_scanner_errors": rag_scanner_errors,
            "rag_budget_skipped_markets": rag_budget_skipped,
            "markets_skipped_before_rag": len(skipped),
            "total_rag_elapsed_ms": total_rag_elapsed_ms,
            "markets_verified_blf": blf_succeeded,
            "blf_enabled": self.config.enable_blf,
            "max_blf_markets_per_tick": self.config.blf_max_markets_per_tick,
            "blf_markets_attempted": blf_attempted,
            "blf_succeeded": blf_succeeded,
            "blf_failed": blf_failed,
            "blf_fallback_count": blf_fallback_count,
            "total_blf_elapsed_ms": total_blf_elapsed_ms,
            "blf_errors": blf_errors,
            "hypothetical_intents_count": len(decisions),
            "trade_count": len(decisions),
            "submit_skipped_due_to_dry_run": self.config.dry_run and bool(decisions),
            "warnings": self.build_warnings(decisions),
            "errors": [],
            "decisions": [decision.to_plan_dict() for decision in decisions],
            "pipeline_debug": pipeline_debug_rows(selected_by_id, signals_by_market, decisions, self.config),
            "threshold_debug_report": threshold_debug_report(
                selected_by_id,
                signals_by_market,
                self.config,
            )
            if self.config.threshold_debug
            else {},
            "opportunity_report": opportunity_report(selected_by_id, signals_by_market),
            "signals": {
                market_id: {
                    "p_market": signals.p_market,
                    "p_stat": signals.p_stat,
                    "p_2402": signals.p_2402,
                    "p_blf": signals.p_blf,
                    "p_final": signals.p_final,
                    "confidence": signals.confidence,
                    "uncertainty": signals.uncertainty,
                    "evidence_quality": signals.evidence_quality,
                    "risk_flags": signals.risk_flags,
                    "reason": signals.reason,
                    "evidence_package": signals.evidence_package,
                    "search_queries": (signals.evidence_package or {}).get("search_queries", []),
                    "raw_search_result_count": (signals.evidence_package or {}).get("raw_search_result_count", 0),
                    "deduped_search_result_count": (
                        signals.evidence_package or {}
                    ).get("deduped_search_result_count", 0),
                    "scanner_elapsed_ms": (signals.evidence_package or {}).get("scanner_elapsed_ms", 0),
                    "llm_attempted": (signals.evidence_package or {}).get("llm_attempted", False),
                    "llm_succeeded": (signals.evidence_package or {}).get("llm_succeeded", False),
                    "llm_failed": (signals.evidence_package or {}).get("llm_failed", False),
                    "llm_fallback": (signals.evidence_package or {}).get("llm_fallback", False),
                    "llm_error_category": (signals.evidence_package or {}).get("llm_error_category"),
                    "llm_elapsed_ms": (signals.evidence_package or {}).get("llm_elapsed_ms", 0),
                    "p_2402_raw": (signals.evidence_package or {}).get("p_2402_raw"),
                    "p_2402_raw_original": (signals.evidence_package or {}).get("p_2402_raw_original"),
                    "p_2402_final_after_shrinkage": (
                        signals.evidence_package or {}
                    ).get("p_2402_final_after_shrinkage"),
                    "market_type": (signals.evidence_package or {}).get(
                        "market_type",
                        (signals.blf_package or {}).get("market_type"),
                    ),
                    "long_horizon_days_to_resolution": (signals.evidence_package or {}).get(
                        "long_horizon_days_to_resolution",
                        (signals.blf_package or {}).get("long_horizon_days_to_resolution"),
                    ),
                    "absence_of_evidence_penalty_detected": (
                        (signals.evidence_package or {}).get("absence_of_evidence_penalty_detected", False)
                        or (signals.blf_package or {}).get("absence_of_evidence_penalty_detected", False)
                    ),
                    "no_direct_evidence_reasoning": (
                        (signals.evidence_package or {}).get("no_direct_evidence_reasoning", False)
                        or (signals.blf_package or {}).get("no_direct_evidence_reasoning", False)
                    ),
                    "future_event_should_anchor_to_market": (
                        (signals.evidence_package or {}).get("future_event_should_anchor_to_market", False)
                        or (signals.blf_package or {}).get("future_event_should_anchor_to_market", False)
                    ),
                    "reasoning_summary": (signals.evidence_package or {}).get("reasoning_summary", ""),
                    "json_parse_error": (signals.evidence_package or {}).get("json_parse_error"),
                    "fallback_reason": (signals.evidence_package or {}).get("fallback_reason"),
                    "llm_prompt_metadata": (signals.evidence_package or {}).get("llm_prompt_metadata", {}),
                    "blf_package": signals.blf_package,
                    "p_final_before_blf": signals.p_final_before_blf,
                    "p_final_after_blf": signals.p_final_after_blf,
                    "blf_adjustment": signals.blf_adjustment,
                    "aggregation_reason": signals.aggregation_reason,
                    "apparent_edge": apparent_edge(selected_by_id.get(market_id), signals),
                }
                for market_id, signals in signals_by_market.items()
            },
            "top_apparent_edges": top_apparent_edges(selected_by_id, signals_by_market),
            "skipped": skipped[:100],
            "intents": [decision.to_intent_dict() for decision in decisions],
        }

    def build_submission_plan(
        self,
        *,
        session: BenchmarkSession,
        lease: TickLease,
        participant_idx: int,
        decisions: list[TradeDecision],
    ) -> dict:
        if not decisions:
            return {
                "accepted": 0,
                "rejected": 0,
                "dry_run": self.config.dry_run,
                "submit_called": False,
                "reason": "no_decisions",
            }
        if self.config.dry_run:
            return {
                "accepted": 0,
                "rejected": 0,
                "dry_run": True,
                "submit_called": False,
                "reason": "dry_run",
            }
        if self.config.threshold_debug:
            return {
                "accepted": 0,
                "rejected": 0,
                "dry_run": True,
                "submit_called": False,
                "reason": "threshold_debug_no_submit",
            }
        if not self.config.allow_live_submit:
            return {
                "accepted": 0,
                "rejected": 0,
                "dry_run": False,
                "submit_called": False,
                "reason": "live_submit_disabled",
            }

        intents = [
            TradeIntentRequest(
                market_id=decision.market_id,
                action=decision.action,
                side=decision.side,
                shares=str(decision.shares),
                idempotency_key="",
            )
            for decision in decisions
        ]
        result = session.submit_intents(lease, participant_idx, intents)
        return {
            "accepted": result.accepted,
            "rejected": result.rejected,
            "dry_run": False,
            "submit_called": True,
            "fills": [fill.model_dump(mode="json") for fill in result.fills],
            "rejections": [rej.model_dump(mode="json") for rej in result.rejections],
        }

    def build_warnings(self, decisions: list[TradeDecision]) -> list[str]:
        warnings: list[str] = []
        if self.config.dry_run:
            warnings.append("dry_run_enabled_no_submit")
        if self.config.threshold_debug:
            warnings.append("threshold_debug_diagnostic_only_no_submit")
        if decisions and not self.config.dry_run and not self.config.allow_live_submit:
            warnings.append("live_submit_disabled")
        if self.config.enable_rag and not os.getenv("BRAVE_SEARCH_API_KEY"):
            warnings.append("rag_enabled_but_brave_key_missing")
        if self.config.enable_llm_rag_summary and not os.getenv("OPENROUTER_API_KEY"):
            warnings.append("llm_rag_enabled_but_openrouter_key_missing")
        if self.config.enable_blf and self.config.blf_provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            warnings.append("blf_enabled_but_openrouter_key_missing")
        return warnings

    def build_error_plan(self, lease: TickLease, candidates_count: int, exc: Exception) -> dict:
        return {
            "strategy": self.config.strategy_name,
            "version": self.config.version,
            "run_mode": "dry_run" if self.config.dry_run else "live",
            "generated_at": datetime.now(UTC).isoformat(),
            "tick_id": lease.tick_id,
            "candidate_set_id": lease.candidate_set_id,
            "dry_run": self.config.dry_run,
            "threshold_debug": self.config.threshold_debug,
            "candidates_loaded": candidates_count,
            "candidates_processed": 0,
            "markets_filtered": 0,
            "rag_markets_scanned": 0,
            "rag_markets_attempted": 0,
            "rag_scanner_errors": [],
            "total_rag_elapsed_ms": 0,
            "llm_rag_enabled": self.config.enable_llm_rag_summary,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "llm_attempted": 0,
            "llm_succeeded": 0,
            "llm_failed": 0,
            "llm_fallback_count": 0,
            "llm_error_categories": [],
            "total_llm_elapsed_ms": 0,
            "blf_enabled": self.config.enable_blf,
            "blf_markets_attempted": 0,
            "blf_succeeded": 0,
            "blf_failed": 0,
            "blf_fallback_count": 0,
            "total_blf_elapsed_ms": 0,
            "blf_errors": [],
            "hypothetical_intents_count": 0,
            "submit_skipped_due_to_dry_run": self.config.dry_run,
            "warnings": self.build_warnings([]),
            "errors": [json_safe(exc)],
            "decisions": [],
            "pipeline_debug": [],
            "threshold_debug_report": {},
            "opportunity_report": {},
            "signals": {},
            "skipped": [],
            "intents": [],
            "submission": {
                "accepted": 0,
                "rejected": 0,
                "dry_run": self.config.dry_run,
                "submit_called": False,
                "reason": "error_before_submit",
            },
        }

    def log_plan_summary(self, plan: dict) -> None:
        logger.info(
            "dry-run summary: candidates_loaded=%s processed=%s filtered=%s "
            "rag_attempted=%s rag_succeeded=%s rag_failed=%s "
            "llm_attempted=%s llm_succeeded=%s llm_failed=%s llm_fallback=%s "
            "blf_attempted=%s blf_succeeded=%s blf_failed=%s blf_fallback=%s "
            "hypothetical_intents=%s",
            plan.get("candidates_loaded"),
            plan.get("candidates_processed"),
            plan.get("markets_filtered"),
            plan.get("rag_markets_attempted"),
            plan.get("rag_markets_scanned"),
            len(plan.get("rag_scanner_errors", [])),
            plan.get("llm_attempted", 0),
            plan.get("llm_succeeded", 0),
            plan.get("llm_failed", 0),
            plan.get("llm_fallback_count", 0),
            plan.get("blf_markets_attempted", 0),
            plan.get("blf_succeeded", 0),
            plan.get("blf_failed", 0),
            plan.get("blf_fallback_count", 0),
            plan.get("hypothetical_intents_count"),
        )
        logger.info("dry-run top_apparent_edges=%s", plan.get("top_apparent_edges", [])[:3])
        report = plan.get("threshold_debug_report", {})
        if report:
            logger.info("dry-run threshold_debug scenarios=%s", report.get("scenarios", [])[:4])
        logger.info(
            "dry-run opportunity_report top_edges=%s blf_changes=%s",
            plan.get("opportunity_report", {}).get("top_by_best_edge", [])[:3],
            plan.get("opportunity_report", {}).get("blf_changed_probability_most", [])[:3],
        )
        logger.info("dry-run warnings=%s errors=%s", plan.get("warnings", []), plan.get("errors", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG-BLF Edge Trader bot.")
    parser.add_argument("--once", action="store_true", help="Process at most one available tick.")
    parser.add_argument("--dry-run", action="store_true", help="Write plans but do not submit intents.")
    parser.add_argument("--slug", help="Experiment slug override.")
    parser.add_argument("--memory-path", type=Path, help="Local JSONL memory path.")
    parser.add_argument("--max-markets", type=int, help="Maximum candidates to process after filtering.")
    parser.add_argument("--enable-blf", action="store_true", help="Enable minimal BLF verifier for this run.")
    parser.add_argument("--max-blf-markets", type=int, help="Maximum markets to verify with BLF per tick.")
    parser.add_argument("--blf-max-steps", type=int, help="Maximum BLF update steps per market.")
    parser.add_argument("--allow-live-submit", action="store_true", help="Allow real submit calls when dry-run is false.")
    parser.add_argument("--threshold-debug", action="store_true", help="Add relaxed-threshold diagnostics; never submit.")
    parser.add_argument("--check-env", action="store_true", help="Print non-secret environment readiness and exit.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    load_env_file()
    args = parse_args()
    if args.check_env:
        log_env_status(get_env_status())
        return
    config = BotConfig.from_env()
    if args.slug:
        config = BotConfig(**{**config.to_experiment_config(), "slug": args.slug})
    if args.dry_run:
        config = BotConfig(**{**config.to_experiment_config(), "dry_run": True})
    if args.max_markets is not None:
        config = BotConfig(**{**config.to_experiment_config(), "max_markets_to_consider": args.max_markets})
    if args.enable_blf:
        config = BotConfig(**{**config.to_experiment_config(), "enable_blf": True})
    if args.max_blf_markets is not None:
        config = BotConfig(**{**config.to_experiment_config(), "blf_max_markets_per_tick": args.max_blf_markets})
    if args.blf_max_steps is not None:
        config = BotConfig(**{**config.to_experiment_config(), "blf_max_steps": args.blf_max_steps})
    if args.allow_live_submit:
        config = BotConfig(**{**config.to_experiment_config(), "allow_live_submit": True})
    if args.threshold_debug:
        config = BotConfig(
            **{
                **config.to_experiment_config(),
                "threshold_debug": True,
                "dry_run": True,
                "allow_live_submit": False,
            }
        )
    EdgeTraderBot(config, memory_path=args.memory_path).run(once=args.once)


def get_env_status() -> dict[str, dict[str, object]]:
    provider = os.getenv("EDGE_TRADER_LLM_PROVIDER", "openrouter")
    model = os.getenv("EDGE_TRADER_LLM_MODEL", "deepseek/deepseek-chat")
    return {
        "PA_SERVER_API_KEY": {"present": bool(os.getenv("PA_SERVER_API_KEY")), "secret": True},
        "BRAVE_SEARCH_API_KEY": {"present": bool(os.getenv("BRAVE_SEARCH_API_KEY")), "secret": True},
        "OPENROUTER_API_KEY": {"present": bool(os.getenv("OPENROUTER_API_KEY")), "secret": True},
        "EDGE_TRADER_LLM_PROVIDER": {"present": bool(provider), "value": provider, "secret": False},
        "EDGE_TRADER_LLM_MODEL": {"present": bool(model), "value": model, "secret": False},
        "EDGE_TRADER_ENABLE_RAG": {
            "present": "EDGE_TRADER_ENABLE_RAG" in os.environ,
            "value": os.getenv("EDGE_TRADER_ENABLE_RAG", "0"),
            "secret": False,
        },
        "EDGE_TRADER_ENABLE_LLM_RAG": {
            "present": "EDGE_TRADER_ENABLE_LLM_RAG" in os.environ,
            "value": os.getenv("EDGE_TRADER_ENABLE_LLM_RAG", "0"),
            "secret": False,
        },
        "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK": {
            "present": "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK" in os.environ,
            "value": os.getenv("EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK"),
            "secret": False,
        },
        "EDGE_TRADER_ENABLE_BLF": {
            "present": "EDGE_TRADER_ENABLE_BLF" in os.environ,
            "value": os.getenv("EDGE_TRADER_ENABLE_BLF", "0"),
            "secret": False,
        },
        "EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK": {
            "present": "EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK" in os.environ,
            "value": os.getenv("EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK"),
            "secret": False,
        },
        "EDGE_TRADER_THRESHOLD_DEBUG": {
            "present": "EDGE_TRADER_THRESHOLD_DEBUG" in os.environ,
            "value": os.getenv("EDGE_TRADER_THRESHOLD_DEBUG", "0"),
            "secret": False,
        },
    }


def log_env_status(status: dict[str, dict[str, object]]) -> None:
    for name, item in status.items():
        if item.get("secret"):
            logger.info("env %s present=%s", name, item["present"])
        else:
            logger.info(
                "env %s present=%s value=%s",
                name,
                item["present"],
                item.get("value"),
            )


def bump_slug(slug: str) -> str:
    match = re.search(r"-v(\d+)$", slug)
    if match:
        number = int(match.group(1)) + 1
        return slug[: match.start()] + f"-v{number:02d}"
    return f"{slug}-v02"


def apparent_edge(market: MarketView | None, signals: ForecastSignals) -> float | None:
    if market is None:
        return None
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    return max(p_final - market.yes_ask, (1.0 - p_final) - market.no_ask)


def top_apparent_edges(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
    limit: int = 3,
) -> list[dict]:
    rows = []
    for market_id, signals in signals_by_market.items():
        edge = apparent_edge(selected_by_id.get(market_id), signals)
        if edge is None:
            continue
        rows.append({
            "market_id": market_id,
            "apparent_edge": edge,
            "p_final": signals.p_final,
            "p_2402": signals.p_2402,
            "confidence": signals.confidence,
            "risk_flags": signals.risk_flags,
        })
    return sorted(rows, key=lambda row: row["apparent_edge"], reverse=True)[:limit]


def pipeline_debug_rows(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
    decisions: list[TradeDecision],
    config: BotConfig,
) -> list[dict]:
    decision_by_id = {decision.market_id: decision for decision in decisions}
    return [
        market_debug_row(market, signals_by_market[market_id], decision_by_id.get(market_id), config)
        for market_id, market in selected_by_id.items()
    ]


def market_debug_row(
    market: MarketView,
    signals: ForecastSignals,
    decision: TradeDecision | None,
    config: BotConfig,
) -> dict:
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    final_edge_yes = p_final - market.yes_ask
    final_edge_no = (1.0 - p_final) - market.no_ask
    if final_edge_yes >= final_edge_no:
        best_side = "YES"
        best_edge = final_edge_yes
        best_price = market.yes_ask
    else:
        best_side = "NO"
        best_edge = final_edge_no
        best_price = market.no_ask
    resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
    market_metadata = classify_market(market)
    raw_flags = list(signals.risk_flags)
    normalized_flags = normalize_risk_flags(raw_flags, resolution_check, market)
    blocker_reasons = blocker_reasons_for(market, signals, config)
    hold_reasons = [] if decision else hold_reasons_for(market, signals, config, blocker_reasons)
    return {
        "market_id": market.market_id,
        "question": market.question,
        **market_metadata,
        "market_midpoint": market.yes_mid,
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "no_bid": market.no_bid,
        "no_ask": market.no_ask,
        "p_market": signals.p_market,
        "p_stat": signals.p_stat,
        "p_2402_raw": (signals.evidence_package or {}).get("p_2402_raw"),
        "p_2402_raw_original": (signals.evidence_package or {}).get("p_2402_raw_original"),
        "p_2402_final_after_shrinkage": (signals.evidence_package or {}).get("p_2402_final_after_shrinkage"),
        "p_final_before_blf": signals.p_final_before_blf,
        "p_blf_raw": (signals.blf_package or {}).get("p_blf_raw"),
        "p_blf_raw_original": (signals.blf_package or {}).get("p_blf_raw_original"),
        "p_blf_final_after_shrinkage": (signals.blf_package or {}).get("p_blf_final_after_shrinkage"),
        "p_final_after_blf": signals.p_final_after_blf,
        "final_edge_yes": final_edge_yes,
        "final_edge_no": final_edge_no,
        "best_side": best_side,
        "best_edge": best_edge,
        "decision": decision.to_plan_dict() if decision else None,
        "hold_reasons": hold_reasons,
        "blocker_reasons": blocker_reasons,
        "evidence_quality": signals.evidence_quality,
        "confidence": signals.confidence,
        "resolution_trade_blocker": resolution_check.get("trade_blocker") is True,
        "absence_of_evidence_penalty_detected": (
            (signals.evidence_package or {}).get("absence_of_evidence_penalty_detected", False)
            or (signals.blf_package or {}).get("absence_of_evidence_penalty_detected", False)
        ),
        "no_direct_evidence_reasoning": (
            (signals.evidence_package or {}).get("no_direct_evidence_reasoning", False)
            or (signals.blf_package or {}).get("no_direct_evidence_reasoning", False)
        ),
        "future_event_should_anchor_to_market": market_metadata["future_event_should_anchor_to_market"],
        "risk_flags": raw_flags,
        "normalized_risk_flags": normalized_flags,
        "aggregation_reason": signals.aggregation_reason,
        "sizing_result": {
            "best_side": best_side,
            "best_price": best_price,
            "shares": size_trade(best_edge, best_price, signals.confidence, config),
            "notional": size_trade(best_edge, best_price, signals.confidence, config) * best_price,
        },
    }


def blocker_reasons_for(market: MarketView, signals: ForecastSignals, config: BotConfig) -> list[str]:
    reasons: list[str] = []
    resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
    if config.block_trade_on_high_resolution_risk and resolution_check.get("trade_blocker") is True:
        reasons.append("resolution_trade_blocker")
    if resolution_check.get("official_source_required") is True and resolution_check.get("official_source_found") is False:
        reasons.append("missing_official_source_requires_larger_edge")
    if market.spread > config.max_spread:
        reasons.append("wide_spread")
    if market.volume_24h < config.min_volume_24h:
        reasons.append("low_liquidity")
    return reasons


def hold_reasons_for(
    market: MarketView,
    signals: ForecastSignals,
    config: BotConfig,
    blocker_reasons: list[str],
) -> list[str]:
    reasons = list(blocker_reasons)
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    best_edge = max(p_final - market.yes_ask, (1.0 - p_final) - market.no_ask)
    threshold = max(config.min_edge, 1.5 * signals.uncertainty)
    if signals.confidence == "low":
        threshold = max(threshold, config.low_confidence_min_edge)
    if best_edge <= threshold:
        reasons.append("edge_below_policy_threshold")
    price = market.yes_ask if p_final - market.yes_ask >= (1.0 - p_final) - market.no_ask else market.no_ask
    if size_trade(best_edge, price, signals.confidence, config) <= 0:
        reasons.append("sizing_zero")
    if signals.evidence_quality <= 2 and signals.p_2402 is not None:
        reasons.append("low_evidence_quality")
    return list(dict.fromkeys(reasons or ["no_buy_signal"]))


def normalize_risk_flags(raw_flags: list[str], resolution_check: dict, market: MarketView | None = None) -> list[str]:
    normalized: list[str] = []
    text = " ".join(str(flag).lower() for flag in raw_flags)
    resolution_flags = set(resolution_check.get("risk_flags", resolution_check.get("flags", [])) or [])
    if "headline_resolution_mismatch" in resolution_flags or "mismatch" in text:
        normalized.append("resolution_mismatch")
    if "official_source_not_confirmed" in resolution_flags or "official" in text:
        normalized.append("missing_official_source")
    if "deadline_unclear" in resolution_flags or "deadline" in text:
        normalized.append("deadline_unclear")
    if "stale_evidence" in resolution_flags or "stale" in text:
        normalized.append("stale_evidence")
    if any(term in text for term in ("weak", "expected", "rumor", "uncertain", "no_external_evidence")):
        normalized.append("weak_evidence")
    if any(term in text for term in ("conflict", "contrary", "opposition")):
        normalized.append("conflicting_evidence")
    if resolution_check.get("ambiguity_level") == "high" or "high_ambiguity" in text:
        normalized.append("high_ambiguity")
    if market is not None and market.spread > 0.20:
        normalized.append("wide_spread")
    if any(term in text for term in ("volume_too_low", "low_liquidity")):
        normalized.append("low_liquidity")
    if any(term in text for term in ("llm", "no direct evidence")):
        normalized.append("llm_uncertainty")
    if any(term in text for term in ("brave_", "search", "missing_brave_api_key", "rag_scanner_error")):
        normalized.append("search_failure")
    if any(term in text for term in ("invalid_json", "json", "fallback")):
        normalized.append("json_fallback")
    if "blf_disagreement" in text:
        normalized.append("blf_disagreement")
    if not normalized and raw_flags:
        normalized.append("other")
    return list(dict.fromkeys(normalized))


def threshold_debug_report(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
    config: BotConfig,
) -> dict:
    scenarios = []
    for min_edge in (0.10, 0.08, 0.06, 0.04):
        for min_quality in (4, 3):
            for use_blf in (True, False):
                scenarios.append(
                    evaluate_threshold_scenario(
                        selected_by_id,
                        signals_by_market,
                        min_edge=min_edge,
                        min_quality=min_quality,
                        use_blf=use_blf,
                        config=config,
                    )
                )
    return {
        "enabled": True,
        "diagnostic_only": True,
        "real_policy_unchanged": True,
        "scenarios": scenarios,
        "rag_market_disagreements_including_absence_penalty": rag_disagreement_rows(
            selected_by_id, signals_by_market, exclude_absence_penalty=False
        )[:10],
        "rag_market_disagreements_excluding_absence_penalty": rag_disagreement_rows(
            selected_by_id, signals_by_market, exclude_absence_penalty=True
        )[:10],
        "top_blf_adjustments": blf_change_rows(selected_by_id, signals_by_market)[:10],
        "negative_apparent_edge_high_evidence_quality": negative_edge_high_quality_rows(
            selected_by_id, signals_by_market
        )[:10],
        "markets_with_no_positive_edge_all_relaxed_thresholds": no_positive_edge_rows(
            selected_by_id, signals_by_market
        )[:10],
        "zero_trade_relaxed_threshold_count": sum(
            1
            for scenario in scenarios
            if scenario["min_edge_threshold"] in {0.08, 0.06, 0.04}
            and scenario["hypothetical_trade_count"] == 0
        ),
    }


def evaluate_threshold_scenario(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
    *,
    min_edge: float,
    min_quality: int,
    use_blf: bool,
    config: BotConfig,
) -> dict:
    opportunities = []
    blocked_resolution = 0
    pass_non_edge = 0
    for market_id, market in selected_by_id.items():
        signals = signals_by_market[market_id]
        p = scenario_probability(signals, use_blf=use_blf)
        edge_yes = p - market.yes_ask
        edge_no = (1.0 - p) - market.no_ask
        side = "YES" if edge_yes >= edge_no else "NO"
        edge = max(edge_yes, edge_no)
        resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
        blocked = resolution_check.get("trade_blocker") is True
        if blocked:
            blocked_resolution += 1
        non_edge_ok = (
            not blocked
            and signals.evidence_quality >= min_quality
            and market.spread <= config.max_spread
        )
        if non_edge_ok:
            pass_non_edge += 1
        if non_edge_ok and edge > min_edge:
            opportunities.append(
                {
                    "market_id": market_id,
                    "question": market.question,
                    "side": side,
                    "edge": edge,
                    "p_scenario": p,
                    "evidence_quality": signals.evidence_quality,
                    "confidence": signals.confidence,
                    "risk_flags": signals.risk_flags,
                    "normalized_risk_flags": normalize_risk_flags(signals.risk_flags, resolution_check, market),
                }
            )
    opportunities = sorted(opportunities, key=lambda row: row["edge"], reverse=True)
    qualities = [row["evidence_quality"] for row in opportunities]
    return {
        "min_edge_threshold": min_edge,
        "min_evidence_quality_to_trade": min_quality,
        "blf_enabled": use_blf,
        "hypothetical_trade_count": len(opportunities),
        "max_edge": opportunities[0]["edge"] if opportunities else None,
        "average_evidence_quality": sum(qualities) / len(qualities) if qualities else 0,
        "blocked_by_resolution_risk": blocked_resolution,
        "pass_all_non_edge_safety_checks": pass_non_edge,
        "top_5_diagnostic_opportunities": opportunities[:5],
    }


def scenario_probability(signals: ForecastSignals, *, use_blf: bool) -> float:
    if use_blf and signals.p_final_after_blf is not None:
        return signals.p_final_after_blf
    if signals.p_final_before_blf is not None:
        return signals.p_final_before_blf
    return signals.p_final if signals.p_final is not None else signals.p_market


def opportunity_report(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
) -> dict:
    rows = [
        market_debug_row(market, signals_by_market[market_id], None, BotConfig())
        for market_id, market in selected_by_id.items()
    ]
    return {
        "top_by_best_edge": sorted(compact_rows(rows), key=lambda row: row["best_edge"], reverse=True)[:10],
        "top_by_evidence_quality": sorted(
            compact_rows(rows), key=lambda row: (row["evidence_quality"], row["best_edge"]), reverse=True
        )[:10],
        "rag_disagrees_with_market_most": rag_disagreement_rows(
            selected_by_id, signals_by_market, exclude_absence_penalty=False
        )[:10],
        "rag_disagrees_with_market_most_excluding_absence_penalty": rag_disagreement_rows(
            selected_by_id, signals_by_market, exclude_absence_penalty=True
        )[:10],
        "blf_changed_probability_most": blf_change_rows(selected_by_id, signals_by_market)[:10],
        "negative_apparent_edge_high_evidence_quality": negative_edge_high_quality_rows(
            selected_by_id, signals_by_market
        )[:10],
        "markets_with_no_positive_edge_all_relaxed_thresholds": no_positive_edge_rows(
            selected_by_id, signals_by_market
        )[:10],
        "blocked_only_by_edge_threshold": [
            compact_row(row) for row in rows if row["hold_reasons"] == ["edge_below_policy_threshold", "sizing_zero"]
            or row["hold_reasons"] == ["edge_below_policy_threshold"]
        ][:10],
        "blocked_by_resolution_risk": [
            compact_row(row) for row in rows if "resolution_trade_blocker" in row["blocker_reasons"]
        ][:10],
        "blocked_by_evidence_quality": [
            compact_row(row) for row in rows if "low_evidence_quality" in row["hold_reasons"]
        ][:10],
        "markets_with_wide_spread": [
            compact_row(row) for row in rows if "wide_spread" in row["normalized_risk_flags"]
        ][:10],
        "markets_with_noisy_free_text_llm_risk_flags": noisy_free_text_flag_rows(rows)[:10],
    }


def compact_rows(rows: list[dict]) -> list[dict]:
    return [compact_row(row) for row in rows]


def compact_row(row: dict) -> dict:
    return {
        "market_id": row["market_id"],
        "question": row["question"],
        "market_type": row["market_type"],
        "best_side": row["best_side"],
        "best_edge": row["best_edge"],
        "evidence_quality": row["evidence_quality"],
        "confidence": row["confidence"],
        "p_final_after_blf": row["p_final_after_blf"],
        "p_blf_final_after_shrinkage": row["p_blf_final_after_shrinkage"],
        "absence_of_evidence_penalty_detected": row["absence_of_evidence_penalty_detected"],
        "normalized_risk_flags": row["normalized_risk_flags"],
        "hold_reasons": row["hold_reasons"],
    }


def rag_disagreement_rows(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
    *,
    exclude_absence_penalty: bool,
) -> list[dict]:
    rows = []
    for market_id, market in selected_by_id.items():
        signals = signals_by_market[market_id]
        if signals.p_2402 is None:
            continue
        absence = (signals.evidence_package or {}).get("absence_of_evidence_penalty_detected", False)
        if exclude_absence_penalty and absence:
            continue
        rows.append(
            {
                "market_id": market_id,
                "question": market.question,
                "market_type": classify_market(market)["market_type"],
                "p_market": signals.p_market,
                "p_2402": signals.p_2402,
                "absolute_disagreement": abs(signals.p_2402 - signals.p_market),
                "evidence_quality": signals.evidence_quality,
                "absence_of_evidence_penalty_detected": absence,
            }
        )
    return sorted(rows, key=lambda row: row["absolute_disagreement"], reverse=True)


def blf_change_rows(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
) -> list[dict]:
    rows = []
    for market_id, market in selected_by_id.items():
        signals = signals_by_market[market_id]
        before = signals.p_final_before_blf
        after = signals.p_final_after_blf
        if before is None or after is None or signals.blf_package is None:
            continue
        rows.append(
            {
                "market_id": market_id,
                "question": market.question,
                "market_type": classify_market(market)["market_type"],
                "p_final_before_blf": before,
                "p_final_after_blf": after,
                "absolute_change": abs(after - before),
                "blf_succeeded": signals.blf_package.get("succeeded"),
            }
        )
    return sorted(rows, key=lambda row: row["absolute_change"], reverse=True)


def negative_edge_high_quality_rows(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
) -> list[dict]:
    rows = []
    for market_id, market in selected_by_id.items():
        signals = signals_by_market[market_id]
        edge = apparent_edge(market, signals)
        if edge is not None and edge < 0 and signals.evidence_quality >= 4:
            rows.append(
                {
                    "market_id": market_id,
                    "question": market.question,
                    "market_type": classify_market(market)["market_type"],
                    "best_edge": edge,
                    "evidence_quality": signals.evidence_quality,
                    "p_final": signals.p_final,
                    "p_2402": signals.p_2402,
                    "p_blf": signals.p_blf,
                }
            )
    return sorted(rows, key=lambda row: row["evidence_quality"], reverse=True)


def no_positive_edge_rows(
    selected_by_id: dict[str, MarketView],
    signals_by_market: dict[str, ForecastSignals],
) -> list[dict]:
    rows = []
    for market_id, market in selected_by_id.items():
        signals = signals_by_market[market_id]
        p_with_blf = signals.p_final_after_blf if signals.p_final_after_blf is not None else signals.p_final
        p_without_blf = signals.p_final_before_blf if signals.p_final_before_blf is not None else signals.p_final
        edges = [
            (p_with_blf or signals.p_market) - market.yes_ask,
            (1.0 - (p_with_blf or signals.p_market)) - market.no_ask,
            (p_without_blf or signals.p_market) - market.yes_ask,
            (1.0 - (p_without_blf or signals.p_market)) - market.no_ask,
        ]
        if max(edges) <= 0:
            rows.append(
                {
                    "market_id": market_id,
                    "question": market.question,
                    "market_type": classify_market(market)["market_type"],
                    "max_edge_across_blf_modes": max(edges),
                    "evidence_quality": signals.evidence_quality,
                }
            )
    return sorted(rows, key=lambda row: row["max_edge_across_blf_modes"], reverse=True)


def noisy_free_text_flag_rows(rows: list[dict]) -> list[dict]:
    noisy = []
    for row in rows:
        free_text = [
            flag
            for flag in row["risk_flags"]
            if isinstance(flag, str) and (" " in flag or len(flag) > 48)
        ]
        if free_text:
            noisy.append(
                {
                    "market_id": row["market_id"],
                    "question": row["question"],
                    "free_text_risk_flags": free_text,
                    "normalized_risk_flags": row["normalized_risk_flags"],
                }
            )
    return noisy


if __name__ == "__main__":
    main()
