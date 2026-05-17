"""Main runner for the RAG-BLF Edge Trader."""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .rag_scanner import RagScanner
from .schemas import ForecastSignals, MarketView, TradeDecision
from .stat_priors import initial_signals
from . import trading_history
from .trading_history import DEFAULT_ARCHIVE_DIR
from .trading_policy import decide_trade, rank_decisions

logger = logging.getLogger(__name__)


class EdgeTraderBot:
    def __init__(
        self,
        config: BotConfig,
        memory_path: Path | None = None,
        archive_dir: Path | None = DEFAULT_ARCHIVE_DIR,
    ) -> None:
        self.config = config
        self.rag = RagScanner(
            enabled=config.enable_rag,
            max_queries=config.rag_max_queries,
            max_results_per_query=config.rag_max_results_per_query,
            enable_llm_summary=config.enable_llm_rag_summary,
        )
        self.blf = BlfVerifier(enabled=config.enable_blf)
        self.memory = JsonlMemory(memory_path or Path(".edge_trader/memory.jsonl"))
        self.archive_dir: Path | None = archive_dir

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
        t_start = time.time()
        run_dir: Path | None = None
        bound_lease: TickLease | None = None
        candidates: list[MarketView] = []
        portfolio = None
        selected: list[MarketView] = []
        decisions: list[TradeDecision] = []
        errors_list: list[dict] = []
        submission_result: dict = {}
        plan: dict | None = None

        try:
            tick = session.load_candidates(lease)
            bound_lease = tick.lease
            candidates = [market_view(market) for market in tick.candidates.markets]
            portfolio = session.get_portfolio(participant_idx)

            # Create archive directory now that we have tick_id
            if self.archive_dir is not None:
                run_dir = trading_history.make_tick_run_dir(self.archive_dir, bound_lease.tick_id)

            experiment_id = getattr(session, "_experiment_id", None) or getattr(
                session, "experiment_id", None
            )
            trading_history.archive_tick_artifact(
                run_dir,
                "metadata",
                trading_history.build_metadata(
                    tick_id=bound_lease.tick_id,
                    candidate_set_id=bound_lease.candidate_set_id,
                    participant_idx=participant_idx,
                    experiment_id=experiment_id,
                    config_dict=self.config.to_experiment_config(),
                ),
            )
            trading_history.archive_tick_artifact(
                run_dir,
                "candidates",
                [json_safe(m) for m in tick.candidates.markets],
            )
            trading_history.archive_tick_artifact(run_dir, "portfolio_before", json_safe(portfolio))

            selected, skipped = self.filter_markets(candidates, portfolio)
            signals_by_market: dict[str, ForecastSignals] = {}

            _rag_lock = threading.Lock()
            _rag_count = [0]

            def _process_market(market: MarketView):
                if time.time() - t_start > self.config.tick_process_deadline_sec:
                    sig = initial_signals(market)
                    sig.risk_flags.append("deadline_skip")
                    logger.warning("Deadline skip for market %s", market.market_id)
                    return market.market_id, sig, None

                sig = initial_signals(market)
                with _rag_lock:
                    do_rag = self.should_run_rag(market, _rag_count[0])
                    if do_rag and self.config.enable_rag:
                        _rag_count[0] += 1
                if do_rag:
                    sig = self.rag.scan(market, sig)
                else:
                    sig.risk_flags.append("rag_skipped_budget")
                sig = self.blf.verify(market, sig)
                sig = combine_signals(sig)
                dec = decide_trade(market, sig, portfolio, self.config)
                return market.market_id, sig, dec

            max_workers = min(self.config.blf_max_workers, max(1, len(selected)))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_process_market, m): m for m in selected}
                for fut in as_completed(futs):
                    try:
                        mid, sig, dec = fut.result()
                        signals_by_market[mid] = sig
                        if dec is not None:
                            decisions.append(dec)
                    except Exception as exc:
                        market = futs[fut]
                        logger.warning("Market %s processing failed: %s", market.market_id, exc)

            trading_history.archive_tick_artifact(
                run_dir,
                "strategy_inputs",
                {
                    "selected_markets": [json_safe(m) for m in selected],
                    "skipped_markets": skipped,
                    "signals": {mid: json_safe(sig) for mid, sig in signals_by_market.items()},
                },
            )
            trading_history.archive_tick_artifact(
                run_dir,
                "market_snapshots",
                {
                    market.market_id: json_safe(market)
                    for market in selected
                },
            )

            ranked_decisions = rank_decisions(decisions, self.config.max_trades_per_tick_target)
            trading_history.archive_tick_artifact(
                run_dir, "decisions", [json_safe(d) for d in ranked_decisions]
            )
            trading_history.archive_tick_artifact(
                run_dir, "intents", [d.to_intent_dict() for d in ranked_decisions]
            )

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
            submission_result = plan["submission"]

            if self.config.dry_run:
                trading_history.archive_tick_artifact(
                    run_dir,
                    "dry_run_result",
                    {
                        "dry_run": True,
                        "hypothetical_intents_count": len(ranked_decisions),
                        "decisions": [json_safe(d) for d in ranked_decisions],
                    },
                )
                trading_history.archive_tick_artifact(run_dir, "submit_result", None)
            else:
                trading_history.archive_tick_artifact(run_dir, "dry_run_result", None)
                trading_history.archive_tick_artifact(run_dir, "submit_result", submission_result)

            safe_plan = json_safe(plan)
            session.put_plan(bound_lease, participant_idx, safe_plan)
            self.memory.append(safe_plan)
            self.log_plan_summary(safe_plan)
            session.finalize(bound_lease, participant_idx)
            trading_history.archive_tick_artifact(run_dir, "finalize_result", {"status": "ok"})

        except Exception as exc:
            errors_list.append(json_safe(exc))
            if plan is None:
                plan = self.build_error_plan(
                    bound_lease or lease, len(candidates), exc  # type: ignore[arg-type]
                )
            else:
                plan.setdefault("errors", []).append(json_safe(exc))
            safe_plan = json_safe(plan)
            try:
                session.put_plan(bound_lease or lease, participant_idx, safe_plan)  # type: ignore[arg-type]
            except Exception:
                logger.exception("Failed to persist error plan for tick %s", lease.tick_id)
            self.memory.append(safe_plan)
            raise
        finally:
            trading_history.archive_tick_artifact(run_dir, "errors", errors_list)
            trading_history.write_summary(
                run_dir,
                trading_history.build_summary(
                    tick_id=bound_lease.tick_id if bound_lease else None,
                    candidate_count=len(candidates),
                    processed_count=len(selected),
                    intent_count=len(decisions),
                    submitted=submission_result.get("submit_called", False),
                    dry_run=self.config.dry_run,
                    errors=errors_list,
                    markets_touched=[d.market_id for d in decisions],
                    pnl_before=json_safe(portfolio).get("pnl") if portfolio else None,
                    elapsed_seconds=time.time() - t_start,
                ),
            )

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
        return {
            "strategy": self.config.strategy_name,
            "version": self.config.version,
            "run_mode": "dry_run" if self.config.dry_run else "live",
            "generated_at": datetime.now(UTC).isoformat(),
            "tick_id": lease.tick_id,
            "candidate_set_id": lease.candidate_set_id,
            "dry_run": self.config.dry_run,
            "allow_live_submit": self.config.allow_live_submit,
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
            "markets_verified_blf": 0,
            "hypothetical_intents_count": len(decisions),
            "trade_count": len(decisions),
            "submit_skipped_due_to_dry_run": self.config.dry_run and bool(decisions),
            "warnings": self.build_warnings(decisions),
            "errors": [],
            "decisions": [decision.to_plan_dict() for decision in decisions],
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
                    "p_2402_final_after_shrinkage": (
                        signals.evidence_package or {}
                    ).get("p_2402_final_after_shrinkage"),
                    "reasoning_summary": (signals.evidence_package or {}).get("reasoning_summary", ""),
                    "json_parse_error": (signals.evidence_package or {}).get("json_parse_error"),
                    "fallback_reason": (signals.evidence_package or {}).get("fallback_reason"),
                    "llm_prompt_metadata": (signals.evidence_package or {}).get("llm_prompt_metadata", {}),
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
        if decisions and not self.config.dry_run and not self.config.allow_live_submit:
            warnings.append("live_submit_disabled")
        if self.config.enable_rag and not os.getenv("BRAVE_SEARCH_API_KEY"):
            warnings.append("rag_enabled_but_brave_key_missing")
        if self.config.enable_llm_rag_summary and not os.getenv("OPENROUTER_API_KEY"):
            warnings.append("llm_rag_enabled_but_openrouter_key_missing")
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
            "hypothetical_intents_count": 0,
            "submit_skipped_due_to_dry_run": self.config.dry_run,
            "warnings": self.build_warnings([]),
            "errors": [json_safe(exc)],
            "decisions": [],
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
            plan.get("hypothetical_intents_count"),
        )
        logger.info("dry-run top_apparent_edges=%s", plan.get("top_apparent_edges", [])[:3])
        logger.info("dry-run warnings=%s errors=%s", plan.get("warnings", []), plan.get("errors", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG-BLF Edge Trader bot.")
    parser.add_argument("--once", action="store_true", help="Process at most one available tick.")
    parser.add_argument("--dry-run", action="store_true", help="Write plans but do not submit intents.")
    parser.add_argument("--slug", help="Experiment slug override.")
    parser.add_argument("--memory-path", type=Path, help="Local JSONL memory path.")
    parser.add_argument("--max-markets", type=int, help="Maximum candidates to process after filtering.")
    parser.add_argument("--allow-live-submit", action="store_true", help="Allow real submit calls when dry-run is false.")
    parser.add_argument("--check-env", action="store_true", help="Print non-secret environment readiness and exit.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory for per-tick JSON archives (default: logs/trading_history). Pass 'none' to disable.",
    )
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
    if args.allow_live_submit:
        config = BotConfig(**{**config.to_experiment_config(), "allow_live_submit": True})
    archive_dir = None if str(args.archive_dir).lower() == "none" else args.archive_dir
    EdgeTraderBot(config, memory_path=args.memory_path, archive_dir=archive_dir).run(once=args.once)


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


if __name__ == "__main__":
    main()
