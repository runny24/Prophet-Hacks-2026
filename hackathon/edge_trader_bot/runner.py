"""Main runner for the RAG-BLF Edge Trader."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TypeVar

from ai_prophet_core import DEFAULT_API_URL, APIClientError, APIError, ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession, TickLease

from .aggregator import combine_signals
from .blf_verifier import BlfVerifier
from .candidate_ranker import (
    candidate_selection_summary,
    disabled_candidate_selection_summary,
    memory_features_from_history,
    rank_candidates,
)
from .config import BotConfig, load_env_file
from .json_utils import json_safe
from .live_readiness import config_from_bot_config, evaluate_tick_live_readiness
from .market_data import market_view
from .market_filter import deterministic_filter
from .memory import JsonlMemory
from .market_classifier import classify_market
from .price_memory import PriceMemory, compute_price_features, market_snapshot
from .price_signal import (
    ForecastMispricingGateResult,
    FreshEventGateResult,
    ShortHorizonSignal,
    compute_short_horizon_trade_signal,
    is_live_tradable_forecast_mispricing_signal,
    is_live_tradable_fresh_event_signal,
    is_live_tradable_signal,
    maybe_price_exit,
    price_signal_to_decision,
)
from .portfolio_risk import positions_by_market
from .rag_scanner import RagScanner
from .schemas import ForecastSignals, MarketView, TradeDecision
from .stat_priors import initial_signals
from .trading_policy import decide_trade, rank_decisions
from .sizer import size_trade

logger = logging.getLogger(__name__)
T = TypeVar("T")


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
        self.price_memory = PriceMemory(Path(".edge_trader/market_history.jsonl"))
        self.candidate_rank_by_id: dict[str, dict] = {}
        self.candidate_selection_summary: dict = disabled_candidate_selection_summary(0, 0)
        self.live_candidate_selection_summary: dict = {}

    def run(self, once: bool = False) -> None:
        env_status = get_env_status()
        log_env_status(env_status)
        if not env_status["PA_SERVER_API_KEY"]["present"]:
            raise RuntimeError("PA_SERVER_API_KEY is required for Prophet Arena server runs")

        api_timeout = int(os.getenv("EDGE_TRADER_API_TIMEOUT", "60"))
        api = ServerAPIClient(
            base_url=os.getenv("PA_SERVER_URL", DEFAULT_API_URL),
            api_key=os.getenv("PA_SERVER_API_KEY"),
            timeout=api_timeout,
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
            logger.warning(
                "LIVE_SUBMIT_ENABLED=%s GUARD_MODE=%s max_trade_size=%s max_position_per_market=%s max_session_loss=%s",
                live_submit_enabled(self.config),
                self.config.live_guard_mode,
                self.config.price_max_trade_size,
                self.config.price_max_position_per_market,
                self.config.price_max_session_loss,
            )

            while True:
                try:
                    lease = session.claim_tick()
                except APIError as exc:
                    logger.warning("API error claiming tick, retrying in 15s: %s", exc)
                    if once:
                        return
                    time.sleep(15)
                    continue

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
                except APIError as exc:
                    logger.warning("API error on tick %s, continuing loop: %s", lease.tick_id, exc)
                    try:
                        session.finalize(
                            lease,
                            participant.participant_idx,
                            status="FAILED",
                            error_code="API_ERROR",
                            error_detail=str(exc)[:1024],
                        )
                    except Exception:
                        logger.warning("Failed to finalize tick %s after API error", lease.tick_id)
                    if once:
                        return
                    time.sleep(5)
                    continue
                except Exception as exc:
                    logger.exception("Tick %s failed", lease.tick_id)
                    try:
                        session.finalize(
                            lease,
                            participant.participant_idx,
                            status="FAILED",
                            error_code="EDGE_TRADER_ERROR",
                            error_detail=str(exc)[:1024],
                        )
                    except Exception:
                        logger.warning("Failed to finalize tick %s", lease.tick_id)
                    raise

                if once:
                    return

    def create_experiment_with_slug_retry(self, session: BenchmarkSession):
        slug = self.config.slug
        for _ in range(50):
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
        tick_started = time.monotonic()
        budget = TickWorkBudget(
            started=tick_started,
            time_budget_seconds=self.config.tick_time_budget_seconds,
            stop_before_deadline_seconds=self.config.stop_new_work_before_deadline_seconds,
        )
        tick = _load_candidates_with_retry(session, lease)
        bound_lease = tick.lease
        candidates = [market_view(market) for market in tick.candidates.markets]
        portfolio = session.get_portfolio(participant_idx)

        plan: dict | None = None
        try:
            selected, skipped = self.filter_markets(candidates, portfolio)
            runtime_diagnostics = RuntimeDiagnostics.from_config(self.config)
            signals_by_market: dict[str, ForecastSignals] = {
                market.market_id: initial_signals(market) for market in selected
            }
            rag_selection = self.rag_selection_metadata(selected)

            signals_by_market = self.run_rag_stage(
                selected,
                signals_by_market,
                rag_selection,
                budget,
                runtime_diagnostics,
            )

            blf_selection: dict[str, dict] = {}
            if self.config.enable_blf:
                blf_selection = self.blf_selection_metadata(selected, signals_by_market)
                signals_by_market = self.run_blf_stage(
                    selected,
                    signals_by_market,
                    blf_selection,
                    budget,
                    runtime_diagnostics,
                )
            else:
                for market in selected:
                    signals = self.blf.verify(market, signals_by_market[market.market_id])
                    signals_by_market[market.market_id] = combine_signals(signals)

            decisions: list[TradeDecision] = []
            price_diagnostics: dict = {}
            if self.config.enable_price_trading:
                decisions, price_diagnostics = self.build_price_trading_decisions(
                    selected,
                    signals_by_market,
                    portfolio,
                    bound_lease,
                )
            else:
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
                rag_selection=rag_selection,
                blf_selection=blf_selection,
                runtime_diagnostics=runtime_diagnostics.to_plan(),
                price_diagnostics=price_diagnostics,
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
            self.append_price_snapshots(bound_lease, selected, signals_by_market, portfolio, ranked_decisions)
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
        eligible: list[MarketView] = []
        skipped: list[dict] = []
        for market in markets:
            ok, reasons = deterministic_filter(market, portfolio, self.config)
            if ok:
                eligible.append(market)
            else:
                skipped.append({
                    "market_id": market.market_id,
                    "reasons": reasons or ["filtered"],
                    "skipped_before_rag": True,
                })
        if self.config.enable_candidate_rerank:
            memory_by_market = {
                market.market_id: memory_features_from_history(
                    self.price_memory.load_recent_market_history(market.market_id, lookback_ticks=8)
                )
                for market in eligible
            }
            ranked = rank_candidates(
                eligible,
                memory_by_market=memory_by_market,
                positions_by_market=positions_by_market(portfolio),
            )
            selected = [market for market, _ in ranked[: self.config.max_markets_to_consider]]
            selected_ids = {market.market_id for market in selected}
            for market, result in ranked:
                self.candidate_rank_by_id[market.market_id] = result.to_dict()
                if market.market_id not in selected_ids:
                    skipped.append({
                        "market_id": market.market_id,
                        "reasons": ["consideration_limit"],
                        "skipped_before_rag": True,
                        "pre_rank_score": result.score,
                        "pre_rank_bucket": result.priority_bucket,
                    })
            self.candidate_selection_summary = candidate_selection_summary(ranked=ranked, selected=selected)
            return selected, skipped

        selected = eligible[: self.config.max_markets_to_consider]
        for market in eligible[self.config.max_markets_to_consider:]:
            skipped.append({
                "market_id": market.market_id,
                "reasons": ["consideration_limit"],
                "skipped_before_rag": True,
            })
        self.candidate_rank_by_id = {}
        self.candidate_selection_summary = disabled_candidate_selection_summary(len(markets), len(selected))
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

    def run_rag_stage(
        self,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
        rag_selection: dict[str, dict],
        budget: "TickWorkBudget",
        diagnostics: "RuntimeDiagnostics",
    ) -> dict[str, ForecastSignals]:
        started = time.monotonic()
        jobs: list[MarketView] = []
        for market in selected:
            signals = signals_by_market[market.market_id]
            if not self.config.enable_rag or selection_is_selected(
                rag_selection.get(market.market_id, {}),
                selected_key="rag_selected",
            ):
                if budget.can_start_work():
                    jobs.append(market)
                else:
                    signals.risk_flags.append("rag_skipped_deadline")
                    signals_by_market[market.market_id] = combine_signals(signals)
                    diagnostics.mark_early_stop("rag")
                    diagnostics.rag_skipped_due_to_deadline_count += 1
            else:
                signals.risk_flags.append("rag_skipped_budget")
                signals_by_market[market.market_id] = combine_signals(signals)

        def scan_one(market: MarketView) -> ForecastSignals:
            signals = signals_by_market[market.market_id]
            try:
                scanned = self.rag.scan(market, signals)
                return self.apply_rag_evidence_recovery(market, scanned)
            except Exception as exc:
                logger.warning("RAG job failed for %s: %s", market.market_id, exc)
                signals.risk_flags.extend(["rag_scanner_exception", type(exc).__name__])
                signals.evidence_package = {
                    "market_id": market.market_id,
                    "question": market.question,
                    "p_2402": None,
                    "evidence_quality": 0,
                    "confidence": "low",
                    "risk_flags": ["rag_scanner_exception"],
                    "scanner_error": f"{type(exc).__name__}: {exc}",
                    "scanner_elapsed_ms": 0,
                }
                return self.apply_rag_evidence_recovery(market, signals)

        concurrency = self.effective_rag_concurrency()
        results = run_bounded_jobs(jobs, scan_one, max_workers=concurrency)
        for market in selected:
            if market.market_id in results:
                signals_by_market[market.market_id] = combine_signals(results[market.market_id])
            else:
                signals_by_market[market.market_id] = combine_signals(signals_by_market[market.market_id])
        elapsed = elapsed_ms(started)
        diagnostics.rag_wall_time_ms = elapsed
        if self.config.enable_llm_rag_summary:
            diagnostics.llm_rag_wall_time_ms = elapsed
        return signals_by_market

    def apply_rag_evidence_recovery(self, market: MarketView, signals: ForecastSignals) -> ForecastSignals:
        package = signals.evidence_package or {}
        if package.get("p_2402") is not None and not package.get("scanner_error"):
            return signals
        current_error = str(package.get("scanner_error") or package.get("fallback_reason") or "")
        if not current_error and not package.get("risk_flags"):
            return signals
        previous_p_final = self.find_previous_p_final(market.market_id)
        if previous_p_final:
            package["previous_p_final_used"] = True
            package["previous_p_final_age_ticks"] = previous_p_final["age_ticks"]
            package["previous_p_final"] = previous_p_final["p_final"]
        if self.config.enable_cached_evidence_fallback:
            cached = self.find_cached_successful_evidence(market.market_id)
            if cached and cached["age_ticks"] <= self.config.max_cached_evidence_age_ticks:
                recovered = dict(cached["evidence_package"])
                original_quality = int(recovered.get("evidence_quality") or 0)
                recovered.update(
                    {
                        "cached_evidence_used": True,
                        "cached_evidence_age_ticks": cached["age_ticks"],
                        "original_evidence_tick_ts": cached.get("tick_id"),
                        "current_rag_error": current_error or "rag_failure",
                        "current_rag_status": "failed_recovered_cached",
                        "scanner_error": None,
                        "evidence_quality": min(original_quality, 3),
                        "confidence": cap_confidence(str(recovered.get("confidence") or "low"), "medium"),
                    }
                )
                recovered["risk_flags"] = list(
                    dict.fromkeys([*(recovered.get("risk_flags") or []), "cached_evidence_used"])
                )
                if previous_p_final:
                    recovered["previous_p_final_used"] = True
                    recovered["previous_p_final_age_ticks"] = previous_p_final["age_ticks"]
                    recovered["previous_p_final"] = previous_p_final["p_final"]
                signals.evidence_package = recovered
                signals.p_2402 = recovered.get("p_2402")
                signals.evidence_quality = int(recovered.get("evidence_quality") or 0)
                signals.confidence = str(recovered.get("confidence") or "low")
                signals.uncertainty = uncertainty_from_cached_quality(signals.evidence_quality, signals.confidence)
                signals.risk_flags = list(
                    dict.fromkeys([flag for flag in signals.risk_flags if not str(flag).startswith("brave_")] + ["cached_evidence_used"])
                )
                signals.reason = "cached RAG evidence recovered current scanner failure"
                return signals
            if cached:
                package["cached_evidence_too_old"] = True
                package["cached_evidence_age_ticks"] = cached["age_ticks"]
                package["current_rag_status"] = "failed_no_fallback"
                signals.risk_flags.append("cached_evidence_too_old")
        metadata = metadata_only_evidence_package(market, package, current_error, previous_p_final)
        signals.evidence_package = metadata
        signals.evidence_quality = min(int(signals.evidence_quality or 0), 1)
        signals.confidence = "low"
        signals.risk_flags = list(dict.fromkeys([*signals.risk_flags, "metadata_only_evidence"]))
        return signals

    def find_cached_successful_evidence(self, market_id: str) -> dict | None:
        if not self.memory.path.exists():
            return None
        lines = _tail_lines(self.memory.path, self.config.max_cached_evidence_age_ticks + 2)
        for age, line in enumerate(reversed(lines), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signal = (record.get("signals") or {}).get(market_id)
            if not isinstance(signal, dict):
                continue
            package = signal.get("evidence_package") or {}
            if successful_evidence_package(package):
                return {
                    "age_ticks": age,
                    "tick_id": record.get("tick_id") or record.get("generated_at"),
                    "evidence_package": package,
                }
        return None

    def find_previous_p_final(self, market_id: str) -> dict | None:
        if not self.memory.path.exists():
            return None
        lines = _tail_lines(self.memory.path, self.config.max_cached_evidence_age_ticks + 2)
        for age, line in enumerate(reversed(lines), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signal = (record.get("signals") or {}).get(market_id)
            if not isinstance(signal, dict):
                continue
            p_final = signal.get("p_final_after_blf") or signal.get("p_final")
            if isinstance(p_final, (int, float)):
                return {"age_ticks": age, "tick_id": record.get("tick_id"), "p_final": float(p_final)}
        return None

    def run_blf_stage(
        self,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
        blf_selection: dict[str, dict],
        budget: "TickWorkBudget",
        diagnostics: "RuntimeDiagnostics",
    ) -> dict[str, ForecastSignals]:
        started = time.monotonic()
        blf_market_ids = {market_id for market_id, item in blf_selection.items() if item["selected"]}
        jobs: list[MarketView] = []
        for market in selected:
            signals = signals_by_market[market.market_id]
            if market.market_id not in blf_market_ids:
                signals.risk_flags.append("blf_skipped_budget")
                signals_by_market[market.market_id] = combine_signals(signals)
                continue
            if budget.can_start_work():
                jobs.append(market)
            else:
                signals.risk_flags.append("blf_skipped_deadline")
                signals_by_market[market.market_id] = combine_signals(signals)
                diagnostics.mark_early_stop("blf")
                diagnostics.blf_skipped_due_to_deadline_count += 1

        def verify_one(market: MarketView) -> ForecastSignals:
            signals = signals_by_market[market.market_id]
            try:
                return self.blf.verify(market, signals)
            except Exception as exc:
                logger.warning("BLF job failed for %s: %s", market.market_id, exc)
                signals.risk_flags.extend(["blf_verifier_exception", type(exc).__name__])
                signals.blf_package = {
                    "market_id": market.market_id,
                    "enabled": True,
                    "attempted": True,
                    "succeeded": False,
                    "fallback_reason": "blf_verifier_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_ms": 0,
                }
                return signals

        results = run_bounded_jobs(jobs, verify_one, max_workers=max(1, self.config.blf_concurrency))
        for market in selected:
            if market.market_id in results:
                signals_by_market[market.market_id] = combine_signals(results[market.market_id])
            else:
                signals_by_market[market.market_id] = combine_signals(signals_by_market[market.market_id])
        diagnostics.blf_wall_time_ms = elapsed_ms(started)
        return signals_by_market

    def effective_rag_concurrency(self) -> int:
        if self.config.enable_llm_rag_summary:
            return max(1, min(self.config.rag_concurrency, self.config.llm_rag_concurrency))
        return max(1, self.config.rag_concurrency)

    def build_price_trading_decisions(
        self,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
        portfolio,
        lease: TickLease,
    ) -> tuple[list[TradeDecision], dict]:
        decisions: list[TradeDecision] = []
        price_rows: list[dict] = []
        exit_rows: list[dict] = []
        positions = positions_by_market(portfolio)
        entry_count = 0
        exit_count = 0
        for market in selected:
            signals = signals_by_market[market.market_id]
            history = self.price_memory.load_recent_market_history(market.market_id, lookback_ticks=8)
            features = compute_price_features(history, market, signals)
            signal = compute_short_horizon_trade_signal(market, signals, features, portfolio, self.config)
            live_tradable, live_block_reasons = is_live_tradable_signal(
                market=market,
                signals=signals,
                features=features,
                signal=signal,
                action="BUY",
            )
            fresh_gate = is_live_tradable_fresh_event_signal(
                market=market,
                signals=signals,
                features=features,
                portfolio=portfolio,
                config=self.config,
            )
            forecast_gate = is_live_tradable_forecast_mispricing_signal(
                market=market,
                signals=signals,
                features=features,
                portfolio=portfolio,
                config=self.config,
            )
            if fresh_gate.allowed:
                channel = "fresh_event"
            elif forecast_gate.allowed:
                channel = "clean_forecast_mispricing"
            elif live_tradable:
                channel = "price_action"
            else:
                channel = "none"
            exit_decision, exit_diag = maybe_price_exit(
                market,
                signals,
                positions.get(market.market_id),
                history,
                self.config,
                current_signal=signal,
            )
            if exit_diag:
                exit_rows.append(exit_diag)
            if exit_decision is not None:
                if self.config.live_guard_mode and exit_count >= 5:
                    exit_diag.setdefault("live_block_reasons", []).append("max_exits_per_tick_guard_cap")
                else:
                    decisions.append(exit_decision)
                    exit_count += 1
            else:
                if self.config.live_guard_mode:
                    if fresh_gate.allowed:
                        signal = fresh_event_signal(market, fresh_gate, signals)
                        live_block_reasons = []
                        live_tradable = True
                    elif forecast_gate.allowed:
                        signal = forecast_mispricing_signal(market, forecast_gate, signals)
                        live_block_reasons = []
                        live_tradable = True
                    elif not live_tradable:
                        signal.blockers.extend(live_block_reasons)
                    freeze_scope = live_freeze_scope()
                    if freeze_scope == "all_entries":
                        signal.blockers.append("live_entries_frozen_all_entries")
                    elif freeze_scope == "forecast_dependent_entries_only" and forecast_dependent_signal(signal, signals):
                        signal.blockers.append("live_entries_frozen_forecast_only")
                    if entry_count >= 3:
                        signal.blockers.append("max_new_entries_per_tick_guard_cap")
                    if total_open_shares(portfolio) >= 12 and market.market_id not in positions:
                        signal.blockers.append("max_total_open_shares_guard_cap")
                    signal.blockers = list(dict.fromkeys(signal.blockers))
                    signal.suggested_size = min(signal.suggested_size, 1)
                if signal.side in ("YES", "NO"):
                    _entry_ask = market.yes_ask if signal.side == "YES" else market.no_ask
                    if _entry_ask >= self.config.near_certainty_block_threshold:
                        signal.blockers.append("near_certainty_blocked")
                entry = price_signal_to_decision(market, signal, signals)
                if entry is not None:
                    decisions.append(entry)
                    entry_count += 1
            row_block_reasons = [*live_block_reasons, *signal.blockers]
            if signal.side == "HOLD" or signal.blockers:
                if not fresh_gate.allowed:
                    row_block_reasons.extend(fresh_gate.reasons)
                if not forecast_gate.allowed:
                    row_block_reasons.extend(forecast_gate.reasons)
            price_rows.append(
                {
                    "market_id": market.market_id,
                    "question": market.question,
                    "action": "BUY",
                    "side": signal.side,
                    "size": signal.suggested_size,
                    "signal_type": signal.signal_type,
                    "entry_price": market.yes_ask if signal.side == "YES" else market.no_ask if signal.side == "NO" else None,
                    "current_mid": market.yes_mid,
                    "previous_mid": features.mid_1_tick_ago,
                    "mid_change_1tick": features.delta_1_tick,
                    "mid_change_2tick": features.delta_2_ticks,
                    "repeated_market_count": features.repeated_seen_count,
                    "spread": market.spread,
                    "expected_tick_edge": signal.expected_tick_edge,
                    "predicted_mid_move": signal.expected_tick_edge,
                    "live_tradable": signal.side != "HOLD" and not signal.blockers,
                    "live_block_reasons": list(dict.fromkeys(row_block_reasons)),
                    "channel": channel if not signal.blockers else "none",
                    "fresh_event_gate": fresh_gate.to_dict(),
                    "forecast_mispricing_gate": forecast_gate.to_dict(),
                    "live_freeze_active": live_freeze_scope() != "none",
                    "live_freeze_reason": os.getenv("EDGE_TRADER_LIVE_FREEZE_REASON", ""),
                    "freeze_scope": live_freeze_scope(),
                    "risk_flags": list(signals.risk_flags or []),
                    "forecast_p_final": signals.p_final_after_blf if signals.p_final_after_blf is not None else signals.p_final,
                    "forecast_confidence": signals.confidence,
                    "forecast_evidence_quality": signals.evidence_quality,
                    "current_rag_status": (signals.evidence_package or {}).get(
                        "current_rag_status",
                        "success" if (signals.evidence_package or {}).get("p_2402") is not None else "not_evaluated",
                    ),
                    "current_rag_error": (signals.evidence_package or {}).get("current_rag_error")
                    or (signals.evidence_package or {}).get("scanner_error"),
                    "cached_evidence_used": (signals.evidence_package or {}).get("cached_evidence_used", False),
                    "cached_evidence_age_ticks": (signals.evidence_package or {}).get("cached_evidence_age_ticks"),
                    "metadata_only_evidence": (signals.evidence_package or {}).get("metadata_only_evidence", False),
                    "previous_p_final_used": (signals.evidence_package or {}).get("previous_p_final_used", False),
                    "previous_p_final_age_ticks": (signals.evidence_package or {}).get("previous_p_final_age_ticks"),
                    "normalized_risk_flags": normalize_risk_flags(
                        signals.risk_flags,
                        (signals.evidence_package or {}).get("resolution_check", {}),
                        market,
                    ),
                    "features": features.to_dict(),
                    "short_horizon_price_signal": signal.to_dict(),
                    "position": position_to_plan_dict(positions.get(market.market_id)),
                    "would_trade": not signal.blockers and signal.side != "HOLD",
                }
            )
        return decisions, price_diagnostics(price_rows, exit_rows, decisions, portfolio)

    def append_price_snapshots(
        self,
        lease: TickLease,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
        portfolio,
        decisions: list[TradeDecision],
    ) -> None:
        traded_market_ids = {decision.market_id for decision in decisions}
        positions = positions_by_market(portfolio)
        generated_at = datetime.now(UTC).isoformat()
        for market in selected:
            self.price_memory.append_market_snapshot(
                market_snapshot(
                    tick_id=lease.tick_id,
                    candidate_set_id=lease.candidate_set_id,
                    generated_at=generated_at,
                    market=market,
                    signals=signals_by_market[market.market_id],
                    position=positions.get(market.market_id),
                    traded=market.market_id in traded_market_ids,
                )
            )

    def rag_selection_metadata(self, selected: list[MarketView]) -> dict[str, dict]:
        eligible: list[tuple[float, MarketView, list[str]]] = []
        metadata: dict[str, dict] = {}
        for idx, market in enumerate(selected):
            ok = 0.05 < market.yes_mid < 0.95
            if self.config.rag_selection_mode == "alpha_diagnostic":
                score, reasons = alpha_rag_selection_score(market)
            else:
                score, reasons = current_rag_selection_score(market, idx)
            metadata[market.market_id] = {
                "rag_selected": False,
                "rag_selection_rank": None,
                "rag_selection_score": score,
                "rag_selection_reason": reasons if ok else [*reasons, "outside_rag_probability_band"],
                "rag_forced_by_live_candidate_selection": False,
                "live_candidate_selection_rank": None,
                "live_candidate_selection_score": None,
            }
            metadata[market.market_id].update(
                {
                    "selected": False,
                    "rank": None,
                    "score": score,
                    "reason": metadata[market.market_id]["rag_selection_reason"],
                }
            )
            if ok:
                eligible.append((score, market, reasons))
        live_ranked = self.live_rag_candidate_order(selected)
        live_rank_by_id = {
            market.market_id: (rank, score, reasons)
            for rank, (market, score, reasons) in enumerate(live_ranked, start=1)
        }
        for market_id, (rank, score, reasons) in live_rank_by_id.items():
            if market_id not in metadata:
                continue
            metadata[market_id]["live_candidate_selection_rank"] = rank
            metadata[market_id]["live_candidate_selection_score"] = score
            metadata[market_id]["live_candidate_selection_reason"] = reasons
        if self.config.rag_selection_mode == "alpha_diagnostic":
            eligible.sort(key=lambda row: row[0], reverse=True)
        else:
            eligible.sort(key=lambda row: row[0], reverse=True)
        chosen: list[MarketView] = []
        chosen_ids: set[str] = set()
        if self.config.enable_price_trading and self.config.live_guard_mode:
            for market, _, _ in live_ranked:
                if len(chosen) >= self.config.rag_max_markets_per_tick:
                    break
                if market.market_id in metadata and 0.05 < market.yes_mid < 0.95:
                    chosen.append(market)
                    chosen_ids.add(market.market_id)
        for _, market, _ in eligible:
            if len(chosen) >= self.config.rag_max_markets_per_tick:
                break
            if market.market_id not in chosen_ids:
                chosen.append(market)
                chosen_ids.add(market.market_id)
        for rank, market in enumerate(chosen, start=1):
            metadata[market.market_id]["rag_selected"] = True
            metadata[market.market_id]["rag_selection_rank"] = rank
            metadata[market.market_id]["selected"] = True
            metadata[market.market_id]["rank"] = rank
            if market.market_id in live_rank_by_id:
                metadata[market.market_id]["rag_forced_by_live_candidate_selection"] = True
                metadata[market.market_id]["reason"] = list(
                    dict.fromkeys([*metadata[market.market_id]["reason"], "rag_forced_by_live_candidate_selection"])
                )
                metadata[market.market_id]["rag_selection_reason"] = metadata[market.market_id]["reason"]
        for record in metadata.values():
            if not record["selected"] and "outside_rag_probability_band" not in record["reason"]:
                record["reason"] = [*record["reason"], "rag_skipped_budget"]
                record["rag_selection_reason"] = record["reason"]
        self.live_candidate_selection_summary = live_candidate_selection_summary(live_ranked, metadata)
        return metadata

    def live_rag_candidate_order(self, selected: list[MarketView]) -> list[tuple[MarketView, float, list[str]]]:
        if not (self.config.enable_price_trading and self.config.live_guard_mode):
            return []
        scored: list[tuple[float, str, MarketView, list[str]]] = []
        for market in selected:
            rank = self.candidate_rank_by_id.get(market.market_id, {})
            bucket = rank.get("priority_bucket", "exploratory")
            components = rank.get("score_components", {}) or {}
            reasons = [f"bucket={bucket}"]
            score = float(rank.get("score") or 0.0)
            if bucket in {"repeated_mover", "near_term_catalyst", "tight_spread_active"}:
                score += 50.0
                reasons.append("live_relevant_bucket")
            if market.spread <= self.config.live_forecast_max_spread:
                score += 25.0
                reasons.append("spread_within_live_forecast_cap")
            if market.spread <= 0.005:
                score += 10.0
                reasons.append("very_tight_spread")
            if components.get("repeated_market_count") or components.get("mid_change_1tick"):
                score += 20.0
                reasons.append("repeated_or_moving")
            if components.get("near_term_catalyst_words"):
                score += 20.0
                reasons.append("near_term_catalyst_words")
            if market.spread > max(0.05, self.config.max_spread):
                score -= 100.0
                reasons.append("wide_spread_penalty")
            scored.append((score, market.market_id, market, reasons))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [(market, score, reasons) for score, _, market, reasons in scored]

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

    def blf_selection_metadata(
        self,
        selected: list[MarketView],
        signals_by_market: dict[str, ForecastSignals],
    ) -> dict[str, dict]:
        scored: list[tuple[float, str, list[str]]] = []
        by_id = {market.market_id: market for market in selected}
        metadata: dict[str, dict] = {}
        for market_id, signals in signals_by_market.items():
            market = by_id[market_id]
            resolution_check = (signals.evidence_package or {}).get("resolution_check", {})
            edge = abs(apparent_edge(market, signals) or 0.0)
            disagreement = abs((signals.p_2402 or signals.p_market) - signals.p_market)
            risk_penalty = 0.10 if resolution_check.get("trade_blocker") else 0.04 if resolution_check.get("risk_flags") else 0.0
            score = edge + 0.01 * signals.evidence_quality + 0.25 * disagreement - risk_penalty
            possible_live_edge = max(
                (signals.p_final_after_blf or signals.p_final or signals.p_market) - market.yes_ask,
                (1.0 - (signals.p_final_after_blf or signals.p_final or signals.p_market)) - market.no_ask,
            )
            if self.config.enable_price_trading and self.config.live_guard_mode:
                if possible_live_edge >= 0.004:
                    score += 0.25 + possible_live_edge
                if (signals.evidence_package or {}).get("fresh_evidence") or (signals.evidence_package or {}).get("near_term_catalyst"):
                    score += 0.15
                rank = self.candidate_rank_by_id.get(market_id, {})
                if rank.get("priority_bucket") in {"repeated_mover", "near_term_catalyst", "tight_spread_active"}:
                    score += 0.05
            reasons = [
                f"abs_edge={edge:.4f}",
                f"evidence_quality={signals.evidence_quality}",
                f"rag_market_disagreement={disagreement:.4f}",
                f"possible_live_edge={possible_live_edge:.4f}",
            ]
            if risk_penalty:
                reasons.append(f"resolution_risk_penalty={risk_penalty:.2f}")
            metadata[market_id] = {
                "blf_selected": False,
                "blf_selection_rank": None,
                "blf_selection_score": score,
                "blf_selection_reason": reasons,
                "selected": False,
                "rank": None,
                "score": score,
                "reason": reasons,
            }
            scored.append((score, market_id, reasons))
        scored.sort(reverse=True)
        for rank, (_, market_id, _) in enumerate(scored[: self.config.blf_max_markets_per_tick], start=1):
            metadata[market_id]["blf_selected"] = True
            metadata[market_id]["blf_selection_rank"] = rank
            metadata[market_id]["selected"] = True
            metadata[market_id]["rank"] = rank
        return metadata

    def build_plan(
        self,
        *,
        lease: TickLease,
        candidates_count: int,
        selected: list[MarketView],
        skipped: list[dict],
        signals_by_market: dict[str, ForecastSignals],
        decisions: list[TradeDecision],
        rag_selection: dict[str, dict] | None = None,
        blf_selection: dict[str, dict] | None = None,
        runtime_diagnostics: dict | None = None,
        price_diagnostics: dict | None = None,
    ) -> dict:
        selected_by_id = {market.market_id: market for market in selected}
        rag_selection = rag_selection or {}
        blf_selection = blf_selection or {}
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
        rag_failure_count_by_reason = dict(
            Counter(
                str((signals.evidence_package or {}).get("current_rag_error") or (signals.evidence_package or {}).get("scanner_error"))
                for signals in signals_by_market.values()
                if (signals.evidence_package or {}).get("scanner_error")
                or (signals.evidence_package or {}).get("current_rag_error")
            )
        )
        cached_evidence_used_count = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("cached_evidence_used")
        )
        metadata_only_fallback_count = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("metadata_only_evidence")
        )
        cached_evidence_too_old_count = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("cached_evidence_too_old")
        )
        previous_p_final_used_count = sum(
            1 for signals in signals_by_market.values() if (signals.evidence_package or {}).get("previous_p_final_used")
        )
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
        runtime_diagnostics = runtime_diagnostics or RuntimeDiagnostics.from_config(self.config).to_plan()
        price_diagnostics = price_diagnostics or empty_price_diagnostics()
        debug_rows = pipeline_debug_rows(
            selected_by_id,
            signals_by_market,
            decisions,
            self.config,
            rag_selection,
            blf_selection,
        )
        for row in debug_rows:
            row.update(pre_rank_debug_fields(self.candidate_rank_by_id.get(row["market_id"], {})))
        live_readiness = evaluate_tick_live_readiness(
            debug_rows,
            config=config_from_bot_config(self.config),
            blf_enabled=self.config.enable_blf,
            allow_live_submit=self.config.allow_live_submit and not self.config.dry_run,
        )
        replay_records = build_replay_records(
            debug_rows,
            tick_id=lease.tick_id,
            candidate_set_id=lease.candidate_set_id,
            generated_at=datetime.now(UTC).isoformat(),
            live_readiness=live_readiness,
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
            "candidate_selection_summary": self.candidate_selection_summary,
            **self.live_candidate_selection_summary,
            "markets_considered": candidates_count,
            "markets_after_filter": len(selected),
            "markets_filtered": len(skipped),
            "rag_markets_attempted": rag_attempted,
            "rag_markets_scanned": rag_scanned,
            "markets_scanned_2402": rag_scanned,
            "rag_scanner_errors": rag_scanner_errors,
            "rag_failure_count_by_reason": rag_failure_count_by_reason,
            "cached_evidence_fallback_count": cached_evidence_used_count,
            "cached_evidence_used_count": cached_evidence_used_count,
            "cached_evidence_too_old_count": cached_evidence_too_old_count,
            "metadata_only_fallback_count": metadata_only_fallback_count,
            "previous_p_final_used_count": previous_p_final_used_count,
            "recovered_rag_candidates_top10": recovered_rag_candidates(signals_by_market, selected_by_id)[:10],
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
            "live_candidates_sent_to_blf_count": sum(
                1 for item in (blf_selection or {}).values() if item.get("selected")
            ),
            "live_candidates_sent_to_blf_top10": [
                {
                    "market_id": market_id,
                    "blf_selection_rank": item.get("blf_selection_rank"),
                    "blf_selection_score": item.get("blf_selection_score"),
                    "blf_selection_reason": item.get("blf_selection_reason"),
                }
                for market_id, item in sorted(
                    (blf_selection or {}).items(),
                    key=lambda kv: kv[1].get("blf_selection_rank") or 10_000,
                )
                if item.get("selected")
            ][:10],
            "total_blf_elapsed_ms": total_blf_elapsed_ms,
            "stage_summed_market_times": {
                "total_rag_elapsed_ms": total_rag_elapsed_ms,
                "total_llm_elapsed_ms": total_llm_elapsed_ms,
                "total_blf_elapsed_ms": total_blf_elapsed_ms,
            },
            "blf_errors": blf_errors,
            **runtime_diagnostics,
            "hypothetical_intents_count": len(decisions),
            "trade_count": len(decisions),
            **price_diagnostics,
            "submit_skipped_due_to_dry_run": self.config.dry_run and bool(decisions),
            "warnings": self.build_warnings(decisions),
            "errors": [],
            "decisions": [decision.to_plan_dict() for decision in decisions],
            "pipeline_debug": debug_rows,
            "trade_readiness_verdict": live_readiness,
            "replay_records": replay_records,
            "threshold_debug_report": threshold_debug_report(
                selected_by_id,
                signals_by_market,
                self.config,
            )
            if self.config.threshold_debug
            else {},
            "opportunity_report": opportunity_report(selected_by_id, signals_by_market, rag_selection, blf_selection),
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
                    "sports_llm_overconfidence_detected": (signals.evidence_package or {}).get(
                        "sports_llm_overconfidence_detected",
                        False,
                    ),
                    "sports_quantitative_support": (signals.evidence_package or {}).get(
                        "sports_quantitative_support",
                        False,
                    ),
                    "sports_support_source_type": (signals.evidence_package or {}).get(
                        "sports_support_source_type",
                        "none",
                    ),
                    "resolution_check": (signals.evidence_package or {}).get("resolution_check", {}),
                    "rag_sources": rag_source_metadata(signals.evidence_package or {}),
                    "blf_package": signals.blf_package,
                    "blf_update_summary": blf_update_summary(signals.blf_package or {}),
                    "p_final_before_blf": signals.p_final_before_blf,
                    "p_final_after_blf": signals.p_final_after_blf,
                    "blf_adjustment": signals.blf_adjustment,
                    "aggregation_reason": signals.aggregation_reason,
                    "apparent_edge": apparent_edge(selected_by_id.get(market_id), signals),
                    **rag_selection.get(market_id, {}),
                    **blf_selection.get(market_id, {}),
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
        if not live_submit_enabled(self.config):
            return {
                "accepted": 0,
                "rejected": 0,
                "dry_run": False,
                "submit_called": False,
                "reason": "live_guard_blocked",
                "live_guard_mode": self.config.live_guard_mode,
                "env_allow_live_submit": os.getenv("EDGE_TRADER_ALLOW_LIVE_SUBMIT", "0"),
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


class TickWorkBudget:
    def __init__(
        self,
        *,
        started: float,
        time_budget_seconds: int,
        stop_before_deadline_seconds: int,
    ) -> None:
        self.started = started
        self.time_budget_seconds = time_budget_seconds
        self.stop_before_deadline_seconds = stop_before_deadline_seconds

    def remaining_seconds(self) -> float:
        return self.time_budget_seconds - (time.monotonic() - self.started)

    def can_start_work(self) -> bool:
        return self.remaining_seconds() >= self.stop_before_deadline_seconds


class RuntimeDiagnostics:
    def __init__(self, config: BotConfig) -> None:
        self.tick_time_budget_seconds = config.tick_time_budget_seconds
        self.stop_new_work_before_deadline_seconds = config.stop_new_work_before_deadline_seconds
        self.early_stop_triggered = False
        self.early_stop_stage = "none"
        self.rag_skipped_due_to_deadline_count = 0
        self.blf_skipped_due_to_deadline_count = 0
        self.rag_wall_time_ms = 0
        self.llm_rag_wall_time_ms = 0
        self.blf_wall_time_ms = 0
        self.rag_concurrency = config.rag_concurrency
        self.llm_rag_concurrency = config.llm_rag_concurrency
        self.blf_concurrency = config.blf_concurrency

    @classmethod
    def from_config(cls, config: BotConfig) -> "RuntimeDiagnostics":
        return cls(config)

    def mark_early_stop(self, stage: str) -> None:
        if not self.early_stop_triggered:
            self.early_stop_triggered = True
            self.early_stop_stage = stage

    def to_plan(self) -> dict:
        skipped = self.rag_skipped_due_to_deadline_count + self.blf_skipped_due_to_deadline_count
        return {
            "tick_time_budget_seconds": self.tick_time_budget_seconds,
            "stop_new_work_before_deadline_seconds": self.stop_new_work_before_deadline_seconds,
            "early_stop_triggered": self.early_stop_triggered,
            "early_stop_stage": self.early_stop_stage,
            "skipped_due_to_deadline_count": skipped,
            "rag_skipped_due_to_deadline_count": self.rag_skipped_due_to_deadline_count,
            "blf_skipped_due_to_deadline_count": self.blf_skipped_due_to_deadline_count,
            "stage_wall_times": {
                "rag_wall_time_ms": self.rag_wall_time_ms,
                "llm_rag_wall_time_ms": self.llm_rag_wall_time_ms,
                "blf_wall_time_ms": self.blf_wall_time_ms,
            },
            "concurrency_settings": {
                "rag_concurrency": self.rag_concurrency,
                "llm_rag_concurrency": self.llm_rag_concurrency,
                "blf_concurrency": self.blf_concurrency,
            },
        }


def _load_candidates_with_retry(session: "BenchmarkSession", lease: "TickLease", max_attempts: int = 5, base_delay: int = 20):
    """Retry load_candidates on transient server errors with linear backoff."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_attempts):
        try:
            return session.load_candidates(lease)
        except APIError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (attempt + 1)
                logger.warning(
                    "load_candidates attempt %d/%d failed, retrying in %ds: %s",
                    attempt + 1, max_attempts, delay, exc,
                )
                time.sleep(delay)
    raise last_exc


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read last n non-empty lines from a JSONL file without loading the full file."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            chunk_size = min(2 * 1024 * 1024, size)  # 2MB chunks
            lines: list[bytes] = []
            pos = size
            remainder = b""
            while pos > 0 and len(lines) < n + 1:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size) + remainder
                parts = chunk.split(b"\n")
                remainder = parts[0]
                lines = parts[1:] + lines
            if remainder:
                lines = [remainder] + lines
            non_empty = [ln.decode("utf-8", errors="replace") for ln in lines if ln.strip()]
            return non_empty[-n:] if len(non_empty) > n else non_empty
    except OSError:
        return []


def run_bounded_jobs(
    items: list[MarketView],
    worker: Callable[[MarketView], ForecastSignals],
    *,
    max_workers: int,
) -> dict[str, ForecastSignals]:
    if not items:
        return {}
    max_workers = max(1, min(max_workers, len(items)))
    results: dict[str, ForecastSignals] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_market = {executor.submit(worker, market): market for market in items}
        for future in as_completed(future_by_market):
            market = future_by_market[future]
            results[market.market_id] = future.result()
    return results


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def empty_price_diagnostics() -> dict:
    return {
        "price_trading_enabled": False,
        "count_price_signals": 0,
        "count_momentum_signals": 0,
        "count_mean_reversion_signals": 0,
        "count_spread_capture_signals": 0,
        "count_forecast_edge_signals": 0,
        "count_trade_intents": 0,
        "count_exit_intents": 0,
        "top_price_signals": [],
        "top_trade_candidates": [],
        "blocked_by_no_history": 0,
        "blocked_by_spread": 0,
        "blocked_by_position_limit": 0,
        "blocked_by_live_guard": 0,
        "blocked_by_edge_threshold": 0,
        "positions_table": [],
        "exit_candidates": [],
        "exits_submitted": 0,
        "positions_held": 0,
        "price_signal_rows": [],
        "live_readiness_verdict": {},
        "live_blocked_reason_counts": {},
        "live_tradable_candidate_count": 0,
        "skipped_by_live_safety_count": 0,
        "live_trade_candidates": [],
        "rejected_trade_candidates_top10": [],
        "signal_type_counts": {},
        "repeated_history_candidate_count": 0,
        "fallback_risk_candidate_count": 0,
        "price_only_candidate_count": 0,
        "forecast_only_candidate_count": 0,
        "live_price_action_candidate_count": 0,
        "live_fresh_event_candidate_count": 0,
        "live_forecast_mispricing_candidate_count": 0,
        "skipped_by_price_action_history_count": 0,
        "skipped_by_fresh_event_safety_count": 0,
        "fresh_event_candidates": [],
        "fresh_event_rejected_top10": [],
        "rejected_forecast_mispricing_top10": [],
        "forecast_gate_rejection_top10": [],
        "not_evaluated_by_forecast_budget_count": 0,
        "analyzed_live_candidate_count": 0,
        "analyzed_live_candidate_top10": [],
        "live_forecast_mispricing_cached_candidate_count": 0,
        "rejected_cached_evidence_top10": [],
        "metadata_only_rejected_top10": [],
        "live_submission_reason": "no_live_tradable_candidates",
        "live_freeze_active": False,
        "live_freeze_reason": "",
        "freeze_scope": "none",
        "clean_price_action_candidates_count": 0,
        "blocked_by_freeze_forecast_only_count": 0,
        "blocked_by_freeze_all_count": 0,
    }


def price_diagnostics(
    price_rows: list[dict],
    exit_rows: list[dict],
    decisions: list[TradeDecision],
    portfolio,
) -> dict:
    signals = [row["short_horizon_price_signal"] for row in price_rows]
    candidates = [row for row in price_rows if row["short_horizon_price_signal"]["side"] != "HOLD"]
    reason_counts = Counter(
        reason
        for row in price_rows
        for reason in row.get("live_block_reasons", [])
    )
    signal_counts = Counter(signal["signal_type"] for signal in signals)
    live_candidates = [row for row in candidates if row.get("live_tradable")]
    rejected_candidates = [row for row in candidates if not row.get("live_tradable")]
    fresh_candidates = [row for row in price_rows if row.get("channel") == "fresh_event"]
    fresh_rejected = [
        row for row in price_rows if (row.get("fresh_event_gate") or {}).get("channel") == "fresh_event"
        and not (row.get("fresh_event_gate") or {}).get("allowed")
    ]
    forecast_rejected = [
        row for row in price_rows
        if (row.get("forecast_mispricing_gate") or {}).get("channel") == "clean_forecast_mispricing"
        and not (row.get("forecast_mispricing_gate") or {}).get("allowed")
    ]
    forecast_analyzed_rows = [row for row in price_rows if forecast_gate_was_analyzed(row)]
    forecast_gate_rejections = [row for row in forecast_rejected if forecast_gate_was_analyzed(row)]
    cached_rows = [row for row in price_rows if row.get("cached_evidence_used")]
    metadata_only_rows = [row for row in price_rows if row.get("metadata_only_evidence")]
    return {
        "price_trading_enabled": True,
        "count_price_signals": sum(1 for signal in signals if signal["signal_type"] != "none"),
        "count_momentum_signals": sum(1 for signal in signals if signal["signal_type"] == "momentum"),
        "count_mean_reversion_signals": sum(1 for signal in signals if signal["signal_type"] == "mean_reversion"),
        "count_spread_capture_signals": sum(1 for signal in signals if signal["signal_type"] == "spread_capture"),
        "count_forecast_edge_signals": sum(1 for signal in signals if signal["signal_type"] == "forecast_edge"),
        "count_trade_intents": sum(1 for decision in decisions if decision.action == "BUY"),
        "count_exit_intents": sum(1 for decision in decisions if decision.action == "SELL"),
        "top_price_signals": sorted(
            candidates,
            key=lambda row: row["short_horizon_price_signal"]["expected_tick_edge"],
            reverse=True,
        )[:10],
        "top_trade_candidates": [
            row
            for row in sorted(
                candidates,
                key=lambda item: item["short_horizon_price_signal"]["expected_tick_edge"],
                reverse=True,
            )
            if not row["short_horizon_price_signal"]["blockers"]
        ][:10],
        "blocked_by_no_history": count_blocker(signals, "no_history"),
        "blocked_by_spread": count_blocker(signals, "spread_too_wide") + count_blocker(signals, "spread_absurd"),
        "blocked_by_position_limit": count_blocker(signals, "position_limit_total")
        + count_blocker(signals, "position_limit_market"),
        "blocked_by_live_guard": 0,
        "blocked_by_edge_threshold": count_blocker(signals, "edge_below_price_threshold"),
        "positions_table": positions_table(portfolio),
        "exit_candidates": exit_rows,
        "exits_submitted": sum(1 for decision in decisions if decision.action == "SELL"),
        "positions_held": len(getattr(portfolio, "positions", []) or []),
        "price_signal_rows": price_rows,
        "live_readiness_verdict": {
            "LIVE_SUBMIT_READY": bool(live_candidates),
            "candidate_count": len(live_candidates),
            "reasons": ["price_action_candidate_present"] if live_candidates else [],
            "blockers": sorted(reason_counts),
        },
        "live_blocked_reason_counts": dict(reason_counts),
        "live_tradable_candidate_count": len(live_candidates),
        "skipped_by_live_safety_count": len(rejected_candidates),
        "live_trade_candidates": live_candidates[:10],
        "rejected_trade_candidates_top10": rejected_candidates[:10],
        "signal_type_counts": dict(signal_counts),
        "repeated_history_candidate_count": sum(
            1 for row in candidates if int(row.get("repeated_market_count") or 0) >= 3
        ),
        "fallback_risk_candidate_count": sum(
            1
            for row in candidates
            if any(
                marker in flag
                for flag in row.get("risk_flags", [])
                for marker in ("rate_limit", "fallback", "llm_rag_failed")
            )
        ),
        "price_only_candidate_count": sum(
            1 for signal in signals if signal["signal_type"] in {"momentum", "mean_reversion"}
        ),
        "forecast_only_candidate_count": sum(
            1 for signal in signals if signal["signal_type"] in {"forecast_edge", "spread_capture"}
        ),
        "live_price_action_candidate_count": sum(
            1 for row in live_candidates if row.get("channel") == "price_action"
        ),
        "live_fresh_event_candidate_count": sum(
            1 for row in live_candidates if row.get("channel") == "fresh_event"
        ),
        "live_forecast_mispricing_candidate_count": sum(
            1 for row in live_candidates if row.get("channel") == "clean_forecast_mispricing"
        ),
        "skipped_by_price_action_history_count": sum(
            1
            for row in price_rows
            if any(
                reason in row.get("live_block_reasons", [])
                for reason in ("no_history", "missing_previous_mid", "insufficient_repeated_history")
            )
        ),
        "skipped_by_fresh_event_safety_count": len(fresh_rejected),
        "fresh_event_candidates": fresh_candidates[:10],
        "fresh_event_rejected_top10": fresh_rejected[:10],
        "rejected_forecast_mispricing_top10": forecast_rejected[:10],
        "forecast_gate_rejection_top10": forecast_gate_rejections[:10],
        "live_forecast_mispricing_cached_candidate_count": sum(
            1 for row in live_candidates if row.get("channel") == "clean_forecast_mispricing" and row.get("cached_evidence_used")
        ),
        "rejected_cached_evidence_top10": [row for row in cached_rows if not row.get("live_tradable")][:10],
        "metadata_only_rejected_top10": [row for row in metadata_only_rows if not row.get("live_tradable")][:10],
        "not_evaluated_by_forecast_budget_count": sum(
            1 for row in price_rows if "not_evaluated_by_forecast_budget" in row.get("live_block_reasons", [])
        ),
        "analyzed_live_candidate_count": len(forecast_analyzed_rows),
        "analyzed_live_candidate_top10": sorted(
            [
                {
                    "market_id": row.get("market_id"),
                    "question": row.get("question"),
                    "current_rag_status": row.get("current_rag_status"),
                    "current_rag_error": row.get("current_rag_error"),
                    "cached_evidence_used": row.get("cached_evidence_used"),
                    "cached_evidence_age_ticks": row.get("cached_evidence_age_ticks"),
                    "metadata_only_evidence": row.get("metadata_only_evidence"),
                    "p_final": row.get("forecast_p_final"),
                    "p_market": (row.get("features") or {}).get("mid_now"),
                    "confidence": row.get("forecast_confidence"),
                    "evidence_quality": row.get("forecast_evidence_quality"),
                    "edge": (row.get("forecast_mispricing_gate") or {}).get("edge"),
                    "spread": row.get("spread"),
                    "risk_flags": row.get("risk_flags", []),
                    "live_gate_status": "allowed" if row.get("live_tradable") else "blocked",
                    "live_gate_block_reasons": row.get("live_block_reasons", []),
                }
                for row in forecast_analyzed_rows
            ],
            key=lambda item: item.get("edge") or -999.0,
            reverse=True,
        )[:10],
        "live_submission_reason": live_submission_reason(live_candidates, fresh_rejected, reason_counts),
        "live_freeze_active": live_freeze_scope() != "none",
        "live_freeze_reason": os.getenv("EDGE_TRADER_LIVE_FREEZE_REASON", ""),
        "freeze_scope": live_freeze_scope(),
        "clean_price_action_candidates_count": sum(
            1
            for row in candidates
            if row.get("signal_type") in {"momentum", "mean_reversion"}
            and not any("fallback_or_rate_limit_risk" in reason for reason in row.get("live_block_reasons", []))
        ),
        "blocked_by_freeze_forecast_only_count": count_blocker(signals, "live_entries_frozen_forecast_only"),
        "blocked_by_freeze_all_count": count_blocker(signals, "live_entries_frozen_all_entries"),
    }


def count_blocker(signals: list[dict], blocker: str) -> int:
    return sum(1 for signal in signals if blocker in signal.get("blockers", []))


def forecast_gate_was_analyzed(row: dict) -> bool:
    return "not_evaluated_by_forecast_budget" not in (row.get("forecast_mispricing_gate") or {}).get("reasons", [])


def live_submission_reason(live_candidates: list[dict], fresh_rejected: list[dict], reason_counts: Counter) -> str:
    if live_candidates:
        return "submitted"
    if reason_counts.get("live_entries_frozen_all_entries") or reason_counts.get("live_entries_frozen_forecast_only"):
        return "live_entries_frozen"
    if fresh_rejected:
        return "only_rejected_by_fresh_event_safety"
    return "no_live_tradable_candidates"


def pre_rank_debug_fields(rank: dict) -> dict:
    return {
        "pre_rank_score": rank.get("score"),
        "pre_rank_bucket": rank.get("priority_bucket"),
        "pre_rank_components": rank.get("score_components", {}),
        "pre_rank_penalty_reasons": rank.get("reject_or_penalty_reasons", []),
    }


def live_candidate_selection_summary(
    live_ranked: list[tuple[MarketView, float, list[str]]],
    rag_selection: dict[str, dict],
) -> dict:
    pool = [
        {
            "market_id": market.market_id,
            "question": market.question,
            "spread": market.spread,
            "yes_mid": market.yes_mid,
            "live_candidate_selection_rank": rank,
            "live_candidate_selection_score": score,
            "live_candidate_selection_reason": reasons,
            "rag_selected": bool((rag_selection.get(market.market_id) or {}).get("rag_selected")),
        }
        for rank, (market, score, reasons) in enumerate(live_ranked, start=1)
    ]
    sent_to_rag = [row for row in pool if row["rag_selected"]]
    return {
        "live_candidate_pool_count": len(pool),
        "live_candidate_pool_top10": pool[:10],
        "live_candidates_sent_to_rag_count": len(sent_to_rag),
        "live_candidates_sent_to_rag_top10": sent_to_rag[:10],
    }


def successful_evidence_package(package: dict) -> bool:
    return (
        isinstance(package, dict)
        and package.get("p_2402") is not None
        and int(package.get("evidence_quality") or 0) > 0
        and not package.get("scanner_error")
        and not package.get("metadata_only_evidence")
        and not package.get("llm_fallback")
    )


def cap_confidence(confidence: str, cap: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    inverse = {0: "low", 1: "medium", 2: "high"}
    return inverse[min(order.get(confidence, 0), order.get(cap, 1))]


def uncertainty_from_cached_quality(evidence_quality: int, confidence: str) -> float:
    if confidence == "medium" and evidence_quality >= 3:
        return 0.06
    return 0.08


def metadata_only_evidence_package(
    market: MarketView,
    failed_package: dict,
    current_error: str,
    previous_p_final: dict | None,
) -> dict:
    metadata = classify_market(market)
    package = {
        "market_id": market.market_id,
        "question": market.question,
        **metadata,
        "p_2402": None,
        "evidence_quality": 1,
        "confidence": "low",
        "evidence_for_yes": [],
        "evidence_for_no": [],
        "open_questions": ["External RAG failed; metadata-only package is diagnostic."],
        "resolution_check": {},
        "risk_flags": ["metadata_only_evidence"],
        "scanner_error": current_error or "rag_failure",
        "current_rag_error": current_error or "rag_failure",
        "current_rag_status": "failed_metadata_only",
        "metadata_only_evidence": True,
        "cached_evidence_used": False,
        "cached_evidence_too_old": bool(failed_package.get("cached_evidence_too_old")),
        "cached_evidence_age_ticks": failed_package.get("cached_evidence_age_ticks"),
        "market_metadata": {
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "yes_mid": market.yes_mid,
            "no_bid": market.no_bid,
            "no_ask": market.no_ask,
            "spread": market.spread,
            "volume_24h": market.volume_24h,
        },
        "queries": failed_package.get("queries", []),
        "search_queries": failed_package.get("search_queries", []),
        "raw_search_result_count": 0,
        "deduped_search_result_count": 0,
        "scanner_elapsed_ms": failed_package.get("scanner_elapsed_ms", 0),
    }
    if previous_p_final:
        package.update(
            {
                "previous_p_final_used": True,
                "previous_p_final_age_ticks": previous_p_final["age_ticks"],
                "previous_p_final": previous_p_final["p_final"],
            }
        )
    return package


def recovered_rag_candidates(
    signals_by_market: dict[str, ForecastSignals],
    selected_by_id: dict[str, MarketView],
) -> list[dict]:
    rows = []
    for market_id, signals in signals_by_market.items():
        package = signals.evidence_package or {}
        if not (package.get("cached_evidence_used") or package.get("metadata_only_evidence")):
            continue
        market = selected_by_id.get(market_id)
        rows.append(
            {
                "market_id": market_id,
                "question": market.question if market else package.get("question"),
                "current_rag_status": package.get("current_rag_status"),
                "current_rag_error": package.get("current_rag_error") or package.get("scanner_error"),
                "cached_evidence_used": package.get("cached_evidence_used", False),
                "cached_evidence_age_ticks": package.get("cached_evidence_age_ticks"),
                "metadata_only_evidence": package.get("metadata_only_evidence", False),
                "evidence_quality": signals.evidence_quality,
                "confidence": signals.confidence,
                "p_final": signals.p_final_after_blf or signals.p_final,
                "p_market": signals.p_market,
                "risk_flags": signals.risk_flags,
            }
        )
    return rows


def positions_table(portfolio) -> list[dict]:
    return [position_to_plan_dict(position) for position in getattr(portfolio, "positions", []) or []]


def total_open_shares(portfolio) -> float:
    total = 0.0
    for position in getattr(portfolio, "positions", []) or []:
        try:
            total += abs(float(getattr(position, "shares", 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def fresh_event_signal(
    market: MarketView,
    gate: FreshEventGateResult,
    signals: ForecastSignals,
) -> ShortHorizonSignal:
    return ShortHorizonSignal(
        market_id=market.market_id,
        side=gate.side,
        signal_type="fresh_event",
        expected_tick_edge=gate.edge,
        confidence=signals.confidence,
        reason="fresh_event: clean catalyst/outcome edge passed guarded gate",
        blockers=[],
        suggested_size=gate.max_size,
    )


def forecast_mispricing_signal(
    market: MarketView,
    gate: ForecastMispricingGateResult,
    signals: ForecastSignals,
) -> ShortHorizonSignal:
    return ShortHorizonSignal(
        market_id=market.market_id,
        side=gate.side,
        signal_type="clean_forecast_mispricing",
        expected_tick_edge=gate.edge,
        confidence=signals.confidence,
        reason="clean_forecast_mispricing: post-shrinkage forecast edge passed guarded gate",
        blockers=[],
        suggested_size=gate.max_size,
    )


def live_freeze_scope() -> str:
    scope = os.getenv("EDGE_TRADER_LIVE_FREEZE_SCOPE", "").lower()
    if scope in {"forecast_dependent_entries_only", "all_entries"}:
        return scope
    if scope == "forecast_only":
        return "forecast_dependent_entries_only"
    if os.getenv("EDGE_TRADER_FREEZE_NEW_ENTRIES", "0").lower() in {"1", "true", "yes"}:
        return "all_entries"
    return "none"


def forecast_dependent_signal(signal, signals: ForecastSignals) -> bool:
    if signal.signal_type in {"forecast_edge", "spread_capture", "fresh_event", "clean_forecast_mispricing"}:
        return True
    market_type = (signals.evidence_package or {}).get("market_type") or (signals.blf_package or {}).get("market_type")
    if market_type == "sports_outcome" and not (signals.evidence_package or {}).get("sports_quantitative_support", False):
        return signal.signal_type not in {"momentum", "mean_reversion"}
    return False


def position_to_plan_dict(position) -> dict | None:
    if position is None:
        return None
    return {
        "market_id": getattr(position, "market_id", None),
        "side": getattr(position, "side", None),
        "shares": str(getattr(position, "shares", "")),
        "avg_entry_price": str(getattr(position, "avg_entry_price", "")),
        "current_price": str(getattr(position, "current_price", "")),
    }


def live_submit_enabled(config: BotConfig) -> bool:
    env_allows = os.getenv("EDGE_TRADER_ALLOW_LIVE_SUBMIT", "0").lower() in {"1", "true", "yes"}
    if config.unsafe_live_override:
        return config.allow_live_submit and env_allows
    return config.allow_live_submit and env_allows and config.live_guard_mode


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
        "EDGE_TRADER_RAG_SELECTION_MODE": {
            "present": "EDGE_TRADER_RAG_SELECTION_MODE" in os.environ,
            "value": os.getenv("EDGE_TRADER_RAG_SELECTION_MODE", "current"),
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
    rag_selection: dict[str, dict] | None = None,
    blf_selection: dict[str, dict] | None = None,
) -> list[dict]:
    decision_by_id = {decision.market_id: decision for decision in decisions}
    rag_selection = rag_selection or {}
    blf_selection = blf_selection or {}
    return [
        market_debug_row(
            market,
            signals_by_market[market_id],
            decision_by_id.get(market_id),
            config,
            rag_selection.get(market_id, missing_selection_record("rag_selection_missing")),
            blf_selection.get(market_id, {}),
        )
        for market_id, market in selected_by_id.items()
    ]


def build_replay_records(
    rows: list[dict],
    *,
    tick_id: str,
    candidate_set_id: str,
    generated_at: str,
    live_readiness: dict,
) -> list[dict]:
    readiness_by_market = {
        item.get("market_id"): item
        for item in live_readiness.get("per_market", [])
        if isinstance(item, dict)
    }
    records = []
    for row in rows:
        records.append(
            {
                "tick_id": tick_id,
                "candidate_set_id": candidate_set_id,
                "generated_at": generated_at,
                "market_id": row.get("market_id"),
                "question": row.get("question"),
                "market_type": row.get("market_type"),
                "prices": {
                    "yes_bid": row.get("yes_bid"),
                    "yes_ask": row.get("yes_ask"),
                    "no_bid": row.get("no_bid"),
                    "no_ask": row.get("no_ask"),
                    "market_midpoint": row.get("market_midpoint"),
                },
                "p_market": row.get("p_market"),
                "p_stat": row.get("p_stat"),
                "p_2402_raw": row.get("p_2402_raw"),
                "p_2402_final_after_shrinkage": row.get("p_2402_final_after_shrinkage"),
                "p_blf_raw": row.get("p_blf_raw"),
                "p_blf_final_after_shrinkage": row.get("p_blf_final_after_shrinkage"),
                "p_final_before_blf": row.get("p_final_before_blf"),
                "p_final_after_blf": row.get("p_final_after_blf"),
                "evidence_quality": row.get("evidence_quality"),
                "confidence": row.get("confidence"),
                "risk_flags": row.get("risk_flags", []),
                "normalized_risk_flags": row.get("normalized_risk_flags", []),
                "resolution_check": row.get("resolution_check", {}),
                "rag_sources": row.get("rag_sources", []),
                "blf_update_summary": row.get("blf_update_summary", {}),
                "decision": row.get("decision"),
                "hold_reasons": row.get("hold_reasons", []),
                "live_readiness": readiness_by_market.get(row.get("market_id"), {}),
                "latency": {
                    "scanner_elapsed_ms": row.get("scanner_elapsed_ms"),
                    "llm_elapsed_ms": row.get("llm_elapsed_ms"),
                    "blf_elapsed_ms": row.get("blf_elapsed_ms"),
                },
            }
        )
    return records


def market_debug_row(
    market: MarketView,
    signals: ForecastSignals,
    decision: TradeDecision | None,
    config: BotConfig,
    rag_selection: dict | None = None,
    blf_selection: dict | None = None,
) -> dict:
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    edge_diag = edge_diagnostics(market, p_final)
    best_side = edge_diag["best_taker_side"]
    best_edge = edge_diag["best_taker_edge"]
    best_price = market.yes_ask if best_side == "YES" else market.no_ask if best_side == "NO" else None
    sizing_shares = (
        size_trade(best_edge, best_price, signals.confidence, config)
        if best_edge is not None and best_price is not None
        else 0
    )
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
        "blf_adjustment": signals.blf_adjustment,
        "final_edge_yes": edge_diag["taker_edge_yes"],
        "final_edge_no": edge_diag["taker_edge_no"],
        **edge_diag,
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
        "sports_llm_overconfidence_detected": (signals.evidence_package or {}).get(
            "sports_llm_overconfidence_detected",
            False,
        ),
        "sports_quantitative_support": (signals.evidence_package or {}).get("sports_quantitative_support", False),
        "sports_support_source_type": (signals.evidence_package or {}).get("sports_support_source_type", "none"),
        "resolution_check": resolution_check,
        "rag_sources": rag_source_metadata(signals.evidence_package or {}),
        "blf_update_summary": blf_update_summary(signals.blf_package or {}),
        "skipped_due_to_deadline": (
            "rag_skipped_deadline" in raw_flags or "blf_skipped_deadline" in raw_flags
        ),
        "risk_flags": raw_flags,
        "normalized_risk_flags": normalized_flags,
        "aggregation_reason": signals.aggregation_reason,
        "scanner_elapsed_ms": (signals.evidence_package or {}).get("scanner_elapsed_ms", 0),
        "llm_elapsed_ms": (signals.evidence_package or {}).get("llm_elapsed_ms", 0),
        "blf_elapsed_ms": (signals.blf_package or {}).get("elapsed_ms", 0),
        "blf_package": signals.blf_package,
        "sizing_result": {
            "best_side": best_side,
            "best_price": best_price,
            "shares": sizing_shares,
            "notional": sizing_shares * best_price if best_price is not None else 0,
        },
            "rag_selected": False,
            "rag_selection_rank": None,
            "rag_selection_score": None,
            "rag_selection_reason": [],
            "blf_selected": False,
            "blf_selection_rank": None,
            "blf_selection_score": None,
            "blf_selection_reason": [],
        **selection_diagnostic_aliases(rag_selection or {}, "rag"),
        **selection_diagnostic_aliases(blf_selection or {}, "blf"),
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


def rag_source_metadata(evidence_package: dict) -> list[dict]:
    sources = []
    for item in [*(evidence_package.get("evidence_for_yes") or []), *(evidence_package.get("evidence_for_no") or [])]:
        sources.append(
            {
                "source": item.get("source"),
                "url": item.get("url"),
                "timestamp": item.get("timestamp"),
                "relevance": item.get("relevance"),
                "supports_resolution_condition": item.get("supports_resolution_condition"),
            }
        )
    return sources[:12]


def blf_update_summary(blf_package: dict) -> dict:
    return {
        "succeeded": blf_package.get("succeeded"),
        "confidence": blf_package.get("confidence"),
        "uncertainty": blf_package.get("uncertainty"),
        "p_blf_raw": blf_package.get("p_blf_raw"),
        "p_blf_final_after_shrinkage": blf_package.get("p_blf_final_after_shrinkage"),
        "risk_flags": blf_package.get("risk_flags", []),
        "fallback_reason": blf_package.get("fallback_reason"),
        "update_steps": [
            {
                "step": step.get("step"),
                "probability_before": step.get("probability_before"),
                "probability_after": step.get("probability_after"),
                "delta": step.get("delta"),
                "reasoning_summary": step.get("reasoning_summary"),
                "risk_flags": step.get("risk_flags", []),
            }
            for step in (blf_package.get("update_steps") or [])[:3]
            if isinstance(step, dict)
        ],
    }


def edge_diagnostics(market: MarketView, p_final: float) -> dict:
    yes_mid = safe_mid(market.yes_bid, market.yes_ask)
    no_mid = safe_mid(market.no_bid, market.no_ask)
    yes_spread = safe_subtract(market.yes_ask, market.yes_bid)
    no_spread = safe_subtract(market.no_ask, market.no_bid)
    taker_edge_yes = safe_subtract(p_final, market.yes_ask)
    taker_edge_no = safe_subtract(1.0 - p_final, market.no_ask)
    maker_edge_yes = safe_subtract(p_final, market.yes_bid)
    maker_edge_no = safe_subtract(1.0 - p_final, market.no_bid)
    mid_edge_yes = safe_subtract(p_final, yes_mid)
    mid_edge_no = safe_subtract(1.0 - p_final, no_mid)
    maker_edge_after_half_spread_yes = safe_subtract(maker_edge_yes, half_value(yes_spread))
    maker_edge_after_half_spread_no = safe_subtract(maker_edge_no, half_value(no_spread))
    maker_edge_after_full_spread_yes = safe_subtract(maker_edge_yes, yes_spread)
    maker_edge_after_full_spread_no = safe_subtract(maker_edge_no, no_spread)
    best_taker_side, best_taker_edge = best_side_and_edge(taker_edge_yes, taker_edge_no)
    best_maker_side, best_maker_edge = best_side_and_edge(maker_edge_yes, maker_edge_no)
    best_mid_side, best_mid_edge = best_side_and_edge(mid_edge_yes, mid_edge_no)
    best_half_spread_side, best_half_spread_edge = best_side_and_edge(
        maker_edge_after_half_spread_yes,
        maker_edge_after_half_spread_no,
    )
    best_full_spread_side, best_full_spread_edge = best_side_and_edge(
        maker_edge_after_full_spread_yes,
        maker_edge_after_full_spread_no,
    )
    best_spread = spread_for_side(best_maker_side, yes_spread, no_spread)
    return {
        "yes_mid_diagnostic": yes_mid,
        "no_mid_diagnostic": no_mid,
        "yes_spread": yes_spread,
        "no_spread": no_spread,
        "best_spread": best_spread,
        "taker_edge_yes": taker_edge_yes,
        "taker_edge_no": taker_edge_no,
        "maker_edge_yes": maker_edge_yes,
        "maker_edge_no": maker_edge_no,
        "mid_edge_yes": mid_edge_yes,
        "mid_edge_no": mid_edge_no,
        "maker_edge_after_half_spread_yes": maker_edge_after_half_spread_yes,
        "maker_edge_after_half_spread_no": maker_edge_after_half_spread_no,
        "maker_edge_after_full_spread_yes": maker_edge_after_full_spread_yes,
        "maker_edge_after_full_spread_no": maker_edge_after_full_spread_no,
        "best_taker_side": best_taker_side,
        "best_taker_edge": best_taker_edge,
        "best_maker_side": best_maker_side,
        "best_maker_edge": best_maker_edge,
        "best_mid_side": best_mid_side,
        "best_mid_edge": best_mid_edge,
        "best_spread_adjusted_maker_side": best_half_spread_side,
        "best_spread_adjusted_maker_edge": best_half_spread_edge,
        "maker_edge_after_half_spread": spread_adjusted_edge_for_side(
            best_maker_side,
            maker_edge_after_half_spread_yes,
            maker_edge_after_half_spread_no,
        ),
        "maker_edge_after_full_spread": spread_adjusted_edge_for_side(
            best_maker_side,
            maker_edge_after_full_spread_yes,
            maker_edge_after_full_spread_no,
        ),
        "best_full_spread_adjusted_maker_side": best_full_spread_side,
        "best_full_spread_adjusted_maker_edge": best_full_spread_edge,
        "passive_opportunity_quality": passive_opportunity_quality(best_mid_edge),
    }


def safe_mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def safe_subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def half_value(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 2.0


def spread_for_side(side: str | None, yes_spread: float | None, no_spread: float | None) -> float | None:
    if side == "YES":
        return yes_spread
    if side == "NO":
        return no_spread
    return None


def spread_adjusted_edge_for_side(
    side: str | None,
    yes_edge: float | None,
    no_edge: float | None,
) -> float | None:
    if side == "YES":
        return yes_edge
    if side == "NO":
        return no_edge
    return None


def passive_opportunity_quality(best_mid_edge: float | None) -> str:
    if best_mid_edge is None or best_mid_edge <= 0:
        return "none"
    if best_mid_edge >= 0.02 - 1e-9:
        return "strong"
    if best_mid_edge >= 0.01 - 1e-9:
        return "medium"
    return "weak"


def best_side_and_edge(yes_edge: float | None, no_edge: float | None) -> tuple[str | None, float | None]:
    if yes_edge is None and no_edge is None:
        return None, None
    if no_edge is None or (yes_edge is not None and yes_edge >= no_edge):
        return "YES", yes_edge
    return "NO", no_edge


def current_rag_selection_score(market: MarketView, idx: int) -> tuple[float, list[str]]:
    return -float(idx), ["current_order", f"selected_index={idx}"]


def alpha_rag_selection_score(market: MarketView) -> tuple[float, list[str]]:
    metadata = classify_market(market)
    score = 0.0
    reasons: list[str] = []
    if 0.05 < market.yes_mid < 0.95:
        score += 1.0
        reasons.append("reasonable_probability_band")
    if market.spread <= 0.10:
        score += 0.5
        reasons.append("tight_spread")
    if market.volume_24h > 0:
        score += min(0.5, market.volume_24h / 10_000.0)
        reasons.append("has_volume")
    days = metadata["long_horizon_days_to_resolution"]
    if 3 <= days <= 365:
        score += 0.5
        reasons.append("short_medium_horizon")
    if metadata["market_type"] in {
        "entertainment_casting",
        "election_control",
        "future_announcement",
        "government_action",
        "appointment_nomination",
        "geopolitical_action",
    }:
        score += 0.4
        reasons.append("news_sensitive_market_type")
    if market.yes_mid < 0.03 or market.yes_mid > 0.97:
        score -= 0.5
        reasons.append("tail_market_penalty")
    return score, reasons or ["alpha_default"]


def selection_is_selected(selection: dict, *, selected_key: str = "rag_selected") -> bool:
    return bool(selection.get("selected", selection.get(selected_key, False)))


def selection_diagnostic_aliases(selection: dict, prefix: str) -> dict:
    if not selection:
        return {}
    aliases = dict(selection)
    selected_key = f"{prefix}_selected"
    rank_key = f"{prefix}_selection_rank"
    score_key = f"{prefix}_selection_score"
    reason_key = f"{prefix}_selection_reason"
    if selected_key not in aliases and "selected" in aliases:
        aliases[selected_key] = aliases["selected"]
    if rank_key not in aliases and "rank" in aliases:
        aliases[rank_key] = aliases["rank"]
    if score_key not in aliases and "score" in aliases:
        aliases[score_key] = aliases["score"]
    if reason_key not in aliases and "reason" in aliases:
        aliases[reason_key] = aliases["reason"]
    if "selected" not in aliases and selected_key in aliases:
        aliases["selected"] = aliases[selected_key]
    if "rank" not in aliases and rank_key in aliases:
        aliases["rank"] = aliases[rank_key]
    if "score" not in aliases and score_key in aliases:
        aliases["score"] = aliases[score_key]
    if "reason" not in aliases and reason_key in aliases:
        aliases["reason"] = aliases[reason_key]
    return aliases


def missing_selection_record(reason: str) -> dict:
    return {
        "selected": False,
        "rank": None,
        "score": None,
        "reason": [reason],
    }


def hold_reasons_for(
    market: MarketView,
    signals: ForecastSignals,
    config: BotConfig,
    blocker_reasons: list[str],
) -> list[str]:
    reasons = list(blocker_reasons)
    p_final = signals.p_final if signals.p_final is not None else signals.p_market
    edge_diag = edge_diagnostics(market, p_final)
    best_edge = edge_diag["best_taker_edge"]
    threshold = max(config.min_edge, 1.5 * signals.uncertainty)
    if signals.confidence == "low":
        threshold = max(threshold, config.low_confidence_min_edge)
    if best_edge is None or best_edge <= threshold:
        reasons.append("edge_below_policy_threshold")
    price = market.yes_ask if edge_diag["best_taker_side"] == "YES" else market.no_ask if edge_diag["best_taker_side"] == "NO" else None
    if best_edge is None or price is None or size_trade(best_edge, price, signals.confidence, config) <= 0:
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
    if "sports_llm_overconfidence" in text:
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
    diagnostic_edge_thresholds = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
    for min_edge in diagnostic_edge_thresholds:
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
        "diagnostic_edge_thresholds": list(diagnostic_edge_thresholds),
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
            if scenario["hypothetical_trade_count"] == 0
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
    rag_selection: dict[str, dict] | None = None,
    blf_selection: dict[str, dict] | None = None,
) -> dict:
    rag_selection = rag_selection or {}
    blf_selection = blf_selection or {}
    rows = [
        market_debug_row(
            market,
            signals_by_market[market_id],
            None,
            BotConfig(),
            rag_selection.get(market_id, {}),
            blf_selection.get(market_id, {}),
        )
        for market_id, market in selected_by_id.items()
    ]
    return {
        "top_by_best_edge": sorted(compact_rows(rows), key=lambda row: none_low(row["best_edge"]), reverse=True)[:10],
        "top_by_taker_edge": sorted(compact_rows(rows), key=lambda row: none_low(row["best_taker_edge"]), reverse=True)[:10],
        "top_by_maker_edge": sorted(compact_rows(rows), key=lambda row: none_low(row["best_maker_edge"]), reverse=True)[:10],
        "top_by_mid_edge": sorted(compact_rows(rows), key=lambda row: none_low(row["best_mid_edge"]), reverse=True)[:10],
        "top_by_spread_adjusted_maker_edge": sorted(
            compact_rows(rows),
            key=lambda row: none_low(row["maker_edge_after_half_spread"]),
            reverse=True,
        )[:10],
        "markets_where_maker_edge_positive_but_taker_edge_negative": [
            compact_row(row)
            for row in sorted(rows, key=lambda item: none_low(item["best_maker_edge"]), reverse=True)
            if none_low(row["best_maker_edge"]) > 0 and none_low(row["best_taker_edge"]) < 0
        ][:10],
        "markets_where_mid_edge_positive_but_taker_edge_negative": [
            compact_row(row)
            for row in sorted(rows, key=lambda item: none_low(item["best_mid_edge"]), reverse=True)
            if none_low(row["best_mid_edge"]) > 0 and none_low(row["best_taker_edge"]) < 0
        ][:10],
        "count_positive_taker_edge": sum(1 for row in rows if none_low(row["best_taker_edge"]) > 0),
        "count_positive_maker_edge": sum(1 for row in rows if none_low(row["best_maker_edge"]) > 0),
        "count_positive_mid_edge": sum(1 for row in rows if none_low(row["best_mid_edge"]) > 0),
        "count_mid_edge_ge_0_005": edge_count_at_least(rows, "best_mid_edge", 0.005),
        "count_mid_edge_ge_0_01": edge_count_at_least(rows, "best_mid_edge", 0.01),
        "count_mid_edge_ge_0_02": edge_count_at_least(rows, "best_mid_edge", 0.02),
        "count_taker_edge_ge_0_005": edge_count_at_least(rows, "best_taker_edge", 0.005),
        "count_taker_edge_ge_0_01": edge_count_at_least(rows, "best_taker_edge", 0.01),
        "count_taker_edge_ge_0_02": edge_count_at_least(rows, "best_taker_edge", 0.02),
        "manual_review_candidates": manual_review_candidates(rows)[:20],
        "manual_review_actionable_count": manual_review_grade_count(rows, "actionable_candidate"),
        "manual_review_watchlist_count": manual_review_grade_count(rows, "watchlist"),
        "manual_review_weak_count": manual_review_grade_count(rows, "weak"),
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
        "top_unscanned_by_taker_edge": top_unscanned(rows, "best_taker_edge")[:10],
        "top_unscanned_by_maker_edge": top_unscanned(rows, "best_maker_edge")[:10],
        "top_unscanned_by_mid_edge": top_unscanned(rows, "best_mid_edge")[:10],
        "top_scanned_by_rag_edge": [
            compact_row(row)
            for row in sorted(rows, key=lambda item: none_low(item["best_taker_edge"]), reverse=True)
            if row.get("rag_selected")
        ][:10],
        "markets_skipped_due_to_rag_budget": [compact_row(row) for row in rows if not row.get("rag_selected")][:25],
        "markets_skipped_due_to_blf_budget": [compact_row(row) for row in rows if not row.get("blf_selected")][:25],
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
        "best_taker_side": row["best_taker_side"],
        "best_taker_edge": row["best_taker_edge"],
        "best_maker_side": row["best_maker_side"],
        "best_maker_edge": row["best_maker_edge"],
        "best_mid_side": row["best_mid_side"],
        "best_mid_edge": row["best_mid_edge"],
        "yes_spread": row.get("yes_spread"),
        "no_spread": row.get("no_spread"),
        "best_spread": row.get("best_spread"),
        "maker_edge_after_half_spread": row.get("maker_edge_after_half_spread"),
        "maker_edge_after_full_spread": row.get("maker_edge_after_full_spread"),
        "best_spread_adjusted_maker_side": row.get("best_spread_adjusted_maker_side"),
        "best_spread_adjusted_maker_edge": row.get("best_spread_adjusted_maker_edge"),
        "passive_opportunity_quality": row.get("passive_opportunity_quality"),
        "evidence_quality": row["evidence_quality"],
        "confidence": row["confidence"],
        "p_market": row.get("p_market"),
        "p_2402": row.get("p_2402_final_after_shrinkage"),
        "p_final_after_blf": row["p_final_after_blf"],
        "p_blf": row.get("p_blf_final_after_shrinkage"),
        "p_blf_final_after_shrinkage": row["p_blf_final_after_shrinkage"],
        "blf_adjustment": row.get("blf_adjustment"),
        "resolution_trade_blocker": row.get("resolution_trade_blocker", False),
        "risk_flags": row.get("risk_flags", []),
        "sports_llm_overconfidence_detected": row.get("sports_llm_overconfidence_detected", False),
        "sports_quantitative_support": row.get("sports_quantitative_support", False),
        "sports_support_source_type": row.get("sports_support_source_type", "none"),
        "skipped_due_to_deadline": row.get("skipped_due_to_deadline", False),
        "absence_of_evidence_penalty_detected": row["absence_of_evidence_penalty_detected"],
        "normalized_risk_flags": row["normalized_risk_flags"],
        "hold_reasons": row["hold_reasons"],
        "rag_selected": row.get("rag_selected"),
        "rag_selection_rank": row.get("rag_selection_rank"),
        "rag_selection_score": row.get("rag_selection_score"),
        "blf_selected": row.get("blf_selected"),
        "blf_selection_rank": row.get("blf_selection_rank"),
        "blf_selection_score": row.get("blf_selection_score"),
    }


def edge_count_at_least(rows: list[dict], field: str, threshold: float) -> int:
    return sum(1 for row in rows if none_low(row.get(field)) >= threshold - 1e-9)


def manual_review_grade_count(rows: list[dict], grade: str) -> int:
    return sum(1 for row in rows if actionable_grade(row) == grade)


def manual_review_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        grade = actionable_grade(row)
        if grade == "none":
            continue
        item = compact_row(row)
        item["actionable_grade"] = grade
        item["p_final"] = row.get("p_final_after_blf")
        item["taker_edge"] = row.get("best_taker_edge")
        item["mid_edge"] = row.get("best_mid_edge")
        item["spread_adjusted_maker_edge"] = row.get("maker_edge_after_half_spread")
        item["reason_not_traded"] = row.get("hold_reasons", [])
        candidates.append(item)
    return sorted(candidates, key=lambda row: none_low(row.get("best_taker_edge")), reverse=True)


def actionable_grade(row: dict) -> str:
    taker_edge = none_low(row.get("best_taker_edge"))
    mid_edge = none_low(row.get("best_mid_edge"))
    evidence_ok = row.get("evidence_quality", 0) >= 4
    confidence_ok = row.get("confidence") in {"medium", "high"}
    safe = not row.get("resolution_trade_blocker") and "low_evidence_quality" not in row.get("hold_reasons", [])
    if not (evidence_ok and confidence_ok and safe):
        return "none"
    if taker_edge >= 0.02 or mid_edge >= 0.02:
        return "actionable_candidate"
    if taker_edge >= 0.005 or mid_edge >= 0.01:
        return "watchlist"
    if taker_edge > 0 or mid_edge > 0:
        return "weak"
    return "none"


def blf_strongly_opposes_rag(row: dict) -> bool:
    p_rag = row.get("p_2402_final_after_shrinkage")
    p_blf = row.get("p_blf_final_after_shrinkage")
    if not isinstance(p_rag, (int, float)) or not isinstance(p_blf, (int, float)):
        return False
    return abs(p_blf - p_rag) >= 0.12


def none_low(value: float | None) -> float:
    return value if isinstance(value, (int, float)) else -999.0


def top_unscanned(rows: list[dict], edge_field: str) -> list[dict]:
    return [
        compact_row(row)
        for row in sorted(rows, key=lambda item: none_low(item.get(edge_field)), reverse=True)
        if not row.get("rag_selected")
    ]


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
