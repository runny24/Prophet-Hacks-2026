"""Runtime configuration for the RAG-BLF Edge Trader."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BotConfig:
    strategy_name: str = "rag_blf_edge_trader"
    version: str = "0.1.0"
    slug: str = "rag-blf-edge-trader-v01"
    model_name: str = "custom:rag-blf-edge-trader"
    n_ticks: int = 96
    starting_cash: float = 10_000.0
    dry_run: bool = False

    min_edge: float = 0.06
    low_confidence_min_edge: float = 0.10
    exit_edge: float = 0.02
    max_spread: float = 0.20
    min_volume_24h: float = 0.0
    max_markets_to_consider: int = 80
    max_trades_per_tick_target: int = 3
    max_open_positions_target: int = 25
    max_new_notional_per_trade: float = 175.0
    max_notional_per_market_target: float = 600.0
    reserve_cash: float = 500.0
    deadline_buffer_sec: int = 90

    enable_rag: bool = False
    enable_blf: bool = False
    blf_max_markets_per_tick: int = 1
    blf_max_steps: int = 2
    blf_provider: str = "openrouter"
    blf_model: str = "deepseek/deepseek-chat"
    blf_timeout_seconds: int = 20
    blf_json_retries: int = 1
    blf_enable_extra_search: bool = False
    rag_max_markets_per_tick: int = 12
    rag_max_queries: int = 3
    rag_max_results_per_query: int = 5
    rag_selection_mode: str = "current"
    llm_provider: str = "openrouter"
    enable_llm_rag_summary: bool = False
    llm_model: str = "deepseek/deepseek-chat"
    llm_timeout_seconds: int = 20
    max_llm_evidence_items: int = 5
    llm_json_retries: int = 1
    block_trade_on_high_resolution_risk: bool = True
    official_source_missing_edge_multiplier: float = 2.0
    allow_live_submit: bool = False
    threshold_debug: bool = False
    max_tick_seconds: int = 600
    latency_warning_seconds: int = 450
    rag_early_stop_after_high_quality: bool = False
    rag_target_runtime_seconds: int | None = None
    rag_concurrency: int = 4
    llm_rag_concurrency: int = 3
    blf_concurrency: int = 2
    tick_time_budget_seconds: int = 600
    stop_new_work_before_deadline_seconds: int = 90

    live_ready_taker_edge: float = 0.03
    live_ready_mid_edge: float = 0.02
    live_ready_min_evidence_quality: int = 4
    live_ready_max_spread: float = 0.02
    live_ready_require_blf: bool = True

    enable_price_trading: bool = False
    price_signal_min_edge: float = 0.001
    forecast_edge_min_edge: float = 0.003
    price_max_trade_size: int = 1
    price_max_position_per_market: int = 3
    price_max_total_open_positions: int = 20
    price_max_session_loss: float = 100.0
    price_exit_after_ticks: int = 4
    price_stop_loss_ticks: float = -0.01
    price_take_profit_ticks: float = 0.005
    fresh_event_max_spread: float = 0.015
    live_forecast_max_spread: float = 0.015
    enable_cached_evidence_fallback: bool = True
    max_cached_evidence_age_ticks: int = 8
    enable_candidate_rerank: bool = False
    candidate_rerank_debug: bool = False
    live_guard_mode: bool = True
    unsafe_live_override: bool = False

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            slug=os.getenv("EDGE_TRADER_SLUG", cls.slug),
            model_name=os.getenv("EDGE_TRADER_MODEL", cls.model_name),
            n_ticks=int(os.getenv("EDGE_TRADER_N_TICKS", str(cls.n_ticks))),
            dry_run=os.getenv("EDGE_TRADER_DRY_RUN", "0").lower() in {"1", "true", "yes"},
            enable_rag=os.getenv("EDGE_TRADER_ENABLE_RAG", "0").lower() in {"1", "true", "yes"},
            enable_blf=os.getenv("EDGE_TRADER_ENABLE_BLF", "0").lower() in {"1", "true", "yes"},
            blf_max_markets_per_tick=int(
                os.getenv("EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK", str(cls.blf_max_markets_per_tick))
            ),
            blf_max_steps=int(os.getenv("EDGE_TRADER_BLF_MAX_STEPS", str(cls.blf_max_steps))),
            blf_provider=os.getenv(
                "EDGE_TRADER_BLF_PROVIDER",
                os.getenv("EDGE_TRADER_LLM_PROVIDER", cls.blf_provider),
            ),
            blf_model=os.getenv(
                "EDGE_TRADER_BLF_MODEL",
                os.getenv("EDGE_TRADER_LLM_MODEL", cls.blf_model),
            ),
            blf_timeout_seconds=int(
                os.getenv("EDGE_TRADER_BLF_TIMEOUT_SECONDS", str(cls.blf_timeout_seconds))
            ),
            blf_json_retries=int(os.getenv("EDGE_TRADER_BLF_JSON_RETRIES", str(cls.blf_json_retries))),
            blf_enable_extra_search=os.getenv("EDGE_TRADER_BLF_ENABLE_EXTRA_SEARCH", "0").lower()
            in {"1", "true", "yes"},
            rag_max_markets_per_tick=int(
                os.getenv(
                    "EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK",
                    os.getenv("EDGE_TRADER_RAG_MAX_MARKETS", str(cls.rag_max_markets_per_tick)),
                )
            ),
            rag_max_queries=int(os.getenv("EDGE_TRADER_RAG_MAX_QUERIES", str(cls.rag_max_queries))),
            rag_max_results_per_query=int(
                os.getenv("EDGE_TRADER_RAG_MAX_RESULTS", str(cls.rag_max_results_per_query))
            ),
            rag_selection_mode=os.getenv("EDGE_TRADER_RAG_SELECTION_MODE", cls.rag_selection_mode),
            enable_llm_rag_summary=os.getenv("EDGE_TRADER_ENABLE_LLM_RAG", "0").lower()
            in {"1", "true", "yes"},
            llm_provider=os.getenv("EDGE_TRADER_LLM_PROVIDER", cls.llm_provider),
            llm_model=os.getenv("EDGE_TRADER_LLM_MODEL", cls.llm_model),
            llm_timeout_seconds=int(
                os.getenv("EDGE_TRADER_LLM_TIMEOUT_SECONDS", str(cls.llm_timeout_seconds))
            ),
            max_llm_evidence_items=int(
                os.getenv("EDGE_TRADER_MAX_LLM_EVIDENCE_ITEMS", str(cls.max_llm_evidence_items))
            ),
            llm_json_retries=int(os.getenv("EDGE_TRADER_LLM_JSON_RETRIES", str(cls.llm_json_retries))),
            allow_live_submit=os.getenv("EDGE_TRADER_ALLOW_LIVE_SUBMIT", "0").lower()
            in {"1", "true", "yes"},
            threshold_debug=os.getenv("EDGE_TRADER_THRESHOLD_DEBUG", "0").lower() in {"1", "true", "yes"},
            max_tick_seconds=int(os.getenv("EDGE_TRADER_MAX_TICK_SECONDS", str(cls.max_tick_seconds))),
            latency_warning_seconds=int(
                os.getenv("EDGE_TRADER_LATENCY_WARNING_SECONDS", str(cls.latency_warning_seconds))
            ),
            rag_early_stop_after_high_quality=os.getenv(
                "EDGE_TRADER_RAG_EARLY_STOP_AFTER_HIGH_QUALITY", "0"
            ).lower()
            in {"1", "true", "yes"},
            rag_target_runtime_seconds=optional_int(os.getenv("EDGE_TRADER_RAG_TARGET_RUNTIME_SECONDS")),
            rag_concurrency=int(os.getenv("EDGE_TRADER_RAG_CONCURRENCY", str(cls.rag_concurrency))),
            llm_rag_concurrency=int(
                os.getenv("EDGE_TRADER_LLM_RAG_CONCURRENCY", str(cls.llm_rag_concurrency))
            ),
            blf_concurrency=int(os.getenv("EDGE_TRADER_BLF_CONCURRENCY", str(cls.blf_concurrency))),
            tick_time_budget_seconds=int(
                os.getenv("EDGE_TRADER_TICK_TIME_BUDGET_SECONDS", str(cls.tick_time_budget_seconds))
            ),
            stop_new_work_before_deadline_seconds=int(
                os.getenv(
                    "EDGE_TRADER_STOP_NEW_WORK_BEFORE_DEADLINE_SECONDS",
                    str(cls.stop_new_work_before_deadline_seconds),
                )
            ),
            live_ready_taker_edge=float(
                os.getenv("EDGE_TRADER_LIVE_READY_TAKER_EDGE", str(cls.live_ready_taker_edge))
            ),
            live_ready_mid_edge=float(
                os.getenv("EDGE_TRADER_LIVE_READY_MID_EDGE", str(cls.live_ready_mid_edge))
            ),
            live_ready_min_evidence_quality=int(
                os.getenv(
                    "EDGE_TRADER_LIVE_READY_MIN_EVIDENCE_QUALITY",
                    str(cls.live_ready_min_evidence_quality),
                )
            ),
            live_ready_max_spread=float(
                os.getenv("EDGE_TRADER_LIVE_READY_MAX_SPREAD", str(cls.live_ready_max_spread))
            ),
            live_ready_require_blf=os.getenv("EDGE_TRADER_LIVE_READY_REQUIRE_BLF", "1").lower()
            in {"1", "true", "yes"},
            enable_price_trading=os.getenv("EDGE_TRADER_ENABLE_PRICE_TRADING", "0").lower()
            in {"1", "true", "yes"},
            price_signal_min_edge=float(
                os.getenv("EDGE_TRADER_PRICE_SIGNAL_MIN_EDGE", str(cls.price_signal_min_edge))
            ),
            forecast_edge_min_edge=float(
                os.getenv("EDGE_TRADER_FORECAST_EDGE_MIN_EDGE", str(cls.forecast_edge_min_edge))
            ),
            price_max_trade_size=int(os.getenv("EDGE_TRADER_MAX_TRADE_SIZE", str(cls.price_max_trade_size))),
            price_max_position_per_market=int(
                os.getenv("EDGE_TRADER_MAX_POSITION_PER_MARKET", str(cls.price_max_position_per_market))
            ),
            price_max_total_open_positions=int(
                os.getenv("EDGE_TRADER_MAX_TOTAL_OPEN_POSITIONS", str(cls.price_max_total_open_positions))
            ),
            price_max_session_loss=float(
                os.getenv(
                    "EDGE_TRADER_MAX_SESSION_LOSS",
                    os.getenv("EDGE_TRADER_MAX_DAILY_LOSS", str(cls.price_max_session_loss)),
                )
            ),
            price_exit_after_ticks=int(
                os.getenv("EDGE_TRADER_EXIT_AFTER_TICKS", str(cls.price_exit_after_ticks))
            ),
            price_stop_loss_ticks=float(
                os.getenv("EDGE_TRADER_STOP_LOSS_TICKS", str(cls.price_stop_loss_ticks))
            ),
            price_take_profit_ticks=float(
                os.getenv("EDGE_TRADER_TAKE_PROFIT_TICKS", str(cls.price_take_profit_ticks))
            ),
            fresh_event_max_spread=float(
                os.getenv("EDGE_TRADER_FRESH_EVENT_MAX_SPREAD", str(cls.fresh_event_max_spread))
            ),
            live_forecast_max_spread=float(
                os.getenv("EDGE_TRADER_LIVE_FORECAST_MAX_SPREAD", str(cls.live_forecast_max_spread))
            ),
            enable_cached_evidence_fallback=os.getenv("EDGE_TRADER_ENABLE_CACHED_EVIDENCE_FALLBACK", "1").lower()
            in {"1", "true", "yes"},
            max_cached_evidence_age_ticks=int(
                os.getenv("EDGE_TRADER_MAX_CACHED_EVIDENCE_AGE_TICKS", str(cls.max_cached_evidence_age_ticks))
            ),
            enable_candidate_rerank=os.getenv("EDGE_TRADER_ENABLE_CANDIDATE_RERANK", "0").lower()
            in {"1", "true", "yes"},
            candidate_rerank_debug=os.getenv("EDGE_TRADER_CANDIDATE_RERANK_DEBUG", "0").lower()
            in {"1", "true", "yes"},
            live_guard_mode=os.getenv("EDGE_TRADER_LIVE_GUARD_MODE", "1").lower()
            in {"1", "true", "yes"},
            unsafe_live_override=os.getenv("EDGE_TRADER_UNSAFE_LIVE_OVERRIDE", "0").lower()
            in {"1", "true", "yes"},
        )

    def to_experiment_config(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        pinned = os.getenv("EDGE_TRADER_CONFIG_HASH", "").strip()
        if pinned:
            return pinned
        payload = json.dumps(self.to_experiment_config(), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def load_env_file(path: str | Path = ".env") -> None:
    """Load a simple dotenv file without overriding existing environment."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
