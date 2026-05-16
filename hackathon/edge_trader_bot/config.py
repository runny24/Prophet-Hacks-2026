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
    rag_max_markets_per_tick: int = 12
    rag_max_queries: int = 3
    rag_max_results_per_query: int = 5
    llm_provider: str = "openrouter"
    enable_llm_rag_summary: bool = False
    llm_model: str = "deepseek/deepseek-chat"
    llm_timeout_seconds: int = 20
    max_llm_evidence_items: int = 5
    llm_json_retries: int = 1
    block_trade_on_high_resolution_risk: bool = True
    official_source_missing_edge_multiplier: float = 2.0
    allow_live_submit: bool = False

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            slug=os.getenv("EDGE_TRADER_SLUG", cls.slug),
            model_name=os.getenv("EDGE_TRADER_MODEL", cls.model_name),
            n_ticks=int(os.getenv("EDGE_TRADER_N_TICKS", str(cls.n_ticks))),
            dry_run=os.getenv("EDGE_TRADER_DRY_RUN", "0").lower() in {"1", "true", "yes"},
            enable_rag=os.getenv("EDGE_TRADER_ENABLE_RAG", "0").lower() in {"1", "true", "yes"},
            enable_blf=os.getenv("EDGE_TRADER_ENABLE_BLF", "0").lower() in {"1", "true", "yes"},
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
        )

    def to_experiment_config(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
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
