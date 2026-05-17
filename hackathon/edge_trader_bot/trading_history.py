"""Archive trading tick data for local replay and backtesting.

Each tick run writes a directory under logs/trading_history/ (or a custom
base dir) containing JSON snapshots of every input, signal, decision, and
API response so that strategy logic can be replayed offline.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REDACT_KEYWORDS = {"KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI", "BRAVE", "API"}
_REDACTED = "[REDACTED]"

DEFAULT_ARCHIVE_DIR = Path("logs/trading_history")


def _should_redact(name: str) -> bool:
    upper = name.upper()
    return any(kw in upper for kw in _REDACT_KEYWORDS)


def redact_config(obj: Any) -> Any:
    """Recursively redact secrets from a config dict."""
    if isinstance(obj, dict):
        return {k: (_REDACTED if _should_redact(k) else redact_config(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_config(v) for v in obj]
    return obj


def safe_json_dump(path: Path, obj: Any) -> None:
    """Write obj to path as JSON, falling back to a string representation on error."""
    from .json_utils import json_safe

    try:
        path.write_text(json.dumps(json_safe(obj), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to serialize %s: %s", path.name, exc)
        try:
            path.write_text(
                json.dumps({"archive_error": str(exc), "repr": repr(obj)[:2000]}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def make_tick_run_dir(base_dir: Path, tick_id: str) -> Path:
    """Create and return a per-tick archive directory."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_id = (
        str(tick_id or "unknown")
        .replace(":", "")
        .replace("+", "")
        .replace("-", "")
        .replace(" ", "")[:16]
    )
    run_dir = base_dir / f"{ts}_tick_{safe_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def archive_tick_artifact(run_dir: Path | None, name: str, obj: Any) -> None:
    """Write a single artifact JSON file into run_dir. No-op if run_dir is None."""
    if run_dir is None:
        return
    safe_json_dump(run_dir / f"{name}.json", obj)


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_summary(run_dir: Path | None, summary: dict) -> None:
    if run_dir is None:
        return
    safe_json_dump(run_dir / "summary.json", summary)


def build_metadata(
    tick_id: str,
    candidate_set_id: str | None,
    participant_idx: int,
    experiment_id: str | None,
    config_dict: dict,
) -> dict:
    """Build the metadata.json artifact."""
    return {
        "tick_id": tick_id,
        "candidate_set_id": candidate_set_id,
        "experiment_id": experiment_id,
        "participant_idx": participant_idx,
        "git_commit": get_git_commit(),
        "config": redact_config(config_dict),
        "created_at": datetime.now(UTC).isoformat(),
        "env_blf_model": os.getenv("EDGE_TRADER_BLF_MODEL", "perplexity/sonar"),
        "env_llm_model": os.getenv("EDGE_TRADER_LLM_MODEL", ""),
        "env_blf_enabled": os.getenv("EDGE_TRADER_ENABLE_BLF", "0"),
        "env_rag_enabled": os.getenv("EDGE_TRADER_ENABLE_RAG", "0"),
    }


def build_summary(
    tick_id: str | None,
    candidate_count: int,
    processed_count: int,
    intent_count: int,
    submitted: bool,
    dry_run: bool,
    errors: list,
    markets_touched: list[str],
    pnl_before: float | None,
    elapsed_seconds: float,
) -> dict:
    """Build the compact summary.json artifact for replay-friendly indexing."""
    return {
        "tick": tick_id,
        "candidate_count": candidate_count,
        "processed_count": processed_count,
        "intent_count": intent_count,
        "submitted": submitted,
        "dry_run": dry_run,
        "errors": errors,
        "markets_touched": markets_touched,
        "pnl_before": pnl_before,
        "pnl_after": None,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "created_at": datetime.now(UTC).isoformat(),
    }
