"""Replay archived trading ticks against the current strategy logic.

Loads per-tick JSON archives produced by the bot and re-runs filtering,
forecasting, and sizing without making live API calls. Useful for
iterating on strategy parameters offline.

Usage
-----
# Replay a single tick directory
python replay_trading_history.py logs/trading_history/20260516_223000_tick_20260516T2230/

# Replay all archived ticks
python replay_trading_history.py --all logs/trading_history/

# Replay and override min_edge
python replay_trading_history.py --all logs/trading_history/ --min-edge 0.03

Outputs are written to logs/replay/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make edge_trader_bot importable when run from hackathon/
sys.path.insert(0, str(Path(__file__).parent))

from edge_trader_bot.aggregator import combine_signals
from edge_trader_bot.config import BotConfig
from edge_trader_bot.json_utils import json_safe
from edge_trader_bot.market_filter import deterministic_filter
from edge_trader_bot.schemas import ForecastSignals, MarketView
from edge_trader_bot.stat_priors import initial_signals
from edge_trader_bot.trading_policy import decide_trade, rank_decisions


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


def _float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def market_view_from_dict(d: dict) -> MarketView | None:
    """Reconstruct a MarketView from a candidates.json entry."""
    try:
        # candidates.json can hold either the raw MarketData wire format or a
        # pre-converted MarketView dict; handle both shapes.
        if "yes_bid" in d:
            # Already converted MarketView shape
            yes_bid = _float(d.get("yes_bid"))
            yes_ask = _float(d.get("yes_ask"))
        else:
            # Raw MarketData shape (nested quote)
            quote = d.get("quote") or {}
            yes_bid = _float(quote.get("best_bid"))
            yes_ask = _float(quote.get("best_ask"))

        yes_mid = (yes_bid + yes_ask) / 2.0
        resolution_time_raw = d.get("resolution_time") or d.get("close_time") or ""
        try:
            from datetime import datetime as _dt
            resolution_time = _dt.fromisoformat(str(resolution_time_raw))
        except Exception:
            resolution_time = datetime.now(UTC)

        return MarketView(
            market_id=str(d.get("market_id") or d.get("id") or ""),
            question=str(d.get("question") or ""),
            description=d.get("description"),
            resolution_time=resolution_time,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_mid=yes_mid,
            no_bid=1.0 - yes_ask,
            no_ask=1.0 - yes_bid,
            no_mid=1.0 - yes_mid,
            spread=yes_ask - yes_bid,
            volume_24h=_float((d.get("quote") or {}).get("volume_24h") or d.get("volume_24h")),
            topic=d.get("topic"),
            family=d.get("family"),
            source_url=d.get("source_url"),
        )
    except Exception as exc:
        logger.warning("Could not reconstruct MarketView from dict: %s", exc)
        return None


def signals_from_archive(archived: dict) -> ForecastSignals | None:
    """Restore a ForecastSignals from a strategy_inputs.json signals entry."""
    try:
        return ForecastSignals(
            market_id=archived["market_id"],
            p_market=_float(archived.get("p_market")),
            p_stat=_float(archived.get("p_stat")),
            p_2402=archived.get("p_2402"),
            p_blf=archived.get("p_blf"),
            p_final=archived.get("p_final"),
            confidence=archived.get("confidence", "low"),
            uncertainty=_float(archived.get("uncertainty"), 0.08),
            evidence_quality=int(archived.get("evidence_quality") or 0),
            risk_flags=list(archived.get("risk_flags") or []),
            reason=archived.get("reason") or "",
        )
    except Exception as exc:
        logger.warning("Could not restore ForecastSignals: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Replay logic
# ---------------------------------------------------------------------------


def replay_tick(tick_dir: Path, config: BotConfig) -> dict:
    """Replay one archived tick directory. Returns a result summary dict."""

    def load(name: str) -> Any:
        path = tick_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)
            return None

    metadata = load("metadata") or {}
    candidates_raw = load("candidates") or []
    portfolio_raw = load("portfolio_before")
    strategy_inputs = load("strategy_inputs") or {}
    archived_signals: dict[str, dict] = strategy_inputs.get("signals") or {}

    # Reconstruct MarketView objects from raw candidates
    markets: list[MarketView] = []
    for entry in candidates_raw:
        mv = market_view_from_dict(entry)
        if mv and mv.market_id:
            markets.append(mv)

    # Filter
    selected: list[MarketView] = []
    skipped: list[dict] = []
    for market in markets:
        ok, reasons = deterministic_filter(market, None, config)
        if ok and len(selected) < config.max_markets_to_consider:
            selected.append(market)
        else:
            skipped.append({"market_id": market.market_id, "reasons": reasons or ["limit"]})

    # Forecast: prefer archived BLF/RAG signals; fall back to stat prior
    decisions = []
    replay_signals: dict[str, dict] = {}

    for market in selected:
        if market.market_id in archived_signals:
            signals = signals_from_archive(archived_signals[market.market_id])
            if signals is None:
                signals = initial_signals(market)
                signals.risk_flags.append("replay_restore_failed")
            else:
                signals.risk_flags.append("replay_from_archive")
        else:
            signals = initial_signals(market)
            signals.risk_flags.append("replay_no_archive_signal")

        signals = combine_signals(signals)
        replay_signals[market.market_id] = json_safe(signals)

        decision = decide_trade(market, signals, None, config)
        if decision is not None:
            decisions.append(decision)

    ranked = rank_decisions(decisions, config.max_trades_per_tick_target)

    return {
        "tick_dir": str(tick_dir),
        "tick_id": metadata.get("tick_id"),
        "git_commit_archived": metadata.get("git_commit"),
        "candidate_count": len(markets),
        "processed_count": len(selected),
        "intent_count": len(ranked),
        "intents": [d.to_intent_dict() for d in ranked],
        "decisions": [json_safe(d) for d in ranked],
        "replay_signals": replay_signals,
        "skipped_count": len(skipped),
        "config_min_edge": config.min_edge,
        "replayed_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay archived trading ticks.")
    p.add_argument(
        "path",
        type=Path,
        help="A single tick directory or (with --all) the trading_history base directory.",
    )
    p.add_argument("--all", action="store_true", help="Replay all tick dirs under path/.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/replay"),
        help="Where to write replay results (default: logs/replay/).",
    )
    p.add_argument("--min-edge", type=float, default=None, help="Override min_edge for replay.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = BotConfig()
    if args.min_edge is not None:
        config = BotConfig(**{**config.__dict__, "min_edge": args.min_edge})

    if args.all:
        tick_dirs = sorted(d for d in args.path.iterdir() if d.is_dir())
    else:
        tick_dirs = [args.path]

    if not tick_dirs:
        logger.error("No tick directories found under %s", args.path)
        sys.exit(1)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"replay_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    jsonl_path = out_dir / "per_tick_results.jsonl"

    for tick_dir in tick_dirs:
        logger.info("Replaying %s", tick_dir.name)
        try:
            result = replay_tick(tick_dir, config)
        except Exception as exc:
            result = {"tick_dir": str(tick_dir), "error": str(exc)}
        all_results.append(result)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(json_safe(result)) + "\n")

    summary_path = out_dir / "replay_summary.json"
    summary = {
        "replayed_at": datetime.now(UTC).isoformat(),
        "tick_count": len(tick_dirs),
        "total_intents": sum(r.get("intent_count", 0) for r in all_results),
        "config_min_edge": config.min_edge,
        "output_dir": str(out_dir),
        "ticks": [
            {
                "tick_id": r.get("tick_id"),
                "intent_count": r.get("intent_count", 0),
                "intents": r.get("intents", []),
            }
            for r in all_results
        ],
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    logger.info(
        "Replay complete: %d ticks, %d total intents → %s",
        len(tick_dirs),
        summary["total_intents"],
        out_dir,
    )


if __name__ == "__main__":
    main()
