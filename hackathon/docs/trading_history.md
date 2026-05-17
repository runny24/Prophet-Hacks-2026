# Trading History: Live Archive & Local Replay

## What is this?

Every time the bot processes a tick it writes a directory of JSON snapshots to
`logs/trading_history/`. These files contain all inputs, forecasts, decisions,
and API responses for that tick. Because the Prophet Arena competition runs on
15-minute intervals, having a local archive lets you iterate on strategy logic
offline without waiting for real ticks.

---

## Where data is saved

```
logs/trading_history/
  YYYYMMDD_HHMMSS_tick_<tick_timestamp>/
    metadata.json        ← config (secrets redacted), git commit, tick IDs
    candidates.json      ← raw market list from load_candidates()
    portfolio_before.json← portfolio state before any decisions
    market_snapshots.json← MarketView fields for every selected market
    strategy_inputs.json ← per-market signals (BLF, RAG, stat prior outputs)
    decisions.json       ← ranked TradeDecision objects
    intents.json         ← intent dicts (market_id, action, side, shares)
    dry_run_result.json  ← hypothetical intents when dry_run=True
    submit_result.json   ← submit_intents() response when live
    finalize_result.json ← finalize status
    portfolio_after.json ← not captured (would require extra round-trip)
    errors.json          ← exceptions caught during the tick
    summary.json         ← compact replay-friendly index entry
```

Secrets (env vars whose names contain KEY, TOKEN, SECRET, PASSWORD, OPENAI,
BRAVE, or API) are replaced with `"[REDACTED]"` in `metadata.json`.

---

## How to collect data during live ticks

The archive is enabled by default at `logs/trading_history/`. Run the bot
normally:

```bash
cd hackathon
PA_SERVER_API_KEY=... \
OPENROUTER_API_KEY=... \
EDGE_TRADER_ENABLE_BLF=1 \
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  -m edge_trader_bot --dry-run
```

To disable archiving (faster, no disk I/O):

```bash
python3 -m edge_trader_bot --archive-dir none
```

To use a custom archive location:

```bash
python3 -m edge_trader_bot --archive-dir /tmp/my_archives
```

---

## How to run replay

Replay re-runs the filtering, sizing, and decision logic against archived
inputs. It uses saved BLF/RAG signals by default (no new API calls).

**Replay a single tick:**

```bash
cd hackathon
python replay_trading_history.py logs/trading_history/20260516_223000_tick_20260516T2230/
```

**Replay all archived ticks:**

```bash
python replay_trading_history.py --all logs/trading_history/
```

**Override strategy parameters for sensitivity analysis:**

```bash
python replay_trading_history.py --all logs/trading_history/ --min-edge 0.03
```

Outputs are written to `logs/replay/replay_<timestamp>/`:
- `per_tick_results.jsonl` — one JSON line per tick
- `replay_summary.json` — aggregated counts and intent list

---

## Limitations

- **No exchange simulation**: replay does not simulate order fills or
  update cash/positions between ticks. It re-runs logic independently per tick.
- **BLF/RAG not re-run**: replay uses the archived signal values from
  `strategy_inputs.json`. This keeps replay fast and free (no API calls).
- **portfolio_after not captured**: fetching the portfolio after finalize would
  require an extra round-trip per tick; it is left as `null` in `summary.json`.
- **MarketView reconstruction**: replay reconstructs markets from
  `candidates.json`. If the raw wire format changes, reconstruction may need
  updating in `replay_trading_history.py::market_view_from_dict`.
