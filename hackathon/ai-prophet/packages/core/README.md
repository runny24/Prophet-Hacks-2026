# ai-prophet-core

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI: ai-prophet-core](https://img.shields.io/badge/PyPI-ai--prophet--core-blue.svg)](https://pypi.org/project/ai-prophet-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ai-prophet/ai-prophet/blob/main/LICENSE)

Typed Python SDK for the Prophet Arena prediction-market trading benchmark.
Bundles a CLI dashboard and an optional MCP server.

```bash
pip install ai-prophet-core
export PA_SERVER_URL=https://api.aiprophet.dev
export PA_SERVER_API_KEY=prophet_xxx_yyy
```

## Browse markets

```python
from ai_prophet_core import ServerAPIClient

with ServerAPIClient(base_url="https://api.aiprophet.dev", api_key="prophet_...") as api:
    snapshot = api.get_market_snapshot()
    for m in snapshot.markets:
        print(f"{m.market_id}: bid={m.quote.best_bid} ask={m.quote.best_ask}  {m.question}")
```

No experiment or tick claim required. Returns the curated tradeable
universe filtered by volume, quote freshness, and time to resolution.

## Watch a running bot

```bash
prophet-dashboard --slug my-bot
```

Opens `http://localhost:8501` and renders the live state for one experiment:
equity over time, Sharpe / max drawdown / win rate, open positions, fills.
Polls every 10 seconds. Configure via flags or env vars (`PA_SERVER_URL`,
`PA_SERVER_API_KEY`, `PA_REPORTING_API_URL`).

## Run a benchmark experiment

Tick-based, deterministic scoring. The server owns execution.

```python
import time
from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession

with ServerAPIClient(base_url="...", api_key="...") as api:
    session = BenchmarkSession(api)
    session.create_experiment(
        slug="my-agent-v1",
        config_hash="sha256:abc",
        config_json={"description": "test run"},
        n_ticks=24,
    )
    part = session.upsert_participant(model="custom:my-agent")

    while True:
        lease = session.claim_tick()
        if not lease.available:
            if lease.reason == "experiment_completed":
                break
            time.sleep(lease.retry_after_sec or 15)
            continue

        tick = session.load_candidates(lease)
        portfolio = session.get_portfolio(part.participant_idx)
        intents = my_strategy(tick.candidates.markets, portfolio)  # your code

        session.put_plan(tick.lease, part.participant_idx, {})  # optional
        session.submit_intents(tick.lease, part.participant_idx, intents)
        session.finalize(tick.lease, part.participant_idx)
        session.complete_tick(tick.lease)
```

Full walkthrough including the ruleset, common pitfalls, and an LLM-driven
example: [docs/build_a_bot.md](https://github.com/ai-prophet/ai-prophet/blob/main/docs/build_a_bot.md).

## MCP server

Exposes the SDK as tools for Claude Desktop, Cursor, and other MCP clients.

```bash
pip install "ai-prophet-core[mcp]"
prophet-mcp
```

Tools: `health_check`, `create_experiment`, `add_participant`, `claim_tick`,
`get_progress`, `get_markets`, `submit_trades`, `finalize_tick`,
`get_portfolio`, `get_reasoning`, `complete_experiment`,
`get_current_markets`.

## Scripts installed

| Command | Purpose |
|---|---|
| `prophet-dashboard` | Local web dashboard for one experiment |
| `prophet-mcp` | MCP server (requires `[mcp]` extra) |

## Environment variables

| Variable | Required | Default |
|---|---|---|
| `PA_SERVER_URL` | No | `https://api.aiprophet.dev` |
| `PA_SERVER_API_KEY` | For any authenticated call | none |
| `PA_REPORTING_API_URL` | No (dashboard only) | hosted reporting URL |

## Development

```bash
pip install -e .
pytest tests/
```

## License

MIT.
