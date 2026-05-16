# Building a Trading Bot with `ai-prophet-core`

How to build a custom prediction-market trading bot against the Prophet
Arena benchmark API using only the typed SDK. No LLM, no Prophet Arena
CLI.

## What this is

Prophet Arena is a paper-trading benchmark for prediction markets. The
server picks the tradeable universe, pins prices into 15-minute snapshots,
and runs deterministic fills against them. Your bot is a thin HTTP client;
the server owns all state (experiments, ticks, fills, portfolio, PnL).

You write the strategy. The SDK gives you a lifecycle wrapper
(`BenchmarkSession`) and typed wire models.

## Install

```bash
pip install ai-prophet-core
```

```bash
export PA_SERVER_URL=https://api.aiprophet.dev          # or your local server
export PA_SERVER_API_KEY=<your prophet api key>
```

API keys are issued out of band. Ask the operator if you don't have one.

## The mental model

- **The server runs on 15-minute ticks** anchored to UTC boundaries
  (`:00`, `:15`, `:30`, `:45`). Each tick is a "decision window".
- **Each tick is bound to a snapshot** of curated markets and prices.
  Every participant sees the same snapshot at the same tick. Fills are
  deterministic against the snapshot prices.
- **You claim a tick, submit trade intents, then finalize.** The server
  enforces ordering with a lease. One claim per `(experiment, tick)`.
- **Settlement happens when the market resolves** (server-side cron).
  Your equity is mark-to-market until then.

## The minimum-viable bot

```python
import hashlib
import json
import os
import time

from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession

API = ServerAPIClient(
    base_url=os.environ["PA_SERVER_URL"],
    api_key=os.environ["PA_SERVER_API_KEY"],
    timeout=30,
)

CONFIG = {"strategy": "buy-cheap-side", "version": "1.0"}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]


def strategy(market) -> tuple[str, str] | None:
    """Return (action, side) or None to HOLD."""
    ask = float(market.quote.best_ask)
    if ask < 0.4:
        return ("BUY", "YES")
    if ask > 0.6:
        return ("BUY", "NO")
    return None


def run() -> None:
    with BenchmarkSession(API) as session:
        session.create_experiment(
            slug="my-bot-v1",
            config_hash=CONFIG_HASH,
            config_json=CONFIG,
            n_ticks=96,
        )
        part = session.upsert_participant(model="custom:my-bot", starting_cash=10_000)

        while True:
            lease = session.claim_tick()
            if not lease.available:
                if lease.reason == "experiment_completed":
                    break
                time.sleep(lease.retry_after_sec or 15)
                continue

            tick = session.load_candidates(lease)
            lease = tick.lease

            intents = []
            for m in tick.candidates.markets:
                decision = strategy(m)
                if decision is None:
                    continue
                action, side = decision
                intents.append(TradeIntentRequest(
                    market_id=m.market_id,
                    action=action,
                    side=side,
                    shares="10",
                    idempotency_key="",   # SDK fills this in
                ))

            session.put_plan(lease, part.participant_idx, {})  # optional
            if intents:
                session.submit_intents(lease, part.participant_idx, intents)
            session.finalize(lease, part.participant_idx)
            session.complete_tick(lease)


if __name__ == "__main__":
    run()
```

Run it as a long-lived process. It blocks on `claim_tick` between ticks
and wakes up every 15 minutes.

## The tick lifecycle

Each tick traverses these calls in order:

| Call | Purpose | Required? |
|---|---|---|
| `session.claim_tick()` | Reserve the next tick. Returns a `TickLease` with `tick_id` and `candidate_set_id`. | **Yes** |
| `session.load_candidates(lease)` | Fetch the market universe and quotes for this tick. | **Yes** |
| `session.get_portfolio(idx)` | Read current cash, equity, positions. | Optional but recommended |
| `session.put_plan(lease, idx, json)` | Persist arbitrary audit JSON. Server doesn't read it. | Optional |
| `session.submit_intents(lease, idx, intents)` | Submit trades. Server executes deterministically against pinned prices. | **Yes** if you want to trade |
| `session.finalize(lease, idx)` | Mark your participant tick as `COMPLETED` (or `FAILED`). | **Yes** |
| `session.complete_tick(lease)` | Advance the experiment after all participants are terminal. | **Yes** |

The server does **not** require forecasts, search results, reasoning
prose, or any LLM output. Those are what the CLI's `prophet trade eval
run` produces; custom bots can skip them entirely.

### Trade intent shape

```python
TradeIntentRequest(
    market_id="kalshi:KXNFLGAME-25NOV23DAL-DAL",   # from candidates
    action="BUY",                                  # BUY or SELL
    side="YES",                                    # YES or NO
    shares="10",                                   # string-encoded decimal
    idempotency_key="",                            # SDK auto-generates
)
```

The SDK fills `idempotency_key` using
`{experiment_id}:{participant_idx}:{tick_id}:{intent_index}`. Reusing
the same key returns a cached fill, so retries are safe.

## Trading rules

Every participant plays by these constants. They're enforced server-side
and importable from `ai_prophet_core.ruleset`, which is the authoritative
source for the version you have installed.

| Constant | Value | Meaning |
|---|---|---|
| `TICK_INTERVAL_SECONDS` | 900 | 15 min between ticks |
| `TICK_SUBMISSION_DEADLINE_SECS` | 540 | Server rejects submits past `tick_ts + 9 min` (HTTP 409) |
| `INITIAL_CASH` | $10,000 | Starting bankroll per participant |
| `MAX_TRADES_PER_TICK` | 20 | Fills above this in a single tick get rejected |
| `MAX_TRADES_PER_DAY` | 100 | Fills above this in a 24h rolling window get rejected |
| `MAX_OPEN_POSITIONS` | 30 | Distinct `(market_id, side)` positions you can hold |
| `MAX_NOTIONAL_PER_MARKET` | $1,000 | Exposure cap per market |
| `MAX_GROSS_EXPOSURE` | $10,000 | Total exposure cap |
| `MAX_INTENTS_PER_TICK_REQUEST` | 50 | HTTP request shape cap; only the first 20 fill |
| `FEE_RATE` | 0.0 | No trading fees |

## Execution semantics

- **BUY YES** fills at `best_ask`. **BUY NO** fills at `1 - best_bid`.
  Prices on the two sides sum to 1, the standard prediction-market
  convention.
- **Positions are keyed by `(market_id, side)`.** Stacking 10 shares of
  YES across multiple ticks accumulates into one position with a
  blended `avg_entry_price`.
- **You cannot hold both YES and NO on the same market.** Submitting a
  BUY on the opposite side of a held position does **not** open a second
  position and is **not** rejected — the server treats it as a reduction
  of the existing position. Shares get netted off until the holding
  reaches zero; any remaining shares from your intent are dropped (they
  do not flip you to the other side). To flip, first SELL down to flat,
  then BUY the new side in a follow-up intent.
- **Unrealized PnL** is mark-to-market at the snapshot's mid-price for
  open positions. **Realized PnL** comes from market resolution: payouts
  are $1 per share on the winning side, $0 on the losing side.

## Common pitfalls

### Slug uniqueness

`(owner_subject, slug)` is unique. Two processes against the same slug
will fight over the tick lease and both lose. **One bot, one unique
slug.** To resume a crashed bot, restart with the same slug and the
same `config_hash`; the server returns the existing experiment.

### Submission deadline

Submit within 9 minutes of `tick_ts`. After that, the server returns
HTTP 409 with `late_by_sec` in the detail. The lease expires shortly
after.

### Future-tick guard

You can't claim a tick more than one interval in the future. After
completing tick `T`, you can claim `T + 15min` immediately, but
`T + 30min` is blocked until the wall clock advances. The server
returns `no_tick_available` with `retry_after_sec`.

### Candidate set ID

The `candidate_set_id` in a trade intent submission must match the
snapshot bound to the tick. `BenchmarkSession.submit_intents` threads
this through automatically. If you're calling `ServerAPIClient`
directly, read it from `claim_tick`'s response and pass it explicitly.

### Markets that "vanish"

Kalshi markets can drop out of the eligible universe between ticks.
Open positions on a dropped market are excluded from equity until the
market resolves (or returns). Plan accordingly.

## Live dashboard

Once your bot is running, launch the dashboard:

```bash
prophet trade dashboard --slug my-bot-v1
```

This starts a local web server (port 8501 by default), opens a browser,
and renders:

- Status, equity, P&L, Sharpe, max drawdown, CAGR, win rate (gated on
  observation count)
- Equity-over-tick chart per participant
- Per-participant leaderboard
- Market ledger (searchable, filter by open/flat/won/lost)
- Reasoning (lazy-loaded from `/experiments/{id}/reasoning`)

The dashboard proxies all API calls through the local server, so your
API key stays in the Python process. It polls every 10 seconds.

Metrics that need history (Sharpe, CAGR, max drawdown) show
**Pending** with an explanation until at least 3 daily P&L observations
have accumulated. Win rate stays pending until 10 trades resolve.

## Beyond the minimum

- **Plan JSON for audit.** Pass any dict to `put_plan`: trading
  rationale, model outputs, debug info. The server stores it as JSON
  and exposes it via `/experiments/{id}/reasoning` for the dashboard.
- **Multiple participants per experiment.** Call `upsert_participant`
  with distinct `(model, rep)` pairs to run strategy variants in
  parallel under one experiment.
- **Network resilience.** The SDK already retries transient errors and
  honours `Retry-After`. Wrap `claim_tick` in your own backoff loop for
  long network blackouts. See `prophet-agent/agent.py` for the pattern.
- **Cross-process idempotency.** Pass `idempotency_key_fn` to
  `submit_intents` if multiple workers drive the same participant.

## When you need something more

- **The LLM pipeline (review, search, forecast, action):** use the
  `ai-prophet` CLI. `prophet trade eval run --models openai:gpt-4o`
  gives you the full 4-stage pipeline with zero custom code.
- **Plug a custom strategy into the CLI's runner** (to inherit its
  tracing, dashboard, local memory): pass `build_pipeline` to
  `ExperimentRunner`. See `ai-prophet/CORE_AUDIT.md` §5.

## API reference

From `ai_prophet_core`:

- `ServerAPIClient`: typed HTTP client. Use directly for raw control or
  via `BenchmarkSession` for the tick lifecycle.
- `BenchmarkSession` (`from ai_prophet_core.arena`): lifecycle wrapper
  that holds the lease and threads `candidate_set_id` and
  `idempotency_key` through automatically.
- Wire models: `TradeIntentRequest`, `TickLease`, `SubmissionResult`,
  `PortfolioResponse`, `MarketSnapshot`.
- Enums: `TradeAction`, `TradeSide`, `SizeType`.

For raw server endpoints, see `ai-prophet/CORE_AUDIT.md` §3.

## A complete worked example

`prophet-agent/agent.py` is the recommended starting point. It uses
Claude for analysis, but the structure (network-resilience loop,
position-aware filtering, structured logging) is strategy-agnostic.
Strip the Anthropic client and drop in your own decision function.
