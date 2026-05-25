# RAG-BLF Price-Aware Edge Trader Strategy

## Goal

Build a Prophet Arena trading-track bot that combines strong forecasting with a real price-aware trading policy.

The important strategy update is:

```text
Forecasting is only one input.
Trading performance is driven by mark-to-market price movement, spread, position management, and exit timing.
```

The current RAG/BLF forecasting stack is useful, but it is now too conservative if used only as a final-event probability estimator. In Prophet Arena, open positions are marked to market at future snapshot mid-prices. Therefore the bot should optimize expected trading return, not just Brier-score-style event accuracy.

The bot should estimate two different quantities:

1. **Outcome fair value**: the long-run fair probability that the market resolves YES.
2. **Trading value**: the expected value of entering, holding, reducing, or exiting a position over the next few ticks.

Core idea:

```text
2402/RAG = wide evidence scanner and rough fair-value estimator
2604/BLF = selective edge verifier and uncertainty reducer
statistical layer = market prior + price history + category/base-rate priors
price layer = short-horizon next-mid / mark-to-market predictor
trading layer = BUY / SELL / HOLD / reduce / add decisions
risk layer = exposure, trade budget, correlation, deadline, and stale-position control
```

---

## Why the strategy changed

The previous version asked mostly:

```text
Is p_final far enough from the executable price to justify a trade?
```

That produced almost no trades because the model was heavily shrunk toward market price, and real executable taker edges were usually below 1%. This is not necessarily a forecasting failure. It means the bot was solving the wrong trading problem.

The better question is:

```text
If I enter or hold this position now, will the future snapshot mid-price move enough in my direction to beat spread, risk, and opportunity cost?
```

Example:

```text
Current market mid: 18.0%
Model fair probability: 18.4%
Final-event edge: only 0.4%, too small for a pure outcome trade

But if the market is drifting up and likely to mark at 18.8% next tick, then a short-horizon trade can still be profitable.
```

So the bot should no longer treat `p_final - ask` as the only edge. It should estimate short-horizon mark-to-market edge as a separate signal.

---

## Objective function

The trading objective should be expected portfolio return under the arena's mark-to-market system.

For a candidate BUY YES:

```text
entry_price = best_ask
predicted_exit_mid = predicted YES mid after N ticks
short_horizon_edge = predicted_exit_mid - entry_price
outcome_edge = p_final_yes - entry_price
trading_score = weighted blend of short_horizon_edge and outcome_edge
```

For a candidate BUY NO:

```text
entry_price = 1 - best_bid
predicted_exit_mid_no = predicted NO mid after N ticks
short_horizon_edge = predicted_exit_mid_no - entry_price
outcome_edge = (1 - p_final_yes) - entry_price
trading_score = weighted blend of short_horizon_edge and outcome_edge
```

For an existing YES position:

```text
exit_price = best_bid
hold_value = predicted_next_yes_mid - exit_price
outcome_hold_edge = p_final_yes - exit_price
sell if hold_value is weak, thesis broke, or capital/trade budget is better used elsewhere
```

For an existing NO position:

```text
exit_price = 1 - best_ask
hold_value = predicted_next_no_mid - exit_price
outcome_hold_edge = (1 - p_final_yes) - exit_price
sell if hold_value is weak, thesis broke, or capital/trade budget is better used elsewhere
```

---

## Architecture

Every 15-minute tick:

```text
1. claim_tick()
2. load_candidates()
3. get_portfolio()
4. update cross-tick market memory
5. compute quote and price-history features
6. update existing-position state
7. deterministic filtering and risk checks
8. run cheap price-action scanner on all processed markets
9. run RAG only on markets where price/evidence/catalyst signals justify it
10. run BLF only on top verified candidates or risky existing positions
11. combine outcome fair value and short-horizon price prediction
12. make BUY / SELL / HOLD / ADD / REDUCE decisions
13. sort intents by expected trading EV and risk
14. put_plan() with audit JSON
15. submit_intents() only if live-submit is explicitly enabled
16. finalize()
17. complete_tick()
```

The default should still be `HOLD`, but not because all outcome edges are too small. It should hold only when both short-horizon and outcome edges are weak after spread and risk adjustment.

---

## Prophet Arena constraints and implications

Key constraints:

- Ticks occur every 15 minutes.
- Submission deadline is 9 minutes after tick timestamp.
- Starting cash is `$10,000`.
- Max trades per tick: `20`.
- Max trades per rolling day: `100`.
- Max open positions: `30`.
- Max notional per market: `$1,000`.
- Max gross exposure: `$10,000`.
- Max intents per tick request: `50`, but only first `20` fills can execute.
- Fee rate appears to be `0.0` in the public docs, but spread still matters.
- `BUY YES` fills at best ask.
- `BUY NO` fills at `1 - best bid`.
- `SELL YES` exits at best bid.
- `SELL NO` exits at `1 - best ask`.
- Positions are keyed by `(market_id, side)`.
- Buying the opposite side reduces an existing position rather than cleanly flipping it.

Implications:

```text
1. Spread is the main execution cost.
2. A small true-probability edge can be untradeable as a taker trade.
3. A small short-horizon price edge may still be useful if the spread is tight.
4. SELL logic is mandatory, not optional.
5. Trade budget is scarce; each fill must compete with future opportunities.
6. Existing positions need active management because they consume open-position and exposure budget.
```

---

## Signal stack

For each market, maintain these values:

```text
p_market       = current YES midpoint
p_stat         = cheap statistical prior / category prior / base rate
p_2402         = RAG rough fair probability
p_blf          = BLF verified fair probability
p_final        = calibrated/shrunk long-run fair YES probability
p_next_mid     = predicted YES midpoint over the next 1-4 ticks
p_exit_mid     = predicted YES midpoint at intended exit horizon
trend_score    = short-horizon price momentum / mean-reversion score
catalyst_score = probability of near-term information update
risk_score     = uncertainty, spread, stale evidence, resolution ambiguity, correlation
trade_score    = final decision score after execution and risk adjustment
```

The major change is adding `p_next_mid` / `p_exit_mid` as first-class objects.

---

## Cross-tick memory

The bot must remember markets across ticks. A pure one-tick forecaster cannot learn price movement.

Cache by `market_id`:

```json
{
  "market_id": "...",
  "last_seen_tick": "...",
  "quote_history": [
    {
      "tick_ts": "...",
      "best_bid": 0.17,
      "best_ask": 0.181,
      "mid": 0.1755,
      "spread": 0.011
    }
  ],
  "last_p_stat": 0.18,
  "last_p_2402": 0.24,
  "last_p_blf": 0.19,
  "last_p_final": 0.185,
  "last_p_next_mid": 0.188,
  "last_trade_score": 0.006,
  "evidence_package": {},
  "belief_state": {},
  "open_questions": [],
  "last_search_time": "...",
  "position_state": {},
  "decision_history": []
}
```

Minimum features from memory:

```text
mid_change_1tick
mid_change_2tick
mid_change_4tick
spread_change_1tick
repeated_market_count
p_final_change_1tick
rag_disagreement_change
realized_mark_to_market_since_last_seen
```

If memory is not yet rich enough, the bot should still run but mark price forecasts as low confidence.

---

## Price-movement layer

The new price layer predicts the next midpoint movement. It should start simple and deterministic, then become learned/calibrated later.

### Inputs

```text
current_mid
current_spread
past mids and spreads
price momentum over 1/2/4 ticks
whether market repeats across ticks
whether RAG/BLF forecast changed materially
whether new evidence was found
whether evidence is stale or fresh
whether market has near-term catalyst
market category
sportsbook/market-odds support type
existing position and unrealized PnL
```

### Initial deterministic model

Start with a conservative heuristic model:

```text
p_next_mid = current_mid
           + momentum_weight * clipped_recent_mid_change
           + forecast_gap_weight * clipped(p_final - current_mid)
           + catalyst_weight * catalyst_direction
           - stale_or_uncertainty_penalty
```

Example defaults:

```text
momentum_weight = 0.25 to 0.50 when movement is confirmed by evidence
forecast_gap_weight = 0.05 to 0.20 because model fair value is noisy
max_predicted_move_without_fresh_evidence = 0.005
max_predicted_move_with_fresh_evidence = 0.015
```

Important rule:

```text
Do not let a large raw RAG probability gap directly create a large short-horizon price forecast.
Most of that gap should be treated as model uncertainty unless price or fresh evidence confirms it.
```

### Momentum vs mean reversion

Use simple rules first:

```text
If price moved and fresh supporting evidence exists:
    weak trend-following signal.

If price moved sharply but evidence is stale/absent and spread widened:
    possible overreaction, but do not blindly fade unless confidence is high.

If price is flat and forecast edge is tiny:
    no trade.
```

---

## Forecasting layer status

The existing forecasting layer is good enough for the next phase.

Keep:

- 2402-style RAG scanner.
- 2604-style BLF verifier.
- Market-aware shrinkage.
- Evidence-quality scoring.
- Resolution-risk checking.
- Sports overconfidence guardrails.
- Absence-of-evidence guardrails for long-horizon announcement/candidacy markets.

Do not spend the next iteration mainly improving RAG prompts. The bottleneck is now trading logic.

Forecasting still matters in three cases:

```text
1. Initial fair-value anchor.
2. Detecting large headline/resolution mismatch.
3. Updating price-movement prediction when new evidence appears.
```

---

## RAG and BLF selection after the strategy change

RAG/BLF should no longer be selected only by apparent final-event edge.

Rank markets for RAG by:

```text
price_move_since_last_tick
spread tightness
repeated-market stability
forecast-market disagreement
possible catalyst / fresh news likelihood
existing position risk
manual category priority
resolution ambiguity opportunity
```

Rank markets for BLF by:

```text
candidate trade_score before BLF
existing position exit importance
large RAG/market disagreement
large recent price move with unclear cause
high evidence quality but high uncertainty
resolution-risk opportunity
```

BLF should also be used to verify exits, not just entries.

---

## Trading decision policy

For each market, compute both taker and passive-style diagnostic edges.

### Entry candidates

For BUY YES:

```text
taker_entry = best_ask
outcome_edge_yes = p_final - taker_entry
price_edge_yes = p_next_mid - taker_entry
trade_score_yes = w_price * price_edge_yes + w_outcome * outcome_edge_yes - risk_penalty
```

For BUY NO:

```text
taker_entry = 1 - best_bid
p_next_mid_no = 1 - p_next_mid
event_prob_no = 1 - p_final
outcome_edge_no = event_prob_no - taker_entry
price_edge_no = p_next_mid_no - taker_entry
trade_score_no = w_price * price_edge_no + w_outcome * outcome_edge_no - risk_penalty
```

Suggested initial weights:

```text
w_price = 0.65
w_outcome = 0.35
```

Use more outcome weight near resolution and more price weight for long-horizon markets.

### Exits

For existing YES:

```text
exit_price = best_bid
hold_price_edge = p_next_mid - exit_price
hold_outcome_edge = p_final - exit_price
sell_score = opportunity_cost + thesis_break_penalty - hold_price_edge - hold_outcome_edge
```

For existing NO:

```text
exit_price = 1 - best_ask
hold_price_edge = (1 - p_next_mid) - exit_price
hold_outcome_edge = (1 - p_final) - exit_price
sell_score = opportunity_cost + thesis_break_penalty - hold_price_edge - hold_outcome_edge
```

Sell when:

```text
1. thesis is broken,
2. predicted next-mid edge is negative after spread,
3. the position is stale and no longer has evidence support,
4. exposure or open-position budget is needed for better trades,
5. the bot accidentally accumulated exposure in a low-quality market,
6. risk flags changed materially.
```

---

## Thresholds and sizing

The old threshold of 3-6% is reasonable for pure final-event taker edge, but too high for short-horizon mark-to-market trading.

Use two thresholds:

```text
outcome_trade_threshold = 0.03 to 0.06
price_trade_threshold = 0.003 to 0.010
```

A small price edge is only acceptable when:

```text
spread is tight,
evidence quality is medium/high,
price prediction has support,
position size is tiny,
and trade budget is not scarce.
```

Starting policy:

| Scenario | Required edge | Size |
|---|---:|---:|
| Pure outcome edge | `>= 0.03` | small/medium |
| Price-movement edge, high confidence, tight spread | `>= 0.005` | tiny |
| Price-movement edge, medium confidence | `>= 0.008` | tiny |
| Weak evidence or stale evidence | no entry | 0 |
| Existing position thesis broke | exit even if edge small | reduce/sell |

Initial sizing:

```text
micro trade: $10-$25 notional
small trade: $25-$75 notional
normal trade: $75-$150 notional
large trade: disabled until backtested
```

Do not chase small edges with large notional. The first goal is to verify that short-horizon price edges produce positive mark-to-market behavior.

---

## Passive/maker diagnostics

The arena docs describe taker-style fills from snapshot quotes. The bot should not assume real passive orders unless the server actually supports them.

However, passive/midpoint diagnostics are still useful:

```text
positive midpoint edge but negative taker edge = market may be efficient but spread blocks entry
positive maker edge but negative taker edge = potential opportunity if future system supports passive execution
```

For current implementation, do not count midpoint/maker edge as executable unless the API supports it. Use it for ranking, diagnostics, and manual review only.

---

## Position and portfolio risk

Hard rules:

```text
Never rely on BUY opposite side to flip a position.
Explicitly SELL existing exposure before opening the opposite side.
Keep max open positions below 30, target 20-25.
Keep category/theme exposure capped.
Do not stack many positions on the same team/event/theme.
Always sort intents by expected trading EV.
Reserve daily trade budget.
```

Suggested caps:

```text
max_new_notional_per_trade = $25 to $75 during price-layer testing
max_notional_per_market_normal = $250
max_notional_per_market_high_confidence = $500
max_category_notional = $1,000 to $1,500
max_trades_per_tick_default = 0 to 3
max_trades_per_tick_burst = 5 only after positive diagnostics
minimum_daily_trade_reserve = 20
```

---

## Diagnostics and evaluation

The next diagnostics should answer:

```text
1. Did we predict next-mid movement correctly?
2. Would small price-edge trades have made money mark-to-market after 1/2/4 ticks?
3. Are RAG/BLF probabilities useful for price movement, or only for fair value?
4. Which categories have actual tradable price motion?
5. Are edges being killed by spread, over-shrinkage, or no price movement?
6. Do SELL/exit rules improve capital efficiency?
```

Add to plan JSON and aggregate diagnostics:

```text
p_next_mid
predicted_mid_move_1tick
predicted_mid_move_2tick
predicted_mid_move_4tick
actual_next_mid_move when available
price_prediction_error
would_trade_price_edge
would_trade_outcome_edge
entry_reason: price_momentum / catalyst / fair_value / exit / risk_reduce
exit_reason
unrealized_pnl_since_entry
mark_to_market_pnl_by_position
```

The four-tick diagnostic should produce a table of hypothetical price-layer trades and their realized next-tick or later mark-to-market PnL when available.

---

## Implementation roadmap

### v1: infrastructure and conservative forecasting

Status: mostly done.

- Tick lifecycle works.
- Candidate loading works.
- Portfolio loading works.
- RAG scanner works with fallback.
- LLM RAG works with rate-limit fallback.
- BLF verifier works with fallback.
- Dry-run diagnostics work.
- Threshold-debug works.
- No unsafe submit detected.

### v2: price-memory layer

Implement next.

- Store quote history per market across ticks.
- Compute mid/spread changes over 1/2/4 ticks.
- Track repeated markets.
- Track previous p_final and RAG/BLF outputs.
- Add actual next-mid evaluation when a market repeats.
- Add diagnostics for predicted vs realized price movement.

### v3: short-horizon price predictor

Implement after memory.

- Add deterministic `price_predictor.py`.
- Predict `p_next_mid` and `expected_mid_move`.
- Use momentum + forecast gap + catalyst/freshness + spread + uncertainty.
- Clamp predicted moves aggressively.
- Add tests for no-history, tight-spread, stale-evidence, fresh-evidence, and large-move cases.

### v4: price-aware trading policy

First guarded-live slice implemented; full predictor still pending.

- Compute outcome edge and price edge separately.
- Use low threshold only for price edge with tight spread and strong support.
- Add micro sizing.
- Add entry reason and risk-adjusted score.
- Keep outcome-only entries conservative.
- Do not allow forecast-edge-only entries to trigger live trades.
- Guarded live now has two entry channels.
- Channel 1: price action.
  - Requires repeated quote history.
  - momentum with repeated history, tight spread, and positive spread-adjusted expected edge,
  - mean reversion only with deeper repeated history, sharp move, and very tight spread.
- Channel 2: fresh-event tiny entry.
  - Does not require previous midpoint or repeated quote history.
  - Requires clean strong evidence/catalyst and no fallback/rate-limit/stale/invalid JSON/timeout flags.
  - Requires `evidence_quality >= 4`.
  - Requires high confidence with executable taker edge at least `0.015`, or medium confidence with edge at least `0.025`.
  - Requires tight spread, default hard cap `0.015`.
  - Blocks long-horizon sports unless explicit near-term quantitative catalyst support exists.
  - Forces size to 1 share in guard mode.
- Block live entries from rate-limit/fallback/stale/invalid-JSON/timeout paths.
- Treat RAG/BLF forecast signals as sanity checks and diagnostics, not direct live-entry authority.
- Guarded live caps:
  - 1 share per order,
  - 3 shares per market,
  - 12 open shares target,
  - 3 new entries per tick,
  - 5 exits per tick.

### v4.1: live-freeze behavior

Implemented after first live attempt showed an over-broad freeze.

- Freeze is tick-local, not permanently sticky.
- Ordinary LLM/RAG/BLF fallback or rate-limit risk freezes only forecast-dependent entries on the next tick.
- Clean price-action entries remain eligible during `forecast_only` freeze.
- Repeated severe rate-limit failures can trigger `all_entries` freeze for the next tick.
- Exits remain allowed when possible.
- Diagnostics include:
  - `live_freeze_active`,
  - `live_freeze_reason`,
  - `freeze_scope`,
  - `clean_price_action_candidates_count`,
  - `blocked_by_freeze_forecast_only_count`,
  - `blocked_by_freeze_all_count`.

### v4.2: fresh-event live channel

Implemented after a guarded live run submitted zero trades because every live entry required repeated quote history.

Observation:

```text
Momentum and mean reversion should require history.
Fresh event/catalyst opportunities should not require prior quote history, but must be much more heavily guarded.
```

Final policy:

- Price-action channel handles repeated-market short-horizon movement.
- Fresh-event channel handles clean catalyst/outcome-edge opportunities with tiny 1-share size.
- Forecast-edge-only remains insufficient under weak evidence.
- Rate-limit/fallback/stale evidence cannot become live entries.
- If no clean price-action or fresh-event candidate passes, submit nothing.

### v4.3: active candidate pre-ranking

Implemented before the final 24-tick guarded live attempt.

Problem:

```text
The live gate was safe but starved. The bot processed many long-horizon, low-catalyst markets, so very few candidates could ever pass price-action or fresh-event live gates.
```

Solution:

- Add deterministic pre-ranking before RAG/BLF/price trading.
- Keep deterministic safety filters.
- Reorder the eligible candidate set so the limited processing budget prefers:
  - existing positions needing exit checks,
  - repeated markets,
  - recent midpoint movers,
  - tight spreads,
  - near-term catalyst wording,
  - previous price-signal or trade-candidate markets.
- Penalize:
  - invalid or very wide quotes,
  - long-horizon no-movement politics/elections/nominations,
  - alien/disclosure low-catalyst markets,
  - long-horizon sports outright markets unless repeated and moving,
  - no-history/no-catalyst markets.
- This does not make the bot more aggressive. It feeds better candidates into the existing conservative live gates.

### v5: position-aware exits

First slice implemented; continue improving with live data.

- Explicit SELL logic for stale or broken positions.
- Exit when predicted hold edge is negative.
- Reduce weakest positions when open-position or exposure budget is scarce.
- Avoid accidental opposite-side netting.

### v6: offline/diagnostic evaluation

- Four-tick realized mark-to-market report.
- Hypothetical trade replay from memory logs.
- Parameter sweeps for price-edge threshold, momentum weight, forecast-gap weight, and shrinkage.
- Category performance breakdown.

---

## Suggested file additions

```text
edge_trader_bot/
  price_memory.py        # quote history and repeated-market state
  price_predictor.py     # short-horizon midpoint prediction
  price_policy.py        # price-aware entry/exit scoring, or fold into trading_policy.py
  pnl_tracker.py         # mark-to-market hypothetical and live PnL attribution
  diagnostics.py         # aggregate predicted-vs-realized price movement tables
```

Existing files to modify:

```text
runner.py                # update memory before scanning; write price diagnostics
trading_policy.py        # separate outcome edge vs price edge; add SELL logic
sizer.py                 # micro sizing for small price edges
schemas.py               # p_next_mid, expected_mid_move, price diagnostics
memory.py                # persistent quote history
scripts/run_four_tick_diagnostic.py  # realized price-move evaluation
```

---

## Main implementation principle

Do not make the bot more aggressive merely by lowering the old forecast-edge threshold.

Instead:

```text
Add a new price-movement edge.
Trade small when the short-horizon price edge is supported.
Keep outcome-only trades conservative.
Use dry-run diagnostics to prove the price layer before live submit.
```

Final strategy summary:

```text
The forecasting stack is now the research engine.
The trading stack must become a price-aware portfolio manager.

Win condition is not strongest final probability forecast.
Win condition is better mark-to-market trading decisions under spread, timing, and risk constraints.
```

## Final Guarded Live Entry Channels

The live bot now has three separate entry channels:

1. `clean_price_action`
   - Momentum / allowed mean reversion only.
   - Requires repeated quote history, previous midpoint movement, tight spread, and positive edge after spread.

2. `fresh_event`
   - No-history catalyst path.
   - Requires clean evidence, medium/high confidence, no fallback/rate-limit/stale risk, no resolution blocker, and tiny size.
   - Thresholds: 0.010 non-sports, 0.015 sports with quantitative support.
   - Sports without odds/model support are blocked.

3. `clean_forecast_mispricing`
   - New micro-entry path for clean forecast/resolution mispricings.
   - Uses only final post-shrinkage probability (`p_final_after_blf` / `p_final`), never raw RAG probability.
   - Requires evidence quality >= 4, medium/high confidence, spread <= 0.015 by default, no fallback/rate-limit/stale/search/JSON/resolution risk, and no selected-failed BLF.
   - Thresholds: 0.008 non-sports, 0.012 sports with quantitative odds/model support.
   - Guard mode forces size to 1 share.

Forecast-only freeze blocks `fresh_event` and `clean_forecast_mispricing` entries. It does not block exits or clean price-action entries. The old broad outcome-edge policy remains conservative; this patch adds only a tiny guarded micro-entry channel for clean post-shrinkage forecast edges.

## Final Live Evidence Recovery

RAG/Brave failures should not erase all evidence for a market that was successfully analyzed in a recent tick, but they also must not create hallucinated live trades.

Evidence recovery policy:

- Reuse previous successful RAG evidence by `market_id` only when it is recent enough.
- Default cache horizon is `EDGE_TRADER_MAX_CACHED_EVIDENCE_AGE_TICKS=8`.
- Cached evidence is capped at quality 3 and confidence medium.
- Cached evidence is explicitly marked and counted in diagnostics.
- Metadata-only fallback is created when RAG fails and no usable cache exists.
- Metadata-only evidence is diagnostic-only and cannot pass `clean_forecast_mispricing`.
- Previous `p_final` is an anchor for diagnostics and price-feature stability, not a standalone live forecast signal.

Live entry remains blocked for rate limits, stale evidence, invalid JSON, malformed responses, timeouts, unresolved resolution risk, selected-failed BLF, sports without quantitative support, or any threshold miss.
