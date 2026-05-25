# RAG-BLF Price-Aware Edge Trader Plan

## Objective

Build an independent Prophet Arena trading-track bot that combines:

- 2402-inspired broad RAG scanning.
- 2604-inspired sequential BLF verification.
- Market/statistical priors.
- Short-horizon price-movement prediction.
- Deterministic BUY / SELL / HOLD / ADD / REDUCE policy.
- Position-aware sizing, exits, and risk controls.

Major direction update:

```text
The forecasting layer is now good enough for the next phase.
The bottleneck is trading: price movement, mark-to-market PnL, exits, and position management.
```

The bot will continue to use `ai-prophet-core` directly through `BenchmarkSession`.

---

## Current Decision

Do not spend the next main iteration on better RAG/BLF prompts unless a bug is found.

The current pipeline can:

- claim ticks,
- load candidates,
- load portfolio,
- scan markets with RAG,
- run BLF on selected markets,
- write plan JSON,
- complete dry-run ticks,
- produce diagnostics,
- avoid unsafe live submit.

However, it mostly computes final-event fair value and compares it to executable price. That is too conservative for a trading contest where open positions are marked to market by future snapshot mid-prices.

New priority:

```text
Add price memory + next-mid prediction + position-aware exits.
```

---

## Implementation Stages

### Stage 1: Runnable MVP

Status: mostly complete.

Done:

- Created `edge_trader_bot`.
- Implemented tick lifecycle.
- Loaded candidates and portfolio.
- Applied deterministic filters.
- Computed market-implied/statistical priors.
- Made conservative BUY/HOLD decisions.
- Wrote plan JSON.
- Added submit/finalize/complete tick flow.
- Added `--once`, `--dry-run`, `--max-markets`, `--check-env`, and safety controls.
- Added local JSONL memory for audit records.
- Added dry-run safety gates so live submit is disabled unless explicitly allowed.

Remaining:

- Keep MVP stable while changing trading policy.

---

### Stage 2: 2402-Lite Scanner

Status: implemented and usable.

Done:

- Deterministic query generation.
- Optional Brave Search adapter.
- Structured RAG evidence package.
- Heuristic fallback when search/LLM fails.
- Evidence-quality scoring.
- Resolution-risk checks.
- Optional OpenRouter/DeepSeek LLM summarization/relevance.
- Rate-limit, timeout, invalid JSON, missing-key, empty-response, and malformed-response fallbacks.
- RAG diagnostics in plan JSON.
- RAG selection modes: `current` and `alpha_diagnostic`.
- RAG package remains advisory only; it does not choose action or size.

Current conclusion:

```text
RAG is useful for ranking and fair-value anchoring, but raw RAG disagreement is not enough for executable taker trades.
```

Next change:

- RAG selection should include price-action and existing-position urgency, not just apparent final-event edge.

---

### Stage 3: 2604-Lite BLF Verifier

Status: first slice implemented and usable.

Done:

- Structured BLF package.
- Belief-state loop.
- Probability clamp/shrinkage.
- Fallback to RAG when BLF fails.
- Deadline-aware limited usage.
- BLF diagnostics in plan JSON.
- BLF remains advisory only; deterministic policy controls trading.

Observed behavior:

- BLF usually moves `p_final` only a few tenths of a percent to about one percentage point.
- BLF often pulls aggressive RAG estimates back toward market-implied probability.
- This is good for forecast calibration but makes pure outcome-edge trading extremely conservative.

Next change:

- Use BLF to verify high-value entries **and exits**.
- Select BLF candidates using price movement, position risk, and catalyst likelihood, not only RAG/market disagreement.

---

### Stage 4: Aggregation and Calibration

Status: partially implemented manually; learned calibration pending.

Done:

- Market-aware shrinkage toward midpoint.
- `p_final_before_blf`, `p_final_after_blf`, `blf_adjustment`, and aggregation reasoning.
- Sports overconfidence guardrails.
- Future-event absence-of-evidence guardrails.

Current conclusion:

```text
The calibration/shrinkage layer is probably doing its job. It prevents bad overconfident trades.
The issue is that final-event edge alone rarely clears a profitable taker threshold.
```

Next change:

- Preserve outcome calibration.
- Add a separate short-horizon `p_next_mid` / price edge instead of weakening calibration.

---

### Stage 5: Position-Aware Trading

Status: pending; now high priority.

Original items:

- Add `SELL` exits for stale/broken positions.
- Avoid accidental opposite-side netting.
- Enforce target exposure below server caps.
- Preserve daily trade budget.
- Add category/theme exposure caps.

Updated scope:

- Add explicit action types:
  - `BUY_NEW`,
  - `ADD`,
  - `SELL_EXIT`,
  - `SELL_REDUCE`,
  - `HOLD`,
  - `SKIP`.
- Compute hold value for existing positions.
- Exit positions when short-horizon hold edge is negative or thesis breaks.
- Never assume BUY opposite side flips exposure cleanly.

---

### Stage 6: Price Memory Layer

Status: first slice implemented.

Goal:

```text
Track market quote history across ticks so the bot can predict price movement and evaluate mark-to-market PnL.
```

Implemented first slice:

- `edge_trader_bot/price_memory.py`.
- Local quote/forecast/position snapshots in `.edge_trader/market_history.jsonl`.
- Recent history loading by `market_id`.
- Basic price features:
  - current midpoint,
  - one/two-tick midpoint history,
  - one/two-tick deltas,
  - spread and spread percent,
  - repeated seen count,
  - simple volatility,
  - forecast-market gap,
  - forecast stability.

Remaining:

- Broaden diagnostics for realized next-tick / 2-tick / 4-tick PnL.
- Improve history compaction if JSONL grows large.
- Store per-market quote history:
  - tick timestamp,
  - best bid,
  - best ask,
  - midpoint,
  - spread,
  - candidate set id,
  - p_final,
  - p_2402,
  - p_blf,
  - decision.
- Compute features:
  - `mid_change_1tick`,
  - `mid_change_2tick`,
  - `mid_change_4tick`,
  - `spread_change_1tick`,
  - `repeated_market_count`,
  - `p_final_change_1tick`,
  - `forecast_market_gap`,
  - `actual_next_mid_move` when available.

Diagnostics:

- Top repeated markets by price movement.
- Top predicted price edges.
- Forecast gap vs actual next-mid move.
- Hypothetical mark-to-market PnL after 1/2/4 ticks.

---

### Stage 7: Short-Horizon Price Predictor

Status: minimal signal layer implemented; proper `p_next_mid` predictor still pending.

Goal:

```text
Predict whether the market midpoint will move enough in our direction over the next few ticks to justify a small trade.
```

Implemented first slice:

- `edge_trader_bot/price_signal.py`.
- Deterministic short-horizon signals:
  - momentum,
  - mean reversion,
  - spread-capture diagnostic,
  - small forecast-edge fallback.
- Tiny suggested size defaults.
- Position-aware blockers for no history, spread, total positions, per-market position, low edge, and forecast contradiction.
- Basic SELL exit logic:
  - take profit,
  - stop loss,
  - time exit,
  - signal reversal,
  - stale/wide quote.
- Runner can enable this layer with `EDGE_TRADER_ENABLE_PRICE_TRADING=1`.
- SELL exits are ranked ahead of new entries.

Still pending:

- Add a dedicated `price_predictor.py` with explicit `p_next_mid`.
- Learn/calibrate whether these signals predict future mid-price movement.

Original intended predictor inputs:

Inputs:

- Current midpoint and spread.
- Quote history.
- RAG/BLF fair value.
- RAG/BLF confidence.
- Evidence freshness.
- Catalyst/risk flags.
- Market category.
- Existing position state.

Initial model:

```text
p_next_mid = current_mid
           + momentum_weight * clipped_recent_mid_change
           + forecast_gap_weight * clipped(p_final - current_mid)
           + catalyst_adjustment
           - uncertainty_penalty
```

Starting constraints:

- Clamp predicted move tightly.
- No-history markets should have near-zero predicted move unless fresh evidence exists.
- Do not let raw RAG disagreement directly create large price edge.
- Sports markets require quantitative odds/market support for any meaningful price-edge signal.

Suggested first thresholds:

```text
price_trade_threshold_high_confidence = 0.005
price_trade_threshold_medium_confidence = 0.008
outcome_trade_threshold = 0.03 to 0.06
```

---

### Stage 8: Price-Aware Trading Policy

Status: new stage; after Stage 7.

Goal:

```text
Separate final-event outcome edge from short-horizon price edge.
```

For each candidate:

```text
outcome_edge_yes = p_final - best_ask
price_edge_yes = p_next_mid - best_ask
outcome_edge_no = (1 - p_final) - (1 - best_bid)
price_edge_no = (1 - p_next_mid) - (1 - best_bid)
```

Decision score:

```text
trade_score = 0.65 * price_edge + 0.35 * outcome_edge - risk_penalty
```

Use more outcome weight near resolution; use more price weight for long-horizon markets.

Sizing:

- Start with micro trades only:
  - `$10-$25` notional for small price edge,
  - `$25-$75` for stronger price edge,
  - larger disabled until dry-run proves positive mark-to-market behavior.

Plan fields to add:

- `p_next_mid`,
- `predicted_mid_move`,
- `price_edge_yes`,
- `price_edge_no`,
- `outcome_edge_yes`,
- `outcome_edge_no`,
- `trade_score`,
- `entry_reason`,
- `exit_reason`,
- `price_signal_confidence`,
- `would_trade_price_edge`,
- `would_trade_outcome_edge`.

---

### Stage 9: Mark-to-Market Evaluation

Status: new stage; after price diagnostics.

Goal:

```text
Evaluate the strategy by whether predicted price edges would have produced positive next-tick / 2-tick / 4-tick PnL.
```

Update `scripts/run_four_tick_diagnostic.py`:

- Keep dry-run by default.
- Preserve live-submit safety.
- Track repeated markets across ticks.
- For each hypothetical trade candidate, compute later mid-price PnL when the market repeats.
- Output:
  - hypothetical entry tick,
  - side,
  - entry price,
  - next mid,
  - 2-tick mid,
  - 4-tick mid,
  - PnL per share,
  - whether signal direction was correct,
  - reason for trade.

This is the key test before live trading.

---

## Progress Log

### Completed earlier

- Created strategy direction update in `rag_blf_edge_trader_strategy_v2.md`.
- Confirmed `references/llm_forecasting` should be used as reference code, not copied wholesale.
- Confirmed 2604 BLF should be implemented as a lightweight belief-state loop.
- Added `strategy.md` alias pointing to canonical strategy file.
- Created `edge_trader_bot` MVP package.
- Added deterministic market filtering, market/statistical prior, forecast combination, sizing, trading policy, and runner.
- Added conservative RAG and BLF interfaces, then implemented first slices.
- Verified package compiles and smoke tests run.

### RAG implementation progress

- Implemented `edge_trader_bot/rag_scanner.py` structured evidence packages.
- Added optional Brave Search adapter via `BRAVE_SEARCH_API_KEY`.
- Added safe fallback when RAG is disabled or search fails.
- Added `edge_trader_bot/resolution_checker.py` for resolution-risk flags.
- Runner records evidence packages in plan JSON and limits RAG markets per tick.
- Added RAG smoke tests.
- Added optional LLM RAG summarization/relevance behind `EDGE_TRADER_ENABLE_LLM_RAG`.
- Standardized LLM config around OpenRouter + `deepseek/deepseek-chat` for cheap testing.
- Hardened LLM RAG failure handling.

### BLF implementation progress

- Implemented `edge_trader_bot/blf_verifier.py`.
- Added structured BLF packages, belief state, update steps, raw/final shrunken `p_blf`, confidence, uncertainty, risk flags, elapsed time, and fallback diagnostics.
- Added BLF config/env support:
  - `EDGE_TRADER_ENABLE_BLF`,
  - `EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK`,
  - `EDGE_TRADER_BLF_MAX_STEPS`,
  - provider/model/timeout/retry controls.
- BLF provider/model defaults to OpenRouter + `deepseek/deepseek-chat`.
- Runner selects top BLF markets and records BLF diagnostics.
- Aggregation records BLF adjustments.

### Diagnostic progress

- Added inspection and threshold-debug diagnostics.
- Added opportunity reports:
  - top taker edges,
  - top maker edges,
  - top mid edges,
  - RAG-market disagreement,
  - BLF probability changes,
  - edge-only blocks,
  - evidence-quality blocks,
  - resolution-risk blocks,
  - raw and normalized risk flags.
- Added market-type classification.
- Added long-horizon future-event guardrails.
- Added 4-tick diagnostic script.
- Added safety gates for live submit.
- Fixed RAG selection schema compatibility and missing-selection crash.
- Added bounded RAG/LLM/BLF concurrency and tick deadline diagnostics.
- Added fast/normal/wide diagnostic profiles.
- Added first price-trading diagnostics into plan JSON and aggregate reports.

### Price-trading implementation progress

- Implemented cross-tick market memory at `.edge_trader/market_history.jsonl`.
- Added price feature computation for repeated markets.
- Added deterministic short-horizon signal generation:
  - momentum,
  - mean reversion,
  - spread-capture diagnostic,
  - forecast-edge fallback.
- Added initial position exit logic:
  - take profit,
  - stop loss,
  - time exit,
  - signal reversal,
  - stale/wide quote.
- Runner now supports `EDGE_TRADER_ENABLE_PRICE_TRADING=1`.
- Diagnostic reports include:
  - `price_trading_summary`,
  - `count_price_signals`,
  - signal-type counts,
  - `top_price_signals`,
  - `top_trade_candidates`,
  - exit candidates,
  - position table,
  - price-signal blockers.
- Added `scripts/run_guarded_live_6h.py` for a bounded guarded live session.
- Live submit still requires both CLI `--allow-live-submit` in the underlying runner and `EDGE_TRADER_ALLOW_LIVE_SUBMIT=1`; guard mode remains on by default.
- Guarded live profile uses tiny trade size and keeps BLF/RAG enabled as sanity checks.
- Added final live-tradable signal gate:
  - rate-limit, fallback, timeout, invalid JSON, stale evidence, and LLM uncertainty paths are blocked from live entries,
  - forecast-edge-only entries are blocked from live submit,
  - sports forecast-dependent entries require quantitative support,
  - live entries require repeated price history and tight spreads,
  - guarded live clamps order size to 1 share,
  - guarded live caps new entries at 3 per tick and exits at 5 per tick.
- Guarded live script supports `--ticks 24`.
- Fixed live-freeze behavior after first live attempt:
  - freeze is tick-local, not permanently sticky,
  - ordinary rate-limit/fallback risk freezes only forecast-dependent entries for the next tick,
  - clean momentum and allowed mean-reversion entries can still pass if history/spread/edge checks pass,
  - only repeated severe rate-limit failures trigger all-entry freeze,
  - added diagnostics for freeze scope and clean price-action candidate counts.
- Added a second guarded live entry channel after live run showed zero submissions:
  - `price_action` channel keeps strict repeated-history momentum / mean-reversion rules,
  - `fresh_event` channel does not require prior quote history,
  - fresh-event entries require clean strong evidence/catalyst, no fallback/rate-limit/stale flags, tight spread, executable taker edge, and 1-share max size,
  - long-horizon sports are blocked from fresh-event entries unless explicit near-term quantitative catalyst support exists.
- Added deterministic candidate pre-ranking before the expensive RAG/BLF/price pipeline:
  - prefers existing positions, repeated movers, tight spreads, recent midpoint movement, and near-term catalyst wording,
  - penalizes invalid/wide quotes, long-horizon no-movement politics, alien/disclosure markets, and long-horizon sports outright markets,
  - enabled by `EDGE_TRADER_ENABLE_CANDIDATE_RERANK=1`,
  - enabled by default in diagnostic/live guarded scripts.

### Key diagnostic observations

Observed across recent one-tick and four-tick dry runs:

- Pipeline is healthy and completes ticks.
- RAG/LLM RAG/BLF mostly succeed, with occasional rate-limit fallbacks.
- Live submit remains disabled as expected.
- No unsafe submit detected.
- Final-event taker edges are tiny.
- Positive midpoint/maker diagnostic edges exist, but most are not executable as taker trades.
- Relaxed threshold scenarios sometimes produce hypothetical trades only when BLF is ignored and thresholds are very low.
- With the normal calibrated BLF/shrinkage path, trades remain zero.
- This looks like a strategy-objective issue rather than a pipeline failure.

Interpretation:

```text
The forecasting stack is conservative and probably not obviously broken.
The bot now needs a short-horizon price/momentum/mark-to-market layer.
```

---

## Recent Run Interpretation

Recent command shape:

```bash
python scripts/run_four_tick_diagnostic.py \
  --ticks 4 \
  --diagnostic-profile fast \
  --continue-on-error
```

Observed summary:

- Processed around 80 markets per tick.
- RAG budget around 8 per tick.
- BLF budget around 2 per tick.
- `trades=0` for all observed ticks.
- Max taker edge remained around 0.001 to 0.0023 in the shown ticks.
- Some rate-limit fallbacks occurred in later ticks.

Conclusion:

```text
Do not fix this by simply lowering the old forecast-edge threshold.
A 0.1%-0.2% final-event edge is not enough to trade as a taker by itself.
Instead, add price-edge prediction and evaluate whether those small edges predict actual next-tick price movement.
```

Latest price-aware fast dry-run:

```bash
python scripts/run_four_tick_diagnostic.py --ticks 1 --diagnostic-profile fast --continue-on-error
```

Observed:

- Tick completed and finalized.
- `candidates_processed=80`.
- `RAG attempted=8`, `RAG failed=0`.
- `LLM attempted=8`, `LLM failed/fallback=5` due to OpenRouter rate limits.
- `BLF attempted=2`, `BLF succeeded=1`, `BLF fallback=1`.
- `trade_count=0`.
- `submit_called=false`.
- `max_edge=0.00656`, mostly sports outcome markets with rate-limit fallback flags.
- Diagnostics written:
  - `outputs/diagnostics/four_tick_20260517_052656.json`
  - `outputs/diagnostics/four_tick_20260517_052656.md`

Interpretation:

```text
The price-aware runner path is live-server compatible and dry-run safe.
No real trade was submitted.
Current first tick still produced no real trades, which is expected because price memory needs repeated market observations.
Rate limits are visible and fall back safely.
Next useful test is a multi-tick dry-run so price memory can generate true repeated-market features.
```

Latest 4-tick live-readiness conclusion:

```text
The server loop and price-memory pipeline are reliable enough for a tiny guarded live test.
However, the largest apparent edges can appear during LLM/BLF rate-limit fallback paths.
Those are watchlist-only and must never become live trades.
Final live policy is therefore: price-action-only live entries, repeated-history required, tight spread required, no forecast-only live entries, no fallback/rate-limit live entries, and exits remain allowed.
Live freeze is tick-local: fallback/rate-limit risk freezes forecast-dependent entries only, while clean price-action entries remain eligible.
```

Latest guarded-live observation:

```text
Live submit was enabled, but submitted trades remained zero because the live gate required repeated quote history for every entry.
Main blockers were no-history / missing previous midpoint / insufficient repeated history, plus normal low-confidence and spread gates.
Conclusion: the old gate was too narrow for fresh event/catalyst opportunities. The final policy now has two channels: strict price-action with history, and tiny fresh-event entries without history but only under clean strong evidence.
```

Latest candidate-selection observation:

```text
The guarded live path was operational but wasted much of the processed-market budget on long-horizon, low-catalyst markets.
The final pre-live patch adds active/tight/repeated/catalyst-aware candidate reranking so the fixed 100-market budget is spent on markets more likely to produce live-tradable price signals.
This does not loosen live safety gates; it only feeds them better candidates.
```

---

## Next Codex Task: Price-Memory + Price Diagnostics

Status: first slice implemented; next work is validation through real repeated ticks.

### Goal

Validate quote-history memory and realized price-move diagnostics before trusting live trading.

### Requirements

1. Run repeated dry-run ticks with price trading enabled.
2. Confirm `.edge_trader/market_history.jsonl` captures repeated market snapshots.
3. Add or verify these fields in per-market debug rows and plan JSON:
   - `current_mid`,
   - `current_spread`,
   - `previous_mid`,
   - `previous_spread`,
   - `mid_change_1tick`,
   - `mid_change_2tick`,
   - `mid_change_4tick`,
   - `spread_change_1tick`,
   - `repeated_market_count`,
   - `previous_p_final`,
   - `p_final_change_1tick`.
4. Continue improving aggregate Markdown/JSON diagnostics:
   - top repeated markets by absolute price move,
   - top repeated markets by p_final movement,
   - count of markets with positive/negative next-tick moves when available,
   - realized next-tick PnL for hypothetical top price-edge candidates when available.
5. Keep all changes dry-run safe.
6. Add more tests using mocked repeated markets and realized next-mid movement.

### Explicit non-goals for this task

- Do not enable live submit.
- Do not make real trades.
- Do not rewrite the RAG/BLF prompts unless needed for schema compatibility.
- Do not lower old outcome-edge thresholds as the main solution.

---

## Next Codex Task After That: Price Predictor

After quote-history memory works, add `price_predictor.py`.

### Goal

Predict `p_next_mid` and `expected_mid_move` for each market.

### Initial heuristic

```text
expected_mid_move =
    0.35 * clipped(mid_change_1tick)
  + 0.15 * clipped(mid_change_4tick)
  + 0.10 * clipped(p_final - current_mid)
  + catalyst_adjustment
  - uncertainty_penalty
```

Clamp:

```text
no fresh evidence: [-0.005, +0.005]
fresh evidence: [-0.015, +0.015]
low confidence: [-0.003, +0.003]
```

Then:

```text
p_next_mid = clamp(current_mid + expected_mid_move, 0.001, 0.999)
```

Add diagnostics before using it for live trades.

---

## Open Questions

1. Does the arena allow any passive/maker-style order behavior, or only snapshot taker fills?
   - Current assumption: only taker fills are executable.
   - Maker/midpoint edges remain diagnostic only.
2. How often do the same markets repeat across ticks?
   - We need memory diagnostics to quantify this.
3. Are short-horizon price moves predictable from RAG/BLF gaps?
   - Four-tick diagnostics should answer this.
4. Which categories produce tradable price motion?
   - Sports long-horizon markets may be too efficient or too stable; announcement/resolution markets may be better.
5. Should we run a tiny live-submit test only after price diagnostics are positive?
   - Not yet. Dry-run first.

---

## Current Recommended Commands

Fast four-tick dry-run for diagnostics:

```bash
python scripts/run_four_tick_diagnostic.py \
  --ticks 4 \
  --diagnostic-profile fast \
  --continue-on-error
```

One-tick fast dry-run:

```bash
python scripts/run_four_tick_diagnostic.py \
  --ticks 1 \
  --diagnostic-profile fast \
  --continue-on-error
```

Normal one-tick dry-run for richer analysis:

```bash
python scripts/run_four_tick_diagnostic.py \
  --ticks 1 \
  --diagnostic-profile normal \
  --continue-on-error
```

Wider one-tick diagnostic when latency is acceptable:

```bash
python scripts/run_four_tick_diagnostic.py \
  --ticks 1 \
  --max-markets 100 \
  --rag-budget 25 \
  --blf-budget 5 \
  --continue-on-error
```

Expected output locations:

```text
outputs/diagnostics/four_tick_*.json
outputs/diagnostics/four_tick_*.md
.edge_trader/memory.jsonl
.edge_trader/market_history.jsonl
```

Guarded six-hour live command, only after dry-run inspection:

```bash
EDGE_TRADER_ALLOW_LIVE_SUBMIT=1 \
EDGE_TRADER_LIVE_GUARD_MODE=1 \
python scripts/run_guarded_live_6h.py \
  --diagnostic-profile live_guarded \
  --duration-hours 6 \
  --max-trade-size 1 \
  --max-position-per-market 3 \
  --continue-on-error
```

The live script enables candidate reranking automatically. To compare without reranking, explicitly set:

```bash
EDGE_TRADER_ENABLE_CANDIDATE_RERANK=0
```

## Final 24-Tick Live Submit Plan

Run this only after confirming `.env` has the intended API keys and after watching the first one or two ticks closely:

```bash
EDGE_TRADER_ALLOW_LIVE_SUBMIT=1 \
EDGE_TRADER_LIVE_GUARD_MODE=1 \
EDGE_TRADER_ENABLE_PRICE_TRADING=1 \
python scripts/run_guarded_live_6h.py \
  --ticks 24 \
  --diagnostic-profile live_guarded \
  --max-trade-size 1 \
  --max-position-per-market 3 \
  --continue-on-error
```

Live-entry rules for this run:

- Live entries have two channels.
- Price-action channel:
  - momentum,
  - mean reversion with at least four repeated observations and tight spread.
- Fresh-event tiny channel:
  - does not require repeated quote history,
  - requires strong clean evidence/catalyst,
  - requires no fallback/rate-limit/stale/invalid JSON/timeout flags,
  - requires `evidence_quality >= 4`,
  - requires high confidence with executable edge at least `0.015`, or medium confidence with edge at least `0.025`,
  - requires spread at or below `EDGE_TRADER_FRESH_EVENT_MAX_SPREAD` default `0.015`,
  - blocks long-horizon sports unless explicit near-term quantitative catalyst support exists,
  - size is forced to 1 share in guard mode.
- Forecast-edge-only signals are diagnostics only.
- Rate-limit/fallback/stale/invalid-JSON/timeout paths are diagnostics only.
- Any `llm_rate_limit`, `blf_rate_limit`, `llm_rag_failed`, `stale_evidence`, `fallback`, or `llm_uncertainty` signal is blocked unless it is clean pure price momentum with enough repeated history.
- Guard mode clamps order size to 1 share.
- Max position per market is 3 shares.
- Max total open shares target is 12.
- Max new entries per tick is 3.
- Max exits per tick is 5.
- If fallback/rate-limit blocks appear, the next tick freezes forecast-dependent entries only.
- If repeated severe LLM/BLF failures appear, the next tick can freeze all new entries.
- Clean price-action momentum and allowed mean reversion remain eligible during forecast-only freeze.
- Exits remain allowed.

Inspect first during live:

- `outputs/live_runs/<timestamp>/ticks.jsonl`
- `submit_called`
- `accepted`
- `trade_count`
- `skipped_by_live_safety_count`
- `blocked_reasons_counts`
- `live_tradable_candidate_count`
- `live_trade_candidates`
- `rejected_trade_candidates_top10`
- `count_exit_intents`
- `submission_reason`
- `live_freeze_active`
- `freeze_scope`
- `live_freeze_reason`
- `clean_price_action_candidates_count`
- `live_price_action_candidate_count`
- `live_fresh_event_candidate_count`
- `live_forecast_mispricing_candidate_count`
- `skipped_by_price_action_history_count`
- `skipped_by_fresh_event_safety_count`
- `fresh_event_candidates`
- `fresh_event_rejected_top10`
- `rejected_forecast_mispricing_top10`
- `candidate_selection_summary`
- `selected_bucket_counts`
- `top_selected_candidates`
- `blocked_by_freeze_forecast_only_count`
- `blocked_by_freeze_all_count`

Guarded live output locations:

```text
outputs/live_runs/<timestamp>/ticks.jsonl
outputs/live_runs/<timestamp>/decisions.jsonl
outputs/live_runs/<timestamp>/trades.jsonl
outputs/live_runs/<timestamp>/summary.md
```

## Final Guarded Micro-Entry Patch

Latest live tests showed that pure price-action plus fresh-event gating was still too narrow:

- `live_submit_enabled=true` and guard mode worked.
- `count_price_signals` appeared, but `live_tradable_candidate_count` stayed at 0.
- `fresh_event` rejected most markets, and Kalshi-style clean forecast/resolution mispricings were not eligible unless they also looked like price momentum.

Final live policy now has three entry channels:

- `clean_price_action`: existing strict momentum / allowed mean-reversion path; requires repeated quote history and tight spreads.
- `fresh_event`: no-history catalyst path; now uses guarded thresholds of 0.010 for non-sports and 0.015 for sports with quantitative support. Sports without quantitative support remain blocked.
- `clean_forecast_mispricing`: new guarded one-share path for clean post-shrinkage forecast edges.

`clean_forecast_mispricing` rules:

- Uses only `p_final_after_blf` / final aggregated probability, never raw `p_2402`.
- Blocks rate-limit, fallback, stale evidence, timeout, invalid JSON, malformed responses, search failure, resolution risk, and ambiguous resolution.
- Requires `evidence_quality >= 4`.
- Requires `confidence` medium or high.
- Requires spread <= `EDGE_TRADER_LIVE_FORECAST_MAX_SPREAD` default 0.015.
- Blocks if BLF was selected and failed.
- Non-sports threshold is 0.008.
- Sports threshold is 0.012 and requires quantitative odds/model support.
- Guard mode forces size to 1 share.
- Forecast-only freeze blocks `fresh_event` and `clean_forecast_mispricing`, but exits and clean price-action are still allowed.

This does not lower the old general outcome-edge threshold. It adds a separate tiny guarded live channel for clean, non-fallback forecast mispricings.

### Final Pipeline Ordering Fix

Latest 1-tick guarded live showed the forecast gates were being evaluated on many markets that had never received RAG/BLF because they were outside the small forecast budget. Those markets now report `not_evaluated_by_forecast_budget` instead of misleading `low_evidence` / `low_confidence` forecast-mispricing blockers.

Live-guarded mode now uses candidate reranking to drive RAG selection:

- Top live-relevant candidates from `repeated_mover`, `near_term_catalyst`, and `tight_spread_active` buckets are forced into RAG budget first.
- Forced RAG records include `rag_forced_by_live_candidate_selection`, `live_candidate_selection_rank`, and `live_candidate_selection_score`.
- BLF selection now prioritizes candidates where post-RAG probabilities create possible live forecast/fresh-event paths.
- Plan and live summaries include:
  - `live_candidate_pool_count`
  - `live_candidate_pool_top10`
  - `live_candidates_sent_to_rag_count`
  - `live_candidates_sent_to_rag_top10`
  - `live_candidates_sent_to_blf_count`
  - `live_candidates_sent_to_blf_top10`
  - `not_evaluated_by_forecast_budget_count`
  - `analyzed_live_candidate_count`
  - `analyzed_live_candidate_top10`
  - `forecast_gate_rejection_top10`

Safety rules are unchanged: no raw RAG live entries, no fallback/rate-limit/stale/error trades, no sports forecast trades without quantitative support, and guard mode keeps order size at 1.

### Final Live Evidence Recovery Patch

Latest 1-tick live run showed candidate selection and RAG budget routing are working:

- `live_candidate_pool_count=100`
- `live_candidates_sent_to_rag_count=8`
- `live_candidates_sent_to_blf_count=2`
- `analyzed_live_candidate_count=8`
- unforecasted markets were correctly labeled as `not_evaluated_by_forecast_budget`

The remaining bottleneck was transient Brave/RAG failure: analyzed live candidates received `brave_http_error`, which forced `evidence_quality=0` and `confidence=low`.

Evidence recovery now works as follows:

- If RAG fails, the bot looks for the latest successful local evidence package for the same `market_id` in memory.
- Cached evidence is only usable up to `EDGE_TRADER_MAX_CACHED_EVIDENCE_AGE_TICKS`, default 8.
- Cached evidence is capped at `evidence_quality <= 3` and `confidence <= medium`.
- Cached packages are marked with `cached_evidence_used`, `cached_evidence_age_ticks`, `original_evidence_tick_ts`, and `current_rag_error`.
- If cache is too old, the bot marks `cached_evidence_too_old` and does not use it for live forecast entry.
- If no usable cache exists, the bot creates `metadata_only_evidence`; this is diagnostic-only and cannot pass `clean_forecast_mispricing`.
- Previous `p_final` can be recorded as `previous_p_final_used`, but cannot by itself create a forecast live entry.

New diagnostics:

- `rag_failure_count_by_reason`
- `cached_evidence_fallback_count`
- `cached_evidence_used_count`
- `cached_evidence_too_old_count`
- `metadata_only_fallback_count`
- `previous_p_final_used_count`
- `live_forecast_mispricing_cached_candidate_count`
- `rejected_cached_evidence_top10`
- `metadata_only_rejected_top10`
- `recovered_rag_candidates_top10`

This improves evidence availability without loosening thresholds, increasing size, or allowing fallback/rate-limit/stale/error signals to trade.

---

## Success Criteria for Next Phase

Before live submit, we want evidence that:

- Repeated-market quote history is being captured correctly.
- `p_next_mid` predictions are recorded.
- Hypothetical price-edge trades have non-random or positive next-tick / multi-tick mark-to-market behavior.
- Existing-position exit logic is simulated and sane.
- Rate-limit fallbacks do not create fake high-confidence price signals.
- The bot still never submits unless `--allow-live-submit` and env safety gates are explicitly enabled.

The next phase is successful even if there are still zero live trades, as long as we can diagnose whether price movement is predictable.

---

## Strategic Summary

The old system was:

```text
forecast fair value -> compare to current price -> trade only if huge final-event edge
```

The new system should be:

```text
forecast fair value
+ predict short-horizon price movement
+ manage positions and exits
+ trade only when expected mark-to-market return beats spread/risk
```

The main engineering move is not more forecasting. It is turning a strong forecasting pipeline into a price-aware trading system.
