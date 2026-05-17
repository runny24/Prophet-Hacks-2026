# RAG-BLF Edge Trader Plan

## Objective

Build an independent Prophet Arena trading-track bot that combines:

- 2402-inspired broad RAG scanning.
- 2604-inspired sequential BLF verification.
- Deterministic trading policy, sizing, exits, and risk controls.

The bot will use `ai-prophet-core` directly through `BenchmarkSession`.

## Current Decision

Do not modify the existing `prophet trade` CLI pipeline for the main strategy.
It is useful reference code, but the RAG-BLF trading strategy needs a custom
control loop with deterministic risk management and explicit exit behavior.

## Implementation Stages

### Stage 1: Runnable MVP

Status: mostly complete

- Create `edge_trader_bot`.
- Implement tick lifecycle.
- Load candidates and portfolio.
- Apply deterministic filters.
- Compute market-implied/statistical priors.
- Make conservative BUY/HOLD decisions.
- Write plan JSON.
- Submit intents, finalize, and complete tick.
- Add `--once` and `--dry-run` runner options.
- Add local JSONL memory for audit records.

### Stage 2: 2402-Lite Scanner

Status: partially implemented

- Generate targeted search queries.
- Retrieve and summarize evidence through a search adapter.
- Score evidence quality and resolution risk.
- Output structured `p_2402` evidence packages.
- Use scanner only for ranking unless edge is exceptionally clear.
- Current implementation has deterministic query generation, optional Brave
  Search adapter, structured fallback, evidence normalization, heuristic
  evidence scoring, optional LLM evidence summarization/relevance, and
  resolution-risk checks.
- LLM summarization/relevance is guarded by config and fallback. If the LLM
  fails, times out, or returns invalid JSON, the scanner falls back to heuristic
  evidence and the deterministic policy remains conservative.
- LLM provider defaults to OpenRouter using `deepseek/deepseek-chat` for cheap
  local testing/training. Direct OpenAI remains possible only via explicit
  `EDGE_TRADER_LLM_PROVIDER=openai`.
- The scanner preserves the structured evidence package format and never chooses
  trade action or share size.

### Stage 3: 2604-Lite BLF Verifier

Status: implemented first slice

- Maintain structured belief state.
- Run short sequential update loop for top markets only.
- Stop on deadline pressure, strong/weak edge, or max steps.
- Add K=3 trial aggregation later if the first slice proves useful.
- Current first slice uses one conservative LLM belief-state verifier on top
  markets only, with strict JSON validation, probability clamp/shrinkage, and
  structured fallback to RAG when BLF fails.

### Stage 4: Aggregation and Calibration

Status: pending

- Aggregate BLF trials in logit space.
- Shrink toward market midpoint.
- Track trial disagreement as model uncertainty.
- Add learned calibration later if enough outcomes accumulate.

### Stage 5: Position-Aware Trading

Status: pending

- Add `SELL` exits for stale/broken positions.
- Avoid accidental opposite-side netting.
- Enforce target exposure below server caps.
- Preserve daily trade budget.
- Add category/theme exposure caps.

## Progress Log

- Created strategy direction update in `rag_blf_edge_trader_strategy_v2.md`.
- Confirmed `references/llm_forecasting` should be used as a reference, not
  copied wholesale.
- Confirmed 2604 BLF should be implemented as a lightweight belief-state loop.
- Added `strategy.md` as a short alias pointing to the canonical strategy file.
- Created `edge_trader_bot` MVP package.
- Added deterministic market filtering, market/statistical prior, forecast
  combination, sizing, trading policy, and runner.
- Added conservative RAG and BLF interfaces as disabled stubs.
- Verified the package compiles and basic policy smoke test runs.
- Implemented Stage 2 / 2.5 first slice:
  - `edge_trader_bot/rag_scanner.py` now returns structured evidence packages.
  - Added optional Brave Search adapter via `BRAVE_SEARCH_API_KEY`.
  - Added safe fallback when RAG is disabled or search fails.
  - Added `edge_trader_bot/resolution_checker.py` for resolution-risk flags.
  - Runner records evidence packages in plan JSON and limits RAG markets per tick.
  - Added `tests/test_rag_scanner.py` smoke tests.
- Added optional LLM RAG summarization/relevance behind
  `EDGE_TRADER_ENABLE_LLM_RAG`.
- Standardized LLM config around `EDGE_TRADER_LLM_PROVIDER=openrouter`,
  `OPENROUTER_API_KEY`, and `EDGE_TRADER_LLM_MODEL=deepseek/deepseek-chat`.
- Added `.env.example` with OpenRouter/DeepSeek defaults.
- Improved resolution checker schema:
  - headline-vs-resolution mismatch,
  - official-source requirement/found,
  - deadline satisfaction,
  - stale evidence,
  - ambiguity level,
  - trade blocker.
- Wired high resolution-risk blockers into deterministic trading policy.
- Added `tests/test_strategy_core.py` for filters, priors, aggregation, sizing,
  and conservative policy behavior.
- Added `tests/test_llm_config.py` for provider defaults and OpenRouter headers
  without network access.
- Added first-dry-run safety controls:
  - `--max-markets`,
  - `--check-env`,
  - `--allow-live-submit`,
  - `.env` loading without overriding existing env,
  - live submit disabled unless explicitly allowed,
  - JSON-safe plan serialization.
- Added `tests/test_runner_safety.py` for dry-run submit safety, max-market caps,
  missing key behavior, env presence reporting, and weird-object JSON safety.
- Ran first real Prophet Arena server dry-run, server loop only:
  - command shape: staged dry-run A,
  - RAG disabled,
  - LLM RAG disabled,
  - `--max-markets 5`,
  - candidates loaded: 256,
  - candidates processed: 5,
  - hypothetical intents: 0,
  - submit called: false,
  - plan persisted, participant finalized, tick completed.
- Prepared and ran staged dry-run B, server loop + Brave RAG:
  - command shape: staged dry-run B,
  - RAG enabled,
  - LLM RAG disabled,
  - max RAG markets per tick: 3,
  - `--max-markets 10`,
  - candidates loaded: 256,
  - candidates processed: 10,
  - markets filtered before RAG: 246,
  - RAG attempted: 3,
  - RAG succeeded: 3,
  - RAG failed: 0,
  - hypothetical intents: 0,
  - submit called: false,
  - plan persisted, participant finalized, tick completed.
- Added RAG diagnostics to plan JSON:
  - `rag_enabled`,
  - `llm_rag_enabled`,
  - `rag_markets_attempted`,
  - `rag_markets_scanned`,
  - `rag_scanner_errors`,
  - `brave_api_present`,
  - search queries,
  - raw/deduped search result counts,
  - evidence quality,
  - `p_2402`,
  - resolution check,
  - scanner elapsed time,
  - total RAG elapsed time,
  - RAG budget skipped markets.
- Added explicit Brave failure categories and structured scanner fallback for:
  missing key, HTTP error, timeout, empty results, malformed response, rate
  limit, and unexpected exceptions.
- Added dry-run summary logging after plan generation.
- Prepared and ran staged dry-run C, server loop + Brave RAG + OpenRouter LLM RAG:
  - command shape: staged dry-run C,
  - RAG enabled,
  - LLM RAG enabled,
  - LLM provider/model: OpenRouter + `deepseek/deepseek-chat`,
  - max RAG markets per tick: 3,
  - `--max-markets 10`,
  - candidates loaded: 256,
  - candidates processed: 10,
  - markets filtered before RAG: 246,
  - RAG attempted: 3,
  - RAG succeeded: 3,
  - RAG failed: 0,
  - LLM attempted: 3,
  - LLM succeeded: 3,
  - LLM failed: 0,
  - LLM fallback: 0,
  - hypothetical intents: 0,
  - submit called: false,
  - plan persisted, participant finalized, tick completed.
- Added LLM RAG diagnostics to plan JSON:
  - provider/model,
  - attempted/succeeded/failed/fallback counts,
  - error categories,
  - per-market elapsed time,
  - total LLM elapsed time,
  - raw and final shrunken `p_2402`,
  - confidence/evidence quality/reasoning summary,
  - JSON parse error and fallback reason,
  - compact prompt metadata without API keys or full prompt text.
- Hardened OpenRouter/DeepSeek LLM RAG failure handling for missing key, HTTP
  error, timeout, rate limit, empty response, invalid JSON, missing required
  fields, invalid probability, and unexpected exceptions. All cases fall back
  to heuristic RAG and do not let the LLM choose action or share size.
- Added tests for OpenRouter invalid JSON fallback, missing OpenRouter key
  fallback, LLM probability clamping, low-quality shrinkage, resolution
  trade-blocker shrinkage, and dry-run submit safety with LLM enabled.
- Implemented minimal Stage 3 BLF verifier:
  - `edge_trader_bot/blf_verifier.py` now returns structured BLF packages,
    belief state, update steps, raw/final shrunken `p_blf`, confidence,
    uncertainty, risk flags, elapsed time, and fallback diagnostics.
  - Added BLF config/env support:
    `EDGE_TRADER_ENABLE_BLF`, `EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK`,
    `EDGE_TRADER_BLF_MAX_STEPS`, `EDGE_TRADER_BLF_PROVIDER`,
    `EDGE_TRADER_BLF_MODEL`, `EDGE_TRADER_BLF_TIMEOUT_SECONDS`,
    `EDGE_TRADER_BLF_JSON_RETRIES`, and
    `EDGE_TRADER_BLF_ENABLE_EXTRA_SEARCH`.
  - BLF provider/model default to OpenRouter +
    `deepseek/deepseek-chat`; extra search is disabled by default.
  - Runner selects top BLF markets by RAG edge, evidence quality, resolution
    risk, and RAG-market disagreement.
  - Plan JSON now includes BLF package per market and plan-level BLF
    diagnostics: attempted, succeeded, failed, fallback, errors, and elapsed
    time.
  - Aggregation now records `p_final_before_blf`, `p_final_after_blf`,
    `blf_adjustment`, and `aggregation_reason`.
  - BLF remains advisory only; deterministic policy still decides BUY/HOLD and
    sizing.
- Added staged dry-run D command:
  `EDGE_TRADER_ENABLE_RAG=1 EDGE_TRADER_ENABLE_LLM_RAG=1 EDGE_TRADER_ENABLE_BLF=1 EDGE_TRADER_MAX_RAG_MARKETS_PER_TICK=5 EDGE_TRADER_MAX_BLF_MARKETS_PER_TICK=1 EDGE_TRADER_BLF_MAX_STEPS=2 python -m edge_trader_bot.runner --once --dry-run --max-markets 25`
- Ran staged dry-run D, server loop + Brave RAG + OpenRouter LLM RAG + minimal
  BLF:
  - RAG enabled,
  - LLM RAG enabled,
  - BLF enabled,
  - max RAG markets per tick: 5,
  - max BLF markets per tick: 1,
  - BLF max steps: 2,
  - `--max-markets 25`,
  - candidates loaded: 256,
  - candidates processed: 25,
  - markets filtered before RAG: 231,
  - RAG attempted: 5,
  - RAG succeeded: 5,
  - RAG failed: 0,
  - LLM attempted: 5,
  - LLM succeeded: 5,
  - LLM failed: 0,
  - LLM fallback: 0,
  - BLF attempted: 1,
  - BLF succeeded: 1,
  - BLF failed: 0,
  - BLF fallback: 0,
  - hypothetical intents: 0,
  - submit called: false,
  - plan persisted, participant finalized, tick completed.
- Added inspection and threshold-debug diagnostics:
  - per-market full pipeline debug rows with prices, priors, RAG/BLF raw and
    shrunken probabilities, final edges, best side, hold/blocker reasons,
    evidence quality, confidence, normalized risk flags, aggregation reason,
    and sizing result.
  - `EDGE_TRADER_THRESHOLD_DEBUG=1` / `--threshold-debug` diagnostic mode that
    never submits, does not alter real policy decisions, and reports relaxed
    threshold scenarios only.
  - relaxed scenario report covers edge thresholds 0.10/0.08/0.06/0.04,
    evidence-quality minimums 4/3, and BLF enabled vs BLF ignored.
  - opportunity report now includes top edges, evidence quality, RAG-market
    disagreement, BLF probability changes, edge-only blocks, resolution-risk
    blocks, evidence-quality blocks, wide-spread markets, and noisy/free-text
    LLM risk flags.
  - raw risk flags are preserved and normalized into fixed vocabulary labels
    for inspection.
- Added tests for threshold-debug safety, diagnostic-only relaxed scenarios,
  resolution blockers in relaxed scenarios, opportunity report BLF fields, and
  normalized risk flags.
- Added market-type classification and long-horizon future-event diagnostics:
  - heuristic tags include future announcement, future candidacy,
    entertainment casting, election control, sports outcome, scientific
    milestone, climate metric, and other.
  - market type, days to resolution, future-event anchoring, and
    absence-as-negative diagnostics are written into RAG package, BLF package,
    per-market signals, and pipeline debug rows.
  - RAG and BLF prompts now explicitly tell the model not to treat "no current
    announcement yet" as strong NO evidence for far-future announcement or
    candidacy markets.
  - Added a guardrail that anchors future-event probabilities near the market
    prior when a low raw probability appears to be driven only by missing
    current evidence, while still allowing explicit decline/ineligibility
    evidence to lower probability.
- Ran expanded threshold-debug dry-run:
  - max RAG markets per tick: 10,
  - max BLF markets per tick: 2,
  - BLF max steps: 2,
  - `--max-markets 50`,
  - candidates loaded: 256,
  - candidates processed: 50,
  - markets filtered before RAG: 206,
  - RAG attempted/succeeded/failed: 10/10/0,
  - LLM attempted/succeeded/failed/fallback: 10/10/0/0,
  - BLF attempted/succeeded/failed/fallback: 2/2/0/0,
  - hypothetical intents: 0,
  - trade count: 0,
  - submit called: false,
  - plan persisted, participant finalized, tick completed.
- Expanded threshold-debug inspection results:
  - top apparent edges were still negative:
    `kalshi:KXG7LEADEROUT-45JAN01-MCAR` around -0.0030,
    `kalshi:KXGREENLANDPRICE-29JAN21-1049B` around -0.0035,
    `kalshi:KXALIENS-27-28` around -0.0037.
  - relaxed threshold scenarios still produced zero trades in 12 cases for
    edge thresholds 0.08/0.06/0.04 across evidence-quality and BLF modes.
  - future-candidacy prompt behavior improved: scanned 2028 candidacy markets
    no longer collapsed to raw 0.02 solely due to lack of current announcement
    evidence; examples stayed near market-prior-neutral values such as 0.50 or
    moderate values.
  - no absence-as-negative guardrail triggered in this live run, which suggests
    the prompt guidance was sufficient for scanned long-horizon candidates.
  - top BLF changes remained modest: Deepika / White Lotus moved p_final by
    about 0.0058; Democratic Senate control moved p_final by about 0.0026.
- Prepared 4-tick / 1-hour diagnostic dry-run mode:
  - Added `scripts/run_four_tick_diagnostic.py`.
  - The script wraps the existing single-tick runner four times with enforced
    dry-run, threshold-debug, RAG, LLM RAG, and minimal BLF settings.
  - It forces `EDGE_TRADER_ALLOW_LIVE_SUBMIT=0` and aborts unsafe configs.
  - It can continue after one tick failure with `--continue-on-error`.
  - It reads each persisted plan from `.edge_trader/memory.jsonl`, prints a
    compact per-tick summary, and writes aggregate JSON + Markdown reports to
    `outputs/diagnostics/`.
  - Aggregate reports include top edges, RAG-market disagreements, BLF
    adjustments, relaxed-threshold summaries, market-type distribution,
    repeated-market p_final movement, and latency/cost diagnostics.
  - Added mocked tests for four-iteration execution, continue-on-error,
    live-submit safety gate, aggregate sums, unsafe submit detection,
    repeated-market tracking, and Markdown report sections.

## Immediate Next Steps

1. Run 4 consecutive diagnostic ticks with RAG/LLM/BLF enabled and
   threshold-debug on:
   `python scripts/run_four_tick_diagnostic.py --ticks 4 --sleep-seconds 900 --max-markets 50 --continue-on-error`
2. Inspect aggregate edge distribution across the generated JSON and Markdown
   reports.
3. If all four ticks still show no positive edge, conclude that this
   configuration is correctly conservative and that edge opportunities are
   sparse.
4. Only consider tiny live submit after a natural positive edge appears under
   strict checks.
5. Keep threshold tuning, K=3 BLF, SELL exits, and calibration after the
   4-tick diagnostic.
