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

Status: pending

- Maintain structured belief state.
- Run short sequential update loop for top markets only.
- Stop on deadline pressure, strong/weak edge, or max steps.
- Run K=3 trials by default.

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

## Immediate Next Steps

1. Inspect C plan records for LLM summary quality, natural-language risk flags,
   probability shrinkage, and resolution-risk consistency.
2. Normalize or constrain LLM-provided `risk_flags` if inspection shows noisy
   free-text flags.
3. Tune thresholds only after RAG + LLM evidence quality is acceptable.
4. Run one threshold-tuning dry-run with hypothetical intents enabled but live
   submit disabled.
5. Only after multiple clean dry-runs, run tiny real submit with max one trade
   and minimum size.
6. Implement minimal BLF verifier after the real RAG + LLM pipeline is stable.
