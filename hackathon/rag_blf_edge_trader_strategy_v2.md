# RAG-BLF Edge Trader Strategy

## Goal

Build a Prophet Arena trading-track bot that combines stronger forecasting with a real trading policy.

The bot should improve on two dimensions:

1. **Forecasting alpha**: estimate a better fair probability for each market.
2. **Trading execution**: convert probability edges into profitable, risk-controlled trades under Prophet Arena constraints.

Forecasting sources:

1. **2402-style batch RAG forecasting**: broad retrieval, relevance filtering, summarization, one-shot probability estimation, and ensemble aggregation.
2. **2604-style BLF / Bayesian Linguistic Forecaster**: sequential belief-state updating, targeted search, multi-trial aggregation, and calibration/shrinkage.
3. **Statistical + judgmental hybrid reasoning**: combine quantitative priors such as market price, base rates, historical analogues, time-to-resolution, spread, liquidity, category behavior, and price movement with LLM-based judgmental reasoning from news/evidence.

The purpose is not simply to improve Brier score. The trading track requires converting forecasts into profitable trades under tick deadlines, position limits, exposure caps, daily trade limits, and market prices.

Core idea:

```text
2402 = wide, cheap market scanner
2604 = expensive, selective edge verifier
statistical layer = market/base-rate/price-action prior
trading layer = conservative position sizing + risk control + exit logic
```

---

## Current engineering direction

This strategy will be implemented as an independent bot package outside the
`ai-prophet` repo, tentatively named `edge_trader_bot`.

The bot will use `ai_prophet_core.arena.BenchmarkSession` directly instead of
modifying the existing `prophet trade` CLI pipeline. The existing CLI pipeline is
useful as a reference for tick orchestration and memory, but it is too rigid for
this strategy because it has a fixed `review -> search -> forecast -> action`
flow and does not naturally support deterministic risk controls, explicit
`SELL` exits, multi-trial BLF verification, or deadline-aware fallback logic.

Implementation principle:

```text
LLM/RAG layer estimates probabilities and evidence quality.
Trading layer deterministically decides whether to trade, size, add, reduce, or exit.
```

The first production path is deliberately conservative:

1. Build a working MVP that can claim ticks, load markets, load portfolio, write
   plan JSON, submit intents, finalize, and complete ticks.
2. Add deterministic filters and a market-implied/statistical prior.
3. Add a 2402-inspired RAG scanner as a broad market ranking tool.
4. Add a 2604-inspired BLF verifier only for the highest-ranked markets.
5. Add multi-trial aggregation, shrinkage toward market price, and position-aware
   exits after the basic bot is stable.

### How the papers map to implementation

The 2402 code in `references/llm_forecasting` should be treated as a reference,
not copied wholesale. Its useful pieces are:

- LLM search query generation.
- Relevance filtering / reranking.
- Article summarization.
- Multiple reasoning prompts.
- Forecast aggregation.

For live trading, the implementation should simplify this into a
`rag_scanner.py` module that emits structured JSON:

```json
{
  "market_id": "...",
  "p_2402": 0.57,
  "evidence_quality": 4,
  "confidence": "medium",
  "evidence_for_yes": [],
  "evidence_for_no": [],
  "open_questions": [],
  "resolution_check": {},
  "risk_flags": []
}
```

The 2604 BLF paper has no code, so we implement the core abstraction rather than
try to exactly reproduce the paper:

```text
belief_state = {
    p,
    confidence,
    evidence_for_yes,
    evidence_for_no,
    open_questions,
    main_risk,
    next_best_search
}
```

The BLF verifier runs a short sequential loop. At each step, it identifies the
single most decision-relevant uncertainty, searches or reads targeted evidence,
updates the belief state, and either continues or submits. In live mode, use
`max_steps = 3 to 4` and `K = 3` trials by default. Use `K = 5` only for
exceptional apparent edges with enough deadline buffer.

### MVP scope

The first implementation is not expected to outperform immediately. It must
first be robust:

- Most markets default to `HOLD`.
- If RAG or BLF fails, fall back to cached beliefs or `HOLD`.
- Never start expensive verification if the deadline buffer is too small.
- Never let LLM output directly control position size.
- Always write an audit plan before finalizing the tick.

---

## High-level architecture

```text
Every 15-minute tick:

1. claim_tick()
2. load_candidates()
3. get_portfolio()
4. update cross-tick memory and existing positions
5. filter markets with deterministic rules
6. run 2402-lite scanner on remaining markets
7. add statistical priors and market-implied prior
8. rank by apparent edge and risk-adjusted score
9. run 2604-style BLF verifier only on top markets
10. aggregate / calibrate / shrink probabilities
11. convert edge into BUY / SELL / HOLD decisions
12. sort intents by expected value
13. put_plan() with audit JSON
14. submit_intents()
15. finalize()
16. complete_tick()
```

The bot should trade selectively. Most markets should be `HOLD`.

The default decision should be no trade unless the forecast edge is large, robust, and executable.

---

## Important Prophet Arena constraints

From the trading docs:

- Ticks occur every 15 minutes.
- Submission deadline is 9 minutes after the tick timestamp.
- Starting cash is `$10,000`.
- Max trades per tick: `20`.
- Max trades per rolling day: `100`.
- Max open positions: `30`.
- Max notional per market: `$1,000`.
- Max gross exposure: `$10,000`.
- Max intents per tick request: `50`, but only the first `20` fills can execute.
- Fee rate: `0.0`.
- `BUY YES` fills at `best_ask`.
- `BUY NO` fills at `1 - best_bid`.
- `SELL YES` effectively exits YES exposure at `best_bid`.
- `SELL NO` effectively exits NO exposure at `1 - best_ask`.
- Positions are keyed by `(market_id, side)`.
- You cannot hold both YES and NO on the same market.
- Buying the opposite side reduces the existing position rather than flipping it. Any remaining shares beyond the current position may be dropped instead of opening the opposite side.

These rules imply that the bot should preserve trade budget and capital for high-conviction opportunities.

Position handling must be explicit and portfolio-aware.

---

## Live tick time budget

The server gives 15-minute ticks, but the bot must submit within 9 minutes.

Recommended live budget:

```text
0:00 - 0:30   claim tick, load candidates, load portfolio
0:30 - 1:30   deterministic filtering, memory lookup, existing-position checks
1:30 - 3:30   2402-lite scanner for remaining markets
3:30 - 4:00   statistical priors, market priors, edge ranking
4:00 - 7:30   BLF verifier only for top markets
7:30 - 8:15   aggregation, shrinkage, sizing, exit checks
8:15 - 8:45   put_plan(), submit_intents(), finalize()
8:45 - 9:00   safety buffer
```

Implementation rule:

```text
Never start expensive BLF verification if less than 90 seconds remain before the submission deadline.
```

If the pipeline is running late, fall back to cached beliefs and high-threshold trades only.

---

## Forecasting engine: probability stack

For each market, estimate a fair probability of YES using several signals.

```text
p_market = market-implied probability from bid/ask midpoint
p_stat   = statistical prior / category prior / base-rate estimate
p_2402   = batch RAG rough forecast
p_blf    = sequential BLF verified forecast
p_final  = calibrated and shrunk trading probability
```

A useful logit-space combination:

```text
logit(p_raw) = w0 * logit(p_market)
             + w1 * logit(p_stat)
             + w2 * logit(p_2402)
             + w3 * logit(p_blf)
```

If there is not enough data to train weights, use rule-based weights.

Suggested weights:

| Market type | Weighting idea |
|---|---|
| Fresh news event | Higher `p_2402` and `p_blf` weight |
| Financial/time-series threshold | Higher `p_stat` and `p_market` weight |
| Election / sports / high-liquidity market | Higher `p_market` and statistical prior weight |
| Ambiguous resolution criteria | Strong shrink toward `p_market` |
| Official source found | Increase `p_blf` weight |
| Thin evidence / low information | Shrink heavily toward `p_market` |

This is the main forecasting innovation beyond the two papers: the bot does not only do judgmental LLM forecasting. It blends judgmental reasoning with statistical and market-implied priors.

---

## Core trading logic

For each market, estimate a fair probability:

```text
p_model = hybrid forecast probability for YES
market_mid = approximate market-implied YES probability
p_final = calibrated/shrunk trading probability for YES
yes_buy_price = best_ask
no_buy_price = 1 - best_bid
```

Expected buy edges:

```text
edge_yes = p_final - yes_buy_price
edge_no  = (1 - p_final) - no_buy_price
```

Trade only if edge clears a conservative threshold:

```text
if edge_yes > threshold:
    BUY YES
elif edge_no > threshold:
    BUY NO
else:
    HOLD
```

Recommended starting threshold:

```text
threshold = max(0.06, 1.5 * model_uncertainty)
```

Use a higher threshold for ambiguous markets, low-quality evidence, wide spreads, stale information, high correlation with existing positions, or categories where the model historically performs poorly.

---

## Trading engine: financial strategy layer

The trading layer should solve a different problem from forecasting.

Forecasting asks:

```text
What is the fair probability?
```

Trading asks:

```text
Is this edge large enough to spend scarce capital, exposure, and trade budget?
```

Important trading ideas:

1. **Selective trading**: no trade is the default.
2. **Expected value**: buy only when probability edge exceeds uncertainty and execution costs.
3. **Fractional-Kelly-style sizing**: larger edge and higher confidence justify larger size, but full Kelly is too aggressive for LLM forecasts.
4. **Risk-adjusted ranking**: prefer high edge per unit uncertainty, not raw edge alone.
5. **Trade budget management**: daily trades are scarce.
6. **Portfolio diversification**: avoid stacking correlated positions.
7. **Exit logic**: sell when edge compresses, thesis breaks, or capital is needed.
8. **Resolution-criteria arbitrage**: exploit markets where headlines differ from exact resolution rules.

---

## Stage 1: deterministic market filter

Before using any LLM, remove bad markets.

Suggested filters:

| Filter | Reason |
|---|---|
| Already at or near `$1,000` notional | Avoid server cap rejection |
| Already holding opposite side | Avoid accidental netting behavior |
| Spread too wide | Apparent edge may be fake |
| Insufficient quote data | Cannot calculate reliable trade price |
| Resolution criteria unclear | LLMs often misprice ambiguous rules |
| Too many open positions | Preserve max-open-position budget |
| Too many trades already today | Preserve daily fill budget |
| Low-confidence categories | Avoid systematic model weakness |
| Highly correlated with existing exposure | Avoid hidden portfolio concentration |
| Market likely to vanish from eligible universe | Avoid mark-to-market blind spots |

Implementation sketch:

```python
def deterministic_filter(market, portfolio, daily_trade_count) -> tuple[bool, list[str]]:
    reasons = []

    quote = market.quote
    best_bid = float(quote.best_bid) if quote.best_bid is not None else None
    best_ask = float(quote.best_ask) if quote.best_ask is not None else None

    if best_bid is None or best_ask is None:
        reasons.append("missing_quote")
        return False, reasons

    spread = best_ask - best_bid
    if spread > 0.20:
        reasons.append("spread_too_wide")

    if daily_trade_count >= 80:
        reasons.append("daily_trade_budget_low")

    if portfolio.open_positions_count >= 25:
        reasons.append("too_many_open_positions")

    # Add position side, category, correlation, and exposure checks here.

    return len(reasons) == 0, reasons
```

---

## Stage 2: statistical prior layer

Before expensive LLM reasoning, compute cheap statistical and market-based signals.

Possible statistical signals:

| Signal | Meaning |
|---|---|
| `market_mid` | Crowd / market-implied probability |
| `spread` | Liquidity and uncertainty proxy |
| `time_to_close` | Urgency and catalyst proximity |
| `price_change_since_last_tick` | Possible new information or overreaction |
| `category_base_rate` | Historical frequency for similar markets |
| `similar_market_outcomes` | Empirical prior from previous markets |
| `volume/liquidity proxy` | Confidence in market price signal |
| `deadline structure` | Event-by-date markets often have asymmetric resolution dynamics |

Market-implied probability:

```python
def market_mid(best_bid: float, best_ask: float) -> float:
    return (best_bid + best_ask) / 2
```

Example statistical prior:

```text
p_stat = weighted blend of:
    market_mid,
    category_base_rate,
    similar_question_base_rate,
    price_momentum_adjustment,
    time_to_resolution_adjustment
```

Important rule:

```text
Statistical signals should anchor the LLM, not blindly override it.
```

---

## Stage 3: 2402-lite batch RAG scanner

Purpose: cheaply scan many markets and estimate rough probabilities.

For each candidate market that passes the deterministic filter:

1. Parse question, title, description, rules, close time, and outcomes.
2. Generate direct search queries.
3. Generate decomposed sub-question queries.
4. Retrieve recent articles/pages/snippets.
5. Relevance-rank the evidence.
6. Summarize the most useful evidence.
7. Produce a rough probability `p_2402` and evidence-quality score.
8. Explicitly check the resolution criteria against the evidence.

The scanner should output structured JSON, not prose.

Example output:

```json
{
  "market_id": "...",
  "question": "...",
  "market_mid": 0.42,
  "p_stat": 0.45,
  "p_2402": 0.57,
  "confidence": "medium",
  "evidence_quality": 4,
  "base_rate": "...",
  "evidence_for_yes": ["..."],
  "evidence_for_no": ["..."],
  "open_questions": [
    "Has the official source updated after the latest article?",
    "Does the resolution criterion require a stricter condition than the headline suggests?"
  ],
  "resolution_check": {
    "headline_matches_resolution": false,
    "deadline_satisfied": "unknown",
    "official_source_required": true,
    "risk": "possible headline/resolution mismatch"
  },
  "risk_flags": ["possible_resolution_ambiguity"]
}
```

The 2402-lite scanner should not be allowed to trade directly except in very obvious cases. Its main job is ranking.

---

## Stage 4: edge ranking

Only send the best few markets to the expensive BLF verifier.

Risk-adjusted ranking score:

```text
score = abs(p_2402 - market_mid)
        * evidence_quality_multiplier
        * liquidity_multiplier
        * time_to_resolution_multiplier
        * resolution_clarity_multiplier
        - ambiguity_penalty
        - exposure_penalty
        - category_penalty
        - correlation_penalty
        - daily_trade_budget_penalty
```

Alternative Sharpe-like score:

```text
score = expected_edge / estimated_uncertainty
```

Practical live rule:

```text
Run BLF verifier on at most top 3-5 markets per tick by default.
Use 5-8 only in burst mode when apparent edge is exceptional and time remains.
```

This keeps the system inside the 9-minute deadline and protects the rolling daily trade limit.

---

## Stage 5: 2604-style BLF edge verifier

Purpose: decide whether the apparent edge from 2402 survives deeper investigation.

Initialize the BLF belief state using the 2402 evidence package and statistical prior.

Initial belief state:

```json
{
  "p": 0.57,
  "market_mid": 0.42,
  "p_stat": 0.45,
  "confidence": "medium",
  "evidence_for_yes": ["..."],
  "evidence_for_no": ["..."],
  "base_rate": "...",
  "open_questions": ["..."],
  "resolution_check": {...},
  "risk_flags": ["..."],
  "next_best_search": null
}
```

At each BLF step:

```text
1. Read current belief state.
2. Identify the single most decision-relevant uncertainty.
3. Search or read only for that uncertainty.
4. Update belief state.
5. Re-check whether evidence satisfies the exact resolution criteria.
6. Stop if:
   - edge is strong and evidence is sufficient,
   - edge disappears,
   - risk flags become too large,
   - max_steps reached,
   - deadline safety buffer is approaching.
```

Recommended max steps:

```text
max_steps = 3 to 5 during live trading
max_steps = 8 to 10 only for offline experiments
```

The BLF verifier should produce:

```json
{
  "market_id": "...",
  "p_blf": 0.61,
  "confidence": "high",
  "model_uncertainty": 0.04,
  "edge_survives": true,
  "best_side": "YES",
  "main_reason": "...",
  "main_risk": "...",
  "resolution_criteria_satisfied": "likely",
  "updated_evidence_for_yes": ["..."],
  "updated_evidence_for_no": ["..."]
}
```

---

## Multi-trial aggregation

LLM forecasts have high run-to-run variance. For top markets, run multiple independent trials.

Recommended live setting:

```text
K = 3 trials for normal top markets
K = 5 trials only for very high apparent edge
```

Aggregate in logit space:

```python
import math

def logit(p: float) -> float:
    eps = 1e-4
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def logit_mean(probs: list[float]) -> float:
    return sigmoid(sum(logit(p) for p in probs) / len(probs))
```

If trials disagree strongly, shrink toward market or reduce size:

```python
def trial_uncertainty(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.10
    mean = sum(probs) / len(probs)
    var = sum((p - mean) ** 2 for p in probs) / (len(probs) - 1)
    return var ** 0.5


def shrink_probability(p_model: float, anchor: float, confidence: str) -> float:
    if confidence == "high":
        w = 0.55
    elif confidence == "medium":
        w = 0.35
    else:
        w = 0.20
    return w * p_model + (1 - w) * anchor
```

For prediction-market trading, `anchor` should usually be `market_mid`, not `0.5`, because market price is a strong crowd prior.

---

## Calibration and market-aware shrinkage

Do not trade directly from raw LLM probability.

Recommended pipeline:

```text
p_market = market_mid
p_stat = statistical prior
p_2402 = batch RAG rough probability
p_trials = [p_blf_trial_1, p_blf_trial_2, p_blf_trial_3]
p_blf_agg = logit_mean(p_trials)
p_raw = combine_market_stat_rag_blf(p_market, p_stat, p_2402, p_blf_agg)
p_final = shrink_probability(p_raw, market_mid, confidence)
```

If we collect enough validation data, replace manual shrinkage with learned calibration:

```text
p_final = hierarchical_or_global_platt_scaling(p_raw, source/category)
```

Before enough data exists, manual shrinkage is safer.

---

## Position sizing

Convert edge into shares conservatively.

Starting table:

| Edge after shrinkage | Shares |
|---:|---:|
| `< 0.04` | 0 |
| `0.04 - 0.07` | 5-10 |
| `0.07 - 0.12` | 10-25 |
| `0.12 - 0.20` | 25-50 |
| `> 0.20` | 50-100 only if high confidence |

Hard caps:

```text
max_new_notional_per_trade = 100 to 250
max_notional_per_market = 600 normally, 1000 only for very high confidence
max_trades_per_tick_default = 0 to 3
max_trades_per_tick_burst = 5 to 8 only for exceptional edges
keep at least 20-30 daily trades in reserve for later ticks
max_open_positions_target = 20 to 25, below server max of 30
```

Fractional-Kelly-style intuition:

```text
For a YES contract with cost c and probability p:
edge = p - c
rough_kelly_fraction ≈ edge / (1 - c)
use only 10% to 25% of rough Kelly because LLM probabilities are noisy
```

Sizing function sketch:

```python
def size_trade(edge: float, price: float, confidence: str, remaining_trade_budget: int) -> int:
    if edge < 0.04:
        return 0
    if confidence == "low" and edge < 0.08:
        return 0

    if edge < 0.07:
        dollars = 50
    elif edge < 0.12:
        dollars = 100
    elif edge < 0.20:
        dollars = 175
    else:
        dollars = 250 if confidence == "high" else 150

    if remaining_trade_budget < 20:
        dollars *= 0.5

    shares = int(dollars / max(price, 0.01))
    return max(shares, 0)
```

---

## Daily trade budget management

The rolling daily trade limit is only `100`, while there may be many ticks per day.

Use a dynamic trade quota:

```text
remaining_trade_budget = 100 - trades_used_rolling_day
remaining_ticks_estimate = estimated ticks left in the current run/day
quota_per_tick = remaining_trade_budget / max(remaining_ticks_estimate, 1)
```

Rules:

```text
Default to 0-3 trades per tick.
Use burst mode only when risk-adjusted edge is very high.
If remaining_trade_budget < 20, raise thresholds and halve position size.
If remaining_trade_budget < 10, trade only exceptional edges.
Always sort intents by expected value before submission.
```

---

## Position-aware execution and exit logic

The bot should not only buy. It should also reduce or exit positions.

Before submitting a BUY:

```text
1. Check whether the same market already has a position.
2. If holding the same side, treat the trade as adding exposure.
3. If holding the opposite side, do not assume BUY flips the position.
4. Either skip, explicitly reduce/SELL existing side, or wait until flat.
```

Exit rules for held YES:

```text
current_exit_price = best_bid
current_edge_to_hold = p_final - current_exit_price

if thesis_broken:
    SELL YES fully
elif current_edge_to_hold < exit_threshold:
    SELL YES partially or fully
elif exposure_needed_for_better_trade:
    SELL weakest positions
```

Exit rules for held NO:

```text
current_exit_price = 1 - best_ask
current_edge_to_hold = (1 - p_final) - current_exit_price

if thesis_broken:
    SELL NO fully
elif current_edge_to_hold < exit_threshold:
    SELL NO partially or fully
elif exposure_needed_for_better_trade:
    SELL weakest positions
```

Suggested exit threshold:

```text
exit_threshold = 0.01 to 0.03
```

Selling can improve capital efficiency and prevent stale positions from consuming exposure.

---

## Resolution-criteria arbitrage

A major LLM-friendly edge is finding cases where the market misunderstands its own resolution criteria.

Add a dedicated checker:

```text
Does the evidence actually satisfy the resolution criteria?
Is the date/time before the deadline?
Does the market require official confirmation?
Does it require announcement, implementation, effectiveness, publication, or finalization?
Is the evidence only a headline, rumor, plan, proposal, or expectation?
Does the market resolve according to a specific source?
```

This should run in both 2402-lite and BLF.

Important trading rule:

```text
If the headline is bullish but the resolution criterion is stricter, do not buy unless the strict criterion is likely satisfied.
```

This can generate real alpha because many participants trade headlines rather than contract text.

---

## Price movement and catalyst handling

Track price movement across ticks.

Signals:

```text
price_change = current_market_mid - previous_market_mid
spread_change = current_spread - previous_spread
```

Use price movement as a trigger, not as a standalone trade signal.

Rules:

```text
Large price move + fresh evidence found:
    allow BLF refresh and possibly trade with trend

Large price move + no evidence found:
    flag possible overreaction, but do not blindly fade

Market approaching known catalyst:
    increase refresh priority, but raise uncertainty buffer
```

Catalysts include:

```text
earnings dates
court rulings
elections
sports matches
Fed/CPI/jobs reports
scheduled announcements
deadline-driven events
```

---

## Portfolio diversification and correlation control

Avoid concentrating in many correlated markets.

Examples of correlated exposure:

```text
same candidate / election outcome
same company or sector
same macro event
same sports team/game
same crypto/stock price regime
same geopolitical conflict
```

Simple implementation:

```text
category = LLM_or_rule_based_classify(market)
if category_exposure > category_cap:
    reduce size or skip
```

Suggested caps:

```text
max_category_notional = 1500 to 2500
max_single_theme_positions = 3 to 5
```

---

## Cross-tick memory

Markets repeat across ticks. Do not recompute everything from scratch.

Cache by `market_id`:

```json
{
  "market_id": "...",
  "last_seen_tick": "...",
  "last_market_mid": 0.42,
  "last_spread": 0.05,
  "last_p_stat": 0.45,
  "last_p_2402": 0.57,
  "last_p_blf": 0.64,
  "last_p_final": 0.61,
  "evidence_package": {...},
  "belief_state": {...},
  "open_questions": [...],
  "last_search_time": "...",
  "position": {...},
  "decision_history": []
}
```

On a later tick, update only if:

```text
1. market price moved materially,
2. spread/liquidity changed materially,
3. new evidence is likely available,
4. position/exposure changed,
5. the market approaches close/resolution/catalyst,
6. previous belief state had unresolved high-value questions.
```

Suggested implementation: start with JSONL or SQLite.

---

## Plan JSON for dashboard / debugging

Use `session.put_plan(...)` to save audit information. The server does not require it, but it is useful for debugging.

Example plan:

```json
{
  "strategy": "rag_blf_edge_trader",
  "version": "0.2",
  "tick_id": "...",
  "markets_considered": 120,
  "markets_after_filter": 35,
  "markets_scanned_2402": 35,
  "markets_verified_blf": 5,
  "trade_count": 3,
  "daily_trades_remaining": 72,
  "decisions": [
    {
      "market_id": "...",
      "market_mid": 0.42,
      "p_stat": 0.45,
      "p_2402": 0.57,
      "p_blf": 0.64,
      "p_final": 0.55,
      "side": "YES",
      "price": 0.46,
      "edge": 0.09,
      "shares": 20,
      "confidence": "medium",
      "action_type": "new_position",
      "reason": "...",
      "main_risk": "...",
      "risk_flags": []
    }
  ],
  "skipped": [
    {
      "market_id": "...",
      "reasons": ["spread_too_wide", "resolution_ambiguous"]
    }
  ]
}
```

---

## Recommended file structure

```text
bot/
  runner.py              # tick lifecycle using BenchmarkSession
  config.py              # thresholds, model settings, risk limits
  market_filter.py       # deterministic filters
  stat_priors.py         # market prior, base rates, price movement, category priors
  rag_scanner.py         # 2402-lite batch retrieval scanner
  belief_agent.py        # 2604-style BLF verifier
  resolution_checker.py  # contract/rule interpretation and headline mismatch checks
  prompts.py             # prompt templates
  aggregator.py          # logit mean, shrinkage, calibration
  trading_policy.py      # BUY/SELL/HOLD logic and intent ranking
  sizer.py               # edge -> shares
  portfolio_risk.py      # exposure, correlation, daily trade budget
  memory.py              # SQLite/JSONL cross-tick cache
  schemas.py             # Pydantic models for structured outputs
  logging_utils.py       # JSONL logs and summaries
  backtest.py            # offline parameter sweeps
```

---

## MVP roadmap

### v1: deterministic + statistical prior + 2402-lite scanner

Goal: get a working bot quickly.

Features:

- Tick lifecycle works.
- Loads candidates and portfolio.
- Filters bad markets.
- Computes market midpoint, spread, and cheap statistical priors.
- Runs 2402-lite scanner.
- Trades only if edge is large, e.g. `> 0.10`.
- Fixed small sizing.
- Writes plan JSON.

### v2: add BLF verifier for top markets

Features:

- Rank markets by apparent edge.
- Run BLF only on top 3-5 by default.
- Max 3-5 steps per market.
- Reject trades when edge disappears.
- Stop BLF early when deadline buffer is approaching.

### v3: add multi-trial aggregation and market-aware shrinkage

Features:

- Run 3 trials for top markets.
- Use logit mean.
- Shrink toward market mid.
- Increase confidence only when trials agree.

### v4: add position-aware sizing, exits, and memory

Features:

- Avoid overexposure.
- Avoid accidental opposite-side netting.
- Add SELL logic for stale or broken positions.
- Reuse evidence across ticks.
- Update only when market price, spread, catalyst timing, or evidence changes.
- Preserve daily trade budget.

### v5: offline sweeps

Tune:

- edge threshold,
- shrinkage weight,
- statistical prior weights,
- max markets verified per tick,
- max steps per BLF,
- position sizing table,
- exit threshold,
- category blacklist/whitelist,
- correlation caps,
- daily trade budget rules.

---

## Pseudocode for main loop

```python
def run_tick(session, participant_idx, memory):
    lease = session.claim_tick()
    if not lease.available:
        return

    tick = session.load_candidates(lease)
    portfolio = session.get_portfolio(participant_idx)

    candidates = []
    for market in tick.candidates.markets:
        ok, reasons = deterministic_filter(market, portfolio, memory.daily_trade_count())
        if ok:
            candidates.append(market)
        else:
            memory.log_skip(market.market_id, reasons)

    scanned = []
    for market in candidates:
        cached = memory.get_market(market.market_id)
        p_stat = compute_statistical_prior(market, cached, portfolio)

        if cached and not should_refresh(market, cached):
            scan = cached["last_scan"]
            scan["p_stat"] = p_stat
        else:
            scan = rag_scanner_2402_lite(market, p_stat=p_stat)
            scan = resolution_checker(scan, market)
            memory.save_scan(market.market_id, scan)
        scanned.append(scan)

    top_edges = rank_for_blf(scanned, portfolio, memory.daily_trade_count())[:5]

    decisions = []
    for scan in top_edges:
        if deadline_buffer_too_small():
            break

        trials = []
        k = 3
        for _ in range(k):
            if deadline_buffer_too_small():
                break
            result = blf_verify(scan, max_steps=4)
            trials.append(result["p_blf"])

        if trials:
            p_blf_agg = logit_mean(trials)
        else:
            p_blf_agg = scan["p_2402"]

        p_raw = combine_market_stat_rag_blf(
            market_mid=scan["market_mid"],
            p_stat=scan["p_stat"],
            p_2402=scan["p_2402"],
            p_blf=p_blf_agg,
        )

        p_final = shrink_probability(
            p_model=p_raw,
            anchor=scan["market_mid"],
            confidence=infer_confidence(trials, scan),
        )

        decision = make_trade_or_exit_decision(scan, p_final, portfolio)
        decisions.append(decision)

    intents = build_trade_intents(
        decisions,
        max_trades_default=3,
        max_trades_burst=8,
    )

    intents = sort_by_expected_value(intents)

    plan = build_plan_json(scanned, decisions, intents)
    session.put_plan(lease, participant_idx, plan)

    if intents:
        session.submit_intents(lease, participant_idx, intents)

    session.finalize(lease, participant_idx)
    session.complete_tick(lease)
```

---

## Main prompts

### 2402-lite scanner prompt

```text
You are a prediction-market research assistant.

Given a binary market, produce a structured evidence package and rough probability.

Focus on:
1. resolution criteria,
2. latest relevant evidence,
3. base rates and statistical priors,
4. evidence for YES,
5. evidence for NO,
6. missing information,
7. risk flags,
8. rough probability of YES,
9. whether the evidence truly satisfies the resolution criteria.

Return JSON only.
```

### BLF verifier prompt

```text
You are a sequential Bayesian forecasting agent.

You maintain a belief state for a binary prediction market.
At each step, update the probability of YES based on the newest observation.
Do not simply append evidence. Compress it into the belief state.

At each step:
1. identify the most decision-relevant unresolved uncertainty,
2. choose one search/read action,
3. update the belief state,
4. check the exact resolution criteria,
5. decide whether to continue or submit.

Return JSON only.
```

### Resolution checker prompt

```text
You are a prediction-market contract interpreter.

Given the market question, background, resolution criteria, deadline, and evidence,
determine whether the evidence actually satisfies the contract.

Distinguish between:
- announced vs implemented,
- expected vs confirmed,
- proposed vs finalized,
- reported by media vs official source,
- before deadline vs after deadline,
- partial fulfillment vs exact fulfillment.

Return JSON only.
```

---

## Strategy summary

The winning strategy is not to make the LLM forecast every market. It is to use the LLM as a selective research engine and combine it with a real trading policy.

Final design:

```text
RAG-BLF Edge Trader

Statistical layer:
    market-implied prior + base rates + price movement + category priors

2402-style batch RAG:
    broad evidence coverage + rough mispricing signal

2604-style BLF:
    targeted sequential belief updates only on top edges

Trading layer:
    market-aware shrinkage + conservative thresholds + position sizing + exits

Risk layer:
    daily trade budget + exposure caps + correlation control + deadline safety

Memory:
    reuse beliefs across 15-minute ticks
```

The bot should be selective, position-aware, deadline-safe, and risk-aware.
