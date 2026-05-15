# Prophet Hacks 2026 — Agent Briefing Document

**Purpose**: This document is the complete context package for an agent tasked with developing a state-of-the-art AI forecasting system at a 36-hour hackathon. Read every section before writing a single line of code.

---

## 1. Repository Map

```
Prophet_Hacks_2026/
├── Development/                          ← YOUR WORKING DIRECTORY
│   └── AGENT_BRIEF.md                    ← This file
│
├── Reference/                            ← Academic papers (read these first)
│   ├── 2402.18563v1.pdf 2604.18576v2.pdf ← BLF paper (primary target architecture)
│   └── FILE_4718.pdf                     ← Halawi et al. 2024 (baseline architecture)
│
└── AI Workshop Code/                     ← Reference implementation code
    ├── AGENTS.md                         ← Agent session protocol (bd/beads workflow)
    ├── README.md                         ← Workshop overview and reading list
    ├── lecture_1/notebooks/
    │   └── openrouter_utils.py           ← OpenRouter REST wrapper (reuse this)
    ├── lecture_2/notebooks/
    │   └── resume_screening.ipynb        ← Vertical-slice MVP pattern
    ├── lecture_3/notebooks/
    │   └── resume_utils.py               ← structured_llm_call() pattern
    └── lecture_4/notebooks/
        ├── agent_utils.py                ← Agent loop + TOOL_REGISTRY pattern (reuse this)
        └── resume_utils.py               ← structured_llm_call() (same as lecture_3)
```

**Development/ is empty.** All new code goes here.

---

## 2. Papers: What You Need to Know

### Paper 1 — BLF: Bayesian Linguistic Forecaster (arXiv:2604.18576v2, Apr 2026)
*Your primary architectural blueprint. This is SOTA on ForecastBench.*

**What it does**: Agentic binary forecasting system achieving Brier Index 73.8 on ForecastBench, beating Cassi, GPT-5, Grok 4.20, and Foresight-32B.

**Three core innovations to implement:**

**1. Linguistic Belief State** (most important — worth ~3.0 BI points)
- At each agent step, the LLM produces both an action AND an updated `belief_state` JSON object
- Belief state contains: `probability` (float 0–1), `confidence_level`, `key_evidence_for`, `key_evidence_against`, `open_questions`
- This is a semi-structured representation — NOT raw text accumulation
- The belief state is passed forward as the structured prior, replacing ever-growing context
- This is approximate sequential Bayesian inference

**2. Multi-Trial Aggregation** (worth ~1.5 BI points)
- Run K=5 independent trials per question
- Aggregate using logit-space mean: `p̂ = σ(α · (1/K) Σ logit(pₖ))`
- Also explore LOO-tuned shrinkage α < 1 toward p=0.5 (empirical Bayes)
- Simple arithmetic mean is the fallback

**3. Hierarchical Calibration** (worth additional BI improvement)
- Apply Platt scaling with per-source intercept offsets δₛ (L2-regularized)
- Avoids global Platt over-shrinking extreme predictions from skewed-base-rate sources
- Use LOO cross-validation to fit

**Agent Loop (Algorithm 1):**
```
Input: question q, cutoff_date d, max_steps T=10
b₀ = {probability: 0.5}; m₀ = [q]
for t = 1..T:
    (action, belief) = LLM(message_history m_{t-1})
    if action == submit(p): return p
    observation = execute_action(action, cutoff_date_filter=d)
    if action == web_search: observation = leak_filter(observation, d)
    m_t = m_{t-1} + [(action, observation, belief)]
Force submit if loop exits
```

**Available tools** (the agent selects one per step):
- `web_search(query)` — with automatic date-based leak filtering
- `summarize_results(url)` — filter and summarize retrieved pages
- `lookup_url(url)` — fetch a specific URL
- `fetch_ts_yfinance(ticker, start, end)` — time-series data for financial questions
- `fetch_wikipedia_section(title, section)` — Wikipedia data
- `submit(probability)` — terminate and return final forecast

**Key ablation findings (use these to prioritize)**:
- Zero-shot baseline: BI=0 (no search, no tools)
- Adding web search: +4.6 BI
- Sequential (vs batch) search: +3.8 BI
- Linguistic belief state: +3.0 BI
- Multi-trial aggregation: +1.5 BI
- Adding crowd signal (market price): +7.1 BI on market questions
- Model: Gemini-3.1-Pro base, but system is model-agnostic

**Problem type**: Binary prediction — estimate P(Y(r)=1 | data ≤ f) for question q with forecast date f and resolution date r.

**Metric**: Brier Index = 100×(1 − √BS), where BS = mean(p_i − o_i)². Higher is better. 50 = always predict 0.5.

---

### Paper 2 — Halawi et al. 2024: "Approaching Human-Level Forecasting with LMs" (arXiv:2402.18563v1)
*Foundational retrieval-augmented baseline. BLF cites and improves upon this.*

**Architecture (simpler, good for MVP):**
1. **Retrieval**: LM generates search queries → news API → LM ranks articles by relevance → LM summarizes top-k
2. **Reasoning**: LM uses scratchpad prompting over question + summaries → produces probability
3. **Aggregation**: trimmed mean across multiple LM runs

**Key dataset — ForecastBench** (the benchmark both papers use):
- Questions from: Polymarket, Manifold, Metaculus, Rand, yfinance, FRED, DBnomics, Wikipedia, ACLED
- Two question types: market (crowd signal available) and dataset (time-series data)
- Test set: questions with begin date after June 1, 2023 (after LM knowledge cutoffs)

**Crowd prediction is a strong baseline**: Brier score ~0.149 for human crowd. Zero-shot LLMs: ~0.208–0.250. Random: 0.250.

---

## 3. Reference Code Patterns — What to Reuse

### OpenRouter API Wrapper
**File**: [AI Workshop Code/lecture_1/notebooks/openrouter_utils.py](../AI%20Workshop%20Code/lecture_1/notebooks/openrouter_utils.py)

Key functions:
- `chat_completion(api_key, model, messages, temperature, max_tokens, response_format)` → use this for all LLM calls
- `safe_chat(api_key, model, prompt, max_retries=2)` → retry wrapper
- `check_credits(api_key)` → monitor budget spend

```python
# Standard call pattern
result = chat_completion(
    api_key=OPENROUTER_API_KEY,
    model="google/gemini-2.5-pro",          # or any model on OpenRouter
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=2000,
    response_format={"type": "json_object"}  # for structured output
)
content = result["content"]       # raw string
parsed  = result["parsed_content"]  # dict if json_object mode
usage   = result["usage"]         # token counts
```

### Structured LLM Call
**File**: [AI Workshop Code/lecture_4/notebooks/agent_utils.py](../AI%20Workshop%20Code/lecture_4/notebooks/agent_utils.py)

```python
result = structured_llm_call(
    api_key=OPENROUTER_API_KEY,
    prompt="Analyze this question and return a probability estimate.",
    context_data={"question": q, "background": bg, "belief_state": json.dumps(belief)},
    output_schema={"probability": 0.5, "confidence": "medium", "evidence_for": [], "evidence_against": [], "open_questions": []},
    model="google/gemini-2.5-pro",
    temperature=0.2
)
```

### Agent Loop + Tool Registry Pattern
**File**: [AI Workshop Code/lecture_4/notebooks/agent_utils.py](../AI%20Workshop%20Code/lecture_4/notebooks/agent_utils.py)

The `TOOL_REGISTRY` dict pattern: `{tool_name: {function, description, parameters}}`.
The agent loop: LLM picks a tool from the registry, you execute it, append (action, observation, belief) to message history, repeat.

### Tech Stack
```
Python ≥ 3.10
httpx==0.27.2       # HTTP client (already in reference code)
pandas==2.2.3       # data handling
python-dotenv==1.0.1 # API key management
jupyter             # notebooks
```

---

## 4. Budget & Constraints

| Constraint | Value |
|---|---|
| OpenRouter credits | **$50 total** |
| Time | **36 hours** |
| Competition format | Unknown until competition starts |
| Metric | Likely Brier Score or Brier Index (see papers) |

**Budget management rules:**
- Use `check_credits()` before and after test runs
- Development/testing: use cheap models (`google/gemini-2.0-flash`, `meta-llama/llama-3.3-70b-instruct`)
- Final runs: use best available (`google/gemini-2.5-pro`, `anthropic/claude-opus-4`, `openai/gpt-4.1`)
- K=5 trials per question multiplies costs by 5 — only use for final evaluation, use K=1 during development
- Target: leave ≥ $10 for final competition run, spend ≤ $40 on development

**Rough cost estimates** (OpenRouter pricing, approximate):
- Gemini Flash: ~$0.10/1M tokens → very cheap for development
- Gemini 2.5 Pro: ~$1.25/1M input → use sparingly
- Claude Sonnet 4: ~$3/1M input → selective use
- Per question (K=5, ~4000 tokens/trial): estimate $0.05–$0.50 depending on model

---

## 5. Implementation Plan

Build in this order (crawl → walk → run):

### Phase 1: MVP Baseline (Hours 0–6)
**Goal**: Get any working end-to-end forecast pipeline outputting valid probabilities.

1. Set up project structure in `Development/`
2. Port `openrouter_utils.py` and `agent_utils.py` patterns
3. Build zero-shot forecaster: question → LLM → probability (no search)
4. Test on 5–10 sample questions, verify output format
5. Establish cost baseline per question

### Phase 2: Retrieval + Belief State (Hours 6–16)
**Goal**: Implement the BLF agent loop with web search.

1. Implement the linguistic belief state schema
2. Build tool registry: `web_search`, `lookup_url`, `submit`
3. Implement the BLF agent loop (Algorithm 1 from paper)
4. Add date-based leak filtering on search results
5. Test on 10–20 questions, compare vs zero-shot baseline

### Phase 3: Aggregation + Calibration (Hours 16–26)
**Goal**: Implement multi-trial aggregation and calibration.

1. Multi-trial runner: K=3 trials per question (cheaper than K=5 during dev)
2. Logit-space aggregation function
3. Shrinkage toward p=0.5 (tunable α)
4. Basic Platt scaling calibration on validation set
5. If crowd/market signal available: inject into prompt as anchor

### Phase 4: Competition Run (Hours 26–36)
**Goal**: Apply to actual competition questions, monitor and submit.

1. Retrieve competition questions and format
2. Run full pipeline with K=5 trials on best model
3. Monitor credit spend closely
4. Review outputs for any obvious errors before submitting
5. Submit and iterate if time/budget allows

---

## 6. Key Prompt Templates

### Zero-Shot Forecast Prompt
```python
ZERO_SHOT_PROMPT = """You are an expert forecaster. Given the following question, estimate the probability that the answer is YES/TRUE.

Question: {question}
Background: {background}
Resolution criteria: {resolution_criteria}
Forecast date: {forecast_date}
Resolution date: {resolution_date}

Reason step by step, then output a probability between 0 and 1.
"""
```

### Belief State Schema (BLF-style)
```json
{
  "probability": 0.5,
  "confidence": "low|medium|high",
  "evidence_for": ["string"],
  "evidence_against": ["string"],
  "open_questions": ["string"],
  "reasoning": "string"
}
```

### Agent Step Prompt Template
```python
AGENT_STEP_PROMPT = """You are forecasting the probability of a future event. 

Question: {question}
Cutoff date (no information after this): {cutoff_date}
Current belief state: {belief_state_json}

Available tools: {tool_descriptions}
Message history: {history}

Produce your next action AND an updated belief state. 
If you have enough information, call submit(probability).
Otherwise, call one tool to gather more evidence.

Output JSON:
{
  "action": "tool_name",
  "action_args": {...},
  "updated_belief": {belief_state_schema}
}
"""
```

---

## 7. Data Sources for Search

When implementing `web_search`, prefer querying:
- **News**: Brave Search, DuckDuckGo, or any search API accessible via OpenRouter
- **Financial**: yfinance (Python library, free) for time-series
- **Wikipedia**: Wikipedia REST API (free, no key needed)
- **Prediction markets**: Polymarket, Metaculus public APIs for crowd signal

**Date filtering is critical**: Always pass `cutoff_date` and strip any results with dates after the cutoff. This prevents leakage from the future (a 1.5% residual leakage rate is acceptable per the BLF paper, but strive lower).

---

## 8. Session Protocol

Per [AI Workshop Code/AGENTS.md](../AI%20Workshop%20Code/AGENTS.md), when ending any work session:
1. File issues for remaining work
2. Run tests/linters if code changed
3. Update issue status
4. **Push to remote** (`git pull --rebase && git push`)
5. Verify with `git status` — must show "up to date with origin"

Branch: `felix` (current). Push to this branch.

---

## 9. What to Do When Competition Details Arrive

When you receive the actual competition problem spec:
1. Identify the question format (binary? continuous? multi-class?)
2. Identify the evaluation metric (Brier score? log score? accuracy?)
3. Identify whether crowd/market signal is available
4. Identify the question sources (geopolitical? financial? time-series?)
5. Adjust the pipeline accordingly — the BLF paper has per-source-type ablations that guide which tools matter most

If the competition is NOT binary forecasting: the Halawi retrieval pattern (search → summarize → reason → aggregate) generalizes to most prediction tasks. The belief state and calibration components are less critical for non-binary tasks.

---

## 10. Quick Reference Checklist

Before submitting any forecast:
- [ ] Cutoff date enforced in search results
- [ ] Output is a valid probability in [0, 1]  
- [ ] Multiple trials run and aggregated (not just K=1)
- [ ] Credits remaining > $10
- [ ] Results spot-checked for obvious hallucinations
- [ ] Git pushed before submitting
