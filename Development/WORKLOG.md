# Prophet Hacks 2026 — Agent Work Log

**How to use this file:**
- Read the STATUS block first — it tells you exactly where things stand right now.
- Append a new entry to the EXPERIMENT LOG every time you make a significant change, run an eval, or hit a decision point. Never edit old entries.
- Update the STATUS block to reflect the current state after each session.
- Add ideas to the BACKLOG rather than acting on them immediately; let the next agent decide priority.

---

## STATUS (update this every session)

```
Last updated:    [YYYY-MM-DD HH:MM]
Updated by:      [agent/human name]
Phase:           [1-MVP | 2-Retrieval+Belief | 3-Aggregation+Calibration | 4-Competition]
Branch:          felix
Credits used:    $0.00 / $50.00
Credits left:    $50.00

What's working:
  - [ ] Nothing yet — project initialized, no code written

What's broken / blocked:
  - Competition problem spec not yet received

Next action (do this first):
  - Wait for competition details, then start Phase 1 (zero-shot MVP) per AGENT_BRIEF.md
```

---

## EXPERIMENT LOG

*Append entries here. Never edit or delete old entries. Newest at the top.*

---

### [ENTRY TEMPLATE — copy this for each new entry]
```
Date/Time:   YYYY-MM-DD HH:MM
Agent:       [name or "human"]
Type:        [code | eval | decision | blocked | discovery]
Credits:     $X.XX spent this session (total: $X.XX / $50.00)

Summary:
  One paragraph describing what you did.

Result / Finding:
  Quantitative result if applicable (Brier Score, Brier Index, cost/question).
  If no metric yet: qualitative outcome.

Files changed:
  - path/to/file.py (created | modified | deleted)

Decision made:
  If you chose between options, record which and why briefly.

Handoff note:
  One sentence: what the next agent should do first.
```

---

### 2026-05-15 — Project Initialization
```
Date/Time:   2026-05-15
Agent:       human (Felix)
Type:        decision
Credits:     $0.00 spent (total: $0.00 / $50.00)

Summary:
  Repository created. AGENT_BRIEF.md written with full architecture target
  (BLF paper, arXiv:2604.18576v2), reference code inventory, budget rules,
  and 4-phase implementation plan. Development/ directory initialized.

Result / Finding:
  No code yet. Architecture target: BLF agent loop with linguistic belief
  state, K=5 multi-trial aggregation, hierarchical Platt calibration.

Files changed:
  - Development/AGENT_BRIEF.md (created)
  - Development/WORKLOG.md (created)

Decision made:
  Use OpenRouter with cheap models (Gemini Flash) for development, best
  available model for final competition run.

Handoff note:
  Wait for competition problem spec, then start Phase 1 — zero-shot
  forecaster per AGENT_BRIEF.md Section 5.
```

---

## BACKLOG

*Ideas and things to try. Not prioritized. Add freely, let the next agent decide order.*

### High Priority (do before competition run)
- [ ] Implement zero-shot forecaster (Phase 1 baseline)
- [ ] Implement linguistic belief state JSON schema
- [ ] Implement agent loop with web_search tool
- [ ] Implement K=3 multi-trial aggregation in logit space
- [ ] Basic Platt scaling calibration

### Medium Priority (do if time allows)
- [ ] Add `fetch_ts_yfinance` tool for financial/time-series questions
- [ ] Add `fetch_wikipedia_section` tool
- [ ] Tune shrinkage parameter α via LOO cross-validation
- [ ] Hierarchical Platt scaling (per-source intercept offsets)
- [ ] Inject crowd/market signal into prompt as anchor for market questions

### Low Priority / Speculative
- [ ] Increase K from 3 → 5 for final run (costs 5× more per question)
- [ ] Experiment with mixture of models (cheap model for search steps, strong model for belief update)
- [ ] Few-shot examples in belief-state prompt (reference lecture_3 techniques)
- [ ] Meta-controller to select tool set per question type (BLF paper Section C.2)
- [ ] Implement 4-layer date-leakage defense (BLF paper Section B)

---

## DECISIONS LOG

*Record any non-obvious choices made so future agents don't re-debate them.*

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-15 | Target BLF architecture over Halawi 2024 | BLF is Apr 2026 SOTA; Halawi is the baseline BLF improves on |
| 2026-05-15 | Development code goes in Development/, not AI Workshop Code/ | AI Workshop Code/ is reference-only for patterns |
| 2026-05-15 | Use Gemini Flash for dev, best model for final run | Budget: ≤$40 dev, ≥$10 final competition run |

---

## KNOWN ISSUES & GOTCHAS

*Things that bit someone. Append, don't edit.*

- (none yet)
