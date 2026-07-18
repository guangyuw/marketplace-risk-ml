# LLM Review-Assist Layer

A production-minded LLM feature bolted onto the risk pipeline **without** letting
the LLM touch the decision. The scoring model still owns routing; the LLM only
drafts a structured brief for the human who reviews a flagged transaction.

## Where it sits

```
             ┌─────────────── scoring model (owns the decision) ───────────────┐
raw txn ──►  build_features ──► XGBoost + Platt ──► risk_score ──► threshold ──► route
             └──────────────────────────────────────────────────────────────────┘
                                                                     │
                       route == manual_review / audit                │ (auto_approve: no-op)
                                                                     ▼
   explain.build_evidence ──► rules.evaluate_rules ──► review_assist.generate_review_brief
   (de-identified evidence,     (policy actions that      (LLM drafts JSON brief;
    percentiles, TreeSHAP)       fired for this txn)        deterministic fallback if no LLM)
```

## Design boundaries (the defensible points)

1. **LLM is not the decision.** Auto-approve / manual-review / audit is decided by
   the calibrated model + business threshold. The LLM never approves or declines.
2. **No PII to the model.** `build_evidence` sends a de-identified payload only:
   banded buyer age, no name/email/payment, no transaction id, no high-cardinality
   section id. In production a redaction step would run before this too.
3. **Facts are computed in code, not by the LLM.** Percentiles come from the
   training/baseline distribution; per-feature contributions come from XGBoost's
   own TreeSHAP (`pred_contribs=True` — no extra dependency). The LLM only
   organizes and phrases these numbers.
4. **Actions are policy-bounded.** The LLM may only recommend actions whose rule
   fired in `review_rules.yaml`. A post-generation sanitizer drops anything else,
   so a hallucinated action can never reach the reviewer.
5. **LLM is off the critical path.** No API key, an API error, or an invalid
   response all fall back to a deterministic template built from the same
   evidence + rules. Serving and evals run fully offline.

## Components

| File | Role |
|------|------|
| `src/explain.py` | Build the de-identified evidence package (percentiles + TreeSHAP). |
| `src/review_rules.yaml` | Declarative `condition → action` review playbook. |
| `src/rules.py` | Load rules; evaluate conditions in a no-builtins namespace. |
| `src/review_assist.py` | Pydantic `ReviewBrief`, prompt, LLM call, sanitizer, fallback. |
| `src/serve.py` | `POST /review-assist` (no-op for auto-approved txns). |
| `evals/` | Gold cases + harness: schema, action recall, hallucination, faithfulness, latency. |

## Evaluation

An LLM surface gets an eval like any other model. `python -m evals.run_eval`
runs fixed gold scenarios through the assist layer and reports:

- `schema_valid_rate` — briefs that parse into `ReviewBrief`
- `action_recall` — expected policy actions that appeared
- `hallucination_rate` — recommended actions **not** backed by a fired rule
- `factor_faithfulness` — risk factors that reference a real evidence feature
- `forbidden_action_hits` — actions that should never appear
- `latency` — mean / p95 per brief

It runs with or without `OPENAI_API_KEY`: with a key it scores the live LLM;
without one it scores the deterministic fallback (faithful by construction), so
CI stays hermetic. Diff the printed metrics to compare prompts/models.

## Running

```bash
# offline (deterministic fallback)
python -m evals.run_eval

# live LLM
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini   # optional
python -m src.serve            # POST /review-assist
```

## Data boundary, said in one line

> In production I wouldn't send complete raw records to an LLM. The explanation
> layer receives only a minimal, de-identified payload — risk score, feature
> contributions, aggregate percentiles, and policy IDs. PII and payment details
> stay in the internal system, and the LLM never makes or executes the decision.
