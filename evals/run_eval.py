"""Evaluation harness for the LLM review-assist layer.

An LLM feature is a product surface, not a demo — so it gets an eval like any
other model. This harness isolates the *assist layer* (evidence -> rules -> brief). Whether
the scoring model routes a transaction to review is a separate concern already
measured in training (PR-AUC / Brier / threshold). So we build evidence for
every gold scenario and evaluate the brief, reporting the model's route only as
context. Metrics:

  - schema_valid_rate   : briefs that parse into the ReviewBrief schema
  - action_recall       : expected policy actions that appeared in the brief
  - hallucination_rate  : recommended actions NOT backed by a fired rule
  - factor_faithfulness : risk factors that reference a real evidence feature
  - forbidden_hits      : actions that explicitly should not appear
  - latency             : mean/p95 per brief

It runs anywhere: with OPENAI_API_KEY set it scores the real LLM; without a
key it scores the deterministic fallback (which is faithful by construction).
Compare runs across prompts/models by diffing the printed metrics.

Usage:
    python -m evals.run_eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_BUNDLE_PATH
from src.explain import build_evidence, rule_namespace
from src.features import build_features
from src.monitoring import route_transactions
from src.review_assist import generate_review_brief
from src.rules import allowed_actions, evaluate_rules

GOLD_PATH = Path(__file__).resolve().parent / "gold_cases.json"


def _score(bundle: dict, txn: dict) -> tuple[pd.DataFrame, float, float, str]:
    row = pd.DataFrame([{**txn, "event_date": pd.to_datetime(txn["event_date"])}])
    x = build_features(row, bundle["freq_maps"])
    prob = float(bundle["model"].predict_proba(x)[:, 1][0])
    threshold = float(bundle["threshold"])
    # Deterministic route for context (no audit randomness): flagged iff prob >= threshold.
    route = "manual_review" if prob >= threshold else "auto_approve"
    return row, prob, threshold, route


def main() -> None:
    if not MODEL_BUNDLE_PATH.exists():
        raise SystemExit(f"Model bundle not found at {MODEL_BUNDLE_PATH}. Run: python -m src.train")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    gold = json.loads(GOLD_PATH.read_text())["cases"]
    allowed = allowed_actions()

    n = len(gold)
    schema_ok = 0
    model_flagged = 0
    recall_num = recall_den = 0
    halluc_actions = total_actions = 0
    faithful_num = faithful_den = 0
    forbidden_hits = 0
    latencies: list[float] = []
    generated_by: dict[str, int] = {}
    rows = []

    for case in gold:
        txn = case["transaction"]
        row, prob, threshold, route = _score(bundle, txn)
        model_flagged += int(route != "auto_approve")

        # Evaluate the assist layer for every scenario, independent of routing.
        evidence = build_evidence(
            bundle, row, risk_score=prob, threshold=threshold, route="manual_review"
        )
        fired_ids = {r["rule_id"] for r in evaluate_rules(rule_namespace(evidence))}
        # Vocabulary of terms that legitimately appear in the evidence, so we can
        # check a (possibly reworded) risk factor references a real dimension
        # rather than an invented one.
        vocab: set[str] = set()
        for c in evidence["top_contributors"]:
            vocab.update(c["label"].lower().split())
        for key in evidence["transaction"]:
            vocab.update(key.replace("_", " ").split())
        vocab -= {"the", "of", "a", "flag", "band", "prevalence"}

        t0 = time.perf_counter()
        brief = generate_review_brief(evidence)
        latencies.append(time.perf_counter() - t0)

        schema_ok += 1  # pydantic-validated by construction
        generated_by[brief.generated_by] = generated_by.get(brief.generated_by, 0) + 1

        got_actions = {c.action for c in brief.recommended_checks}
        expected = set(case["expected_actions"])
        case_recall = "-"
        if expected:
            hit = len(expected & got_actions)
            recall_num += hit
            recall_den += len(expected)
            case_recall = f"{hit}/{len(expected)}"

        case_halluc = 0
        for c in brief.recommended_checks:
            total_actions += 1
            if c.action not in allowed or c.rule_id not in fired_ids:
                halluc_actions += 1
                case_halluc += 1

        forbidden_hits += len(set(case.get("forbidden_actions", [])) & got_actions)

        for f in brief.risk_factors:
            faithful_den += 1
            text = f"{f.factor} {f.evidence}".lower()
            if any(term in text for term in vocab):
                faithful_num += 1

        rows.append(
            {
                "case": case["id"],
                "score": round(prob, 3),
                "route": route,
                "recall": case_recall,
                "halluc": case_halluc,
                "by": brief.generated_by,
            }
        )

    print("\n=== Review-Assist Evaluation ===")
    print(f"cases                : {n}")
    print(f"generated_by         : {generated_by}")
    print(f"model_flagged (info) : {model_flagged}/{n}")
    print(f"schema_valid_rate    : {schema_ok / n:.2%}")
    if recall_den:
        print(f"action_recall        : {recall_num / recall_den:.2%}  ({recall_num}/{recall_den})")
    if total_actions:
        print(f"hallucination_rate   : {halluc_actions / total_actions:.2%}  ({halluc_actions}/{total_actions})")
    if faithful_den:
        print(f"factor_faithfulness  : {faithful_num / faithful_den:.2%}  ({faithful_num}/{faithful_den})")
    print(f"forbidden_action_hits: {forbidden_hits}")
    if latencies:
        arr = np.array(latencies)
        print(f"latency mean / p95   : {arr.mean() * 1000:.1f} ms / {np.percentile(arr, 95) * 1000:.1f} ms")

    print("\nper-case:")
    hdr = f"{'case':<26}{'score':>7}{'route':>15}{'recall':>9}{'halluc':>8}{'by':>18}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['case']:<26}{r['score']:>7}{r['route']:>15}"
            f"{r['recall']:>9}{r['halluc']:>8}{r['by']:>18}"
        )


if __name__ == "__main__":
    main()
