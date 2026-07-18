"""Tests for the LLM review-assist layer.

These run without any API key: they exercise the rules engine, the evidence
package, and the deterministic fallback brief (which is what serving returns
whenever the LLM is unavailable).
"""

import joblib
import pandas as pd
import pytest

from src.config import MODEL_BUNDLE_PATH
from src.explain import build_evidence, rule_namespace
from src.review_assist import ReviewBrief, _fallback_brief, generate_review_brief
from src.rules import allowed_actions, evaluate_rules


def _evidence_from(**overrides):
    """Build an evidence dict directly (no model needed) for rule tests."""
    base = {
        "risk_score": 0.5,
        "threshold": 0.05,
        "route": "manual_review",
        "transaction": {
            "ticket_price": 150.0,
            "quantity": 2,
            "venue_type": "arena",
            "event_category": "concert",
            "seller_prior_dispute_rate": 0.02,
            "buyer_prior_purchases": 5,
            "buyer_age_band": "25-34",
        },
        "distribution_context": {
            "ticket_price_percentile": 50,
            "seller_dispute_percentile": 50,
            "quantity_percentile": 50,
        },
        "top_contributors": [
            {
                "feature": "seller_prior_dispute_rate",
                "label": "seller historical dispute rate",
                "value": 0.02,
                "contribution": 0.1,
                "direction": "increases_risk",
                "percentile": 50,
            }
        ],
    }
    base["transaction"].update(overrides.get("transaction", {}))
    base["distribution_context"].update(overrides.get("distribution_context", {}))
    if "risk_score" in overrides:
        base["risk_score"] = overrides["risk_score"]
    return base


def test_high_dispute_rule_fires():
    ev = _evidence_from(transaction={"seller_prior_dispute_rate": 0.3})
    fired = {r["rule_id"] for r in evaluate_rules(rule_namespace(ev))}
    assert "high_seller_dispute" in fired


def test_clean_transaction_fires_no_rules():
    ev = _evidence_from()
    assert evaluate_rules(rule_namespace(ev)) == []


def test_bulk_first_purchase_rule():
    ev = _evidence_from(transaction={"quantity": 8, "buyer_prior_purchases": 0})
    fired = {r["rule_id"] for r in evaluate_rules(rule_namespace(ev))}
    assert "bulk_first_purchase" in fired


def test_fallback_brief_is_faithful_and_schema_valid():
    # Test the deterministic fallback directly so the test is hermetic
    # regardless of whether an API key is configured.
    ev = _evidence_from(
        transaction={"seller_prior_dispute_rate": 0.3, "venue_type": "festival"},
        risk_score=0.7,
    )
    fired = evaluate_rules(rule_namespace(ev))
    brief = _fallback_brief(ev, fired)
    assert isinstance(brief, ReviewBrief)
    assert brief.generated_by == "fallback_template"
    # Every recommended action must be an allowed, policy-backed action.
    assert all(c.action in allowed_actions() for c in brief.recommended_checks)
    fired_ids = {r["rule_id"] for r in fired}
    assert all(c.rule_id in fired_ids for c in brief.recommended_checks)


def test_generate_review_brief_never_hallucinates_actions():
    """Whatever the source (LLM or fallback), served checks must be policy-backed."""
    ev = _evidence_from(
        transaction={"seller_prior_dispute_rate": 0.3, "venue_type": "festival"},
        risk_score=0.7,
    )
    brief = generate_review_brief(ev)
    fired_ids = {r["rule_id"] for r in evaluate_rules(rule_namespace(ev))}
    assert all(c.action in allowed_actions() for c in brief.recommended_checks)
    assert all(c.rule_id in fired_ids for c in brief.recommended_checks)


def test_elevated_score_triggers_escalation():
    ev = _evidence_from(risk_score=0.8)
    fired = evaluate_rules(rule_namespace(ev))
    brief = _fallback_brief(ev, fired)
    assert brief.recommended_disposition == "escalate"


@pytest.mark.skipif(not MODEL_BUNDLE_PATH.exists(), reason="model bundle not trained")
def test_build_evidence_excludes_pii_and_ranks_contributors():
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    row = pd.DataFrame(
        [
            {
                "ticket_price": 1400.0,
                "quantity": 8,
                "buyer_age": 22,
                "venue_type": "festival",
                "event_category": "concert",
                "section_code": "S0003",
                "seller_prior_dispute_rate": 0.4,
                "buyer_prior_purchases": 0,
                "event_date": pd.to_datetime("2025-07-20"),
            }
        ]
    )
    ev = build_evidence(bundle, row, risk_score=0.7, threshold=0.05, route="manual_review")
    txn = ev["transaction"]
    # De-identified: banded age, no raw age / section id / transaction id.
    assert "buyer_age_band" in txn
    assert "buyer_age" not in txn
    assert "section_code" not in txn
    assert "transaction_id" not in txn
    # Contributors are ranked by absolute contribution.
    contribs = [abs(c["contribution"]) for c in ev["top_contributors"]]
    assert contribs == sorted(contribs, reverse=True)
