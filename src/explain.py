"""De-identified evidence package for the LLM review-assist layer.

The LLM that drafts a manual-review brief never sees the raw record. It only
sees the structured evidence assembled here:

- the calibrated risk score, threshold, and route (from the scoring model)
- per-feature contributions to *this* prediction (TreeSHAP, computed by
  XGBoost itself — no extra dependency)
- where each value sits in the training/baseline distribution (percentile)
- a de-identified view of the transaction (no PII: age is banded, the
  high-cardinality section id and transaction id are dropped)

Every number here is computed in code. The LLM's only job downstream is to
organize and phrase these facts — it does not decide what is anomalous, does
not compute contributions, and does not invent rules or actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features import build_features

# Engineered feature -> human-readable label used in the brief.
_FEATURE_LABELS = {
    "seller_prior_dispute_rate": "seller historical dispute rate",
    "log_price": "ticket price",
    "quantity": "quantity ordered",
    "buyer_prior_purchases": "buyer purchase history",
    "buyer_age": "buyer age",
    "venue_type_freq": "venue type prevalence",
    "event_category_freq": "event category prevalence",
    "section_code_freq": "section prevalence",
    "dow": "day of week",
    "month": "month",
    "seller_prior_dispute_rate_isna": "seller history missing flag",
    "buyer_prior_purchases_isna": "buyer history missing flag",
}


def _age_band(age: int) -> str:
    """Band a raw age into a coarse range to avoid sending exact PII-ish values."""
    bands = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64)]
    for lo, hi in bands:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "65+"


def _percentile(baseline: np.ndarray, value: float) -> int:
    """Percentile rank of value within a baseline distribution (0-100)."""
    baseline = np.asarray(baseline, dtype=float)
    if baseline.size == 0:
        return 0
    return int(round(float((baseline < value).mean()) * 100))


def _tree_shap(bundle: dict, x_row: pd.DataFrame) -> dict[str, float]:
    """Per-feature contributions to the raw margin via XGBoost's own TreeSHAP.

    Uses ``pred_contribs=True`` on the underlying booster so we get exact
    TreeSHAP values without adding the ``shap`` package. Contributions are on
    the model's log-odds margin (before Platt calibration), which is what we
    want for "which features pushed this score up".
    """
    model = bundle["model"]
    base = getattr(model, "base_model", model)  # PlattCalibrated wraps base_model
    cols = bundle["feature_columns"]
    try:
        import xgboost as xgb

        booster = base.get_booster()
        dmatrix = xgb.DMatrix(x_row[cols], feature_names=list(cols))
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]
        # Last column is the bias term; drop it.
        return {c: float(v) for c, v in zip(cols, contribs[:-1])}
    except Exception:
        # Fallback: global gain importances scaled by direction is not available
        # without SHAP; return zeros so the brief still renders from percentiles.
        return {c: 0.0 for c in cols}


def build_evidence(
    bundle: dict,
    raw_row: pd.DataFrame,
    *,
    risk_score: float,
    threshold: float,
    route: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Assemble the de-identified evidence package for one transaction.

    Parameters
    ----------
    bundle : dict
        Loaded model bundle (model, freq_maps, feature_columns, baseline_features).
    raw_row : DataFrame
        Single-row raw transaction (same schema as /predict input).
    """
    cols = bundle["feature_columns"]
    baseline: pd.DataFrame = bundle["baseline_features"]
    x_row = build_features(raw_row, bundle["freq_maps"])[cols]

    r = raw_row.iloc[0]
    contributions = _tree_shap(bundle, x_row)

    # Percentiles for the fields the review rules and brief care about.
    price_pct = _percentile(baseline["log_price"].values, float(x_row["log_price"].iloc[0]))
    dispute_pct = _percentile(
        baseline["seller_prior_dispute_rate"].values,
        float(r["seller_prior_dispute_rate"]),
    )
    qty_pct = _percentile(baseline["quantity"].values, float(r["quantity"]))

    # Rank engineered features by absolute contribution to this prediction.
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_contributors = []
    for feat, contrib in ranked[:top_k]:
        val = float(x_row[feat].iloc[0])
        pct: int | None = None
        if feat in baseline.columns:
            pct = _percentile(baseline[feat].values, val)
        top_contributors.append(
            {
                "feature": feat,
                "label": _FEATURE_LABELS.get(feat, feat),
                "value": round(val, 4),
                "contribution": round(contrib, 4),
                "direction": "increases_risk" if contrib > 0 else "decreases_risk",
                "percentile": pct,
            }
        )

    evidence = {
        "risk_score": round(float(risk_score), 4),
        "threshold": round(float(threshold), 4),
        "route": route,
        # De-identified business view: no name/email/payment, no exact age,
        # no transaction id, no high-cardinality section id.
        "transaction": {
            "ticket_price": round(float(r["ticket_price"]), 2),
            "quantity": int(r["quantity"]),
            "venue_type": str(r["venue_type"]),
            "event_category": str(r["event_category"]),
            "seller_prior_dispute_rate": round(float(r["seller_prior_dispute_rate"]), 4),
            "buyer_prior_purchases": int(r["buyer_prior_purchases"]),
            "buyer_age_band": _age_band(int(r["buyer_age"])),
        },
        "distribution_context": {
            "ticket_price_percentile": price_pct,
            "seller_dispute_percentile": dispute_pct,
            "quantity_percentile": qty_pct,
        },
        "top_contributors": top_contributors,
    }
    return evidence


def rule_namespace(evidence: dict[str, Any]) -> dict[str, Any]:
    """Flatten an evidence package into the variables review rules can reference."""
    txn = evidence["transaction"]
    dist = evidence["distribution_context"]
    return {
        "ticket_price": txn["ticket_price"],
        "quantity": txn["quantity"],
        "venue_type": txn["venue_type"],
        "event_category": txn["event_category"],
        "seller_prior_dispute_rate": txn["seller_prior_dispute_rate"],
        "buyer_prior_purchases": txn["buyer_prior_purchases"],
        "ticket_price_percentile": dist["ticket_price_percentile"],
        "seller_dispute_percentile": dist["seller_dispute_percentile"],
        "quantity_percentile": dist["quantity_percentile"],
        "risk_score": evidence["risk_score"],
    }
