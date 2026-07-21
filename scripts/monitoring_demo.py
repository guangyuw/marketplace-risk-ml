#!/usr/bin/env python3
"""Post-deployment monitoring demo.

Mirrors what a production monitoring job would do on a schedule:
  1. Load the model bundle that is currently in production
  2. Score a batch of incoming transactions
  3. Compare score and feature distributions against the launch-time baseline (PSI)
  4. Show the routing breakdown and what action each PSI level implies

Run:
    python scripts/monitoring_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET
from src.data import generate_synthetic_transactions
from src.features import build_features
from src.monitoring import route_transactions, psi

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"


def main() -> None:
    # ── 1. Load the production model bundle ──────────────────────────────────
    if not BUNDLE_PATH.exists():
        print(f"Model bundle not found at {BUNDLE_PATH}")
        print("Run  python -m src.train  first.")
        sys.exit(1)

    bundle = joblib.load(BUNDLE_PATH)
    model             = bundle["model"]
    freq_maps         = bundle["freq_maps"]
    threshold         = bundle["threshold"]
    feature_cols      = bundle["feature_columns"]
    baseline_scores   = bundle["baseline_scores"]    # test-set scores at launch time
    baseline_features = bundle["baseline_features"]  # test-set features at launch time

    print(f"Loaded  : {BUNDLE_PATH}")
    print(f"Deployed threshold: {threshold}")
    print()

    # ── 2. Simulate a batch of current transactions ───────────────────────────
    # In production: read today's transactions from the database or request log.
    # Use the last 20% by time to match the baseline's distributional maturity
    # (point-in-time features stabilise as sellers accumulate history).
    _sim = generate_synthetic_transactions(n=20_000, seed=99)
    _sim = _sim.sort_values("event_date")
    current_df = _sim.iloc[int(len(_sim) * 0.8):].copy()
    x_current  = build_features(current_df, freq_maps)[feature_cols]

    # ── 3. Score with the production model ───────────────────────────────────
    current_scores = model.predict_proba(x_current)[:, 1]

    # ── 4. PSI drift check ────────────────────────────────────────────────────
    # Baseline = test-set scores from training time (out-of-sample at launch).
    # Current  = scores on today's traffic.
    score_psi = psi(baseline_scores, current_scores)
    print(f"Score PSI = {score_psi:.4f}", end="  →  ")
    if score_psi < 0.1:
        print("stable, no action needed")
    elif score_psi < 0.25:
        print("moderate drift — investigate which features are shifting")
    else:
        print("HIGH drift — schedule retraining")
    print()

    print("Feature PSI  (launch baseline → current traffic):")
    for col in ["log_price", "seller_prior_dispute_rate", "buyer_age"]:
        if col in baseline_features.columns and col in x_current.columns:
            val = psi(baseline_features[col].values, x_current[col].values)
            action = "  ← investigate" if val > 0.1 else ""
            print(f"  {col:35s} {val:.4f}{action}")
    print()

    # ── 5. Routing breakdown with the deployed threshold ─────────────────────
    routes = route_transactions(current_scores, threshold=threshold, audit_fraction=0.02)
    print(f"Routing breakdown  (threshold={threshold}):")
    print(pd.Series(routes).value_counts(normalize=True).round(4).to_string())
    print()
    print("Actions by PSI level:")
    print("  < 0.10  no action")
    print("  0.10–0.25  check feature PSI, verify data pipeline")
    print("  > 0.25  retrain, update bundle, redeploy")


if __name__ == "__main__":
    main()
