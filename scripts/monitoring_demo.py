#!/usr/bin/env python3
"""Monitoring demo: threshold selection, PSI drift, and transaction routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FALSE_CLEAR_TOLERANCE, TARGET
from src.data import generate_synthetic_transactions, temporal_train_test_split
from src.features import prepare_datasets
from src.model import train_xgboost
from src.monitoring import choose_threshold_by_tolerance, psi, route_transactions


def main() -> None:
    df = generate_synthetic_transactions()
    train_df, test_df = temporal_train_test_split(df)
    x_train, y_train, x_test, y_test, _freq_maps = prepare_datasets(
        train_df, test_df, TARGET
    )

    model = train_xgboost(x_train, y_train)
    p_train = model.predict_proba(x_train)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]

    # 1) business threshold
    t, table = choose_threshold_by_tolerance(
        y_test, p_test, tolerance=FALSE_CLEAR_TOLERANCE
    )
    print("threshold =", t)
    print(table.head())

    # 2) score PSI: train vs test (proxy for post-deploy traffic)
    print("score PSI =", round(psi(p_train, p_test), 4))

    # 3) feature PSI
    for col in ["log_price", "seller_prior_dispute_rate", "buyer_age"]:
        print(col, "PSI =", round(psi(x_train[col].values, x_test[col].values), 4))

    # 4) routing: auto / manual / audit
    routes = route_transactions(p_test, threshold=t, audit_fraction=0.02)
    print(pd.Series(routes).value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
