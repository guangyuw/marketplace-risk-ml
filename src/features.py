"""Feature engineering — all mappings fit on train only."""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_FREQ_COLS = ["venue_type", "event_category", "section_code"]
HIST_COLS = ["seller_prior_dispute_rate", "buyer_prior_purchases"]


def fit_freq_maps(train_df: pd.DataFrame, cols: list[str] | None = None) -> dict[str, pd.Series]:
    cols = cols or CAT_FREQ_COLS
    return {c: train_df[c].value_counts(normalize=True) for c in cols}


def build_features(df: pd.DataFrame, freq_maps: dict[str, pd.Series]) -> pd.DataFrame:
    d = df.copy()

    for c in HIST_COLS:
        d[f"{c}_isna"] = d[c].isna().astype(int)
        d[c] = d[c].fillna(0)

    d["log_price"] = np.log1p(d["ticket_price"])
    d["dow"] = d["event_date"].dt.dayofweek
    d["month"] = d["event_date"].dt.month

    for c in CAT_FREQ_COLS:
        d[f"{c}_freq"] = d[c].map(freq_maps[c]).fillna(0.0)

    feature_cols = (
        [
            "log_price",
            "quantity",
            "buyer_age",
            "seller_prior_dispute_rate",
            "buyer_prior_purchases",
            "seller_prior_dispute_rate_isna",
            "buyer_prior_purchases_isna",
            "dow",
            "month",
        ]
        + [f"{c}_freq" for c in CAT_FREQ_COLS]
    )
    return d[feature_cols]


def prepare_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, pd.Series]]:
    freq_maps = fit_freq_maps(train_df)
    x_train = build_features(train_df, freq_maps)
    x_test = build_features(test_df, freq_maps)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    return x_train, y_train, x_test, y_test, freq_maps
