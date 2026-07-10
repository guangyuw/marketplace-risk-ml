"""Data loading and temporal train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TARGET, TRAIN_QUANTILE


def generate_synthetic_transactions(n: int = 40_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic marketplace transactions for offline development.

    Schema mirrors a secondary ticket marketplace:
    - long-tail prices, high-cardinality categories, imbalanced risk label
    - historical seller/buyer features computed only from past events (no leakage)
    """
    rng = np.random.default_rng(seed)

    venue_type = rng.choice(
        ["arena", "stadium", "theater", "club", "festival"],
        n,
        p=[0.35, 0.25, 0.2, 0.12, 0.08],
    )
    event_category = rng.choice(
        ["concert", "sports", "comedy", "theater"],
        n,
    )
    section_code = rng.choice([f"S{c:04d}" for c in range(300)], n)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "event_date": pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 540, n), unit="D"),
            "ticket_price": np.round(rng.lognormal(mean=4.5, sigma=0.9, size=n), 2),
            "quantity": rng.poisson(2, n) + 1,
            "buyer_age": np.clip(rng.normal(34, 11, n), 18, 75).astype(int),
            "venue_type": venue_type,
            "event_category": event_category,
            "section_code": section_code,
            # Pretend these come from a leakage-safe rolling window in SQL
            "seller_prior_dispute_rate": np.clip(rng.beta(2, 20, n), 0, 1),
            "buyer_prior_purchases": rng.poisson(3, n),
        }
    )

    logit = (
        -4.0
        + 6.5 * df["seller_prior_dispute_rate"]
        + 0.9 * (df["ticket_price"] > df["ticket_price"].quantile(0.75)).astype(float)
        + 0.8 * df["venue_type"].isin(["festival", "club"]).astype(float)
        + 0.05 * df["buyer_prior_purchases"]
        + rng.normal(0, 0.4, n)
    )
    p = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "event_date",
    quantile: float = TRAIN_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time — never random-split transactional scoring problems."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    cutoff = ordered[date_col].quantile(quantile)
    train = ordered[ordered[date_col] <= cutoff].copy()
    test = ordered[ordered[date_col] > cutoff].copy()
    return train, test
