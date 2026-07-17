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
    - bimodal seller quality (verified-style low-risk sellers vs. a risky minority)
      plus a price x seller-risk interaction, matching how risk actually concentrates
      in resale marketplaces rather than varying smoothly across all sellers
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

    # Two seller tiers: most behave like verified/professional resellers (very low
    # historical dispute rate), a smaller share are chronically disputed. Real seller
    # bases look like this far more than a single smooth distribution — trust & safety
    # data consistently shows a small minority of accounts driving most disputes.
    is_risky_seller = rng.random(n) < 0.10
    seller_prior_dispute_rate = np.where(
        is_risky_seller,
        rng.beta(5, 8, n),   # risky tier: mean ~38%
        rng.beta(1, 80, n),  # clean tier: mean ~1.2%
    )

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
            "seller_prior_dispute_rate": np.clip(seller_prior_dispute_rate, 0, 1),
            "buyer_prior_purchases": rng.poisson(3, n),
        }
    )

    price_hi = (df["ticket_price"] > df["ticket_price"].quantile(0.75)).astype(float)
    risky_venue = df["venue_type"].isin(["festival", "club"]).astype(float)
    dispute = df["seller_prior_dispute_rate"]
    # Risky sellers unloading expensive tickets compound risk beyond either factor
    # alone — the kind of interaction a linear model can't represent but trees can.
    risky_seller_high_price = (dispute > 0.15).astype(float) * price_hi * 3.6
    # Diminishing-returns trust effect: loyalty matters most for a buyer's first
    # few purchases, then flattens out.
    buyer_trust = -0.3 * np.log1p(df["buyer_prior_purchases"])

    logit = (
        -3.85
        + 7.0 * dispute
        + 0.5 * price_hi
        + 0.5 * risky_venue
        + risky_seller_high_price
        + buyer_trust
        + rng.normal(0, 0.19, n)
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


def temporal_three_way_split(
    df: pd.DataFrame,
    date_col: str = "event_date",
    train_q: float = 0.6,
    calib_q: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal 60/20/20 split: train → calibration → test.

    All three windows are strictly time-ordered with no overlap:
      - train  : [0%,  60%) — fit the model
      - calib  : [60%, 80%) — fit the probability calibrator (Platt scaling)
      - test   : [80%, 100%] — held-out evaluation only

    Using a separate calib window prevents the calibrator from seeing
    training data (which would over-fit the sigmoid) and keeps the test
    set truly unseen.
    """
    ordered = df.sort_values(date_col).reset_index(drop=True)
    cutoff_train = ordered[date_col].quantile(train_q)
    cutoff_calib = ordered[date_col].quantile(calib_q)
    train = ordered[ordered[date_col] <= cutoff_train].copy()
    calib = ordered[
        (ordered[date_col] > cutoff_train) & (ordered[date_col] <= cutoff_calib)
    ].copy()
    test = ordered[ordered[date_col] > cutoff_calib].copy()
    return train, calib, test
