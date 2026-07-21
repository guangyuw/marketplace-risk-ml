"""Data loading and temporal train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TARGET, TRAIN_QUANTILE

def generate_synthetic_transactions(n: int = 40_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic marketplace transactions for offline development.

    Schema mirrors a secondary ticket marketplace:
    - long-tail prices, high-cardinality categories, imbalanced risk label
    - seller entities with persistent latent risk; per-transaction is_disputed
      outcomes are aggregated point-in-time to produce seller_prior_dispute_rate
      (strictly past rows only — no leakage)
    - bimodal seller quality (verified-style low-risk sellers vs. a risky minority)
      plus a price x seller-risk interaction, matching how risk actually concentrates
      in resale marketplaces rather than varying smoothly across all sellers
    """
    rng = np.random.default_rng(seed)

    # Scale seller/buyer pool with n so each seller keeps ~20 transactions on
    # average. This ensures the point-in-time dispute rate has enough history
    # regardless of whether n is 5k (monitoring demo) or 40k (training).
    n_sellers = max(50, n // 20)
    n_buyers  = max(200, n // 4)

    # ── Seller entities: bimodal latent risk ─────────────────────────────────
    # Most sellers behave like verified/professional resellers (very low dispute
    # rate); a small minority are chronically high-risk. Real trust-and-safety
    # data shows a small fraction of accounts driving the majority of disputes.
    is_risky_seller = rng.random(n_sellers) < 0.10
    seller_latent_risk = np.where(
        is_risky_seller,
        rng.beta(5, 8, n_sellers),   # risky tier: mean ~38%
        rng.beta(1, 80, n_sellers),  # clean tier: mean ~1.2%
    )

    # ── Assign sellers and buyers to transactions ─────────────────────────────
    seller_ids = rng.integers(0, n_sellers, n)
    buyer_ids = rng.integers(0, n_buyers, n)
    per_txn_seller_risk = seller_latent_risk[seller_ids]

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

    event_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 540, n), unit="D"
    )

    # ── Per-transaction dispute outcome ───────────────────────────────────────
    # is_disputed is the raw 0/1 event recorded after each transaction settles.
    # It is used only to compute the historical prior — NOT as a feature itself
    # (that would leak the current transaction's outcome into the model).
    is_disputed = (rng.random(n) < per_txn_seller_risk).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "seller_id": seller_ids,
            "buyer_id": buyer_ids,
            "event_date": event_date,
            "ticket_price": np.round(rng.lognormal(mean=4.5, sigma=0.9, size=n), 2),
            "quantity": rng.poisson(2, n) + 1,
            "buyer_age": np.clip(rng.normal(34, 11, n), 18, 75).astype(int),
            "venue_type": venue_type,
            "event_category": event_category,
            "section_code": section_code,
            "is_disputed": is_disputed,
        }
    )

    # ── Point-in-time seller prior dispute rate ───────────────────────────────
    # Sort by date so past rows always come before current. For each transaction,
    # compute the seller's dispute rate using only strictly prior transactions
    # (expanding mean shifted by 1). Sellers with no history get NaN → filled
    # downstream in features.py with an is_missing indicator.
    df = df.sort_values("event_date").reset_index(drop=True)
    df["seller_prior_dispute_rate"] = (
        df.groupby("seller_id")["is_disputed"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # ── Point-in-time buyer prior purchases ──────────────────────────────────
    df["buyer_prior_purchases"] = (
        df.groupby("buyer_id")["transaction_id"]
        .transform(lambda s: (s.expanding().count() - 1).astype(int))
    )

    # ── Risk label ────────────────────────────────────────────────────────────
    # Derived from seller latent risk (not from is_disputed of THIS transaction).
    # Look up latent risk via seller_id so the mapping survives the sort above.
    # Risky sellers unloading expensive tickets compound risk beyond either factor
    # alone — the kind of interaction a linear model can't represent but trees can.
    seller_risk_col = df["seller_id"].map(pd.Series(seller_latent_risk)).values
    price_hi = (df["ticket_price"] > df["ticket_price"].quantile(0.75)).astype(float)
    risky_venue = df["venue_type"].isin(["festival", "club"]).astype(float)
    risky_seller_high_price = (seller_risk_col > 0.15).astype(float) * price_hi * 3.6
    # Diminishing-returns trust effect: loyalty matters most for a buyer's first
    # few purchases, then flattens out.
    buyer_trust = -0.3 * np.log1p(df["buyer_prior_purchases"].fillna(0))

    logit = (
        -3.85
        + 7.0 * seller_risk_col
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
