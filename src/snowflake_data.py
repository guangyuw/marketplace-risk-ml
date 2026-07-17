"""Snowflake integration: upload raw tables, extract point-in-time features.

Production flow this module replicates
---------------------------------------
  Raw event log  →  Snowflake warehouse  →  feature SQL (window fns)  →  training DataFrame

In production, the raw event log comes from application DBs / Kafka / S3.
Here we generate a realistic synthetic version and write it to Snowflake so
the rest of the pipeline (feature SQL → train → MLflow) is identical to prod.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    _SNOWFLAKE_AVAILABLE = True
except ImportError:
    _SNOWFLAKE_AVAILABLE = False

SQL_DIR = Path(__file__).parent.parent / "sql"

N_SELLERS = 500
N_BUYERS = 2_000


def _require_snowflake():
    if not _SNOWFLAKE_AVAILABLE:
        raise ImportError(
            "snowflake-connector-python not installed.\n"
            "Run: pip install 'snowflake-connector-python[pandas]'"
        )


# ── 1. Raw data generation ───────────────────────────────────────────────────

def generate_raw_tables(
    n: int = 40_000,
    n_sellers: int = N_SELLERS,
    n_buyers: int = N_BUYERS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate raw transactions and buyers with no pre-computed features.

    Unlike generate_synthetic_transactions() in data.py (which fakes the
    aggregated features), this produces the *raw event log* that would live in
    a production warehouse:
      - Each seller has a latent risk level that governs dispute probability.
      - is_disputed is an outcome of each individual transaction.
      - is_high_risk (the label) also depends on seller risk but is NOT derived
        from is_disputed directly — Snowflake computes seller_prior_dispute_rate
        from past rows only via a window function, so there is no leakage.

    Returns
    -------
    transactions : DataFrame with columns matching MARKETPLACE.ANALYTICS.TRANSACTIONS
    buyers       : DataFrame with columns matching MARKETPLACE.ANALYTICS.BUYERS
    """
    rng = np.random.default_rng(seed)

    # Persistent risk level per seller, bimodal: most sellers behave like
    # verified/professional resellers (very low risk), a minority are
    # chronically disputed. Mirrors src.data.generate_synthetic_transactions.
    is_risky_seller = rng.random(n_sellers) < 0.10
    seller_risk = np.where(
        is_risky_seller,
        rng.beta(5, 8, n_sellers),   # risky tier: mean ~38 %
        rng.beta(1, 80, n_sellers),  # clean tier: mean ~1.2 %
    )

    seller_ids = rng.integers(0, n_sellers, n)
    buyer_ids = rng.integers(0, n_buyers, n)
    per_txn_seller_risk = seller_risk[seller_ids]

    venue_type = rng.choice(
        ["arena", "stadium", "theater", "club", "festival"],
        n,
        p=[0.35, 0.25, 0.2, 0.12, 0.08],
    )
    event_category = rng.choice(["concert", "sports", "comedy", "theater"], n)
    section_code = rng.choice([f"S{c:04d}" for c in range(300)], n)
    ticket_price = np.round(rng.lognormal(mean=4.5, sigma=0.9, size=n), 2)

    # is_disputed: raw outcome, used by the feature SQL to build seller history
    is_disputed = rng.random(n) < per_txn_seller_risk

    # is_high_risk: training label — correlated with seller risk but NOT with
    # is_disputed of this same transaction (that would be leakage).
    # Same interaction + trust structure as generate_synthetic_transactions.
    price_hi = (ticket_price > np.quantile(ticket_price, 0.75)).astype(float)
    risky_venue = np.isin(venue_type, ["festival", "club"]).astype(float)
    risky_seller_high_price = (per_txn_seller_risk > 0.15).astype(float) * price_hi * 3.6
    logit = (
        -3.85
        + 7.0 * per_txn_seller_risk
        + 0.5 * price_hi
        + 0.5 * risky_venue
        + risky_seller_high_price
        + rng.normal(0, 0.19, n)
    )
    p_risk = 1.0 / (1.0 + np.exp(-logit))
    is_high_risk = rng.random(n) < p_risk

    # Snowflake write_pandas expects uppercase column names
    transactions = pd.DataFrame(
        {
            "TRANSACTION_ID": np.arange(1, n + 1),
            "SELLER_ID": (seller_ids + 1).astype(int),
            "BUYER_ID": (buyer_ids + 1).astype(int),
            "EVENT_DATE": (
                pd.Timestamp("2024-01-01")
                + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
            ),
            "TICKET_PRICE": ticket_price,
            "QUANTITY": (rng.poisson(2, n) + 1).astype(int),
            "VENUE_TYPE": venue_type,
            "EVENT_CATEGORY": event_category,
            "SECTION_CODE": section_code,
            "IS_DISPUTED": is_disputed,
            "IS_HIGH_RISK": is_high_risk,
        }
    )

    buyers = pd.DataFrame(
        {
            "BUYER_ID": np.arange(1, n_buyers + 1),
            "BUYER_AGE": np.clip(rng.normal(34, 11, n_buyers), 18, 75).astype(int),
        }
    )

    return transactions, buyers


# ── 2. Snowflake connection ──────────────────────────────────────────────────

def get_connection():
    """Create Snowflake connection from environment variables.

    Required env vars  : SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    Optional env vars  : SNOWFLAKE_AUTHENTICATOR
                           - username_password_mfa (default): password + TOTP passcode
                           - snowflake: password only (no MFA)
                           - externalbrowser: browser SSO (often fails on trial)
                         SNOWFLAKE_PASSCODE: optional; if missing, prompts in terminal
                         SNOWFLAKE_WAREHOUSE (default COMPUTE_WH),
                         SNOWFLAKE_DATABASE  (default MARKETPLACE),
                         SNOWFLAKE_SCHEMA    (default ANALYTICS)
    """
    _require_snowflake()
    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "username_password_mfa")
    kwargs = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "authenticator": authenticator,
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.environ.get("SNOWFLAKE_DATABASE", "MARKETPLACE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
    }
    if authenticator == "username_password_mfa":
        passcode = os.environ.get("SNOWFLAKE_PASSCODE", "").strip()
        if not passcode:
            passcode = input(
                "Open Google Authenticator → enter the 6-digit Snowflake code: "
            ).strip()
        kwargs["passcode"] = passcode
        kwargs["client_request_mfa_token"] = True
    return snowflake.connector.connect(**kwargs)


# ── 3. Schema setup & upload ─────────────────────────────────────────────────

_DDL_TRANSACTIONS = """
CREATE OR REPLACE TABLE TRANSACTIONS (
    TRANSACTION_ID  INTEGER,
    SELLER_ID       INTEGER,
    BUYER_ID        INTEGER,
    EVENT_DATE      DATE,
    TICKET_PRICE    FLOAT,
    QUANTITY        INTEGER,
    VENUE_TYPE      STRING,
    EVENT_CATEGORY  STRING,
    SECTION_CODE    STRING,
    IS_DISPUTED     BOOLEAN,
    IS_HIGH_RISK    BOOLEAN
)
"""

_DDL_BUYERS = """
CREATE OR REPLACE TABLE BUYERS (
    BUYER_ID   INTEGER,
    BUYER_AGE  INTEGER
)
"""


def _setup_schema(cur) -> None:
    for stmt in [
        "CREATE DATABASE IF NOT EXISTS MARKETPLACE",
        "CREATE SCHEMA IF NOT EXISTS MARKETPLACE.ANALYTICS",
        "USE DATABASE MARKETPLACE",
        "USE SCHEMA ANALYTICS",
    ]:
        cur.execute(stmt)


def upload_raw_tables(
    conn,
    transactions: pd.DataFrame,
    buyers: pd.DataFrame,
) -> None:
    """Create tables in Snowflake and bulk-upload DataFrames.

    Uses write_pandas (internal Snowflake stage + COPY INTO) — same mechanism
    as production ELT pipelines.
    """
    _require_snowflake()
    cur = conn.cursor()
    _setup_schema(cur)
    cur.execute(_DDL_TRANSACTIONS)
    cur.execute(_DDL_BUYERS)

    # write_pandas serializes timestamps as epoch ns VARIANT; Snowflake DATE
    # cannot cast that. Send ISO date strings instead.
    transactions = transactions.copy()
    buyers = buyers.copy()
    transactions["EVENT_DATE"] = pd.to_datetime(transactions["EVENT_DATE"]).dt.strftime(
        "%Y-%m-%d"
    )
    # Native Python bools avoid similar VARIANT cast issues
    for col in ("IS_DISPUTED", "IS_HIGH_RISK"):
        transactions[col] = transactions[col].astype(bool)

    success_t, _, nrows_t, _ = write_pandas(
        conn, transactions, "TRANSACTIONS",
        database="MARKETPLACE", schema="ANALYTICS",
        quote_identifiers=False,
    )
    success_b, _, nrows_b, _ = write_pandas(
        conn, buyers, "BUYERS",
        database="MARKETPLACE", schema="ANALYTICS",
        quote_identifiers=False,
    )

    if not (success_t and success_b):
        raise RuntimeError("write_pandas failed — check Snowflake logs.")

    print(f"  Uploaded {nrows_t:,} rows → TRANSACTIONS")
    print(f"  Uploaded {nrows_b:,} rows → BUYERS")


# ── 4. Feature extraction ────────────────────────────────────────────────────

def extract_features(conn) -> pd.DataFrame:
    """Run point-in-time feature SQL inside Snowflake, return as DataFrame.

    The SQL uses window functions with ROWS BETWEEN UNBOUNDED PRECEDING AND
    1 PRECEDING so that each row only sees history that was available *before*
    that transaction — no leakage into training features.
    """
    _require_snowflake()
    sql = (SQL_DIR / "feature_extraction_snowflake.sql").read_text()

    cur = conn.cursor()
    _setup_schema(cur)
    cur.execute(sql)
    df = cur.fetch_pandas_all()

    # Snowflake returns uppercase column names; lowercase to match local pipeline
    df.columns = [c.lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["event_date"])
    print(f"  Pulled {len(df):,} feature rows from Snowflake")
    return df
