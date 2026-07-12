"""End-to-end Snowflake → feature extraction → model training pipeline.

This script replicates the production ML workflow for a marketplace risk model:

  [1] Generate synthetic raw event log (transactions + buyers)
       ↓  (in production: app DB / Kafka / S3 → warehouse ETL)
  [2] Upload raw tables to Snowflake (MARKETPLACE.ANALYTICS)
       ↓  (in production: this data already lives in the warehouse)
  [3] Run point-in-time feature SQL inside Snowflake
       - Window functions ensure each row only sees history before its date
       - No leakage: ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
       ↓  (in production: this runs as a scheduled dbt model or Snowflake task)
  [4] Pull feature snapshot to Python
       ↓
  [5] Temporal 60/20/20 train/calib/test split
       ↓
  [6] Train XGBoost + Platt calibration, log to MLflow
       ↓  (in production: logged to Databricks managed MLflow)
  [7] Save model bundle (serve.py uses it unchanged)

Setup
-----
Copy .env.example to .env and fill in your Snowflake credentials, then:

    cd marketplace-risk-ml
    pip install -r requirements.txt
    python scripts/snowflake_pipeline.py

Or pass --no-upload to skip re-uploading data (reuse existing Snowflake tables):

    python scripts/snowflake_pipeline.py --no-upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing src/ when run from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed

from src.snowflake_data import (
    extract_features,
    generate_raw_tables,
    get_connection,
    upload_raw_tables,
)
from src.train import train_pipeline


def main(upload: bool = True) -> None:
    print("=" * 60)
    print("Snowflake → Train Pipeline")
    print("=" * 60)

    # ── Step 1: Generate raw synthetic data ──────────────────────────────────
    print("\n[1/5] Generating raw synthetic data (40k transactions)...")
    transactions, buyers = generate_raw_tables(n=40_000)
    print(f"      {len(transactions):,} transactions | {buyers['BUYER_ID'].nunique():,} unique buyers")

    # ── Step 2: Connect to Snowflake ─────────────────────────────────────────
    print("\n[2/5] Connecting to Snowflake...")
    conn = get_connection()
    print("      Connected.")

    # ── Step 3: Upload raw tables (optional) ─────────────────────────────────
    if upload:
        print("\n[3/5] Uploading raw tables to Snowflake...")
        upload_raw_tables(conn, transactions, buyers)
    else:
        print("\n[3/5] Skipping upload (--no-upload); using existing Snowflake tables.")

    # ── Step 4: Extract point-in-time features from Snowflake ────────────────
    print("\n[4/5] Extracting point-in-time features from Snowflake...")
    df = extract_features(conn)
    conn.close()

    # Rename label column to match train pipeline convention
    df = df.rename(columns={"label": "is_high_risk"})
    print(f"      Feature shape: {df.shape}  |  label rate: {df['is_high_risk'].mean():.2%}")

    # ── Step 5: Train model using the Snowflake feature snapshot ─────────────
    print("\n[5/5] Training model (same pipeline as local — data source = Snowflake)...")
    result = train_pipeline(df=df)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  MLflow run_id : {result['run_id']}")
    print(f"  PR-AUC        : {result['metrics']['pr_auc']:.4f}")
    print(f"  Threshold     : {result['threshold']:.4f}")
    print(f"  Model bundle  : {result['model_path']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snowflake → train pipeline")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading raw tables (reuse existing Snowflake tables)",
    )
    args = parser.parse_args()
    main(upload=not args.no_upload)
