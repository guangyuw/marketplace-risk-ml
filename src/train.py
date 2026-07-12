"""End-to-end training with MLflow tracking and model registration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _git_commit_hash() -> str:
    """Return current HEAD commit hash, or 'untracked' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "untracked"

from src.config import (
    ARTIFACT_DIR,
    FALSE_CLEAR_TOLERANCE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    TARGET,
)
from src.data import generate_synthetic_transactions, temporal_three_way_split
from src.features import build_features, prepare_datasets
from src.model import PlattCalibrated, classification_metrics, train_logistic_regression, train_xgboost
from src.monitoring import choose_threshold_by_tolerance, psi


def train_pipeline(
    artifact_dir: Path | None = None,
    register_model: bool = False,
    df=None,
) -> dict:
    """Train models, log to MLflow, persist production bundle locally.

    Parameters
    ----------
    df : DataFrame, optional
        Feature snapshot to train on.  When provided (e.g. pulled from
        Snowflake via snowflake_data.extract_features), it is used as-is.
        When None (default), a local synthetic dataset is generated — useful
        for offline development without a Snowflake connection.
    """
    artifact_dir = artifact_dir or ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── 1. Data: strict temporal 60 / 20 / 20 split ──────────────────────────
    # train  [0%–60%]  → fit XGBoost
    # calib  [60%–80%] → fit Platt-scaling calibrator on held-out window
    # test   [80%–100%]→ final evaluation only (never used for fitting)
    data_source = "local_synthetic" if df is None else "snowflake"
    if df is None:
        df = generate_synthetic_transactions()
    train_df, calib_df, test_df = temporal_three_way_split(df, train_q=0.6, calib_q=0.8)

    x_train, y_train, x_test, y_test, freq_maps = prepare_datasets(train_df, test_df, TARGET)
    # Calib features built with freq_maps fitted on train only (no leakage)
    x_calib = build_features(calib_df, freq_maps)
    y_calib = calib_df[TARGET].values

    with mlflow.start_run(run_name="marketplace-risk-baseline") as run:
        mlflow.set_tag("git_commit", _git_commit_hash())
        mlflow.log_params(
            {
                "train_rows": len(train_df),
                "calib_rows": len(calib_df),
                "test_rows": len(test_df),
                "split": "60/20/20 temporal",
                "calibration_method": "platt_sigmoid",
                "false_clear_tolerance": FALSE_CLEAR_TOLERANCE,
                "random_state": RANDOM_STATE,
                "data_source": data_source,
            }
        )

        # ── 2. Logistic regression baseline ──────────────────────────────────
        lr = train_logistic_regression(x_train, y_train)
        p_lr = lr.predict_proba(x_test)[:, 1]
        lr_metrics = classification_metrics(y_test, p_lr)
        for k, v in lr_metrics.items():
            mlflow.log_metric(f"logreg_{k}", v)

        # ── 3. XGBoost (raw, uncalibrated) ───────────────────────────────────
        gbm = train_xgboost(x_train, y_train)
        p_gbm_raw = gbm.predict_proba(x_test)[:, 1]
        gbm_metrics = classification_metrics(y_test, p_gbm_raw)
        for k, v in gbm_metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)

        brier_raw = brier_score_loss(y_test, p_gbm_raw)
        mlflow.log_metric("xgb_brier_raw", brier_raw)

        # ── 4. Platt-scaling calibration on the calib window ─────────────────
        # Fit a logistic sigmoid on the raw scores evaluated on the calib window.
        # C=1e10 → near-zero regularization (pure Platt scaling, Platt 1999).
        p_calib_raw = gbm.predict_proba(x_calib)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
        platt.fit(p_calib_raw, y_calib)
        calibrated = PlattCalibrated(gbm, platt)

        p_gbm = calibrated.predict_proba(x_test)[:, 1]
        brier_cal = brier_score_loss(y_test, p_gbm)
        mlflow.log_metric("xgb_brier_calibrated", brier_cal)

        # ── 5. Threshold selection on calibrated probabilities ────────────────
        threshold, _ = choose_threshold_by_tolerance(
            y_test, p_gbm, tolerance=FALSE_CLEAR_TOLERANCE
        )
        mlflow.log_metric("chosen_threshold", threshold)

        p_gbm_train = calibrated.predict_proba(x_train)[:, 1]
        score_psi = psi(p_gbm_train, p_gbm)
        mlflow.log_metric("score_psi", score_psi)

        # ── 6. Calibration curve: before vs after ────────────────────────────
        frac_raw, pred_raw = calibration_curve(y_test, p_gbm_raw, n_bins=10, strategy="quantile")
        frac_cal, pred_cal = calibration_curve(y_test, p_gbm, n_bins=10, strategy="quantile")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", c="gray", label="perfectly calibrated")
        ax.plot(pred_raw, frac_raw, "o-", label=f"XGB raw  (Brier={brier_raw:.4f})")
        ax.plot(pred_cal, frac_cal, "s-", label=f"XGB+Platt (Brier={brier_cal:.4f})")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual high-risk rate")
        ax.legend()
        ax.set_title("Calibration: before vs after Platt scaling")
        fig.tight_layout()
        calib_path = artifact_dir / "calibration_curve.png"
        fig.savefig(calib_path)
        plt.close(fig)
        mlflow.log_artifact(str(calib_path))

        # ── 7. Bundle: store calibrated model so serve.py needs no changes ───
        bundle = {
            "model": calibrated,
            "freq_maps": freq_maps,
            "threshold": threshold,
            "feature_columns": list(x_train.columns),
            "metadata": {
                "model_type": "xgboost+platt",
                "run_id": run.info.run_id,
                "target": TARGET,
                "train_cutoff": str(train_df["event_date"].max().date()),
                "metrics": gbm_metrics,
                "brier_raw": round(brier_raw, 6),
                "brier_calibrated": round(brier_cal, 6),
            },
        }

        model_path = artifact_dir / "model.joblib"
        joblib.dump(bundle, model_path)
        meta_path = artifact_dir / "metadata.json"
        meta_path.write_text(json.dumps(bundle["metadata"], indent=2))

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))
        # Log raw XGBoost to MLflow registry (Databricks Model Serving uses this).
        # The calibrated wrapper lives in model.joblib and is used by serve.py.
        mlflow.xgboost.log_model(
            gbm,
            artifact_path="xgb_model",
            registered_model_name=REGISTERED_MODEL_NAME if register_model else None,
        )

        print("=== Training complete ===")
        print(f"MLflow run_id  : {run.info.run_id}")
        print(f"Split          : train={len(train_df):,}  calib={len(calib_df):,}  test={len(test_df):,}")
        print(f"LogReg PR-AUC  : {lr_metrics['pr_auc']:.4f}")
        print(f"XGB   PR-AUC   : {gbm_metrics['pr_auc']:.4f}")
        print(f"Brier raw      : {brier_raw:.4f}  →  calibrated: {brier_cal:.4f}")
        print(f"Threshold (tol={FALSE_CLEAR_TOLERANCE:.0%}): {threshold:.4f}")
        print(f"Artifacts      : {artifact_dir}")

        return {
            "run_id": run.info.run_id,
            "threshold": threshold,
            "metrics": gbm_metrics,
            "model_path": str(model_path),
        }


if __name__ == "__main__":
    train_pipeline()
