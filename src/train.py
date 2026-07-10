"""End-to-end training with MLflow tracking and model registration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.calibration import calibration_curve
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
from src.data import generate_synthetic_transactions, temporal_train_test_split
from src.features import prepare_datasets
from src.model import classification_metrics, train_logistic_regression, train_xgboost
from src.monitoring import choose_threshold_by_tolerance, psi


def train_pipeline(
    artifact_dir: Path | None = None,
    register_model: bool = False,
) -> dict:
    """Train models, log to MLflow, persist production bundle locally."""
    artifact_dir = artifact_dir or ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = generate_synthetic_transactions()
    train_df, test_df = temporal_train_test_split(df)
    x_train, y_train, x_test, y_test, freq_maps = prepare_datasets(train_df, test_df, TARGET)

    with mlflow.start_run(run_name="marketplace-risk-baseline") as run:
        mlflow.set_tag("git_commit", _git_commit_hash())
        mlflow.log_params(
            {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "false_clear_tolerance": FALSE_CLEAR_TOLERANCE,
                "random_state": RANDOM_STATE,
            }
        )

        lr = train_logistic_regression(x_train, y_train)
        p_lr = lr.predict_proba(x_test)[:, 1]
        lr_metrics = classification_metrics(y_test, p_lr)
        for k, v in lr_metrics.items():
            mlflow.log_metric(f"logreg_{k}", v)

        gbm = train_xgboost(x_train, y_train)
        p_gbm = gbm.predict_proba(x_test)[:, 1]
        gbm_metrics = classification_metrics(y_test, p_gbm)
        for k, v in gbm_metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)

        threshold, _ = choose_threshold_by_tolerance(
            y_test, p_gbm, tolerance=FALSE_CLEAR_TOLERANCE
        )
        mlflow.log_metric("chosen_threshold", threshold)
        p_gbm_train = gbm.predict_proba(x_train)[:, 1]
        score_psi = psi(p_gbm_train, p_gbm)
        mlflow.log_metric("score_psi", score_psi)

        # Step 9: probability calibration — Brier score + calibration curve
        brier = brier_score_loss(y_test, p_gbm)
        mlflow.log_metric("xgb_brier_score", brier)

        frac_pos, mean_pred = calibration_curve(y_test, p_gbm, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "--", c="gray", label="perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "o-", label=f"GBM (Brier={brier:.4f})")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual high-risk rate")
        ax.legend()
        ax.set_title("Calibration curve")
        fig.tight_layout()
        calib_path = artifact_dir / "calibration_curve.png"
        fig.savefig(calib_path)
        plt.close(fig)
        mlflow.log_artifact(str(calib_path))

        bundle = {
            "model": gbm,
            "freq_maps": freq_maps,
            "threshold": threshold,
            "feature_columns": list(x_train.columns),
            "metadata": {
                "model_type": "xgboost",
                "run_id": run.info.run_id,
                "target": TARGET,
                "train_cutoff": str(train_df["event_date"].max().date()),
                "metrics": gbm_metrics,
            },
        }

        model_path = artifact_dir / "model.joblib"
        joblib.dump(bundle, model_path)
        meta_path = artifact_dir / "metadata.json"
        meta_path.write_text(json.dumps(bundle["metadata"], indent=2))

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))
        mlflow.xgboost.log_model(
            gbm,
            artifact_path="xgb_model",
            registered_model_name=REGISTERED_MODEL_NAME if register_model else None,
        )

        print("=== Training complete ===")
        print(f"MLflow run_id: {run.info.run_id}")
        print(f"LogReg PR-AUC: {lr_metrics['pr_auc']:.4f}")
        print(f"XGB   PR-AUC: {gbm_metrics['pr_auc']:.4f}")
        print(f"XGB   Brier:  {brier:.4f}")
        print(f"Threshold (tolerance={FALSE_CLEAR_TOLERANCE:.0%}): {threshold:.4f}")
        print(f"Artifacts: {artifact_dir}")

        return {
            "run_id": run.info.run_id,
            "threshold": threshold,
            "metrics": gbm_metrics,
            "model_path": str(model_path),
        }


if __name__ == "__main__":
    train_pipeline()
