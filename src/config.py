"""Central configuration for the marketplace risk-scoring pipeline."""

from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Target: high-risk transaction / listing (maps to improper claim, fraud, bad listing)
TARGET = "is_high_risk"

# Temporal split: train on past, score future (production-forward evaluation)
TRAIN_QUANTILE = 0.8

# Business decision: auto-approve when risk probability < threshold
FALSE_CLEAR_TOLERANCE = 0.03
AUDIT_FRACTION = 0.02

# Cost-sensitive routing (optional threshold selection)
C_FALSE_CLEAR = 500.0
C_MANUAL_REVIEW = 8.0

# MLflow (sqlite avoids new MLflow's file-store UI restrictions)
MLFLOW_EXPERIMENT = "marketplace-risk-scoring"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "marketplace_risk_scorer"

# Serving defaults (overwritten after training)
DEFAULT_THRESHOLD = 0.05
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACT_DIR / "model.joblib"
