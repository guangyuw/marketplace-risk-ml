"""Model training utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


class PlattCalibrated:
    """Platt-scaling wrapper around a pre-fitted classifier.

    Fits a logistic sigmoid (Platt, 1999) on a held-out calibration window so
    that predicted probabilities are better aligned with observed frequencies.
    Interface matches sklearn: .predict_proba(X) returns shape (n, 2).

    Defined here (not in train.py) so joblib can always find the class at
    src.model.PlattCalibrated regardless of which script loaded the model.
    """

    def __init__(self, base_model, platt: LogisticRegression) -> None:
        self.base_model = base_model
        self.platt = platt

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt.predict_proba(raw)


def train_logistic_regression(x_train, y_train) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(x_train, y_train) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    sample_weight = np.where(y_train == 1, neg / max(pos, 1), 1.0)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "positive_rate": float(np.mean(y_true)),
    }
