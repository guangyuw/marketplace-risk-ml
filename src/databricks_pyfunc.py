"""MLflow pyfunc for Databricks Model Serving — same scoring path as FastAPI.

Logs a self-contained model that:
  raw transaction fields → feature build → Platt-calibrated risk_score → route

Does not depend on importing ``src.*`` at inference time (Serving containers
only have the logged code + pip deps). Feature logic is duplicated here on
purpose so the UC-registered model is deployable without the Git folder.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd


RAW_INPUT_COLUMNS = [
    "ticket_price",
    "quantity",
    "buyer_age",
    "venue_type",
    "event_category",
    "section_code",
    "seller_prior_dispute_rate",
    "buyer_prior_purchases",
    "event_date",
]

_CAT_FREQ_COLS = ["venue_type", "event_category", "section_code"]
_HIST_COLS = ["seller_prior_dispute_rate", "buyer_prior_purchases"]


def _build_features(df: pd.DataFrame, freq_maps: dict[str, pd.Series]) -> pd.DataFrame:
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["event_date"]):
        d["event_date"] = pd.to_datetime(d["event_date"])

    for c in _HIST_COLS:
        d[f"{c}_isna"] = d[c].isna().astype(int)
        d[c] = d[c].fillna(0)

    d["log_price"] = np.log1p(d["ticket_price"].astype(float))
    d["dow"] = d["event_date"].dt.dayofweek.astype(int)
    d["month"] = d["event_date"].dt.month.astype(int)

    for c in _CAT_FREQ_COLS:
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
        + [f"{c}_freq" for c in _CAT_FREQ_COLS]
    )
    return d[feature_cols]


def _route(scores: np.ndarray, threshold: float, audit_fraction: float = 0.02) -> list[str]:
    """Match FastAPI routing (auto_approve / manual_review / audit)."""
    rng = np.random.default_rng()
    auto = scores < threshold
    audit = auto & (rng.random(len(scores)) < audit_fraction)
    out: list[str] = []
    for is_auto, is_audit in zip(auto, audit):
        if is_audit:
            out.append("audit")
        elif is_auto:
            out.append("auto_approve")
        else:
            out.append("manual_review")
    return out


class MarketplaceRiskScorer(mlflow.pyfunc.PythonModel):
    """Production scorer for Databricks Model Serving / UC registration."""

    def load_context(self, context: Any) -> None:
        import joblib

        payload = joblib.load(context.artifacts["serving_payload"])
        self.base_model = payload["base_model"]
        self.platt = payload["platt"]
        self.freq_maps = payload["freq_maps"]
        self.threshold = float(payload["threshold"])
        self.feature_columns = list(payload["feature_columns"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        missing = [c for c in RAW_INPUT_COLUMNS if c not in model_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        x = _build_features(model_input[RAW_INPUT_COLUMNS], self.freq_maps)
        x = x[self.feature_columns]

        raw = self.base_model.predict_proba(x)[:, 1].reshape(-1, 1)
        scores = self.platt.predict_proba(raw)[:, 1]
        routes = _route(scores, self.threshold)

        return pd.DataFrame(
            {
                "risk_score": scores,
                "route": routes,
                "threshold": self.threshold,
            }
        )


def make_raw_input_example(n: int = 3) -> pd.DataFrame:
    """Small raw-input example for signature / Serving UI."""
    rows = [
        {
            "ticket_price": 185.0,
            "quantity": 2,
            "buyer_age": 29,
            "venue_type": "arena",
            "event_category": "concert",
            "section_code": "S0042",
            "seller_prior_dispute_rate": 0.5,
            "buyer_prior_purchases": 3,
            "event_date": "2025-03-15",
        },
        {
            "ticket_price": 90.0,
            "quantity": 1,
            "buyer_age": 41,
            "venue_type": "stadium",
            "event_category": "sports",
            "section_code": "S0010",
            "seller_prior_dispute_rate": 0.02,
            "buyer_prior_purchases": 12,
            "event_date": "2025-06-01",
        },
        {
            "ticket_price": 420.0,
            "quantity": 4,
            "buyer_age": 22,
            "venue_type": "theater",
            "event_category": "comedy",
            "section_code": "S0099",
            "seller_prior_dispute_rate": 0.35,
            "buyer_prior_purchases": 0,
            "event_date": "2025-01-20",
        },
    ]
    return pd.DataFrame(rows[:n])


# Models-from-code entrypoint for mlflow.pyfunc.log_model(python_model=<this file>).
mlflow.models.set_model(MarketplaceRiskScorer())
