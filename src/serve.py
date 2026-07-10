"""FastAPI inference service for real-time risk scoring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_BUNDLE_PATH
from src.features import build_features
from src.monitoring import route_transactions

_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_BUNDLE_PATH}. Run: python -m src.train"
        )
    _bundle.update(joblib.load(MODEL_BUNDLE_PATH))
    yield
    _bundle.clear()


app = FastAPI(
    title="Marketplace Risk Scorer",
    description="Production-style API for transaction risk scoring and routing",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionFeatures(BaseModel):
    ticket_price: float = Field(..., gt=0, examples=[185.0])
    quantity: int = Field(..., ge=1, examples=[2])
    buyer_age: int = Field(..., ge=18, le=100, examples=[29])
    venue_type: str = Field(..., examples=["arena"])
    event_category: str = Field(..., examples=["concert"])
    section_code: str = Field(..., examples=["S0042"])
    seller_prior_dispute_rate: float = Field(..., ge=0, le=1, examples=[0.04])
    buyer_prior_purchases: int = Field(..., ge=0, examples=[3])
    event_date: str = Field(..., examples=["2025-03-15"])


class PredictionResponse(BaseModel):
    risk_score: float
    route: Literal["auto_approve", "manual_review", "audit"]
    threshold: float
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures) -> PredictionResponse:
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    row = pd.DataFrame(
        [
            {
                **features.model_dump(),
                "event_date": pd.to_datetime(features.event_date),
            }
        ]
    )
    x = build_features(row, _bundle["freq_maps"])
    prob = float(_bundle["model"].predict_proba(x)[:, 1][0])
    threshold = float(_bundle["threshold"])
    route = route_transactions(np.array([prob]), threshold)[0]

    metadata = _bundle.get("metadata", {})
    model_version = metadata.get("run_id") or metadata.get("model_type", "unknown")

    return PredictionResponse(
        risk_score=prob,
        route=route,
        threshold=threshold,
        model_version=model_version,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
