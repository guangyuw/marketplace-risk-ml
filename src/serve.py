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
from src.explain import build_evidence
from src.features import build_features
from src.monitoring import route_transactions
from src.review_assist import ReviewBrief, generate_review_brief

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
    ticket_price: float = Field(..., gt=0, examples=[220.0])
    quantity: int = Field(..., ge=1, examples=[2])
    buyer_age: int = Field(..., ge=18, le=100, examples=[28])
    venue_type: str = Field(..., examples=["festival"])
    event_category: str = Field(..., examples=["concert"])
    section_code: str = Field(..., examples=["S0003"])
    seller_prior_dispute_rate: float = Field(..., ge=0, le=1, examples=[0.0])
    buyer_prior_purchases: int = Field(..., ge=0, examples=[2])
    event_date: str = Field(..., examples=["2025-06-01"])


class PredictionResponse(BaseModel):
    risk_score: float
    route: Literal["auto_approve", "manual_review", "audit"]
    threshold: float
    model_version: str


class ReviewAssistResponse(BaseModel):
    risk_score: float
    route: Literal["auto_approve", "manual_review", "audit"]
    threshold: float
    assist_applicable: bool
    evidence: dict | None = None
    brief: ReviewBrief | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _score_row(features: "TransactionFeatures") -> tuple[pd.DataFrame, float, float, str]:
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
    return row, prob, threshold, route


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures) -> PredictionResponse:
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    _, prob, threshold, route = _score_row(features)

    metadata = _bundle.get("metadata", {})
    model_version = metadata.get("run_id") or metadata.get("model_type", "unknown")

    return PredictionResponse(
        risk_score=prob,
        route=route,
        threshold=threshold,
        model_version=model_version,
    )


@app.post("/review-assist", response_model=ReviewAssistResponse)
def review_assist(features: TransactionFeatures) -> ReviewAssistResponse:
    """Draft a structured review brief for a flagged transaction.

    The scoring model decides the route; this endpoint only adds a reviewer
    aid. It is a no-op for auto-approved transactions — assist is reserved for
    manual_review / audit, where a human is already in the loop.
    """
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")

    row, prob, threshold, route = _score_row(features)

    if route == "auto_approve":
        return ReviewAssistResponse(
            risk_score=prob,
            route=route,
            threshold=threshold,
            assist_applicable=False,
        )

    evidence = build_evidence(
        _bundle, row, risk_score=prob, threshold=threshold, route=route
    )
    brief = generate_review_brief(evidence)

    return ReviewAssistResponse(
        risk_score=prob,
        route=route,
        threshold=threshold,
        assist_applicable=True,
        evidence=evidence,
        brief=brief,
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
