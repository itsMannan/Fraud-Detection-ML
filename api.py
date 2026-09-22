#!/usr/bin/env python3
"""Real-time scoring API.

    uvicorn api:app --host 0.0.0.0 --port 8000

POST /predict
{
  "transactions": [
    {"Time": 0, "Amount": 149.62, "V1": -1.36, "...": "..."}
  ]
}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.config import MODELS  # noqa: E402
from fraud_detection.persist import load_bundle  # noqa: E402
from predict import score_frame_with_bundle  # noqa: E402

app = FastAPI(
    title="Credit Card Fraud Risk API",
    description="Scores ULB-format transactions with the champion fraud model.",
    version="1.0.0",
)
_BUNDLE: dict | None = None


class PredictRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(..., min_length=1)


def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        path = MODELS / "champion_bundle.joblib"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model bundle not found. Run `python train.py` first.",
            )
        _BUNDLE = load_bundle(path)
    return _BUNDLE


@app.get("/health")
def health() -> dict:
    path = MODELS / "champion_bundle.joblib"
    return {"status": "ok" if path.exists() else "untrained", "model_present": path.exists()}


@app.get("/model")
def model_info() -> dict:
    bundle = get_bundle()
    return {
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "feature_names": bundle["feature_names"],
        "metrics": bundle.get("metrics", {}),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    bundle = get_bundle()
    df = pd.DataFrame(req.transactions)
    try:
        scored = score_frame_with_bundle(df, bundle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "results": scored.to_dict(orient="records"),
    }
