"""
FastAPI wrapper exposing /predict for the React frontend.
Run locally:
  uvicorn fastapi_app:app --reload --port 8000
"""
from typing import Optional, List, Any
import numpy as np
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prediction_model import predict_next_for_ticker, walk_forward_backtest

app = FastAPI(title="Stock Predictor API", version="1.0.0")

# Allow browser apps; tighten origins in production via env or config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    ticker: str
    horizon: int = 1
    model_type: str = "rf"


class PredictResponse(BaseModel):
    ticker: str
    model_type: str
    horizon: int
    pred_next_ret: float
    pred_next_price: float
    confidence_score: float | None = None
    prob_up: float | None = None
    prob_down: float | None = None
    prob_up_gaf: float | None = None
    last_close: float | None = None
    vol_20d: float | None = None
    walk_forward: dict[str, Any] | None = None
    feature_importance: list[dict[str, Any]] | None = None


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _serialize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest = Body(...)):
    res = predict_next_for_ticker(
        payload.ticker,
        period="5y",
        model_type=payload.model_type,
        horizon=payload.horizon,
    )

    # optional: walk-forward metrics for the same ticker/model
    wf_metrics = None
    try:
        wf_metrics = walk_forward_backtest(payload.ticker, model_type=payload.model_type, horizon=payload.horizon)
    except Exception:
        wf_metrics = None

    # feature importance if present
    feat_imp_raw = res.get("feature_importance") if isinstance(res, dict) else None
    feature_importance = None
    if isinstance(feat_imp_raw, dict):
        feature_importance = [
            {"feature": k, "importance": _to_float(v)} for k, v in feat_imp_raw.items()
        ]

    return PredictResponse(
        ticker=payload.ticker.upper(),
        model_type=payload.model_type,
        horizon=payload.horizon,
        pred_next_ret=_to_float(res.get("pred_next_ret")),
        pred_next_price=_to_float(res.get("pred_next_price")),
        confidence_score=_to_float(res.get("confidence_score")),
        prob_up=_to_float(res.get("prob_up")),
        prob_down=_to_float(res.get("prob_down")),
        prob_up_gaf=_to_float(res.get("prob_up_gaf")),
        last_close=_to_float(res.get("last_close")),
        vol_20d=_to_float(res.get("vol_20d")),
        walk_forward=_serialize(wf_metrics) if wf_metrics else None,
        feature_importance=feature_importance,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
