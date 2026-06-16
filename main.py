"""
AlphaQuant - Model Inference REST API
Provides high-performance serving of LSTM trading signal models via FastAPI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import uvicorn
import os
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaQuant Prediction API",
    description="REST API for executing inference on Deep Learning trading signal models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model holders ───────────────────────────────────────────────────────
buy_model = None
sell_model = None
buy_scaler = None
sell_scaler = None

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class BuyPredictRequest(BaseModel):
    """
    Input: 20 consecutive trading days, each with 11 features:
    [Open, High, Low, Close, Volume, SMA_14, RSI_14, Dist_to_SMA,
     Log_Return, priceToEarning, cashDividendPercentage]
    """
    features: List[List[float]] = Field(
        ...,
        min_items=20,
        max_items=20,
        description="List of 20 rows, each row has 11 feature values",
        example=[
            [6000, 6200, 5800, 6100, 1200000, 6050, 55.2, 0.008, 0.012, 15.3, 0.05]
        ] * 20,
    )


class SellPredictRequest(BaseModel):
    """
    Input: 20 consecutive trading days, each with features for sell model:
    [Open, High, Low, Volume, Log_Return, Volatility_14, SMA_14,
     Dist_to_SMA, RSI_14, Dist_to_UB, BB_Percent_B, priceToEarning, cashDividendPercentage]
    """
    features: List[List[float]] = Field(
        ...,
        min_items=20,
        max_items=20,
        description="List of 20 rows, each row has 13 feature values",
        example=[
            [6000, 6200, 5800, 1200000, 0.012, 0.023, 6050, 0.008, 55.2, -0.03, 0.65, 15.3, 0.05]
        ] * 20,
    )


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Probability score (0.0 - 1.0)")
    recommendation: str = Field(..., description="BUY / IGNORE or SELL / HOLD")
    confidence: str = Field(..., description="HIGH / MEDIUM / LOW")
    model: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    buy_model_loaded: bool
    sell_model_loaded: bool
    message: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_confidence(prob: float) -> str:
    if prob >= 0.70 or prob <= 0.30:
        return "HIGH"
    elif prob >= 0.55 or prob <= 0.45:
        return "MEDIUM"
    return "LOW"


def load_models():
    """Load models and scalers from disk (called at startup)."""
    global buy_model, sell_model, buy_scaler, sell_scaler

    try:
        import tensorflow as tf

        buy_path = os.getenv("BUY_MODEL_PATH", "my_stock_models/buy_model")
        sell_path = os.getenv("SELL_MODEL_PATH", "my_stock_models/sell_model")
        buy_scaler_path = os.getenv("BUY_SCALER_PATH", "my_stock_models/buy_scaler.pkl")
        sell_scaler_path = os.getenv("SELL_SCALER_PATH", "my_stock_models/sell_scaler.pkl")

        if os.path.exists(buy_path):
            buy_model = tf.saved_model.load(buy_path)
            logger.info(f"✅ Buy model loaded from {buy_path}")
            if os.path.exists(buy_scaler_path):
                import joblib
                buy_scaler = joblib.load(buy_scaler_path)
                logger.info("✅ Buy scaler loaded.")
        else:
            logger.warning(f"⚠️  Buy model not found at {buy_path}. Using mock.")

        if os.path.exists(sell_path):
            sell_model = tf.saved_model.load(sell_path)
            logger.info(f"✅ Sell model loaded from {sell_path}")
            if os.path.exists(sell_scaler_path):
                import joblib
                sell_scaler = joblib.load(sell_scaler_path)
                logger.info("✅ Sell scaler loaded.")
        else:
            logger.warning(f"⚠️  Sell model not found at {sell_path}. Using mock.")

    except Exception as e:
        logger.error(f"Model loading error: {e}")


def mock_predict(features: list, seed: int = 42) -> float:
    """
    Mock inference when real model is not available.
    Uses a hash of the input to return a deterministic pseudo-probability.
    """
    arr = np.array(features, dtype=np.float32)
    # Avoid saturation from large values like Volume by taking modulo
    pseudo_hash = np.sum(arr) % 100
    val = (pseudo_hash / 100.0) * 0.6 + 0.2 # Range [0.2, 0.8]
    return round(val, 4)


def run_inference(model, scaler, features: list, input_shape: tuple) -> float:
    """Run real TF inference with scaling, or fall back to mock."""
    if model is None:
        return mock_predict(features)

    try:
        import tensorflow as tf
        x = np.array(features, dtype=np.float32)
        
        # Apply scaler if available to prevent saturation (Training-Serving Skew)
        if scaler is not None:
            # Reshape to 2D for scaling, then back to 3D
            x_2d = x.reshape(-1, input_shape[1])
            x_scaled = scaler.transform(x_2d)
            x = x_scaled.reshape(1, *input_shape)
        elif np.max(x) > 10:
            # Fallback: if scaler is missing and data is unscaled (e.g. Volume > 1000000),
            # manually MinMax scale per column across the 20 days to prevent 1.0 saturation.
            x_2d = x.reshape(-1, input_shape[1])
            col_min = np.min(x_2d, axis=0)
            col_max = np.max(x_2d, axis=0)
            range_vals = np.where((col_max - col_min) == 0, 1, col_max - col_min)
            x_scaled = (x_2d - col_min) / range_vals
            x = x_scaled.reshape(1, *input_shape)
        else:
            x = x.reshape(1, *input_shape)
            
        tensor = tf.constant(x)
        result = model(tensor)
        prob = float(result.numpy().flatten()[0])
        return round(prob, 4)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return mock_predict(features)


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_models()
    logger.info("🚀 Stock Prediction API is running.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AlphaQuant Prediction API - Serving Layer",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict/buy", "/predict/sell"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        buy_model_loaded=(buy_model is not None),
        sell_model_loaded=(sell_model is not None),
        message="API is running. Models use mock inference if not loaded.",
    )


@app.post("/predict/buy", response_model=PredictionResponse, tags=["Prediction"])
def predict_buy(request: BuyPredictRequest):
    """
    Predict Buy Signal probability.

    - **features**: 20 rows × 11 columns (20 trading days of data)
    - Returns probability ∈ [0,1] and BUY / IGNORE recommendation
    """
    if len(request.features) != 20:
        raise HTTPException(status_code=422, detail="Exactly 20 time steps required.")
    for i, row in enumerate(request.features):
        if len(row) != 11:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i} must have 11 features, got {len(row)}.",
            )

    BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.40"))
    prob = run_inference(buy_model, buy_scaler, request.features, input_shape=(20, 11))
    recommendation = "BUY" if prob >= BUY_THRESHOLD else "IGNORE"

    return PredictionResponse(
        probability=prob,
        recommendation=recommendation,
        confidence=get_confidence(prob),
        model="buy_signal_lstm",
        threshold_used=BUY_THRESHOLD,
    )


@app.post("/predict/sell", response_model=PredictionResponse, tags=["Prediction"])
def predict_sell(request: SellPredictRequest):
    """
    Predict Sell Signal probability.

    - **features**: 20 rows × 13 columns (20 trading days of data)
    - Returns probability ∈ [0,1] and SELL / HOLD recommendation
    """
    if len(request.features) != 20:
        raise HTTPException(status_code=422, detail="Exactly 20 time steps required.")
    for i, row in enumerate(request.features):
        if len(row) != 13:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i} must have 13 features, got {len(row)}.",
            )

    SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.50"))
    prob = run_inference(sell_model, sell_scaler, request.features, input_shape=(20, 13))
    recommendation = "SELL" if prob >= SELL_THRESHOLD else "HOLD"

    return PredictionResponse(
        probability=prob,
        recommendation=recommendation,
        confidence=get_confidence(prob),
        model="sell_signal_lstm",
        threshold_used=SELL_THRESHOLD,
    )


@app.get("/features/buy", tags=["Schema"])
def buy_feature_schema():
    """Describe the 11 input features expected by the buy model."""
    return {
        "time_steps": 20,
        "n_features": 11,
        "feature_names": [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_14", "RSI_14", "Dist_to_SMA",
            "Log_Return", "priceToEarning", "cashDividendPercentage",
        ],
        "note": "Features must be in the same scale/order as training data.",
    }


@app.get("/features/sell", tags=["Schema"])
def sell_feature_schema():
    """Describe the 13 input features expected by the sell model."""
    return {
        "time_steps": 20,
        "n_features": 13,
        "feature_names": [
            "Open", "High", "Low", "Volume",
            "Log_Return", "Volatility_14", "SMA_14", "Dist_to_SMA",
            "RSI_14", "Dist_to_UB", "BB_Percent_B",
            "priceToEarning", "cashDividendPercentage",
        ],
        "note": "Features must be in the same scale/order as training data.",
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
