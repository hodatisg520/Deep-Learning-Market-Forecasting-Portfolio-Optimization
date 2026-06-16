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

class LivePredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol (e.g., SAM.VN)")
    mode: str = Field("buy", description="'buy' or 'sell' mode")

class LivePredictionResponse(BaseModel):
    probability: float
    recommendation: str
    confidence: str
    model: str
    threshold_used: float
    features_used: List[List[float]]


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
            logger.warning(f"⚠️  Buy model not found at {buy_path}.")

        if os.path.exists(sell_path):
            sell_model = tf.saved_model.load(sell_path)
            logger.info(f"✅ Sell model loaded from {sell_path}")
            if os.path.exists(sell_scaler_path):
                import joblib
                sell_scaler = joblib.load(sell_scaler_path)
                logger.info("✅ Sell scaler loaded.")
        else:
            logger.warning(f"⚠️  Sell model not found at {sell_path}.")

    except Exception as e:
        logger.error(f"Model loading error: {e}")


def run_inference(model, scaler, features: list, input_shape: tuple) -> float:
    """Run real TF inference with scaling."""
    if model is None or scaler is None:
        raise RuntimeError("Model or Scaler is not loaded on the server.")

    import tensorflow as tf
    x = np.array(features, dtype=np.float32)
    
    # Apply scaler (Training-Serving Skew Prevention)
    x_2d = x.reshape(-1, input_shape[1])
    
    # Check if the scaler expects more features (e.g. 14) and pad with zeros
    expected_features = getattr(scaler, "n_features_in_", input_shape[1])
    pad_width = expected_features - x_2d.shape[1]
    
    if pad_width > 0:
        padding = np.zeros((x_2d.shape[0], pad_width))
        x_2d_padded = np.hstack((x_2d, padding))
        x_scaled_padded = scaler.transform(x_2d_padded)
        x_scaled = x_scaled_padded[:, :input_shape[1]]
    else:
        x_scaled = scaler.transform(x_2d)
        
    x = x_scaled.reshape(1, *input_shape).astype(np.float32)
        
    tensor = tf.constant(x)
    
    # Call using serving_default signature to avoid _UserObject callable errors
    infer = model.signatures['serving_default']
    result = infer(tensor)
    prob = float(list(result.values())[0].numpy().flatten()[0])
    return round(prob, 4)


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
        message="API is running.",
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
    try:
        prob = run_inference(buy_model, buy_scaler, request.features, input_shape=(20, 11))
    except Exception as e:
        logger.error(f"Buy Inference Error: {e}")
        raise HTTPException(status_code=503, detail="Model inference failed. Models may not be loaded.")

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
    try:
        prob = run_inference(sell_model, sell_scaler, request.features, input_shape=(20, 13))
    except Exception as e:
        logger.error(f"Sell Inference Error: {e}")
        raise HTTPException(status_code=503, detail="Model inference failed. Models may not be loaded.")

    recommendation = "SELL" if prob >= SELL_THRESHOLD else "HOLD"

    return PredictionResponse(
        probability=prob,
        recommendation=recommendation,
        confidence=get_confidence(prob),
        model="sell_signal_lstm",
        threshold_used=SELL_THRESHOLD,
    )


@app.post("/predict/live", response_model=LivePredictionResponse, tags=["Prediction"])
def predict_live(request: LivePredictRequest):
    """
    Fetch live data from Yahoo Finance and predict instantly.
    """
    import yfinance as yf
    import pandas as pd
    
    ticker = request.ticker
    try:
        df = yf.download(ticker, period="60d", progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.reset_index(inplace=True)
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        df['sma_14'] = df['close'].rolling(window=14).mean()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_14'] = df['log_return'].rolling(window=14).std()
        df.dropna(inplace=True)
        
        df_recent = df.tail(20)
        if len(df_recent) < 20:
            raise HTTPException(status_code=400, detail="Not enough data after rolling windows.")
            
        features = df_recent[['open', 'high', 'low', 'close', 'volume', 'sma_14', 'log_return', 'volatility_14']].values
        
        # Pad to 14 columns to satisfy the scaler
        pad_width = 14 - features.shape[1]
        padding = np.zeros((features.shape[0], pad_width))
        features_14d = np.hstack((features, padding))

        features_11d = features_14d[:, :11].tolist()
        features_13d = features_14d[:, :13].tolist()
        
        if request.mode == "buy":
            BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.40"))
            prob = run_inference(buy_model, buy_scaler, features_11d, input_shape=(20, 11))
            recommendation = "BUY" if prob >= BUY_THRESHOLD else "IGNORE"
            return LivePredictionResponse(
                probability=prob, recommendation=recommendation, confidence=get_confidence(prob),
                model="buy_signal_lstm", threshold_used=BUY_THRESHOLD, features_used=features_11d
            )
        else:
            SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.50"))
            prob = run_inference(sell_model, sell_scaler, features_13d, input_shape=(20, 13))
            recommendation = "SELL" if prob >= SELL_THRESHOLD else "HOLD"
            return LivePredictionResponse(
                probability=prob, recommendation=recommendation, confidence=get_confidence(prob),
                model="sell_signal_lstm", threshold_used=SELL_THRESHOLD, features_used=features_13d
            )
    except Exception as e:
        logger.error(f"Live inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
