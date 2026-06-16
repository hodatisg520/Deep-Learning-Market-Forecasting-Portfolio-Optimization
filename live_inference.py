"""
Live Inference Script

Fetches real-time market data from Yahoo Finance, computes technical
indicators, and runs the pre-trained LSTM model for Buy signal prediction.
Designed for local execution and rapid prototyping.
"""

import os
import logging

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Suppress TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_TICKER = "SAM.VN"  # SAM Holdings on HOSE (Ho Chi Minh Stock Exchange)
DATA_PERIOD = "60d"
TIME_STEPS = 20
SCALER_FEATURES = 14
MODEL_FEATURES = 11
BUY_THRESHOLD = 0.40

MODEL_PATH = "my_stock_models/buy_model"
SCALER_PATH = "my_stock_models/buy_scaler.pkl"


def fetch_live_data(ticker: str) -> pd.DataFrame:
    """Download recent historical data from Yahoo Finance."""
    logger.info(f"Fetching {DATA_PERIOD} of data for {ticker} from Yahoo Finance...")
    df = yf.download(ticker, period=DATA_PERIOD, progress=False)

    # yfinance may return MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True)
    df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume', 'Date': 'date'
    }, inplace=True)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators required by the model."""
    logger.info("Computing technical indicators (SMA_14, Log Return, Volatility)...")

    df['sma_14'] = df['close'].rolling(window=14).mean()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_14'] = df['log_return'].rolling(window=14).std()

    # Drop rows with NaN values from rolling window calculations
    df.dropna(inplace=True)
    return df


def main() -> None:
    """Execute the live inference pipeline."""
    ticker = DEFAULT_TICKER

    # 1. Fetch and prepare data
    df = fetch_live_data(ticker)
    df = compute_features(df)

    df_recent = df.tail(TIME_STEPS).copy()
    if len(df_recent) < TIME_STEPS:
        logger.error(f"Insufficient data: need {TIME_STEPS} rows, got {len(df_recent)}.")
        return

    logger.info(f"Successfully retrieved {TIME_STEPS} recent trading days.")

    # 2. Preprocessing and padding
    features = df_recent[[
        'open', 'high', 'low', 'close', 'volume',
        'sma_14', 'log_return', 'volatility_14'
    ]].values

    # Pad to 14 columns (scaler was trained on 14 features)
    pad_width = SCALER_FEATURES - features.shape[1]
    padding = np.zeros((features.shape[0], pad_width))
    features_padded = np.hstack((features, padding))

    # 3. Load scaler and model
    logger.info("Loading model and scaler...")
    try:
        scaler = joblib.load(SCALER_PATH)
        model = tf.saved_model.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        return

    # 4. Scale and slice
    features_scaled = scaler.transform(features_padded)
    features_model = features_scaled[:, :MODEL_FEATURES]

    x_input = features_model.astype(np.float32).reshape(1, TIME_STEPS, MODEL_FEATURES)
    tensor_input = tf.constant(x_input)

    # 5. Run inference using the serving signature
    logger.info("Running model inference...")
    infer = model.signatures['serving_default']
    result = infer(tensor_input)
    probability = float(list(result.values())[0].numpy().flatten()[0])

    # 6. Display results
    recommendation = "BUY" if probability >= BUY_THRESHOLD else "IGNORE"
    latest_date = df_recent['date'].iloc[-1].strftime('%Y-%m-%d')

    print("=" * 55)
    print(f"  PREDICTION RESULT FOR {ticker}")
    print(f"  Latest Trading Date : {latest_date}")
    print(f"  Buy Probability     : {probability * 100:.2f}%")
    print(f"  Recommendation      : {recommendation}")
    print("=" * 55)


if __name__ == "__main__":
    main()
