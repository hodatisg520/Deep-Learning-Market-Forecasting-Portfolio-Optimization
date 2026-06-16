"""
Stock Prediction Inference Pipeline

Extracts transformed features from the data warehouse (via dbt),
executes inference using the TensorFlow LSTM model, and stores
the prediction results back into the database.
"""

import os
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment Configuration
DB_URI = os.getenv("DATABASE_URI", "postgresql://user:pass@localhost:5432/stockdb")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/buy_model")
REQUIRED_TIME_STEPS = 20
TOTAL_FEATURES = 11

class DatabaseClient:
    """Handles all database interactions for the prediction pipeline."""

    def __init__(self, uri: str):
        self.uri = uri
        self.engine: Optional[Engine] = None

    def connect(self) -> None:
        """Establishes the database connection."""
        logger.info("Connecting to the database...")
        self.engine = create_engine(self.uri)

    def fetch_features(self) -> pd.DataFrame:
        """
        Retrieves the most recent 20 days of stock features.
        
        Returns:
            pd.DataFrame: DataFrame containing recent market data.
        """
        logger.info("Fetching recent features from public.stock_features...")
        query = f"""
            SELECT date, ticker, open, high, low, close, volume, sma_14, log_return, volatility_14 
            FROM public.stock_features 
            ORDER BY date DESC 
            LIMIT {REQUIRED_TIME_STEPS}
        """
        if self.engine is None:
            raise ConnectionError("Database engine is not initialized. Call connect() first.")
            
        return pd.read_sql(query, self.engine)

    def store_prediction(self, date: pd.Timestamp, ticker: str, probability: float, recommendation: str) -> None:
        """
        Stores the resulting prediction back into the warehouse.
        """
        logger.info(f"Storing prediction for {ticker} on {date}...")
        prediction_df = pd.DataFrame({
            'date': [date],
            'ticker': [ticker],
            'probability': [probability],
            'recommendation': [recommendation],
            'model_used': ['buy_signal_lstm']
        })
        
        if self.engine is None:
            raise ConnectionError("Database engine is not initialized.")
            
        prediction_df.to_sql('stock_predictions', self.engine, if_exists='append', index=False)


class StockPredictor:
    """Manages data preprocessing and TensorFlow model inference."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """Loads the saved TensorFlow model from disk."""
        import tensorflow as tf
        logger.info(f"Loading TensorFlow model from {self.model_path}...")
        try:
            self.model = tf.saved_model.load(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw DataFrame rows into LSTM-compatible 3D tensors.
        
        Args:
            df (pd.DataFrame): Raw features dataframe.
            
        Returns:
            np.ndarray: A 3D tensor of shape (batch_size, time_steps, features).
        """
        if len(df) < REQUIRED_TIME_STEPS:
            raise ValueError(f"Insufficient data. Expected {REQUIRED_TIME_STEPS} rows, got {len(df)}.")

        features = df[['open', 'high', 'low', 'close', 'volume', 'sma_14', 'log_return', 'volatility_14']].values
        
        # Pad missing columns to reach the 11 features required by the Buy Signal model architecture
        pad_width = TOTAL_FEATURES - features.shape[1]
        if pad_width > 0:
            padding = np.zeros((features.shape[0], pad_width))
            features = np.hstack((features, padding))

        # Reshape for LSTM: (batch_size, time_steps, features)
        x_input = np.array(features, dtype=np.float32).reshape(1, REQUIRED_TIME_STEPS, TOTAL_FEATURES)
        return x_input

    def predict(self, x_input: np.ndarray) -> Tuple[float, str]:
        """
        Executes inference to determine the buy signal probability.
        
        Returns:
            Tuple[float, str]: The probability score and the discrete recommendation.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
            
        import tensorflow as tf
        logger.info("Running TensorFlow inference...")
        
        tensor_input = tf.constant(x_input)
        infer = self.model.signatures['serving_default']
        result = infer(tensor_input)
        probability = float(list(result.values())[0].numpy().flatten()[0])
        
        recommendation = "BUY" if probability >= 0.4 else "IGNORE"
        logger.info(f"Prediction complete. Prob: {probability:.4f} | Action: {recommendation}")
        
        return probability, recommendation


def main() -> None:
    """Main execution orchestrator."""
    try:
        # 1. Database Operations
        db_client = DatabaseClient(DB_URI)
        db_client.connect()
        df_features = db_client.fetch_features()

        # 2. Model Inference
        predictor = StockPredictor(MODEL_PATH)
        predictor.load_model()
        
        x_input = predictor.preprocess(df_features)
        probability, recommendation = predictor.predict(x_input)

        # 3. Store Results
        target_date = df_features['date'].iloc[0]
        target_ticker = df_features['ticker'].iloc[0]
        
        db_client.store_prediction(target_date, target_ticker, probability, recommendation)
        logger.info("Pipeline executed successfully.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
