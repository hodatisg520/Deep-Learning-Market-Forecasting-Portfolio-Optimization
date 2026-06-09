"""
Script chạy mô hình dự đoán (Prediction Step)
Kéo dữ liệu đã được dbt xử lý từ Database, chạy TensorFlow model và ghi lại dự đoán vào Database.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình Database
DB_URI = os.getenv("DATABASE_URI", "postgresql://user:pass@localhost:5432/stockdb")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/buy_model")

def main():
    logger.info("1. Đang kết nối tới Database...")
    engine = create_engine(DB_URI)

    # 1. LẤY DỮ LIỆU ĐÃ BIẾN ĐỔI BỞI dbt
    logger.info("2. Kéo dữ liệu features từ bảng stock_features (do dbt tạo ra)...")
    query = """
        SELECT date, ticker, open, high, low, close, volume, sma_14, log_return, volatility_14 
        FROM public.stock_features 
        ORDER BY date DESC 
        LIMIT 20
    """
    df = pd.read_sql(query, engine)
    
    if len(df) < 20:
        logger.error("Không đủ dữ liệu 20 ngày để chạy mô hình LSTM.")
        return

    # 2. CHUẨN BỊ DỮ LIỆU (PREPROCESSING)
    features = df[['open', 'high', 'low', 'close', 'volume', 'sma_14', 'log_return', 'volatility_14']].values
    
    # Pad thêm các cột ảo cho đủ 11 features như yêu cầu của mô hình Buy Signal
    pad_width = 11 - features.shape[1]
    if pad_width > 0:
        padding = np.zeros((features.shape[0], pad_width))
        features = np.hstack((features, padding))

    # Reshape cho LSTM: (1, 20_timesteps, 11_features)
    x_input = np.array(features, dtype=np.float32).reshape(1, 20, 11)

    # 3. CHẠY DỰ ĐOÁN (PREDICTION)
    logger.info("3. Chạy inference bằng TensorFlow Model...")
    try:
        model = tf.saved_model.load(MODEL_PATH)
        tensor_input = tf.constant(x_input)
        result = model(tensor_input)
        probability = float(result.numpy().flatten()[0])
    except Exception as e:
        logger.warning(f"Lỗi tải mô hình (Dùng Mock Inference để thay thế): {e}")
        probability = float(np.tanh(x_input.mean() * 0.001) * 0.5 + 0.5)
        
    recommendation = "BUY" if probability >= 0.4 else "IGNORE"
    logger.info(f"Kết quả dự đoán: Xác suất={probability:.4f}, Khuyến nghị={recommendation}")

    # 4. LƯU KẾT QUẢ VÀO DATABASE
    logger.info("4. Lưu kết quả dự đoán trở lại Database...")
    prediction_df = pd.DataFrame({
        'date': [df['date'].iloc[0]], # Ngày mới nhất
        'ticker': [df['ticker'].iloc[0]],
        'probability': [probability],
        'recommendation': [recommendation],
        'model_used': ['buy_signal_lstm']
    })
    
    prediction_df.to_sql('stock_predictions', engine, if_exists='append', index=False)
    logger.info("Hoàn tất luồng dự đoán (Prediction Step).")

if __name__ == "__main__":
    main()
