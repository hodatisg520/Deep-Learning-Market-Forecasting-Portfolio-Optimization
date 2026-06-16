import os
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def fetch_live_data(ticker: str) -> pd.DataFrame:
    print(f"📡 Lấy dữ liệu thực tế 60 ngày gần nhất của mã {ticker} từ Yahoo Finance...")
    # Tải dữ liệu 60 ngày để có đủ khoảng lùi tính toán các đường trung bình (14 ngày)
    df = yf.download(ticker, period="60d", progress=False)
    
    # yfinance trả về MultiIndex column, ta chỉ lấy các cột cơ bản
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.reset_index(inplace=True)
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Date': 'date'
    }, inplace=True)
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Đang tính toán các chỉ báo kỹ thuật (SMA_14, Log_Return, Volatility_14)...")
    
    # 1. Simple Moving Average 14 days
    df['sma_14'] = df['close'].rolling(window=14).mean()
    
    # 2. Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. Volatility 14 days (Standard Deviation of Log Returns)
    df['volatility_14'] = df['log_return'].rolling(window=14).std()
    
    # Bỏ đi những ngày ban đầu bị thiếu dữ liệu (NaN) do tính rolling
    df.dropna(inplace=True)
    return df

def main():
    ticker = "SAM.VN" # SAM Holdings trên sàn HOSE
    
    # 1. Fetch Data
    df = fetch_live_data(ticker)
    df = feature_engineering(df)
    
    # Chỉ lấy đúng 20 ngày giao dịch gần nhất
    df_recent = df.tail(20).copy()
    
    if len(df_recent) < 20:
        print("❌ Lỗi: Không lấy đủ 20 ngày dữ liệu.")
        return

    print("✅ Đã lấy đủ 20 ngày dữ liệu giao dịch gần nhất!")
    print(df_recent[['date', 'close', 'volume', 'sma_14', 'volatility_14']].tail(3))
    print("--------------------------------------------------")

    # 2. Preprocessing & Padding
    # Trích xuất 8 đặc trưng cơ bản
    features = df_recent[['open', 'high', 'low', 'close', 'volume', 'sma_14', 'log_return', 'volatility_14']].values
    
    # LƯU Ý QUAN TRỌNG: 
    # Scaler (Bộ chuẩn hóa) mà bạn đã lưu đang yêu cầu 14 cột.
    # Nhưng Model LSTM lại chỉ ăn 11 cột.
    # Do đó, ta sẽ bù 6 cột số 0 vào (8 + 6 = 14 cột) để chạy qua Scaler.
    pad_width = 14 - features.shape[1]
    padding = np.zeros((features.shape[0], pad_width))
    features_14d = np.hstack((features, padding))

    # 3. Load Scaler & Model
    print("🧠 Đang tải Model AI và Scaler (Bộ chuẩn hóa) từ thư mục my_stock_models...")
    scaler_path = "my_stock_models/buy_scaler.pkl"
    model_path = "my_stock_models/buy_model"
    
    try:
        scaler = joblib.load(scaler_path)
        model = tf.saved_model.load(model_path)
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return

    # 4. Scale Data
    # Áp dụng Scaler đã huấn luyện (14 cột) lên dữ liệu hôm nay
    features_scaled_14d = scaler.transform(features_14d)
    
    # Chỉ cắt lấy 11 cột đầu tiên để nạp vào AI
    features_scaled_11d = features_scaled_14d[:, :11]
    
    # Đóng gói thành dạng Tensor 3D (1, 20, 11) cho LSTM
    x_input = np.array(features_scaled_11d, dtype=np.float32).reshape(1, 20, 11)
    tensor_input = tf.constant(x_input)

    # 5. Inference (Dự đoán)
    print("🔮 AI ĐANG SUY NGHĨ DỰA TRÊN THỊ TRƯỜNG HIỆN TẠI...")
    infer = model.signatures['serving_default']
    result_dict = infer(tensor_input)
    # Lấy ra kết quả đầu tiên (thường key là 'output_0' hoặc 'dense_x')
    probability = float(list(result_dict.values())[0].numpy().flatten()[0])
    
    # Hiển thị Kết quả
    print("==================================================")
    print(f"📊 KẾT QUẢ DỰ BÁO CHO CỔ PHIẾU {ticker}")
    print(f"🕒 Ngày giao dịch gần nhất: {df_recent['date'].iloc[-1].strftime('%d/%m/%Y')}")
    print(f"💡 Xác suất Tăng giá (Buy Signal): {probability * 100:.2f}%")
    
    if probability >= 0.4:
        print("🚀 LỜI KHUYÊN AI: NÊN MUA (BUY) - Tín hiệu tích cực!")
    else:
        print("🛑 LỜI KHUYÊN AI: BỎ QUA (IGNORE) - Thị trường rủi ro, nên đứng ngoài.")
    print("==================================================")

if __name__ == "__main__":
    main()
