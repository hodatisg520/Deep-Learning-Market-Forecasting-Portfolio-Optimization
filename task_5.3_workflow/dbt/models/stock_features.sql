-- Mô hình dbt (Data Build Tool) dùng để biến đổi dữ liệu (Transformation)
-- File này thực thi các lệnh SQL trực tiếp trên Data Warehouse (PostgreSQL/MongoDB)

{{ config(materialized='table') }}

WITH raw_stock_data AS (
    -- Bước 1: Lấy dữ liệu thô đã được Airbyte nạp vào database
    SELECT 
        date,
        ticker,
        open,
        high,
        low,
        close,
        volume
    FROM {{ source('stock_raw', 'raw_daily_prices') }}
),

calculated_features AS (
    -- Bước 2: Dùng SQL Window Functions để tính toán các đặc trưng (Features) cho Deep Learning
    SELECT 
        *,
        -- Tính Simple Moving Average (SMA) 14 ngày
        AVG(close) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as sma_14,

        -- Tính Lợi nhuận Log (Log Return) giữa ngày hôm nay và hôm qua
        LN(close / NULLIF(LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date), 0)) as log_return,
        
        -- Tính Độ biến động (Volatility) trong 14 ngày (Độ lệch chuẩn của Log Return)
        STDDEV(LN(close / NULLIF(LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date), 0))) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as volatility_14

        -- Ghi chú: RSI và các chỉ báo phức tạp hơn có thể được tạo bằng CTEs nâng cao hoặc dbt macros
    FROM raw_stock_data
)

-- Bước 3: Trả về bảng dữ liệu cuối cùng (sạch và đã có features) để mô hình Python sử dụng
SELECT * 
FROM calculated_features
WHERE sma_14 IS NOT NULL -- Lọc bỏ những dòng chưa đủ 14 ngày đầu tiên để tính SMA
