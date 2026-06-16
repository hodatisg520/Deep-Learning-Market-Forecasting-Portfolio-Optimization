-- dbt Model: Stock Features
-- This model applies SQL transformations directly on the Data Warehouse.
-- Computes rolling technical indicators necessary for deep learning inference.

{{ config(materialized='table') }}

WITH raw_stock_data AS (
    -- Step 1: Extract raw daily price data ingested via Airbyte
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
    -- Step 2: Compute technical features using SQL Window Functions
    SELECT 
        *,
        -- 14-day Simple Moving Average (SMA)
        AVG(close) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS sma_14,

        -- Daily Logarithmic Return
        LN(close / NULLIF(LAG(close, 1) OVER (
            PARTITION BY ticker 
            ORDER BY date
        ), 0)) AS log_return,
        
        -- 14-day Volatility (Standard Deviation of Log Returns)
        STDDEV(LN(close / NULLIF(LAG(close, 1) OVER (
            PARTITION BY ticker 
            ORDER BY date
        ), 0))) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS volatility_14

        -- Note: Additional advanced indicators (e.g., RSI) can be implemented 
        -- using dbt macros or Python preprocessing layers.
    FROM raw_stock_data
)

-- Step 3: Materialize the final analytical dataset
SELECT * 
FROM calculated_features
WHERE sma_14 IS NOT NULL -- Exclude initial rows that lack sufficient history for SMA calculation
