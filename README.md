# Task 5.1 — Model Deployment as REST API

Deploy your LSTM buy/sell signal models (Task 3.1 & 3.2) as a FastAPI REST service.

---

## Project Structure

```
task5_1_deployment/
├── app/
│   └── main.py            # FastAPI application
├── models/                # Put exported TF SavedModels here
│   ├── buy_model/         # Exported from Task 3.1
│   └── sell_model/        # Exported from Task 3.2
├── export_models.py       # Colab script to export your models
├── test_api.py            # Python test client
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Step 1 — Export Your Models from Colab

Copy and run `export_models.py` in your Colab notebook **after** Task 3 training:

```python
# Exports to: /content/drive/MyDrive/my_stock_models/
# - buy_model/     (TF SavedModel)
# - sell_model/    (TF SavedModel)
# - buy_scaler.pkl
# - sell_scaler.pkl
```

Then download the `my_stock_models/` folder from Google Drive to your local machine and place it:

```
task5_1_deployment/
└── models/
    ├── buy_model/
    └── sell_model/
```

---

## Step 2 — Run Locally (without Docker)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API server

```bash
# From the task5_1_deployment/ directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     ✅ Buy model loaded from models/buy_model
INFO:     ✅ Sell model loaded from models/sell_model
INFO:     🚀 Stock Prediction API is running.
```

### Open the interactive API docs

Visit: **http://localhost:8000/docs**

---

## Step 3 — Run with Docker

### Build the image

```bash
docker build -t stock-prediction-api .
```

### Run the container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name stock-api \
  stock-prediction-api
```

### Check logs

```bash
docker logs stock-api
```

### Stop the container

```bash
docker stop stock-api && docker rm stock-api
```

---

## Step 4 — Send Prediction Requests

### Using curl

#### Health check
```bash
curl http://localhost:8000/health
```
Expected output:
```json
{
  "status": "ok",
  "buy_model_loaded": true,
  "sell_model_loaded": true,
  "message": "API is running. Models use mock inference if not loaded."
}
```

#### Buy Signal Prediction
```bash
curl -X POST http://localhost:8000/predict/buy \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [6000, 6200, 5800, 6100, 1200000, 6050.0, 55.2, 0.008, 0.012, 15.3, 0.05],
      [6100, 6300, 6000, 6250, 1350000, 6080.0, 57.1, 0.010, 0.015, 15.3, 0.05],
      [6200, 6400, 6100, 6300, 1100000, 6120.0, 58.4, 0.012, 0.008, 15.3, 0.05],
      [6250, 6350, 6150, 6200,  980000, 6150.0, 54.0, 0.006,-0.016, 15.3, 0.05],
      [6150, 6280, 6100, 6180, 1050000, 6160.0, 52.5, 0.004,-0.003, 15.3, 0.05],
      [6100, 6200, 6000, 6090, 1200000, 6140.0, 49.8, 0.002,-0.015, 15.3, 0.05],
      [6050, 6150, 5950, 6020, 1300000, 6110.0, 47.2,-0.001,-0.011, 15.3, 0.05],
      [6000, 6100, 5900, 6000, 1400000, 6070.0, 45.5,-0.004,-0.003, 15.3, 0.05],
      [5950, 6050, 5850, 5980, 1500000, 6020.0, 43.8,-0.007,-0.003, 15.3, 0.05],
      [5900, 6000, 5800, 5950, 1600000, 5970.0, 42.1,-0.010,-0.005, 15.3, 0.05],
      [5850, 5980, 5780, 5900, 1700000, 5920.0, 40.5,-0.013,-0.008, 15.3, 0.05],
      [5800, 5930, 5730, 5870, 1800000, 5880.0, 39.0,-0.015,-0.005, 15.3, 0.05],
      [5820, 5950, 5750, 5900, 1600000, 5850.0, 41.2,-0.013, 0.005, 15.3, 0.05],
      [5870, 6000, 5800, 5960, 1400000, 5840.0, 44.0,-0.010, 0.010, 15.3, 0.05],
      [5930, 6050, 5860, 6020, 1300000, 5850.0, 47.3,-0.006, 0.010, 15.3, 0.05],
      [6000, 6100, 5930, 6080, 1200000, 5870.0, 50.5,-0.002, 0.010, 15.3, 0.05],
      [6050, 6150, 6000, 6120, 1100000, 5900.0, 53.4, 0.002, 0.007, 15.3, 0.05],
      [6100, 6200, 6050, 6180, 1000000, 5940.0, 56.2, 0.006, 0.010, 15.3, 0.05],
      [6150, 6250, 6100, 6220,  950000, 5980.0, 58.9, 0.009, 0.007, 15.3, 0.05],
      [6200, 6300, 6150, 6280,  900000, 6020.0, 61.5, 0.013, 0.010, 15.3, 0.05]
    ]
  }'
```

Expected output:
```json
{
  "probability": 0.6234,
  "recommendation": "BUY",
  "confidence": "MEDIUM",
  "model": "buy_signal_lstm",
  "threshold_used": 0.4
}
```

#### Sell Signal Prediction
```bash
curl -X POST http://localhost:8000/predict/sell \
  -H "Content-Type: application/json" \
  -d '{"features": [... 20 rows × 13 features ...]}'
```

### Using Python test script

```bash
# Make sure the server is running first
python test_api.py
```

---

## API Endpoints Summary

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Root / welcome |
| GET | `/health` | Model load status |
| GET | `/docs` | Interactive Swagger UI |
| POST | `/predict/buy` | Buy signal prediction |
| POST | `/predict/sell` | Sell signal prediction |
| GET | `/features/buy` | Feature schema for buy model |
| GET | `/features/sell` | Feature schema for sell model |

---

## Input / Output Schema

### POST /predict/buy

**Input:**
```json
{
  "features": [
    [Open, High, Low, Close, Volume, SMA_14, RSI_14, Dist_to_SMA, Log_Return, priceToEarning, cashDividendPercentage],
    ... (20 rows total)
  ]
}
```

**Output:**
```json
{
  "probability": 0.72,
  "recommendation": "BUY",
  "confidence": "HIGH",
  "model": "buy_signal_lstm",
  "threshold_used": 0.4
}
```

### POST /predict/sell

**Input:**
```json
{
  "features": [
    [Open, High, Low, Volume, Log_Return, Volatility_14, SMA_14, Dist_to_SMA, RSI_14, Dist_to_UB, BB_Percent_B, priceToEarning, cashDividendPercentage],
    ... (20 rows total)
  ]
}
```

**Output:**
```json
{
  "probability": 0.81,
  "recommendation": "SELL",
  "confidence": "HIGH",
  "model": "sell_signal_lstm",
  "threshold_used": 0.5
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUY_MODEL_PATH` | `models/buy_model` | Path to buy SavedModel |
| `SELL_MODEL_PATH` | `models/sell_model` | Path to sell SavedModel |
| `BUY_THRESHOLD` | `0.40` | Probability threshold for BUY |
| `SELL_THRESHOLD` | `0.50` | Probability threshold for SELL |

Override at runtime:
```bash
BUY_THRESHOLD=0.45 uvicorn app.main:app --port 8000
```

---

## Note: Running without pre-trained models

If the model files are not present, the API will start with **mock inference** — it returns deterministic pseudo-probabilities based on input statistics. This lets you test the API structure without a real model.
