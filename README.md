<div align="center">
  
# AlphaQuant: Deep Learning for Market Forecasting & Portfolio Optimization

**An End-to-End AI-Driven Quantitative Trading and Portfolio Management System**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Project Overview

**AlphaQuant** is a comprehensive, end-to-end quantitative finance project that leverages Deep Learning (LSTM networks) to analyze, forecast, and trade on both the **Nasdaq (US)** and **VN-Index (Vietnam)** stock markets. 

Developed as the final project for *CS313 - Deep Learning for Artificial Intelligence*, this system goes beyond simple price prediction. It encompasses a full quantitative trading pipeline: from raw time-series data processing and advanced feature engineering to AI-driven buy/sell signal classification, rigorous risk management, mathematical portfolio optimization (Markowitz), and finally, a production-ready REST API deployment with a SaaS web interface.

This project demonstrates strong capabilities in **Data Science, Deep Learning, Quantitative Finance, and Machine Learning Operations (MLOps)**.

---

## Key Features & Methodologies

### 1. Multi-Horizon Stock Price Forecasting (Regression)
- **Multi-Feature Inputs:** Utilizes OHLCV data combined with macroeconomic indicators.
- **Architectures:** Custom LSTM neural networks designed for non-stationary financial time-series data.
- **Forecasting Scenarios:** 
  - Next-day prediction (Baseline).
  - k-day-ahead forecasting (k=3, k=7).
  - Consecutive k-day sequence forecasting (predicting the full trajectory of the next k days).
- **Performance:** Achieved highly accurate price tracking with a Mean Absolute Percentage Error (MAPE) as low as **1.93%** on Vietnamese stocks (SAM).

### 2. Algorithmic Trading Signal Detection (Classification)
- **Technical Indicators Engineering:** Engineered complex features including Log Returns, 14-day Rolling Volatility, SMA, RSI, and Bollinger Bands.
- **Fundamental Integration:** Combined technical indicators with fundamental metrics (P/E ratio, Dividend Yield).
- **Binary Classification:** Trained separate, optimized LSTM models to predict highly profitable **BUY** entry points and risk-mitigating **SELL** exit signals.
- **Class Imbalance Handling:** Implemented custom class weights to penalize false negatives in buying opportunities.

### 3. AI-Driven Portfolio Construction & Risk Management
- **Composite Risk Scoring:** Developed a proprietary risk metric combining AI Sell probabilities (50%), Volatility (30%), and Technical Risk (20%).
- **Automated Filtering:** Systematically filters out high-risk assets from a universe of Vietnamese stocks.
- **Markowitz Efficient Frontier:** Utilized `PyPortfolioOpt` with Ledoit-Wolf shrinkage to calculate the optimal capital allocation, maximizing the theoretical Sharpe Ratio (0.13) and achieving an expected annual return of **7.3%** under strict risk constraints.

### 4. SaaS Deployment & MLOps (Task 5)
- **REST API:** Deployed the trained LSTM Buy-Signal model using **FastAPI** and **Uvicorn**, ensuring low-latency inference.
- **Data Validation:** Enforced strict JSON schema validation for 20-day time-series inputs using **Pydantic**.
- **Web Interface:** Built a lightweight, zero-dependency HTML5/Vanilla JS frontend that communicates directly with the FastAPI backend to provide real-time trading recommendations to end-users.

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | TensorFlow, Keras, LSTM |
| **Data Processing & ML** | Pandas, NumPy, Scikit-learn |
| **Quantitative Finance** | PyPortfolioOpt, Technical Analysis (RSI, Bollinger Bands, SMA) |
| **Backend & API** | FastAPI, Uvicorn, Pydantic, Python 3.11 |
| **Frontend UI** | HTML5, CSS3, Vanilla JavaScript (ES6) |
| **Environment** | Google Colab, Jupyter Notebook, Docker (Optional) |

---

## Repository Structure

```text
AlphaQuant/
├── data/                             # (To be mounted via Google Drive)
│   ├── stock-historical-data/        # Raw CSV OHLCV data
│   ├── financial-ratio/              # P/E, ROE, ROA datasets
│   └── ticker-overview.csv           # Company metadata
├── models/
│   └── buy_model/                    # Exported TF SavedModel for API deployment
├── app/
│   └── main.py                       # FastAPI application and inference endpoints
├── index.html                        # SaaS Web Interface (Frontend)
├── Final_project_DL4AI.ipynb         # Main Jupyter Notebook (Training & Evaluation)
├── 240112-project-report.pdf         # Comprehensive Academic Project Report
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Highlighted Results

- **Superior Sequence Forecasting:** Proved that predicting a sequence of k days yields a richer gradient signal and lower error (MAPE: 2.85%) than predicting a single k-th day independently.
- **Robust Risk Mitigation:** The automated portfolio construction successfully identified and excluded 7 out of 10 stocks exhibiting high downside risk based on the AI Composite Score.
- **Seamless Deployment:** The production API successfully processes 220 data points (20 days x 11 features) in milliseconds, returning actionable probability scores and confidence intervals.

---

## Getting Started

### 1. Training & Evaluation (Google Colab)
1. Clone this repository to your local machine or Google Drive.
2. Upload the required datasets to your Google Drive following the structure in `Data Structure`.
3. Open `Final_project_DL4AI.ipynb` via Google Colab.
4. Update the `base_path` variable to point to your data directory.
5. Install optimization dependencies: `!pip install PyPortfolioOpt`.
6. Run all cells to execute the pipeline from data preprocessing to portfolio optimization.

### 2. Running the REST API & Web Interface Locally
1. Create a Python 3.11 virtual environment:
   ```bash
   conda create -n alphaquant python=3.11 -y
   conda activate alphaquant
   ```
2. Install requirements:
   ```bash
   pip install fastapi uvicorn numpy tensorflow pydantic
   ```
3. Start the backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. **Launch the UI:** Double-click `index.html` in your browser. Paste your JSON formatted time-series data and click *Analyze & Predict*.

---

## Author

**Nguyen Hong Dang**
- **Student ID:** 240112
- **Course:** CS313 - Deep Learning for Artificial Intelligence (Spring 2026)

> *This project was developed for academic purposes. The financial models and predictions provided are for educational demonstration and should not be considered financial advice.*
