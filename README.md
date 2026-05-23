<div align="center">
  
# AlphaQuant: Deep Learning for Market Forecasting and Portfolio Optimization

**An End-to-End AI-Driven Quantitative Trading and Portfolio Management System**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE.svg?logo=apache-airflow)](https://airflow.apache.org/)
[![dbt](https://img.shields.io/badge/dbt-1.8-FF694B.svg?logo=dbt)](https://www.getdbt.com/)

</div>

---

## Project Overview

**AlphaQuant** is a comprehensive, end-to-end quantitative finance and artificial intelligence system designed to analyze, forecast, and execute trading strategies on the Nasdaq and Vietnam (VN-Index) stock markets. 

Developed as an academic capstone, this project bridges the gap between theoretical deep learning models and industry-grade engineering practices. It encompasses the entire data lifecycle: from automated data ingestion and transformation pipelines (ELT) to the training of Long Short-Term Memory (LSTM) neural networks, risk-adjusted portfolio optimization, and final deployment as a production-ready Software-as-a-Service (SaaS) application.

## Core Capabilities and Technical Highlights

### 1. Multi-Horizon Time-Series Forecasting
- **Architecture:** Developed custom LSTM networks optimized for non-stationary financial time-series data.
- **Features:** Engineered multi-dimensional input tensors incorporating Open, High, Low, Close, Volume (OHLCV) alongside calculated macroeconomic and technical indicators.
- **Results:** Achieved a Mean Absolute Percentage Error (MAPE) of 1.93% on sequential predictions, outperforming baseline single-variable models by leveraging k-day trajectory forecasting.

### 2. Algorithmic Trading Signal Classification
- **Feature Engineering:** Implemented advanced statistical indicators including 14-day Rolling Volatility, Simple Moving Average (SMA), Relative Strength Index (RSI), Log Returns, and Bollinger Bands.
- **Model Design:** Trained distinct binary classification models to detect optimal entry (Buy) and exit (Sell) signals. 
- **Optimization:** Addressed class imbalance by applying custom loss weights to minimize false negatives during critical market entry points.

### 3. Quantitative Risk Management and Portfolio Construction
- **Risk Assessment:** Formulated a proprietary composite risk score utilizing AI-derived sell probabilities (50%), historical volatility (30%), and technical risk factors (20%).
- **Asset Allocation:** Applied Modern Portfolio Theory (Markowitz Efficient Frontier) using `PyPortfolioOpt` with Ledoit-Wolf shrinkage to dynamically construct portfolios.
- **Performance:** Systematically excluded high-risk assets, maximizing the theoretical Sharpe Ratio and achieving an expected annual return of 7.3% under stringent risk constraints.

### 4. MLOps and Production Deployment (API & SaaS)
- **Backend Architecture:** Deployed the trained TensorFlow models as a highly concurrent REST API utilizing **FastAPI** and **Uvicorn**, hosted on Render.
- **Data Validation:** Enforced strict schema validation using **Pydantic** to handle complex 20-day multi-feature JSON payloads.
- **Frontend Interface:** Developed a lightweight, zero-dependency web interface utilizing HTML5 and Vanilla JavaScript. The application communicates asynchronously with the FastAPI backend to deliver real-time, actionable trading insights to end-users.
- **Live Access:** The production environment is publicly accessible via GitHub Pages.

### 5. Automated Data Engineering Workflow
Designed a modern ELT (Extract, Load, Transform) data pipeline to ensure data integrity and continuous model readiness:
- **Data Ingestion:** Simulated automated extraction from external APIs into a relational database using **Airbyte**.
- **Data Transformation:** Utilized **dbt (Data Build Tool)** to perform in-database transformations, calculating technical features (SMA, RSI, Volatility) via optimized SQL Window Functions.
- **Orchestration:** Authored Directed Acyclic Graphs (DAGs) using **Apache Airflow** to orchestrate the sequential execution of data ingestion, dbt transformations, and Python-based model inference on a daily schedule.

## Technology Stack

- **Deep Learning & Mathematics:** TensorFlow, Keras, NumPy, Pandas, Scikit-learn, PyPortfolioOpt
- **Backend Engineering:** Python 3.11, FastAPI, Uvicorn, Pydantic
- **Data Engineering & MLOps:** Apache Airflow, dbt, Airbyte, PostgreSQL (Architecture Design)
- **Frontend Development:** HTML5, CSS3, JavaScript (ES6), Fetch API
- **Version Control & CI/CD:** Git, GitHub Pages, Render

## Repository Architecture

```text
AlphaQuant/
├── app/
│   └── main.py                       # FastAPI application for production model serving
├── index.html                        # SaaS Web Application (Frontend Client)
├── task_5.3_workflow/                # Automated Data Engineering Pipeline
│   ├── airflow/dags/                 # Apache Airflow orchestration scripts
│   ├── dbt/models/                   # dbt SQL transformation models
│   └── scripts/                      # Python inference and data synchronization scripts
├── task_5.3_report.md                # System architecture diagrams and workflow documentation
├── Final_project_DL4AI.ipynb         # Model development, training, and evaluation notebook
├── 240112-project-report.pdf         # Academic project report and methodological justification
├── requirements.txt                  # Production backend dependencies
└── README.md                         # Project documentation
```

## Setup and Execution Guidelines

### 1. Accessing the Live SaaS Application (Production)
The system has been fully deployed to the cloud. You do not need to install any local dependencies to use the prediction model.
1. Navigate to the public SaaS portal: **[AlphaQuant Prediction Web Application](https://hodatisg520.github.io/Deep-Learning-Market-Forecasting-Portfolio-Optimization/)**
2. Click the "Tải dữ liệu mẫu" (Load Sample Data) button to populate the matrix with 20 days of historical financial indicators.
3. Click "Dự Đoán Kết Quả" (Predict Result) to send the payload to the cloud-hosted FastAPI inference engine.
4. The system will asynchronously return the predicted probability and trading recommendation (BUY/IGNORE/SELL).

### 2. Executing the Production API Locally (Development)
If you wish to test the FastAPI backend on your local machine instead of the cloud deployment:
1. Initialize a Python 3.11 virtual environment.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the ASGI server via Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Access the auto-generated Swagger UI API documentation at `http://localhost:8000/docs` to test endpoints manually.
5. To connect the local UI to your local API, open `index.html` in your browser and ensure the `API_BASE_URL` variable inside the JavaScript code points to `http://localhost:8000`.

### 3. Model Development and Training Environment
To replicate the deep learning training, evaluation, and portfolio optimization procedures:
1. Open `Final_project_DL4AI.ipynb` in Google Colab or your preferred Jupyter environment.
2. Upload the required CSV datasets (historical data, financial ratios, ticker overview) into the corresponding directory structure defined in the code.
3. Ensure the `base_path` variable is correctly configured to point to your data directory.
4. Execute the cells sequentially to observe the pipeline from raw data preprocessing to model convergence and Markowitz optimization outputs.

## Author

**Nguyen Hong Dang**
- CS313 - Deep Learning for Artificial Intelligence
- Spring 2026

*Disclaimer: This repository and its associated models are developed strictly for academic and portfolio demonstration purposes. The predictions generated by these models do not constitute financial advice.*
