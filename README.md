<div align="center">
  
# AlphaQuant: Deep Learning for Market Forecasting & Portfolio Optimization

**An End-to-End AI-Driven Quantitative Trading and Portfolio Management System**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE.svg?logo=apache-airflow)](https://airflow.apache.org/)
[![dbt](https://img.shields.io/badge/dbt-1.8-FF694B.svg?logo=dbt)](https://www.getdbt.com/)

</div>

---

## 📖 Project Overview

**AlphaQuant** is a comprehensive quantitative finance project that leverages Deep Learning (LSTM networks) to analyze, forecast, and trade on both the **Nasdaq (US)** and **VN-Index (Vietnam)** stock markets. 

Developed for *CS313 - Deep Learning for Artificial Intelligence*, this system goes beyond notebook-level price prediction. It encompasses a full MLOps pipeline: from raw time-series data processing to API deployment, a web-based SaaS interface, and an automated data engineering workflow.

🌐 **Live Web Application (SaaS):** [AlphaQuant Prediction Portal](https://hodatisg520.github.io/Deep-Learning-Market-Forecasting-Portfolio-Optimization/)

---

## 🚀 Key Milestones & Tasks

### 🧠 Task 1-4: Core AI Models & Portfolio Optimization
- **Multi-Horizon Forecasting:** Custom LSTM neural networks for 1-day, 3-day, and 7-day ahead predictions with MAPE as low as 1.93%.
- **Trading Signal Detection:** Binary classification models predicting highly profitable **BUY** entry points and risk-mitigating **SELL** exits based on complex technical indicators (SMA, RSI, Volatility).
- **Risk Management & Portfolio:** Automated Markowitz Efficient Frontier filtering using PyPortfolioOpt to achieve optimal expected annual returns under strict risk constraints.

### 🌐 Task 5.1 & 5.2: Model Deployment & SaaS Application (Extra Credit)
Instead of keeping the model in a Jupyter Notebook, the system has been fully deployed to the cloud:
- **Task 5.1 (API Deployment):** The trained LSTM Buy/Sell model is served as a high-performance **REST API** using **FastAPI** and **Uvicorn**, deployed on **Render**. It processes live 20-day time-series data and returns JSON predictions instantly.
- **Task 5.2 (SaaS UI):** A lightweight, responsive web application was built and hosted publicly on **GitHub Pages**. Users can interactively test the model by entering stock indicators or loading sample data, receiving real-time Buy/Hold/Sell recommendations.

### ⚙️ Task 5.3: AI Engineering Workflow (Extra Credit)
To ensure the model receives fresh data automatically, a modern **Data Engineering Pipeline (ELT)** was designed using industry-standard tools:
- **Airbyte:** Automates data ingestion from Nasdaq/Vietnam APIs into the database.
- **dbt (Data Build Tool):** Transforms raw data directly in SQL, computing critical technical indicators (SMA, RSI) efficiently.
- **Python (TensorFlow):** Pulls the transformed features to run the prediction model and stores results back to the database.
- **Apache Airflow:** Acts as the orchestrator, executing the entire pipeline (`Airbyte -> dbt -> Python`) on a daily schedule.
> *Skeleton code for this pipeline is available in the `task_5.3_workflow/` directory.*

---

## 🛠️ Repository Structure

```text
AlphaQuant/
├── app/
│   └── main.py                       # (Task 5.1) FastAPI application for model serving
├── index.html                        # (Task 5.2) SaaS Web Interface (Frontend)
├── task_5.3_workflow/                # (Task 5.3) Automated AI Engineering Pipeline
│   ├── airflow/dags/                 # Airflow DAG for orchestration
│   ├── dbt/models/                   # dbt SQL models for feature engineering
│   └── scripts/                      # Python prediction scripts
├── task_5.3_report.md                # System diagram and workflow documentation
├── Final_project_DL4AI.ipynb         # Main Jupyter Notebook (Training & Evaluation)
├── 240112-project-report.pdf         # Comprehensive Academic Project Report
├── requirements.txt                  # Python API dependencies
└── README.md                         # Project documentation
```

---

## 💻 Getting Started (Setup Instructions)

### 1. View the Live SaaS Demo
Simply visit the public GitHub Pages link to use the AI model from your browser:
👉 [**AlphaQuant Live Demo**](https://hodatisg520.github.io/Deep-Learning-Market-Forecasting-Portfolio-Optimization/)

### 2. Running the API Locally (Backend)
If you want to run the FastAPI backend on your own machine:
1. Ensure you have Python 3.11 installed.
2. Open your terminal in the project directory and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python main.py
   ```
   *(The API will be available at `http://localhost:8000`)*

### 3. Model Training & Evaluation
To view the Deep Learning experiments:
1. Open `Final_project_DL4AI.ipynb` via Google Colab.
2. Upload your datasets to Google Drive.
3. Run the notebook cells sequentially to train the LSTM models and evaluate the Markowitz portfolio optimization.

---

## 👨‍🎓 Author
**Nguyen Hong Dang**
- **Course:** CS313 - Deep Learning for Artificial Intelligence (Spring 2026)
- *Disclaimer: This project was developed for academic purposes. The financial predictions provided are for educational demonstration and should not be considered financial advice.*
