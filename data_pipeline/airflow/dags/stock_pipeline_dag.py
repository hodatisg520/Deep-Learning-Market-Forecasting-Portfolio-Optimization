from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess

# Thông số cấu hình mặc định cho DAG
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

def run_prediction_script():
    """Hàm Python gọi script dự đoán bằng mô hình TensorFlow"""
    # Trong thực tế, bạn sẽ import hàm từ scripts/predict_and_store.py
    # hoặc chạy file python riêng biệt. Ở đây minh họa bằng subprocess.
    subprocess.run(["python", "/opt/airflow/scripts/predict_and_store.py"], check=True)

with DAG(
    'stock_prediction_workflow',
    default_args=default_args,
    description='Automated workflow for Stock Price Prediction (Task 5.3)',
    schedule_interval='@daily', # Chạy mỗi ngày 1 lần
    start_date=days_ago(1),
    catchup=False,
    tags=['stock', 'deep_learning'],
) as dag:

    # 1. AIRBYTE: Kích hoạt đồng bộ dữ liệu tĩnh (từ file CSV/API vào Database thô)
    # Airbyte có API để kích hoạt một Connection. Ở đây ta dùng BashOperator gọi curl.
    trigger_airbyte_sync = BashOperator(
        task_id='airbyte_data_ingestion',
        bash_command='curl -X POST http://airbyte-server:8000/api/v1/connections/sync -d \'{"connectionId": "your-connection-id"}\' -H "Content-Type: application/json"'
    )

    # 2. DBT: Làm sạch dữ liệu thô và tính toán đặc trưng (SMA, RSI, Volatility)
    run_dbt_transformation = BashOperator(
        task_id='dbt_data_transformation',
        bash_command='cd /opt/airflow/dbt_project && dbt run --models stock_features'
    )

    # 3. PYTHON SCRIPT: Chạy mô hình dự đoán (TensorFlow) trên dữ liệu đã biến đổi
    run_prediction_model = PythonOperator(
        task_id='run_tensorflow_prediction',
        python_callable=run_prediction_script
    )

    # ĐỊNH NGHĨA THỨ TỰ THỰC THI CỦA WORKFLOW (Pipeline)
    trigger_airbyte_sync >> run_dbt_transformation >> run_prediction_model
