"""
Stock Prediction Airflow DAG

Orchestrates the daily pipeline combining Airbyte data ingestion, 
dbt data transformation, and Python/TensorFlow machine learning inference.
"""

from datetime import timedelta
import subprocess

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import pendulum

# Default arguments applied to all tasks in this DAG
DEFAULT_ARGS = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def execute_inference_script() -> None:
    """
    Executes the TensorFlow prediction script.
    In a production environment, this could be refactored into a DockerOperator 
    or an ECSOperator to isolate deep learning dependencies.
    """
    script_path = "/opt/airflow/scripts/predict_and_store.py"
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Prediction script failed with exit code {e.returncode}")

with DAG(
    dag_id='stock_prediction_workflow',
    default_args=DEFAULT_ARGS,
    description='End-to-end automated workflow for Stock Price Prediction',
    schedule_interval='@daily',
    start_date=pendulum.today('UTC').add(days=-1),
    catchup=False,
    tags=['finance', 'deep_learning', 'production'],
) as dag:

    # Task 1: Airbyte Sync Trigger
    # Initiates the ELT sync from external sources (e.g., CSV, APIs) to the raw data warehouse.
    trigger_airbyte_ingestion = BashOperator(
        task_id='airbyte_data_ingestion',
        bash_command=(
            "curl -X POST http://airbyte-server:8000/api/v1/connections/sync "
            "-d '{\"connectionId\": \"your-connection-id\"}' "
            "-H 'Content-Type: application/json'"
        )
    )

    # Task 2: dbt Transformation
    # Executes SQL models to clean raw data and compute financial indicators (SMA, Volatility, etc.)
    execute_dbt_models = BashOperator(
        task_id='dbt_data_transformation',
        bash_command='cd /opt/airflow/dbt_project && dbt run --models stock_features'
    )

    # Task 3: Machine Learning Inference
    # Fetches transformed features, runs the TensorFlow LSTM model, and stores the results.
    run_ml_inference = PythonOperator(
        task_id='run_tensorflow_prediction',
        python_callable=execute_inference_script
    )

    # Define the DAG execution sequence (Pipeline Dependencies)
    trigger_airbyte_ingestion >> execute_dbt_models >> run_ml_inference
