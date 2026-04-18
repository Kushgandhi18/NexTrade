"""
retrain_dag.py
Airflow DAG: daily retraining for all tracked stocks.
Fetches fresh data, retrains, and checks for drift.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ------------------------------------------------------------------
# Default args
# ------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "stock_prediction",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

STOCKS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
MODELS = ["gru", "lstm", "ensemble"]


def retrain_stock(stock: str, model: str, **kwargs):
    """Retrain a single stock/model pair and check drift."""
    import sys
    sys.path.insert(0, "/app")

    from backend.services.training_pipeline import TrainingPipeline
    from backend.utils.drift_detector import DriftDetector
    import numpy as np

    pipeline = TrainingPipeline()
    result = pipeline.run(stock=stock, model_name=model)

    test_rmse = result["test_metrics"].get("test_rmse", float("inf"))
    drift = DriftDetector()
    is_drifting = drift.check(current_rmse=test_rmse, threshold=5.0)

    if is_drifting:
        print(f"WARNING: Drift detected for {stock}/{model} — RMSE={test_rmse:.4f}")
    else:
        print(f"OK: {stock}/{model} | RMSE={test_rmse:.4f}")

    return result


# ------------------------------------------------------------------
# DAG
# ------------------------------------------------------------------
with DAG(
    dag_id="daily_stock_retraining",
    default_args=DEFAULT_ARGS,
    description="Daily retraining pipeline for all stock models",
    schedule_interval="0 2 * * 1-5",  # 2 AM on weekdays (market days)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["stock_prediction", "retraining"],
) as dag:

    for stock in STOCKS:
        for model in MODELS:
            task_id = f"retrain_{stock}_{model}".lower()
            PythonOperator(
                task_id=task_id,
                python_callable=retrain_stock,
                op_kwargs={"stock": stock, "model": model},
            )
