"""
training_pipeline.py
Orchestrates the full training flow:
  DataService → FeatureService → Model → Evaluate → MLflow log → Save
"""

import logging
import os
import time
from datetime import datetime

import numpy as np
import mlflow
import mlflow.sklearn

from backend.services.data_service import DataService
from backend.services.feature_service import FeatureService
from backend.models.model_factory import ModelFactory
from backend.utils.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

MODEL_STORE_DIR = os.environ.get("MODEL_STORE_DIR", "model_store")


class TrainingPipeline:
    """Full training orchestration triggered by POST /train."""

    def __init__(self):
        self.data_service = DataService()
        self.feature_service = FeatureService()
        self.model_factory = ModelFactory()
        self.regime_detector = RegimeDetector()

    def run(
        self,
        stock: str,
        model_name: str = "auto",
        sequence_length: int = 60,
        hyperparams: dict = None,
    ) -> dict:
        """
        Full training pipeline.

        Args:
            stock: Ticker symbol e.g. 'AAPL'
            model_name: 'lstm', 'gru', 'arima_svm', 'transformer', 'ensemble', or 'auto'
            sequence_length: sliding window size
            hyperparams: optional dict to override model defaults

        Returns:
            dict with metrics + model path
        """
        start_time = time.time()
        logger.info(f"=== Training Pipeline START | stock={stock}, model={model_name} ===")

        # 1. Fetch data
        df_raw = self.data_service.fetch_stock_data(stock)

        # 2. Detect regime (for auto model selection)
        regime = self.regime_detector.detect(df_raw["Close"])
        if model_name == "auto":
            model_name = self.model_factory.select_by_regime(regime)
            logger.info(f"Auto-selected model: {model_name} (regime: {regime})")

        # 3. Feature engineering
        df_features = self.feature_service.add_indicators(df_raw)

        # 4. Walk-forward split
        train_df, val_df, test_df = self.data_service.train_val_test_split(df_features)

        # 5. Create sequences (fit scaler on train only → avoid data leakage)
        X_train, y_train, scaler = self.feature_service.create_sequences(
            train_df, sequence_length=sequence_length, fit_scaler=True
        )
        X_val, y_val, _ = self.feature_service.create_sequences(
            val_df, sequence_length=sequence_length, fit_scaler=False
        )
        X_test, y_test, _ = self.feature_service.create_sequences(
            test_df, sequence_length=sequence_length, fit_scaler=False
        )

        # 6. Get model
        kwargs = hyperparams or {}
        model = self.model_factory.get_model(model_name, **kwargs)

        # 7. Train
        with mlflow.start_run(run_name=f"{stock}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "stock": stock,
                "model": model_name,
                "regime": regime,
                "sequence_length": sequence_length,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                **(hyperparams or {}),
            })

            train_metrics = model.train(X_train, y_train)

            # 8. Evaluate on test set
            test_preds = model.predict(X_test)
            test_metrics = self._evaluate(y_test, test_preds, prefix="test_")

            # 9. Log metrics
            mlflow.log_metrics({**train_metrics, **test_metrics})

            # 10. Save model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(MODEL_STORE_DIR, stock, model_name, f"v_{version}")
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)
            mlflow.log_artifact(model_path)

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"=== Training Pipeline DONE in {elapsed}s ===")

            return {
                "stock": stock,
                "model": model_name,
                "regime": regime,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "model_path": model_path,
                "version": version,
                "elapsed_seconds": elapsed,
            }

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
        """Compute MAE, RMSE, R², and Direction Accuracy."""
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9))
        direction = float(np.mean(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true))))
        return {
            f"{prefix}mae": mae,
            f"{prefix}rmse": rmse,
            f"{prefix}r2": r2,
            f"{prefix}direction_accuracy": direction,
        }
