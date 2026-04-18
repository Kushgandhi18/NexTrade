"""
ensemble_model.py
Weighted ensemble that combines GRU, LSTM, ARIMA-SVM, and Transformer predictions.
Final = 0.4*GRU + 0.3*LSTM + 0.2*ARIMA + 0.1*Transformer
"""

import logging
import os
import pickle
import numpy as np

from backend.models.base_model import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "gru": 0.4,
    "lstm": 0.3,
    "arima_svm": 0.2,
    "transformer": 0.1,
}


class EnsembleModel(BaseModel):
    """
    Weighted ensemble of multiple base models.
    Each sub-model must be pre-trained before calling EnsembleModel.train().
    """

    def __init__(self, models: dict = None, weights: dict = None):
        """
        Args:
            models: dict of {name: BaseModel instance}
            weights: dict of {name: float}, must sum to 1.0
        """
        self.models = models or {}
        self.weights = weights or DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Ensemble weights sum to {total:.4f}, normalizing...")
            factor = 1.0 / total
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Trains all sub-models sequentially.
        Returns aggregated metrics.
        """
        all_metrics = {}
        for name, model in self.models.items():
            logger.info(f"Training ensemble sub-model: {name}")
            metrics = model.train(X, y)
            all_metrics[name] = metrics

        # Ensemble evaluation
        preds = self.predict(X)
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r2 = float(1 - np.sum((y - preds) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-9))
        direction = float(np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y))))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "direction_accuracy": direction,
            "sub_model_metrics": all_metrics,
        }
        logger.info(f"Ensemble trained | MAE={mae:.4f}, RMSE={rmse:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of all sub-model predictions."""
        if not self.models:
            raise RuntimeError("No sub-models loaded. Add models before predicting.")

        total_weight = 0.0
        weighted_sum = None

        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            preds = model.predict(X)
            if weighted_sum is None:
                weighted_sum = weight * preds
            else:
                weighted_sum += weight * preds
            total_weight += weight

        return weighted_sum / (total_weight + 1e-9)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save weights + model paths
        model_paths = {}
        for name, model in self.models.items():
            sub_path = os.path.join(os.path.dirname(path), f"ensemble_{name}")
            model.save(sub_path)
            model_paths[name] = sub_path

        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "model_paths": model_paths}, f)
        logger.info(f"EnsembleModel saved to {path}")

    def load(self, path: str) -> None:
        from backend.models.model_factory import ModelFactory

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.weights = data["weights"]
        factory = ModelFactory()
        self.models = {}
        for name, sub_path in data["model_paths"].items():
            model = factory.get_model(name)
            model.load(sub_path)
            self.models[name] = model
        logger.info(f"EnsembleModel loaded from {path}")
