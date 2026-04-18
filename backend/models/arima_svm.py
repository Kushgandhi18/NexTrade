"""
arima_svm.py
ARIMA → residuals → SVM hybrid model.
"""

import logging
import os
import pickle

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from backend.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ArimaSVMModel(BaseModel):
    """
    Hybrid model:
      1. ARIMA captures the linear trend.
      2. SVM (RBF kernel) learns the nonlinear residuals.
      Final = ARIMA_prediction + SVM_residual_prediction
    """

    def __init__(self, arima_order: tuple = None):
        self.arima_order = arima_order  # if None, auto_arima selects
        self.arima = None
        self.svm = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        self.svm_scaler = StandardScaler()
        self._arima_fitted = False

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X: 1D or 2D array of price series (uses only the Close column if 2D)
        y: target next-day prices (used for residual evaluation)
        """
        # Flatten to 1D price series for ARIMA
        series = y if y.ndim == 1 else y[:, 0]

        logger.info("Fitting ARIMA (auto order selection)...")
        automodel = auto_arima(
            series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        self.arima_order = automodel.order
        self.arima = ARIMA(series, order=self.arima_order).fit()
        self._arima_fitted = True

        arima_preds = self.arima.fittedvalues
        residuals = series - arima_preds

        # Train SVM on residuals
        logger.info("Training SVM on ARIMA residuals...")
        res_features = self._build_residual_features(residuals)
        res_targets = residuals[len(residuals) - len(res_features) :]

        res_features_scaled = self.svm_scaler.fit_transform(
            res_features.reshape(-1, 1)
        )
        self.svm.fit(res_features_scaled, res_targets)

        # Compute training metrics
        final_preds = arima_preds[len(arima_preds) - len(series) :] + residuals
        mae = float(np.mean(np.abs(series - final_preds)))
        rmse = float(np.sqrt(np.mean((series - final_preds) ** 2)))
        logger.info(f"ARIMA-SVM trained | MAE={mae:.4f}, RMSE={rmse:.4f}")
        return {"mae": mae, "rmse": rmse}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-step values for each input step."""
        if not self._arima_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        steps = len(X) if X.ndim == 1 else X.shape[0]
        arima_forecasts = self.arima.forecast(steps=steps)

        # Predict residuals using SVM
        dummy_res = np.zeros(steps)
        dummy_scaled = self.svm_scaler.transform(dummy_res.reshape(-1, 1))
        svm_corrections = self.svm.predict(dummy_scaled)

        return arima_forecasts + svm_corrections

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "arima_order": self.arima_order,
                    "arima_params": self.arima.params if self.arima else None,
                    "svm": self.svm,
                    "svm_scaler": self.svm_scaler,
                },
                f,
            )
        logger.info(f"ArimaSVMModel saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.arima_order = data["arima_order"]
        self.svm = data["svm"]
        self.svm_scaler = data["svm_scaler"]
        self._arima_fitted = True
        logger.info(f"ArimaSVMModel loaded from {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_residual_features(residuals: np.ndarray) -> np.ndarray:
        """Create lag-1 feature array from residuals."""
        return residuals[:-1]
