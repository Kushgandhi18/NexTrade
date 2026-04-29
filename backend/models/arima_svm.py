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
        X: 3D array (samples, timesteps, features)
        y: target next-day prices
        """
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

        # Train SVM on exogenous features (last timestep of each sequence) + lag-1 residual
        logger.info("Training SVM on exogenous features and ARIMA residuals...")
        X_last_step = X[:, -1, :] if X.ndim == 3 else X
        
        # We need lag-1 residual for each sample. 
        # Shift residuals by 1 to get lag-1. We drop the first sample.
        lag_1_residuals = residuals[:-1].reshape(-1, 1)
        res_targets = residuals[1:]
        X_exogenous = X_last_step[1:]
        
        # Combine exogenous features with lag-1 residuals
        svm_features = np.hstack((X_exogenous, lag_1_residuals))

        svm_features_scaled = self.svm_scaler.fit_transform(svm_features)
        self.svm.fit(svm_features_scaled, res_targets)

        # Compute training metrics
        svm_pred_residuals = self.svm.predict(svm_features_scaled)
        final_preds = arima_preds[1:] + svm_pred_residuals
        
        mae = float(np.mean(np.abs(series[1:] - final_preds)))
        rmse = float(np.sqrt(np.mean((series[1:] - final_preds) ** 2)))
        logger.info(f"ARIMA-SVM trained | MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # Save last known residual for live prediction
        self._last_residual = residuals[-1]
        return {"mae": mae, "rmse": rmse}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-step values using ARIMA + SVM."""
        if not self._arima_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        steps = len(X) if X.ndim == 1 else X.shape[0]
        arima_forecasts = self.arima.forecast(steps=steps).values

        # Build feature vector for SVM using X and last known residual
        X_last_step = X[:, -1, :] if X.ndim == 3 else X
        svm_corrections = []
        
        current_residual = getattr(self, "_last_residual", 0.0)
        
        for i in range(steps):
            exog = X_last_step[i].reshape(1, -1)
            feat = np.hstack((exog, [[current_residual]]))
            feat_scaled = self.svm_scaler.transform(feat)
            
            # Predict next residual
            pred_res = self.svm.predict(feat_scaled)[0]
            svm_corrections.append(pred_res)
            
            # Autoregressive update for the next step
            current_residual = pred_res

        return arima_forecasts + np.array(svm_corrections)

    def save(self, path: str) -> None:
        import joblib
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # Using joblib for efficient sci-kit learn model saving
        joblib.dump(
            {
                "arima_order": self.arima_order,
                "arima": self.arima,
                "svm": self.svm,
                "svm_scaler": self.svm_scaler,
                "last_residual": getattr(self, "_last_residual", 0.0)
            },
            path,
        )
        logger.info(f"ArimaSVMModel saved to {path}")

    def load(self, path: str) -> None:
        import joblib
        data = joblib.load(path)
        self.arima_order = data["arima_order"]
        self.arima = data.get("arima")
        self.svm = data["svm"]
        self.svm_scaler = data["svm_scaler"]
        self._last_residual = data.get("last_residual", 0.0)
        self._arima_fitted = True
        logger.info(f"ArimaSVMModel loaded from {path}")
