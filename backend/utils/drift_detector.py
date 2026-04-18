"""
drift_detector.py
Monitors prediction error over time and triggers retraining if drift is detected.
"""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects model drift by comparing current error metrics against a baseline.

    Strategy:
      - Maintain a rolling window of RMSE values
      - If current RMSE > (baseline_rmse * threshold_multiplier) → trigger retraining
      - Uses Page-Hinkley test for sequential drift detection
    """

    def __init__(
        self,
        rmse_threshold_multiplier: float = 1.5,
        window_size: int = 30,
        on_drift: Optional[Callable] = None,
    ):
        self.rmse_threshold_multiplier = rmse_threshold_multiplier
        self.window_size = window_size
        self.on_drift = on_drift  # callback e.g. trigger_retraining()
        self._error_history: list = []
        self._baseline_rmse: Optional[float] = None

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Update error history with latest predictions.
        Returns a status dict with drift status.
        """
        current_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        self._error_history.append(current_rmse)

        # Keep rolling window
        if len(self._error_history) > self.window_size:
            self._error_history.pop(0)

        # Set baseline on first fill
        if self._baseline_rmse is None and len(self._error_history) >= self.window_size:
            self._baseline_rmse = float(np.mean(self._error_history))
            logger.info(f"Drift baseline set: RMSE={self._baseline_rmse:.4f}")

        drift_detected = False
        if self._baseline_rmse:
            threshold = self._baseline_rmse * self.rmse_threshold_multiplier
            drift_detected = current_rmse > threshold

            if drift_detected:
                logger.warning(
                    f"Model Drift Detected! current_rmse={current_rmse:.4f} > "
                    f"threshold={threshold:.4f}"
                )
                if self.on_drift:
                    self.on_drift()

        return {
            "current_rmse": round(current_rmse, 4),
            "baseline_rmse": round(self._baseline_rmse, 4) if self._baseline_rmse else None,
            "drift_detected": drift_detected,
            "history_length": len(self._error_history),
        }

    def reset_baseline(self):
        """Call after successful retraining to reset the drift baseline."""
        self._error_history.clear()
        self._baseline_rmse = None
        logger.info("Drift baseline reset.")

    def check(self, current_rmse: float, threshold: float) -> bool:
        """
        Simple threshold check (for programmatic use in pipelines).
        Returns True if retraining should be triggered.
        """
        if current_rmse > threshold:
            logger.warning(
                f"Drift check: RMSE {current_rmse:.4f} exceeds threshold {threshold:.4f}"
            )
            return True
        return False
