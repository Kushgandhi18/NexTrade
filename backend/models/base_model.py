"""
base_model.py
Abstract interface that all ML models must implement.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """All prediction models inherit from this interface."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the model.
        Returns a dict of training metrics (MAE, RMSE, etc.).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for input X."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
