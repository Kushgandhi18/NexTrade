"""
lstm_model.py
LSTM model for stock price prediction.
Best for long-term dependencies and cyclical patterns.
"""

import logging
import os
import numpy as np
import pickle

from backend.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM architecture:
      Input (60 timesteps × features)
      → LSTM(128) → Dropout(0.2)
      → LSTM(64)  → Dropout(0.2)
      → Dense(32) → Dense(1)
    Loss: Huber (handles outliers better than MSE)
    Optimizer: Adam
    """

    def __init__(
        self,
        units: list = None,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        self.units = units or [128, 64]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self, input_shape: tuple):
        """Builds the Keras LSTM model."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(self.units[0], return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.units[1], return_sequences=False),
                Dropout(self.dropout),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=["mae"],
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X: [samples, timesteps, features]
        y: [samples] — next-day price
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        logger.info(f"Training LSTM | X={X.shape}, y={y.shape}")
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ]

        self.history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )

        preds = self.model.predict(X).flatten()
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r2 = float(1 - np.sum((y - preds) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-9))
        direction = float(np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y))))

        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "direction_accuracy": direction}
        logger.info(f"LSTM trained | {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X).flatten()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        # Save hyperparams separately
        meta_path = path + ".meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "units": self.units,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                },
                f,
            )
        logger.info(f"LSTMModel saved to {path}")

    def load(self, path: str) -> None:
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"LSTMModel loaded from {path}")
