"""
transformer_model.py
Transformer model using multi-head self-attention for stock price prediction.
Best for capturing complex long-range dependencies.
"""

import logging
import os
import numpy as np
import pickle

from backend.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TransformerModel(BaseModel):
    """
    Transformer architecture for time-series:
      Input (60 timesteps × features)
      → Positional Encoding
      → Multi-Head Attention (4 heads)
      → Feed-Forward Network
      → Global Average Pooling
      → Dense(64) → Dense(1)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dff: int = 128,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self, input_shape: tuple):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        inputs = layers.Input(shape=input_shape)

        # Project to d_model dims
        x = layers.Dense(self.d_model)(inputs)

        # Positional encoding (simple learned)
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=self.d_model)(positions)
        x = x + pos_embed

        # Multi-Head Attention block
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
        )(x, x)
        attn_output = layers.Dropout(self.dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward block
        ffn_output = layers.Dense(self.dff, activation="relu")(x)
        ffn_output = layers.Dense(self.d_model)(ffn_output)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Pooling + output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        output = layers.Dense(1)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=["mae"],
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        logger.info(f"Training Transformer | X={X.shape}, y={y.shape}")
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ]

        self.model.fit(
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
        logger.info(f"Transformer trained | {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X).flatten()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"TransformerModel saved to {path}")

    def load(self, path: str) -> None:
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"TransformerModel loaded from {path}")
