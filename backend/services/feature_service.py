"""
feature_service.py
Adds technical indicators and creates sliding-window sequences for deep learning.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureService:
    """Responsible for feature engineering and sequence creation."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        Returns a new dataframe with indicator columns appended.
        """
        df = df.copy()

        # --- Moving Averages ---
        for window in [5, 10, 20, 50, 200]:
            df[f"ma_{window}"] = df["Close"].rolling(window).mean()
            df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

        # --- RSI (14-period) ---
        df["rsi"] = self._compute_rsi(df["Close"], period=14)

        # --- MACD ---
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- Bollinger Bands (20-period) ---
        bb_mid = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_width"] + 1e-9)

        # --- Volatility ---
        df["daily_return"] = df["Close"].pct_change()
        df["volatility_5"] = df["daily_return"].rolling(5).std()
        df["volatility_20"] = df["daily_return"].rolling(20).std()
        df["volatility_60"] = df["daily_return"].rolling(60).std()

        # --- ATR (Average True Range) ---
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # --- Momentum ---
        for m in [5, 10, 20]:
            df[f"momentum_{m}"] = df["Close"] - df["Close"].shift(m)

        # --- Time-Cyclical Features ---
        day_of_week = None
        if "Date" in df.columns:
            day_of_week = pd.to_datetime(df["Date"]).dt.dayofweek
        elif "date" in df.columns:
            day_of_week = pd.to_datetime(df["date"]).dt.dayofweek
        elif hasattr(df.index, "dayofweek"):
            day_of_week = df.index.dayofweek
            
        if day_of_week is not None:
            df["day_of_week"] = day_of_week
            df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # --- Lag Features ---
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)

        # --- Multi-scale Rolling Stats ---
        for window in [5, 20, 60]:
            df[f"rolling_mean_{window}"] = df["Close"].rolling(window).mean()
            df[f"rolling_std_{window}"] = df["Close"].rolling(window).std()
            df[f"rolling_max_{window}"] = df["Close"].rolling(window).max()
            df[f"rolling_min_{window}"] = df["Close"].rolling(window).min()

        # --- Volume features ---
        df["volume_ma_5"] = df["Volume"].rolling(5).mean()
        df["volume_ma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma_20"] + 1e-9)

        # Drop rows with NaNs introduced by indicators
        df = df.dropna()
        logger.info(f"Features added. Shape: {df.shape}, Columns: {list(df.columns)}")
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
        sequence_length: int = 60,
        fit_scaler: bool = True,
    ):
        """
        Scale data and create sliding-window sequences.

        Returns:
            X: np.ndarray of shape [samples, timesteps, features]
            y: np.ndarray of shape [samples] (next-day close, unscaled)
            scaler: fitted MinMaxScaler (needed for inverse transform)
        """
        feature_cols = [c for c in df.columns if c != target_col]
        all_cols = feature_cols + [target_col]

        data = df[all_cols].values

        if fit_scaler:
            scaled = self.scaler.fit_transform(data)
            self._fitted = True
        else:
            if not self._fitted:
                raise RuntimeError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled = self.scaler.transform(data)

        X, y_scaled = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i - sequence_length : i])
            y_scaled.append(scaled[i, -1])  # last column = Close (scaled)

        X = np.array(X)
        y_scaled = np.array(y_scaled)

        # Inverse-transform y back to real price for interpretability
        dummy = np.zeros((len(y_scaled), data.shape[1]))
        dummy[:, -1] = y_scaled
        y = self.scaler.inverse_transform(dummy)[:, -1]

        logger.info(f"Sequences created → X: {X.shape}, y: {y.shape}")
        return X, y, self.scaler

    def inverse_transform_predictions(
        self, predictions: np.ndarray, n_features: int
    ) -> np.ndarray:
        """Inverse-scale a 1D prediction array back to actual prices."""
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, -1] = predictions
        return self.scaler.inverse_transform(dummy)[:, -1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))
