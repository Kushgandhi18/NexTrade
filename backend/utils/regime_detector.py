"""
regime_detector.py
Classifies a stock price series into: trending / cyclical / volatile / stable.
Used by ModelFactory to auto-select the best model.
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects the current market regime of a price series.

    Regimes:
      'trending'  → strong directional movement (GRU best)
      'cyclical'  → mean-reverting oscillation  (LSTM best)
      'volatile'  → high variability, no clear trend (Ensemble best)
      'stable'    → low volatility, weak trend  (ARIMA-SVM sufficient)
    """

    def detect(self, price_series: pd.Series, window: int = 60) -> str:
        """
        Analyze the last `window` days and return a regime string.
        """
        series = price_series.dropna().tail(window)
        if len(series) < 20:
            logger.warning("Not enough data for regime detection. Defaulting to 'volatile'.")
            return "volatile"

        returns = series.pct_change().dropna()
        volatility = float(returns.std())
        trend_strength = self._hurst_exponent(series.values)
        is_stationary = self._is_stationary(series)

        logger.info(
            f"Regime metrics | volatility={volatility:.4f}, "
            f"hurst={trend_strength:.4f}, stationary={is_stationary}"
        )

        # Classification rules based on research paper insights
        if volatility > 0.03:
            return "volatile"

        if is_stationary and trend_strength < 0.45:
            return "cyclical"

        if trend_strength > 0.55:
            return "trending"

        if volatility < 0.01:
            return "stable"

        return "volatile"  # default fallback

    @staticmethod
    def _is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
        """ADF test: True if series is stationary (mean-reverting / cyclical)."""
        try:
            result = adfuller(series.dropna(), autolag="AIC")
            p_value = result[1]
            return p_value < significance
        except Exception:
            return False

    @staticmethod
    def _hurst_exponent(series: np.ndarray) -> float:
        """
        Hurst Exponent via R/S analysis.
        H < 0.5 → mean-reverting (cyclical)
        H = 0.5 → random walk
        H > 0.5 → trending
        """
        n = len(series)
        if n < 20:
            return 0.5

        lags = range(2, min(20, n // 2))
        tau = []
        for lag in lags:
            sub_series = series[:lag]
            if len(sub_series) < 2:
                continue
            mean_sub = np.mean(sub_series)
            deviation = np.cumsum(sub_series - mean_sub)
            rescaled = (np.max(deviation) - np.min(deviation)) / (np.std(sub_series) + 1e-9)
            tau.append(rescaled)

        if len(tau) < 2:
            return 0.5

        lags_used = list(range(2, 2 + len(tau)))
        try:
            poly = np.polyfit(np.log(lags_used), np.log(tau), 1)
            return float(poly[0])
        except Exception:
            return 0.5
