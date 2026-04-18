"""
backtesting.py
Simulates trading from model predictions.
Returns profit, Sharpe ratio, and max drawdown.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.04  # 4% annual


class BacktestingService:
    """
    Simulates a simple buy/sell strategy based on model predictions.
    
    Strategy:
      - BUY if predicted_price > current_price
      - SELL (exit) if predicted_price <= current_price
      - No short selling (long-only)
    """

    def __init__(self, initial_capital: float = 10_000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% per trade

    def simulate(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        threshold: float = 0.0,
    ) -> dict:
        """
        Run backtesting simulation.

        Args:
            predictions: array of predicted next-day prices
            actual_prices: array of actual prices (aligned with predictions)
            threshold: minimum predicted % change to trigger a trade

        Returns:
            dict with profit, sharpe_ratio, max_drawdown, direction_accuracy, trade_count
        """
        n = min(len(predictions), len(actual_prices))
        predictions = predictions[:n]
        actual_prices = actual_prices[:n]

        capital = self.initial_capital
        position = 0.0  # shares held
        portfolio_values = [capital]
        trades = 0

        for i in range(1, n):
            current_price = actual_prices[i - 1]
            predicted_price = predictions[i - 1]
            next_actual = actual_prices[i]

            pred_change = (predicted_price - current_price) / (current_price + 1e-9)

            if pred_change > threshold:
                # BUY signal
                if position == 0.0 and capital > 0:
                    shares_to_buy = (capital * (1 - self.commission)) / current_price
                    position = shares_to_buy
                    capital = 0.0
                    trades += 1

            else:
                # SELL signal
                if position > 0.0:
                    proceeds = position * current_price * (1 - self.commission)
                    capital = proceeds
                    position = 0.0
                    trades += 1

            # Portfolio value at end of day
            portfolio_value = capital + position * next_actual
            portfolio_values.append(portfolio_value)

        # Close any open position at end
        if position > 0.0:
            capital += position * actual_prices[-1] * (1 - self.commission)
            portfolio_values[-1] = capital

        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-9)

        total_profit = float(portfolio_values[-1] - self.initial_capital)
        total_return_pct = float(total_profit / self.initial_capital * 100)
        max_dd = float(self._max_drawdown(portfolio_values))
        sharpe = float(self._sharpe_ratio(returns))
        direction_accuracy = float(
            np.mean(np.sign(np.diff(predictions)) == np.sign(np.diff(actual_prices)))
        )

        result = {
            "initial_capital": self.initial_capital,
            "final_capital": round(float(portfolio_values[-1]), 2),
            "total_profit": round(total_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "direction_accuracy": round(direction_accuracy, 4),
            "trade_count": trades,
            "portfolio_values": portfolio_values.tolist(),
        }

        logger.info(
            f"Backtest | Profit={total_profit:.2f}, Sharpe={sharpe:.4f}, "
            f"MaxDD={max_dd*100:.2f}%, Trades={trades}"
        )
        return result

    @staticmethod
    def _sharpe_ratio(daily_returns: np.ndarray) -> float:
        """Annualized Sharpe Ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        excess = daily_returns - RISK_FREE_RATE / 252
        return float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))

    @staticmethod
    def _max_drawdown(portfolio_values: np.ndarray) -> float:
        """Maximum Peak-to-Trough drawdown."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / (peak + 1e-9)
        return float(np.min(drawdown))
