"""
data_service.py
Fetches OHLCV data from Yahoo Finance and cleans it.
Adds optional macroeconomic context (S&P 500, NASDAQ).
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Market-wide context tickers
MARKET_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
}


class DataService:
    """Responsible for data fetching and cleaning."""

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "10y",
        include_market: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for `symbol` via yfinance.
        Optionally joins S&P 500 and NASDAQ returns as extra features.
        """
        logger.info(f"Fetching data for {symbol} | period={period}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for symbol: {symbol}")

        # Keep only relevant columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # strip timezone

        if include_market:
            df = self._join_market_features(df, period)

        df = self.clean_data(df)
        return df

    def _join_market_features(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Join S&P 500, NASDAQ, and VIX daily returns as extra columns."""
        for name, ticker_sym in MARKET_TICKERS.items():
            try:
                mkt = yf.Ticker(ticker_sym).history(period=period)[["Close"]]
                mkt.index = pd.to_datetime(mkt.index).tz_localize(None)
                mkt = mkt.rename(columns={"Close": f"{name}_close"})
                mkt[f"{name}_return"] = mkt[f"{name}_close"].pct_change()
                df = df.join(mkt[[f"{name}_return"]], how="left")
            except Exception as e:
                logger.warning(f"Could not fetch market data for {name}: {e}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataframe:
        - Forward-fill then back-fill missing values
        - Remove duplicate indices
        - Drop rows still missing after fill
        - Remove extreme outliers via IQR clipping
        """
        logger.info("Cleaning data...")

        # Remove duplicate dates
        df = df[~df.index.duplicated(keep="first")]

        # Forward fill then back fill
        df = df.ffill().bfill()

        # Drop any remaining NaN rows
        df = df.dropna()

        # Clip outliers per numeric column (IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q3)

        logger.info(f"Clean data shape: {df.shape}")
        return df

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        """
        Walk-forward chronological split (NEVER random).
        Returns (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"Split sizes → train: {len(train)}, val: {len(val)}, test: {len(test)}"
        )
        return train, val, test
