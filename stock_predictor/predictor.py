"""Stock price prediction using AutoGluon's time series forecasting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import MultiWindowSplitter


class StockPredictor:
    """A class for predicting stock prices using AutoGluon's time series forecasting."""

    def __init__(self, prediction_length: int = 30, time_limit: int = 120) -> None:
        """Initialize the StockPredictor.

        Args:
            prediction_length: Number of days to predict into the future.
            time_limit: Maximum training time for the model in seconds.
        """
        self.prediction_length = prediction_length
        self.time_limit = time_limit
        self.model: Optional[TimeSeriesPredictor] = None
        self.training_data: Optional[TimeSeriesDataFrame] = None
        self.training_params = {
            "path": "ag_models",
            "target": "Close",
            "prediction_length": prediction_length,
            "eval_metric": "sMAPE",
            "freq": "D",
        }

    def fetch_stock_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame containing the historical stock data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                raise ValueError(
                    f"No data found for {ticker} in the specified date range."
                )

            # Reset index to make Date a column
            df = df.reset_index()
            df = df.rename(columns={"Date": "date"})

            # Ensure we have the required columns
            if "Close" not in df.columns:
                raise ValueError(
                    "No 'Close' price data available for the selected date range."
                )

            return df

        except Exception as e:
            raise RuntimeError(f"Error fetching stock data: {str(e)}")

    def prepare_data(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """Prepare the data for AutoGluon time series forecasting.

        Args:
            df: DataFrame containing the stock data with a 'date' and 'Close' column

        Returns:
            TimeSeriesDataFrame formatted for AutoGluon
        """
        # Ensure date is in datetime format and timezone-naive
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        # Create a time series with a single time series ID
        df["item_id"] = 1

        # Sort by date
        df = df.sort_values("date")

        # Convert to TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="date",
        )

        return ts_df

    def train_model(self, train_data: TimeSeriesDataFrame) -> Dict[str, Any]:
        """Train the AutoGluon time series model.

        Args:
            train_data: Training data in TimeSeriesDataFrame format

        Returns:
            Dictionary containing training results and metrics
        """
        # Initialize the predictor
        self.model = TimeSeriesPredictor(**self.training_params)

        # Train the model
        self.model.fit(train_data, presets="fast_training", time_limit=self.time_limit, num_val_windows=1)

        # Get model leaderboard
        leaderboard = self.model.leaderboard(train_data)

        return {
            "leaderboard": leaderboard,
            "best_model": leaderboard.index[0],
        }

    def predict(self, data: TimeSeriesDataFrame) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Make predictions using the trained model.

        Args:
            data: Time series data to make predictions on

        Returns:
            Tuple of (predictions, actuals) TimeSeriesDataFrames
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train_model() first.")

        # Make predictions
        predictions = self.model.predict(data)

        # Get actual values for the prediction period
        actuals = data.slice_by_timestep(-self.prediction_length, None)

        return predictions, actuals

    def evaluate_model(
        self, train_data: TimeSeriesDataFrame, test_data: TimeSeriesDataFrame
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            train_data: Training data in TimeSeriesDataFrame format
            test_data: Test data in TimeSeriesDataFrame format

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train_model() first.")

        # Evaluate the model
        evaluation = self.model.evaluate(test_data)

        # Calculate additional metrics if needed
        # ...

        return evaluation

    def predict_future(
        self, data: TimeSeriesDataFrame, prediction_length: Optional[int] = None
    ) -> TimeSeriesDataFrame:
        """Predict future stock prices beyond the training data.

        Args:
            data: Historical data in TimeSeriesDataFrame format
            prediction_length: Optional override for number of days to predict

        Returns:
            TimeSeriesDataFrame containing the predictions
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train_model() first.")

        if prediction_length is None:
            prediction_length = self.prediction_length

        # Make predictions
        predictions = self.model.predict(data)

        return predictions
