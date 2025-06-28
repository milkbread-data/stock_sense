import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from stock_predictor.predictor import StockPredictor
from autogluon.timeseries import TimeSeriesDataFrame

class TestStockPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = StockPredictor(prediction_length=10, time_limit=60)

    def test_initialization(self):
        self.assertEqual(self.predictor.prediction_length, 10)
        self.assertEqual(self.predictor.time_limit, 60)

    @patch('yfinance.Ticker')
    def test_fetch_stock_data(self, mock_ticker):
        # Mock the yfinance Ticker object
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'Close': [100, 102]
        })
        mock_ticker.return_value = mock_instance

        df = self.predictor.fetch_stock_data('AAPL', '2023-01-01', '2023-01-02')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Close', df.columns)

    def test_prepare_data(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'Close': [100, 102]
        })
        ts_df = self.predictor.prepare_data(df)
        self.assertIsInstance(ts_df, TimeSeriesDataFrame)
        self.assertEqual(ts_df.shape[0], 2)

    @patch('stock_predictor.predictor.TimeSeriesPredictor')
    def test_train_model(self, mock_predictor):
        # Mock the TimeSeriesPredictor
        mock_instance = MagicMock()
        mock_predictor.return_value = mock_instance

        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=20))
        df = pd.DataFrame({
            'date': dates,
            'Close': range(20)
        })
        train_data = self.predictor.prepare_data(df)

        # Call train_model
        self.predictor.train_model(train_data)

        # Check that fit was called
        mock_instance.fit.assert_called_once()

if __name__ == '__main__':
    unittest.main()