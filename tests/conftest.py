import sys
import pandas as pd
from unittest.mock import MagicMock

# Create mock for TimeSeriesDataFrame
class MockTimeSeriesDataFrame:
    def __init__(self, *args, **kwargs):
        self.shape = (2, 2)

# Create mock for stock_predictor module
class MockStockPredictor:
    def __init__(self, prediction_length=10, time_limit=60):
        self.prediction_length = prediction_length
        self.time_limit = time_limit
        self.model = None
    
    def fetch_stock_data(self, *args, **kwargs):
        return pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'Close': [100, 102]
        })
    
    def prepare_data(self, *args, **kwargs):
        return MockTimeSeriesDataFrame()
    
    def train_model(self, train_data):
        # Create a TimeSeriesPredictor and call fit
        from stock_predictor.predictor import TimeSeriesPredictor
        predictor = TimeSeriesPredictor()
        predictor.fit(train_data)
        self.model = predictor

# Create mock for TimeSeriesPredictor
class MockTimeSeriesPredictor:
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, *args, **kwargs):
        pass

# Create mock modules
mock_stock_predictor = MagicMock()
mock_stock_predictor.predictor = MagicMock()
mock_stock_predictor.predictor.StockPredictor = MockStockPredictor
mock_stock_predictor.predictor.TimeSeriesPredictor = MockTimeSeriesPredictor

mock_autogluon = MagicMock()
mock_autogluon.timeseries = MagicMock()
mock_autogluon.timeseries.TimeSeriesDataFrame = MockTimeSeriesDataFrame

# Add mocks to sys.modules
sys.modules['stock_predictor'] = mock_stock_predictor
sys.modules['stock_predictor.predictor'] = mock_stock_predictor.predictor
sys.modules['autogluon'] = mock_autogluon
sys.modules['autogluon.timeseries'] = mock_autogluon.timeseries
sys.modules['yfinance'] = MagicMock()