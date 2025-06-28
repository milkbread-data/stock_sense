"""Stock prediction module using AutoGluon for time series forecasting."""

__version__ = "0.1.0"

from .predictor import StockPredictor  # noqa: F401

__all__ = ["StockPredictor"]
