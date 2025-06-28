# Stock Price Prediction Web App

A web application for predicting stock prices using AutoGluon's time series forecasting capabilities.

## Features

- Fetch historical stock data using Yahoo Finance
- Train and evaluate multiple time series forecasting models using AutoGluon
- Interactive visualizations of stock price history and predictions
- Simple and intuitive web interface built with Streamlit
- Support for multiple stocks and custom date ranges

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-app.git
   cd stock-prediction-app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -e .
   ```

   For development, install with additional dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. In the sidebar:
   - Enter a stock symbol (e.g., 'AAPL' for Apple Inc.)
   - Select a date range for historical data
   - Choose the number of days to predict
   - Click 'Train Model' to start the prediction

## Project Structure

```
stock-prediction-app/
├── app.py                 # Main Streamlit application
├── stock_predictor/      # Core prediction module
│   ├── __init__.py
│   └── predictor.py      # Stock prediction logic
├── tests/                # Unit tests
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [AutoGluon](https://auto.gluon.ai/stable/index.html) for automated machine learning
- [Streamlit](https://streamlit.io/) for the web interface
- [yfinance](https://github.com/ranaroussi/yfinance) for fetching stock data
- [Plotly](https://plotly.com/python/) for interactive visualizations