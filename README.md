# Stock Sense

A web application for stock price prediction using AutoGluon's time series forecasting capabilities.

## Features

- Fetch historical stock data from Yahoo Finance
- Train machine learning models to predict future stock prices
- Visualize historical data and predictions
- Evaluate model performance with metrics

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-sense.git
cd stock-sense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

This application can be easily deployed on Streamlit Community Cloud:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
3. Click "New app" and select this repository
4. Select the main branch and enter the path to the app: `app.py`
5. Click "Deploy"

## Dependencies

- Python 3.12+
- Streamlit
- Pandas
- NumPy
- YFinance
- Plotly
- AutoGluon
- Matplotlib
- Scikit-learn

## License

MIT