"""
Streamlit web application for stock price prediction using AutoGluon.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

from stock_predictor.predictor import StockPredictor

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stDateInput>div>div>input {
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>div>div>div>div>div {
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAlert {
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_dates() -> Tuple[date, date]:
    """Get default date range (1 year ago to today)."""
    end_date = date.today()
    start_date = end_date - relativedelta(years=1)
    return start_date, end_date


def plot_stock_data(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None,
    title: str = "Stock Price Prediction",
) -> None:
    """Plot stock price history and predictions using Plotly.

    Args:
        history: DataFrame with historical data
        predictions: DataFrame with predictions
        actuals: Optional DataFrame with actual values for the prediction period
        title: Plot title
    """
    # Create figure
    fig = go.Figure()

    # Prepare and add historical data
    history_plot = history.copy()
    history_plot = history_plot.rename(columns={"Close": "value"})
    fig.add_trace(
        go.Scatter(
            x=history_plot["date"],
            y=history_plot["value"],
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4"),
        )
    )

    # Prepare and add predictions
    pred_df_plot = predictions.copy()
    pred_df_plot = pred_df_plot.rename(columns={"mean": "value"})
    
    # Use timestamp column for x-axis
    x_column = "timestamp" if "timestamp" in pred_df_plot.columns else "date"
    
    fig.add_trace(
        go.Scatter(
            x=pred_df_plot[x_column],
            y=pred_df_plot["value"],
            mode="lines",
            name="Predicted",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    # Prepare and add actual values if available
    if actuals is not None and not actuals.empty:
        actuals_plot = actuals.copy()
        actuals_plot = actuals_plot.rename(columns={"Close": "value"})
        
        # Use timestamp column for x-axis
        x_column = "timestamp" if "timestamp" in actuals_plot.columns else "date"
        
        fig.add_trace(
            go.Scatter(
                x=actuals_plot[x_column],
                y=actuals_plot["value"],
                mode="lines",
                name="Actual",
                line=dict(color="#2ca02c"),
            )
        )

    # Add confidence interval if available
    if "0.9" in predictions.columns:
        # Use timestamp column for x-axis
        x_column = "timestamp" if "timestamp" in predictions.columns else "date"
        
        fig.add_trace(
            go.Scatter(
                x=pd.concat([predictions[x_column], predictions[x_column][::-1]]),
                y=pd.concat([predictions["0.9"], predictions["0.1"][::-1]]),
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="90% Confidence",
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            y=1.1,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_model_metrics(metrics: Dict[str, float]) -> None:
    """Display model evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    if not metrics:
        return

    st.subheader("Model Performance")

    # Create columns for metrics
    cols = st.columns(len(metrics))

    for (metric_name, metric_value), col in zip(metrics.items(), cols):
        col.metric(
            label=metric_name.upper(),
            value=f"{metric_value:.4f}",
        )


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("📈 Stock Price Predictor")
    st.markdown(
        "Predict future stock prices using AutoGluon's time series forecasting capabilities."
    )

    # Initialize session state
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "actuals" not in st.session_state:
        st.session_state.actuals = None

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Settings")

        # Stock symbol input
        ticker = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol",
        ).upper()

        # Date range selection
        default_start, default_end = get_default_dates()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=date(2000, 1, 1),
                max_value=date.today() - timedelta(days=1),
                help="Start date for historical data",
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=start_date + timedelta(days=7),
                max_value=date.today(),
                help="End date for historical data",
            )

        # Prediction settings
        prediction_days = st.number_input(
            "Days to Predict",
            min_value=1,
            max_value=90,
            value=30,
            help="Number of days to predict into the future",
        )

        time_limit = st.slider(
            "Training Time (seconds)",
            min_value=60,
            max_value=600,
            value=120,
            step=30,
            help="Set the maximum training time for the model.",
        )

        # Train button
        train_clicked = st.button("🚀 Train & Predict", use_container_width=True)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This app uses AutoGluon's time series forecasting to predict stock prices. "
            "The model is trained on historical price data and makes predictions for the specified time period."
        )
        st.markdown("### How to Use")
        st.markdown(
            "1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)\n"
            "2. Select a date range for historical data\n"
            "3. Choose the number of days to predict\n"
            "4. Click 'Train & Predict'"
        )

    # Main content area
    if train_clicked and ticker:
        with st.spinner(f"Fetching {ticker} data and training model..."):
            try:
                # Initialize predictor
                predictor = StockPredictor(
                    prediction_length=prediction_days, time_limit=time_limit
                )

                # Fetch stock data
                stock_data = predictor.fetch_stock_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )

                if stock_data.empty:
                    st.error(f"No data found for {ticker} in the specified date range.")
                    return

                # Prepare data for AutoGluon
                ts_data = predictor.prepare_data(stock_data)

                # Split data into train and test sets (80/20 split) for evaluation
                split_point = int(len(ts_data) * 0.8)
                train_data_eval = ts_data.slice_by_timestep(0, split_point)
                test_data_eval = ts_data.slice_by_timestep(split_point + 1, None)

                # Ensure enough data for training and validation
                if len(train_data_eval) < prediction_days + 5:  # Add a small buffer
                    st.error(
                        f"Not enough historical data to train the model for {prediction_days} days of prediction. "
                        "Please select a longer date range or reduce the prediction days."
                    )
                    return

                if test_data_eval.empty:
                    st.error(
                        "Not enough data for evaluation. Please select a longer date range or reduce the prediction days."
                    )
                    return

                # Train model for evaluation
                predictor.train_model(train_data_eval)

                # Make predictions on test set for evaluation
                test_predictions_ts, test_actuals_ts = predictor.predict(test_data_eval)
                test_predictions = test_predictions_ts.reset_index()
                test_actuals = test_actuals_ts.reset_index()

                # Calculate metrics on test set
                metrics = predictor.evaluate_model(train_data_eval, test_data_eval)

                # Train final model on all available data for future predictions
                predictor.train_model(ts_data)

                # Predict future prices
                future_predictions_ts = predictor.predict_future(ts_data)
                future_predictions = future_predictions_ts.reset_index()

                # Update session state
                st.session_state.predictor = predictor
                st.session_state.stock_data = stock_data
                st.session_state.predictions = future_predictions
                st.session_state.actuals = test_actuals
                st.session_state.metrics = metrics

                st.success("Model trained and predictions generated successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

    # Display results if available
    if (
        st.session_state.stock_data is not None
        and st.session_state.predictions is not None
    ):
        # Display metrics
        if hasattr(st.session_state, "metrics") and st.session_state.metrics:
            display_model_metrics(st.session_state.metrics)

        # Plot predictions
        st.subheader(f"{ticker} Stock Price Prediction")
        plot_stock_data(
            history=st.session_state.stock_data,
            predictions=st.session_state.predictions,
            actuals=st.session_state.actuals,
            title=f"{ticker} Stock Price Forecast",
        )

        # Show raw data in expandable section
        with st.expander("View Raw Data"):
            st.subheader("Historical Data")
            st.dataframe(st.session_state.stock_data)

            st.subheader("Predictions")
            st.dataframe(st.session_state.predictions)

    # Show instructions if no data is available yet
    elif not train_clicked:
        st.info(
            "👈 Enter a stock symbol and adjust the settings in the sidebar, "
            "then click 'Train & Predict' to get started."
        )

    # Add footer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This application is for educational purposes only and "
        "should not be used for making financial decisions. Stock market investments "
        "are subject to market risks. Past performance is not indicative of future results."
    )


if __name__ == "__main__":
    main()
