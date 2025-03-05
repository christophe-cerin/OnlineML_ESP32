#
# This code provides a basic implementation of the TBATS
# (Trigonometric Box-Cox, ARMA Errors, Trend, and Seasonality)
# model. It includes a simple Box-Cox transformation, seasonal
# decomposition, trend estimation using a moving average, and AR(1)
# model for residuals. For a more robust implementation, you would
# need to handle more complex seasonal patterns, multiple
# seasonalities, and potentially use more advanced models for trend
# and seasonality estimation.
#
# Here, I'll provide a simplified version of the TBATS model. Note
# that this implementation will be a basic one and may not cover all
# the nuances and optimizations of the full TBATS model. For a
# production-ready implementation, using a library like sktime is
# highly recommended.
#
# Let's break down the components and implement them step by step.
# Step 1: Box-Cox Transformation
#   Box-Cox transformation is used to stabilize the variance of a time series.
# Step 2: Seasonal Decomposition
#   The TBATS model handles multiple seasonalities. Here, we'll use a
#   simple seasonal decomposition for demonstration purposes.
# Step 3: Trend Estimation
#   We'll use a simple moving average to estimate the trend.
# Step 4: ARMA Model
#   For simplicity, we'll use a simple AR(1) model to model the residuals.
# Step 5: Forecasting
#   Combine the components to make a forecast.
#

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA


def box_cox_transform(series, lambda_value=None):
    if lambda_value is None:
        transformed_series, lambda_value = boxcox(
            series + 1
        )  # Adding 1 to avoid zero values
    else:
        transformed_series = (series**lambda_value - 1) / lambda_value
    return transformed_series, lambda_value


def inverse_box_cox_transform(series, lambda_value):
    if lambda_value == 0:
        return np.exp(series)
    else:
        return (series * lambda_value + 1) ** (1 / lambda_value)


def seasonal_decompose(series, seasonal_periods):
    seasonal_component = np.zeros_like(series)
    for i in range(len(series)):
        seasonal_component[i] = (
            series[i - seasonal_periods] if i >= seasonal_periods else series[i]
        )
    return seasonal_component


def estimate_trend(series, window=12):
    trend_component = np.convolve(series, np.ones(window) / window, mode="valid")
    trend_component = np.pad(trend_component, (window // 2, window // 2), mode="edge")
    return trend_component[: len(series)]


def fit_arima_model(residuals, order=(1, 0, 0)):
    model = ARIMA(residuals, order=order)
    model_fit = model.fit()
    return model_fit


def tbats_forecast(
    series,
    seasonal_periods,
    forecast_steps=12,
    box_cox_lambda=None,
    arima_order=(1, 0, 0),
):
    # Box-Cox Transform
    transformed_series, lambda_value = box_cox_transform(series, box_cox_lambda)
    # print(len(transformed_series),transformed_series)

    # Seasonal Decomposition
    seasonal_component = seasonal_decompose(transformed_series, seasonal_periods)
    # print(len(seasonal_component),seasonal_component)

    # Trend Estimation
    trend_component = estimate_trend(transformed_series, window=seasonal_periods)
    # print(len(trend_component),seasonal_periods,trend_component)

    # Residuals
    residuals = transformed_series - trend_component - seasonal_component

    # Fit ARIMA model to residuals
    arima_model = fit_arima_model(residuals, arima_order)

    # Forecasting
    forecast_residuals = arima_model.forecast(steps=forecast_steps)

    # Reconstruct the forecasted series
    last_seasonal = seasonal_component[-seasonal_periods:]
    last_trend = trend_component[-1]
    forecasted_series = last_trend + last_seasonal + forecast_residuals

    # Inverse Box-Cox Transform
    forecasted_series = inverse_box_cox_transform(forecasted_series, lambda_value)

    return forecasted_series


def NormalizeData(data):
    # Normalize in [0-1]
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    # Example usage
    # Create a sample time series
    np.random.seed(0)
    series = np.sin(np.linspace(0, 10, 150)) + np.random.normal(0, 0.1, 150)
    time_series = NormalizeData(series)

    # Fit TBATS and forecast
    forecasted_series = tbats_forecast(
        time_series, seasonal_periods=12, forecast_steps=12
    )

    # Print the forecasted values
    print(forecasted_series)
