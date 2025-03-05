#
# Facebook's Prophet is a popular open-source library for time series
# forecasting. It is implemented in Python and R. Below is a pure
# Python implementation of the core functionality of Prophet, focusing
# on the main steps involved in time series forecasting. However, it's
# important to note that a full implementation of Prophet is quite
# complex and involves many components such as trend, seasonality, and
# holiday effects. For simplicity, I'll provide a basic example that
# captures the essence of the Prophet algorithm.
#
# Explanation:
#   Fourier Series: This function generates the seasonal
#      component using a Fourier series expansion.
#   Trend Component: This function adds a piecewise linear trend
#      with changepoints.
#   Seasonality Component: This function adds a seasonal component
#      using the Fourier series.
#   Fit Prophet: This function fits a linear regression model to
#      the data with the trend and seasonality components.
#   Predict Prophet: This function predicts future values using
#      the fitted model.
#
# Note:
#
# This is a simplified version and does not include all the features
# of the original Prophet library, such as holidays, growth models, or
# robust parameter tuning.
#
# The example uses a synthetic dataset with a sinusoidal trend for
# demonstration purposes.  For a production-ready solution, it is
# recommended to use the official Prophet library (fbprophet or
# prophet) from Facebook. You can install the official Prophet library
# using:
# pip install prophet
# This will provide you with a fully-featured and optimized version
# of Prophet.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta


def fourier_series(t, a, b, freq, order):
    """Fourier series for seasonality."""
    return sum(
        a[i] * np.cos(freq * (i + 1) * t) + b[i] * np.sin(freq * (i + 1) * t)
        for i in range(order)
    )


def add_trend(df, changepoints, changepoint_prior_scale):
    """Add trend component to the dataframe."""
    df["t"] = (df["ds"] - df["ds"].min()) / timedelta(days=1)
    df["trend"] = 0.0
    df["cp"] = 0.0
    for cp in changepoints:
        df["cp"] = df["cp"] + (df["ds"] >= cp).astype(int)
        df["trend"] += (
            np.log1p(
                df["t"]
                - (df["ds"] >= cp).astype(int)
                * (cp - df["ds"].min())
                / timedelta(days=1)
            )
            * changepoint_prior_scale
            * df["cp"]
        )

    return df


def add_seasonality(df, period, fourier_order):
    """Add seasonality component to the dataframe."""
    df["seasonality"] = fourier_series(
        (df["ds"] - df["ds"].min()).dt.total_seconds() / (period * 86400),
        np.random.randn(fourier_order),
        np.random.randn(fourier_order),
        2 * np.pi / (period * 86400),
        fourier_order,
    )
    return df


def fit_prophet(df, changepoints, changepoint_prior_scale, period, fourier_order):
    """Fit the Prophet model."""
    df = add_trend(df, changepoints, changepoint_prior_scale)
    df = add_seasonality(df, period, fourier_order)

    X = df[["trend", "seasonality"]].values
    y = df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    return model, df


def predict_prophet(model, df, future_periods):
    """Predict using the fitted Prophet model."""
    # future_df = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max(), periods=future_periods + 1, closed='right')})
    future_df = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=df["ds"].max(), periods=future_periods + 1, inclusive="right"
            )
        }
    )
    future_df = add_trend(future_df, changepoints, changepoint_prior_scale)
    future_df = add_seasonality(future_df, period, fourier_order)

    future_df["yhat"] = model.predict(future_df[["trend", "seasonality"]].values)
    return future_df


if __name__ == "__main__":
    # Example usage
    my_size = 150

    df = pd.DataFrame(
        {
            "ds": pd.date_range(start="2022-01-01", periods=my_size),
            "y": np.sin(np.linspace(0, 2 * np.pi, my_size))
            + np.random.normal(0, 0.1, my_size),
        }
    )

    # changepoints = pd.date_range(start="2022-01-01", periods=5, freq="20D")
    changepoints = pd.date_range(start="2022-01-01", periods=my_size, freq="30D")
    changepoint_prior_scale = 0.05
    # changepoint_prior_scale = 1.0
    # period = 365
    period = 0.05
    fourier_order = 6

    model, df = fit_prophet(
        df, changepoints, changepoint_prior_scale, period, fourier_order
    )
    future_df = predict_prophet(model, df, future_periods=my_size)

    print(df)
    # print(changepoints)
    print(future_df)

    from prophet import Prophet

    m = Prophet()
    m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
    future_prophet = m.make_future_dataframe(periods=my_size)
    Le_futur = m.predict(future_prophet)[my_size:]
    print(Le_futur["yhat"])

    labels = ["Original Signal", "Pure Prophet", "Prophet Lib"]
    colors = ["r", "g", "b"]

    fig, axs = plt.subplots(3, 1, layout="constrained")

    for i, ax in enumerate(axs):
        if i == 0:
            axs[i].plot(df["ds"], df["y"], color=colors[i], label=labels[i])
            axs[i].legend(loc="upper right")
            # axs[0].set_xlim(0, 2)
            # axs1.set_xlabel("Date")
            # axs1.set_ylabel("Original signal")
            axs[i].grid(True)
        elif i == 1:
            axs[i].plot(
                future_df["ds"], future_df["yhat"], color=colors[i], label=labels[i]
            )
            axs[i].legend(loc="upper right")
            axs[i].set_ylabel("Forecast")
            axs[i].grid(True)
        else:
            axs[i].plot(
                Le_futur["ds"], Le_futur["yhat"], color=colors[i], label=labels[i]
            )
            axs[i].legend(loc="upper right")
            axs[i].set_ylabel("Forecast")
            axs[i].grid(True)

    # plt.figure(figsize=(10, 6))
    # plt.plot(df['ds'], df['y'], label='Original')
    # plt.plot(future_df['ds'], future_df['yhat'], label='Forecast', linestyle='--')
    # plt.legend()

    plt.show()
