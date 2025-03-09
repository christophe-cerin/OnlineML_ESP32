#
# The TBATS (Trigonometric Seasonal, Box-Cox Transformation, ARMA
# errors, Trend, and Seasonal components) model is a powerful method
# for time series forecasting that can handle multiple seasonalities
# and complex patterns. If you want to implement TBATS in Python
# without using the sktime library, you can use other libraries like
# statsmodels for ARMA modeling and custom implementations for the
# other components.
#
# Below is a simplified version of TBATS using Python. This
# implementation will cover the basic structure and logic of the TBATS
# model, but it may not be as optimized or feature-complete as the
# sktime version.
#
# Step-by-Step Implementation
#   Box-Cox Transformation: To stabilize the variance.
#   Trend and Seasonal Decomposition: Using moving averages or other
#   decomposition methods.
#   ARMA Modeling: For residuals.
#   Back-Transformation: To get the final forecast.
#
# Required Libraries
#   numpy and pandas for data manipulation.
#   scipy for Box-Cox transformation.
#   statsmodels for ARMA modeling.
#

import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import savgol_filter
from scipy.optimize import minimize

#
# See https://thingsdaq.org/2023/04/18/circular-buffer-in-python/
#
import numpy as np
import matplotlib.pyplot as plt


class RingBuffer:
    """Class that implements a not-yet-full buffer."""

    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.data = []

    class __Full:
        """Class that implements a full buffer."""

        def add(self, x):
            """Add an element overwriting the oldest one."""
            self.data[self.currpos] = x
            self.currpos = (self.currpos + 1) % self.bufsize

        def get(self):
            """Return list of elements in correct order."""
            return self.data[self.currpos :] + self.data[: self.currpos]

    def add(self, x):
        """Add an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.bufsize:
            # Initializing current position attribute
            self.currpos = 0
            # Permanently change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """Return a list of elements from the oldest to the newest."""
        return self.data


# Box-Cox transformation
def boxcox_transform(series):
    transformed, _ = boxcox(series + 1e-3)  # Adding small constant to avoid zero
    return transformed


def invboxcox(y, ld):
    if ld == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(ld * y + 1) / ld)


# Inverse Box-Cox transformation
def inverse_boxcox_transform(series, lam):
    if lam == 0:
        return np.exp(series)
    else:
        return np.power(series * lam + 1, 1 / lam)


# Savitzky-Golay filter for trend extraction
def extract_trend(series, window_length=51, polyorder=3):
    return savgol_filter(series, window_length, polyorder)


# Seasonal decomposition using LOESS or similar
def extract_seasonality(series, trend, window_length=51, polyorder=3):
    detrended = series - trend
    seasonal = savgol_filter(detrended, window_length, polyorder)
    return seasonal


# Fit ARMA model to residuals
def fit_arma(residuals, order=(1, 1, 1)):
    model = ARIMA(residuals, order=order)
    model_fit = model.fit()
    return model_fit


# TBATS model implementation
def tbats_forecast(
    series,
    trend_window=51,
    trend_polyorder=3,
    seasonal_window=51,
    seasonal_polyorder=3,
    arma_order=(1, 1, 1),
    n_forecast=10,
):
    # Box-Cox transformation
    transformed_series = boxcox_transform(series)

    # Extract trend
    trend = extract_trend(
        transformed_series, window_length=trend_window, polyorder=trend_polyorder
    )

    # Extract seasonality
    seasonality = extract_seasonality(
        transformed_series,
        trend,
        window_length=seasonal_window,
        polyorder=seasonal_polyorder,
    )

    # Extract residuals
    residuals = transformed_series - trend - seasonality

    # Fit ARMA model to residuals
    arma_fit = fit_arma(residuals, order=arma_order)

    # Forecast residuals
    residual_forecast = arma_fit.forecast(steps=n_forecast)

    # Reconstruct forecast
    forecast = trend[-1] + seasonality[-1] + residual_forecast
    # print(trend[-1])
    # print(seasonality[-1])
    # print(residual_forecast)
    forecast = inverse_boxcox_transform(
        forecast, np.mean(transformed_series - np.log(transformed_series + 1e-3))
    )
    # forecast = inverse_boxcox_transform(
    #    forecast, np.mean(series - np.log(series + 1e-3))
    # )
    # fcast = invboxcox(forecast,np.mean(transformed_series- np.log(transformed_series + 1e-3)))
    # print(fcast + np.mean(series - np.log(series + 1e-3)))
    # print('Mean:',np.mean(series - np.log(series + 1e-3)))
    return forecast + np.mean(series - np.log(series + 1e-3))


if __name__ == "__main__":

    # Size of the ring buffer
    my_size = 360
    # Motif length
    mm = 74
    # Number of windows of size my_size.
    chunk_size = 14

    #
    # Initialization
    #
    from numpy import genfromtxt

    #
    # Humidity is at position 3, and temperature at position 5
    #
    data = genfromtxt(
        "TourPerret.csv", delimiter=";", comments="#", usecols=(3), skip_header=1
    )

    ringbuffer = RingBuffer(my_size)

    for i in range(my_size * chunk_size):

        if i > 0 and i % my_size == 0:

            # Forecasting
            # forecast = tbats_forecast(data[i:i+my_size], n_forecast=5)
            tab = np.array(ringbuffer.get())
            forecast = tbats_forecast(tab, n_forecast=5)

            # Print forecast
            print(
                "Forecast based on data between [%d,%d[: %s"
                % (i - my_size, i, ["{0:0.2f}".format(i) for i in forecast])
            )

            #
            # https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html#
            #
            from sktime.forecasting.tbats import TBATS
            import pandas as pd

            forecaster = TBATS(
                use_box_cox=False,
                use_trend=False,
                use_damped_trend=False,
                sp=12,
                use_arma_errors=False,
                n_jobs=1,
            )

            forecaster.fit(pd.Series(tab))

            y_pred = forecaster.predict(fh=[1, 2, 3, 4, 5])

            out = []
            for row in y_pred:
                out = out + [row]

            print(
                "Forecast based on SKTIME + TBATS:", ["{0:0.2f}".format(i) for i in out]
            )

        else:

            ringbuffer.add(data[i])

    plt.grid(visible=True, axis="x", color="black", linestyle=":", linewidth=1)
    plt.plot(data[0 : my_size * (chunk_size - 1)], color="red", label="Input data")
    plt.xticks(rotation=-45)
    plt.xticks(np.arange(0, my_size * (chunk_size - 1) + 1, my_size))
    plt.legend()
    plt.show()
