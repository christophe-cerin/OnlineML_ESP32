## Time Series Forecasting for MicroPython-Ulab on ESP32
This repository contains a collection of classes and methods for time series forecasting. 
It includes implementations of various forecasting models and metrics, as well as utilities for evaluation and handling time 
series data. The library is designed to be flexible and extendable, allowing users to experiment with different 
forecasting techniques and metrics.

It's specifically designed for an ESP32 microcontroller. These algorithms are conversions of the main [River](https://riverml.xyz/latest/) algorithms, which can be viewed [here](https://github.com/online-ml/river/tree/main/river/time_series). Due to the limited functionality of micropython-ulab, which does not support the full River algorithms, we have converted the River code into micropython-ulab compatible versions as described in this repository. 

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in [MICROPYTHON-ULAB](https://github.com/online-ml/river/tree/main/river/feature_extraction)

### Overview & Usage
The repository includes the following components:

#### Metrics:

HorizonMetric: A class for calculating metrics over a forecasting horizon.  <br /> 
HorizonAggMetric: Aggregates horizon metrics using a specified aggregation function.  <br /> 
DummyMetric: A placeholder metric for testing purposes.
```
#Example
y_true = [3, 5, 7, 9]
y_pred = [2.5, 5.5, 6.5, 8.5]

metric = DummyMetric()
horizon_metric = HorizonMetric(metric)
horizon_metric.update(y_true, y_pred)
print(horizon_metric)

agg_metric = HorizonAggMetric(metric, mean)
agg_metric.update(y_true, y_pred)
print(agg_metric)
```

#### Time Series Forecasting Models:

Holt-Winters: Implements the Holt-Winters method for exponential smoothing, supporting both additive and multiplicative seasonal components.  <br /> 
SNARIMAX: Implements a Seasonal Non-Additive AutoRegressive Integrated Moving Average model (SNARIMAX) with differencing.

```
#SNARIMAX Example
from snarimax import SNARIMAX

# Initialize the SNARIMAX model
model = SNARIMAX(p=12, d=1, q=12, m=12)

# Learn from the dataset
for y in airline_passengers:
    model.learn_one(y)

# Forecast the next 12 periods
forecast = model.forecast(horizon=12)
print(forecast)

```
```
#Hot Winters Example
from holt_winters import HoltWinters

# Initialize the Holt-Winters model
model = HoltWinters(alpha=0.3, beta=0.1, gamma=0.6, seasonality=12, multiplicative=False)

# Learn from the time series data
for value in time_series_data:
    model.learn_one(value)

# Forecast the next 12 periods
forecasts = model.forecast(horizon=12)
print(forecasts)
```

#### Evaluation:

Iterative Evaluation: Functions for evaluating models over a rolling horizon, including an example of calculating mean absolute error.
```
# Example usage
def airline_passengers():
    data = [
        ({'month': (1949, 1, 1)}, 112),
        ({'month': (1949, 2, 1)}, 118),
        ({'month': (1949, 3, 1)}, 132),
        ({'month': (1949, 4, 1)}, 129),
        ({'month': (1949, 5, 1)}, 121),
        ({'month': (1949, 6, 1)}, 135),
        ({'month': (1949, 7, 1)}, 148),
        ({'month': (1949, 8, 1)}, 148),
        # Add more data as needed
    ]
    for item in data:
        yield item

# Instantiate the forecaster, metric, and dataset
model = SimpleForecaster()
metric = mean_absolute_error
dataset = airline_passengers()
horizon = 3

# Perform the evaluation
result = evaluate(dataset, model, metric, horizon)
print('Final Metric:', result)

```



