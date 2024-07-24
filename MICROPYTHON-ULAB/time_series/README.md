## Time Series Forecasting for MicroPython-Ulab on ESP32
This repository contains a collection of classes and methods for time series forecasting. 
It includes implementations of various forecasting models and metrics, as well as utilities for evaluation and handling time 
series data. The library is designed to be flexible and extendable, allowing users to experiment with different 
forecasting techniques and metrics.

It's specifically designed for an ESP32 microcontroller. These algorithms are conversions of the main [River](https://riverml.xyz/latest/) algorithms, which can be viewed [here](https://github.com/online-ml/river/tree/main/river/time_series). Due to the limited functionality of micropython-ulab, which does not support the full River algorithms, we have converted the River code into micropython-ulab compatible versions as described in this repository. 

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in [MICROPYTHON-ULAB](https://github.com/online-ml/river/tree/main/river/feature_extraction)

### Overview
The repository includes the following components:

#### Time Series Forecasting Models:

Holt-Winters: Implements the Holt-Winters method for exponential smoothing, supporting both additive and multiplicative seasonal components.
SNARIMAX: Implements a Seasonal Non-Additive AutoRegressive Integrated Moving Average model (SNARIMAX) with differencing.

#### Metrics:

HorizonMetric: A class for calculating metrics over a forecasting horizon.
HorizonAggMetric: Aggregates horizon metrics using a specified aggregation function.
DummyMetric: A placeholder metric for testing purposes.

#### Utilities:

FixedQueue: A fixed-size queue implementation for handling data with a maximum length.
Differencer: A class for differencing time series data to make it stationary.

#### Evaluation:

Iterative Evaluation: Functions for evaluating models over a rolling horizon, including an example of calculating mean absolute error.




