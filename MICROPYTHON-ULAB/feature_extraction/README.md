## Feature Extraction code for MicroPython on ESP32

This repository contains Python code for various feature extraction techniques useful for data preprocessing and model improvement, specifically designed to run on the ESP32. These algorithms are conversions of the main [River](https://riverml.xyz/latest/) algorithms, which can be viewed [here](https://github.com/online-ml/river/tree/main/river/feature_extraction). Due to the limited functionality of micropython-ulab, which does not support the full River algorithms, we have converted the River code into micropython-ulab compatible versions as described in this repository. 
The tools are designed for use with MicroPython on ESP32 devices, leveraging the micropython-ulab library for numerical operations.

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in MICROPYTHON-ULAB.

### Example Usage

#### TimeRollingAgg
The TimeRollingAgg class computes rolling statistics for time-series data. It calculates the mean of values over a specified time window, grouped by certain features.

Features:
- Window-Based Aggregation: Aggregates data within a rolling time window.
- Group-Based: Groups data by specified features.

 ``` 
# Example usage
agg = TimeRollingAgg(on="value", by=["group"], window=7 * 24 * 3600)  # 7 days in seconds
start_time = time.mktime((2023, 1, 1, 0, 0, 0, 0, 0, -1))

for day in range(366):
    g = chr(random.randint(97, 122))
    x = {"group": g, "value": ord(g) - 97 + random.random()}
    t = start_time + day * 24 * 3600
    agg.learn_one(x, t=t)

print("Number of unique groups:", len(agg.state))
print("Current state:", agg.state)

```

#### RBFSampler
The RBFSampler class performs random Fourier feature mapping for approximating the radial basis function (RBF) kernel. It transforms data into a higher-dimensional space for better performance with linear models.

Features:
- Gamma Parameter: Controls the width of the RBF kernel.
- Random Fourier Features: Uses random weights and offsets to approximate the kernel.
```
# Example Usage
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
Y = [0, 0, 1, 1]

model = RBFSampler(seed=3)
log_reg = LogisticRegression()

for x, y in zip(X, Y):
    transformed_x = model.transform_one(x)
    log_reg.learn_one(list(transformed_x.values()), y)
    y_pred = log_reg.predict_one(list(transformed_x.values()))
    print(y, int(y_pred))
```

#### PolynomialExtender
The PolynomialExtender class generates polynomial features from input data. It expands features up to a specified polynomial degree, including interaction terms and optional bias.

Features:
- Polynomial Degree: Defines the maximum degree of polynomial features.
- Interaction Terms: Generates features that are interactions between input features.
- Bias Term: Optionally adds a bias term to the features.

```
# Example Usage
poly = PolynomialExtender(degree=2, include_bias=True)
X = [{'x': 0, 'y': 1}, {'x': 2, 'y': 3}, {'x': 4, 'y': 5}]

for x in X:
    transformed = poly.transform_one(x)
    print(transformed)
```
