
# Anomaly Detection Algorithms for MicroPython on ESP32
This repository contains implementations of various anomaly detection algorithms for MicroPython, specifically designed to run on the ESP32. The algorithms included are:

1) Local Outlier Factor (LOF)
2) Half-Space Trees (HST)
3) Standard Absolute Deviation (SAD)
4) One-Class SVM

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in MICROPYTHON-ULAB.

### Example Usage

####  Local Outlier Factor (LOF)
LOF detects anomalies by measuring the local density deviation of a given data point with respect to its neighbors.
 ``` 
import ulab
import ulab as np
import math
import copy

# Insert the LOF implementation code here

# Example usage:
X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
lof = LocalOutlierFactor()

for x in X[:3]:
    lof.learn_one({'x': x})  # Warming up

for x in X:
    features = {'x': x}
    print(f'Anomaly score for x={x:.3f}: {lof.score_one(features):.3f}')
    lof.learn_one(features)
 ```

#### Half-Space Trees (HST)
HST uses a forest of randomly created binary trees to detect anomalies
```
import random

# Insert the HST implementation code here

# Example usage:
values = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
hst = HalfSpaceTrees(n_trees=5, height=3, window_size=3, seed=42)

for x in values[:3]:
    hst.learn_one({'x': x})

for x in values:
    features = {'x': x}
    hst.learn_one(features)
    print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')
 ```

####  Standard Absolute Deviation (SAD)
SAD uses the standard deviation to measure the deviation of a data point from the mean.
 ```
import micropython_random

# Insert the SAD implementation code here

# Example usage:
def example_usage():
    seed = 42
    _rng = micropython_random.Random(seed)
 
    model = StandardAbsoluteDeviation()
    
    for _ in range(150):
        y = _rng.gauss(0, 1)  # Generate random data
        model.update(y)  # Update model with data

    # Calculate scores for different values
    score1 = model.score(2)
    score2 = model.score(0)
    score3 = model.score(1)

    # Print scores
    print("Score 1:", score1)
    print("Score 2:", score2)
    print("Score 3:", score3)

# Run example usage
example_usage()
 ```

#### One-Class SVM
One-Class SVM learns the boundary of the normal data and scores the anomalies based on their distance from this boundary.
```
import random

#Insert the One-Class SVM implementation code here
 
# Example usage:
def example_usage():
    import random

    model = OneClassSVM(nu=0.2)

    # Simulating data generation and learning
    for _ in range(100):
        x = {'feature1': random.uniform(-1, 1), 'feature2': random.uniform(-1, 1)}
        model.learn_one(x)

    # Scoring some new data points
    score1 = model.score_one({'feature1': 0.5, 'feature2': -0.2})
    score2 = model.score_one({'feature1': -0.1, 'feature2': 0.3})
    score3 = model.score_one({'feature1': 0.0, 'feature2': 0.0})

    print("Score 1:", score1)
    print("Score 2:", score2)
    print("Score 3:", score3)

# Run example usage
example_usage()
 ```
