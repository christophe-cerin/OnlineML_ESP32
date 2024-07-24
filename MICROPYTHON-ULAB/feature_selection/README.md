## Feature Selection code for MicroPython on ESP32
This repository contains micropython code for various feature extraction techniques useful for data preprocessing and model improvement, specifically designed to run on the ESP32. These algorithms are conversions of the main [River](https://riverml.xyz/latest/) algorithms, which can be viewed [here](https://github.com/online-ml/river/tree/main/river/feature_extraction). Due to the limited functionality of micropython-ulab, which does not support the full River algorithms, we have converted the River code into micropython-ulab compatible versions as described in this repository. 
The tools are designed for use with MicroPython on ESP32 devices, leveraging the micropython-ulab library for numerical operations.

### Pre-requisites
MicroPython-ulab: Ensure you have MicroPython-ulab installed on your ESP32. You can find installation instructions in [MICROPYTHON-ULAB](https://github.com/online-ml/river/tree/main/river/feature_extraction)

### Example Usage

#### Poisson Correlation
Computes the Pearson correlation coefficient between two variables to assess their linear relationship.
 ```
def example():
    X = [
        {0: 0.5, 1: 2.3, 2: 2.9},
        {0: 0.6, 1: 2.1, 2: 2.8},
        {0: 0.8, 1: 2.0, 2: 2.5},
        {0: 1.0, 1: 1.9, 2: 2.3},
    ]
    y = [1.0, 1.1, 1.2, 1.3]

    selector = SelectKBest(similarity=PearsonCorr(), k=2)

    for xi, yi in zip(X, y):
        selector.learn_one(xi, yi)

    print(selector.leaderboard)
    print(selector.transform_one(X[-1]))

example()

 ```

#### Poisson Inclusion
Randomly selects features based on a probability of inclusion. Features are added with a probability p using a geometric distribution.
 ```
def example():
    X = [
        {0: 0.5, 1: 2.3, 2: 2.9},
        {0: 0.6, 1: 2.1, 2: 2.8},
        {0: 0.8, 1: 2.0, 2: 2.5},
        # Add more samples
    ]
    y = [1.0, 1.1, 1.2, 1.3]

    selector = PoissonInclusion(p=0.1, seed=42)
    feature_names = X[0].keys()
    n = 0

    for xi, yi in zip(X, y):
        xt = selector.transform_one(xi)
        if set(xt.keys()) == feature_names:
            break
        n += 1

    print(f"Number of iterations until all features are included: {n}")

example()

 ```

#### Variance Threshold
Removes features with variance below a certain threshold. Features are assessed based on their variance across the dataset.
 ```
def example():
    X = [
        {0: 0, 1: 2, 2: 0, 3: 3},
        {0: 0, 1: 1, 2: 4, 3: 3},
        {0: 0, 1: 1, 2: 1, 3: 3}
    ]

    selector = VarianceThreshold()

    for x in X:
        selector.learn_one(x)
        print(selector.transform_one(x))

example()
 ```
