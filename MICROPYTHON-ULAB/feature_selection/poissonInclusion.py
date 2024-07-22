import random

class PoissonInclusion:
    """Randomly selects features with an inclusion trial.

    When a new feature is encountered, it is selected with probability `p`. The number of times a
    feature needs to be seen before it is added to the model follows a geometric distribution with
    expected value `1 / p`. This feature selection method is meant to be used when you have a
    very large amount of sparse features.

    Parameters
    ----------
    p : float
        Probability of including a feature the first time it is encountered.
    seed : int | None
        Random seed value used for reproducibility.
    """

    def __init__(self, p: float, seed: int = None):
        self.p = p
        if seed is not None:
            random.seed(seed)
        self.included = set()

    def transform_one(self, x):
        xt = {}
        for i, xi in x.items():
            if i in self.included:
                xt[i] = xi
            elif random.random() < self.p:
                self.included.add(i)
                xt[i] = xi
        return xt

# Example usage
def example():
    # Simulated dataset
    X = [
        {0: 0.5, 1: 2.3, 2: 2.9},
        {0: 0.6, 1: 2.1, 2: 2.8},
        {0: 0.8, 1: 2.0, 2: 2.5},
        {0: 1.0, 1: 1.9, 2: 2.3},
        {0: 1.2, 1: 1.8, 2: 2.1},
        {0: 1.4, 1: 1.7, 2: 1.9},
        {0: 1.6, 1: 1.6, 2: 1.7},
        {0: 1.8, 1: 1.5, 2: 1.5},
        {0: 2.0, 1: 1.4, 2: 1.3},
        {0: 2.2, 1: 1.3, 2: 1.1},
        {0: 2.4, 1: 1.2, 2: 0.9},
        {0: 2.6, 1: 1.1, 2: 0.7},
    ]
    y = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]

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
