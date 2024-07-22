import ulab as np

class VarianceThreshold:
    """Removes low-variance features."""

    def __init__(self, threshold=0, min_samples=2):
        self.threshold = threshold
        self.min_samples = min_samples
        self.variances = {}

    def learn_one(self, x):
        for i, xi in x.items():
            if i not in self.variances:
                self.variances[i] = {
                    'n': 0,
                    'mean': 0,
                    'm2': 0
                }
            self.variances[i]['n'] += 1
            delta = xi - self.variances[i]['mean']
            self.variances[i]['mean'] += delta / self.variances[i]['n']
            delta2 = xi - self.variances[i]['mean']
            self.variances[i]['m2'] += delta * delta2

    def check_feature(self, feature):
        if feature not in self.variances:
            return True
        if self.variances[feature]['n'] < self.min_samples:
            return True
        variance = self.variances[feature]['m2'] / (self.variances[feature]['n'] - 1)
        return variance > self.threshold

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if self.check_feature(i)}

# Example usage
def example():
    # Simulated dataset
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
