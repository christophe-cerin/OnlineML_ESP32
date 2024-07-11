# One-class SVM for anomaly detection in MicroPython

import random

class OneClassSVM:
    def __init__(self, nu=0.1, learning_rate=0.01, clip_gradient=1e12):
        self.nu = nu
        self.learning_rate = learning_rate
        self.clip_gradient = clip_gradient
        self.weights = {}
        self.intercept = 1.0
        self.n_iterations = 0

    def _initialize_weights(self, x):
        if not self.weights:
            self.weights = {k: 0.0 for k in x}

    def _clip(self, value, threshold):
        if value > threshold:
            return threshold
        if value < -threshold:
            return -threshold
        return value

    def _raw_dot(self, x):
        return sum(self.weights.get(k, 0.0) * v for k, v in x.items())

    def learn_one(self, x):
        self._initialize_weights(x)
        self.n_iterations += 1

        y = 1  # In one-class SVM, we treat the target as always 1
        pred = self._raw_dot(x) - self.intercept
        loss_gradient = -y if pred < 1 else 0

        for k, v in x.items():
            grad = loss_gradient * v + 2.0 * self.nu * self.weights[k]
            grad = self._clip(grad, self.clip_gradient)
            self.weights[k] -= self.learning_rate * grad

        intercept_update = loss_gradient + 2.0 * self.learning_rate * self.nu
        self.intercept -= self.learning_rate * intercept_update

    def score_one(self, x):
        return self._raw_dot(x) - self.intercept

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
#example_usage()
