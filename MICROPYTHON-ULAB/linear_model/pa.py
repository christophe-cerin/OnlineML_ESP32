from ulab import numpy as np

def sign(x):
    """Returns the sign of x."""
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))

class BasePA:
    def __init__(self, C, mode, learn_intercept):
        self.C = C
        self.mode = mode
        self.calc_tau = {0: self._calc_tau_0, 1: self._calc_tau_1, 2: self._calc_tau_2}[mode]
        self.learn_intercept = learn_intercept
        self.weights = np.zeros(0)
        self.intercept = 0.0

    def _calc_tau_0(self, x, loss):
        norm = np.linalg.norm(x) ** 2
        if norm > 0:
            return loss / norm
        return 0

    def _calc_tau_1(self, x, loss):
        norm = np.linalg.norm(x) ** 2
        if norm > 0:
            return min(self.C, loss / norm)
        return 0

    def _calc_tau_2(self, x, loss):
        return loss / (np.linalg.norm(x) ** 2 + 0.5 / self.C)

    def calc_tau(self, x, loss):
        return self.calc_tau(x, loss)

class PARegressor(BasePA):
    """Passive-Aggressive Regressor."""

    def __init__(self, C=1.0, mode=1, eps=0.1, learn_intercept=True):
        super().__init__(C=C, mode=mode, learn_intercept=learn_intercept)
        self.eps = eps
        self.loss = lambda y, y_pred: max(0, abs(y - y_pred) - eps)  # Epsilon-insensitive loss

    def learn_one(self, x, y):
        if self.weights.size == 0:
            self.weights = np.zeros(len(x))
        y_pred = self.predict_one(x)
        loss = self.loss(y, y_pred)
        tau = self.calc_tau(np.array(list(x.values())), loss)
        step = tau * sign(y - y_pred)  # Use the custom sign function

        for i, xi in x.items():
            self.weights[i] += step * xi
        if self.learn_intercept:
            self.intercept += step

    def predict_one(self, x):
        x = np.array(list(x.values()))
        return np.dot(x, self.weights) + self.intercept

class PAClassifier(BasePA):
    """Passive-Aggressive Classifier."""

    def __init__(self, C=1.0, mode=1, learn_intercept=True):
        super().__init__(C=C, mode=mode, learn_intercept=learn_intercept)
        self.loss = lambda y, y_pred: max(0, 1 - y * y_pred)  # Hinge loss

    def learn_one(self, x, y):
        if self.weights.size == 0:
            self.weights = np.zeros(len(x))
        y_pred = self.predict_proba_one(x)[True]
        tau = self.calc_tau(np.array(list(x.values())), self.loss(y, y_pred))
        step = tau * (1 if y else -1)  # y == False becomes -1

        for i, xi in x.items():
            self.weights[i] += step * xi
        if self.learn_intercept:
            self.intercept += step

    def predict_proba_one(self, x):
        x = np.array(list(x.values()))
        y_pred = 1 / (1 + np.exp(- (np.dot(x, self.weights) + self.intercept)))  # Sigmoid function
        return {False: 1.0 - y_pred, True: y_pred}

# Example usage
if __name__ == "__main__":
    # Example for PARegressor
    print("PARegressor Example")

    # Initialize PARegressor
    regressor = PARegressor(C=0.01, mode=1, eps=0.1, learn_intercept=False)

    # Example data
    X_reg = [{0: 1.0, 1: 2.0}, {0: 2.0, 1: 3.0}, {0: 3.0, 1: 4.0}]
    y_reg = [4.0, 5.0, 6.0]

    # Training
    for xi, yi in zip(X_reg, y_reg):
        regressor.learn_one(xi, yi)

    # Prediction
    x_new_reg = {0: 4.0, 1: 5.0}
    prediction_reg = regressor.predict_one(x_new_reg)
    print(f"PARegressor Prediction: {prediction_reg}")

    # Example for PAClassifier
    print("\nPAClassifier Example")

    # Initialize PAClassifier
    classifier = PAClassifier(C=0.01, mode=1)

    # Example data
    X_cls = [{0: 1.0, 1: 2.0}, {0: 2.0, 1: 3.0}, {0: 3.0, 1: 4.0}]
    y_cls = [True, False, True]

    # Training
    for xi, yi in zip(X_cls, y_cls):
        classifier.learn_one(xi, yi)

    # Prediction
    x_new_cls = {0: 2.0, 1: 3.0}
    probabilities_cls = classifier.predict_proba_one(x_new_cls)
    prediction_cls = max(probabilities_cls, key=probabilities_cls.get)
    print(f"PAClassifier Prediction: {prediction_cls}")
    print(f"Probabilities: {probabilities_cls}")
