from ulab import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, l2=0.0, l1=0.0, intercept_init=0.0, intercept_lr=0.01):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.l1 = l1
        self.weights = None
        self.intercept = intercept_init
        self.intercept_lr = intercept_lr

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba_one(self, x):
        z = np.dot(x, self.weights) + self.intercept
        p = self.sigmoid(z)
        return {False: 1.0 - p, True: p}

    def sign(self, x):
        return np.vectorize(lambda x: -1 if x < 0 else (1 if x > 0 else 0))(x)

    def fit(self, X, y, epochs=1):
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_features)

        for _ in range(epochs):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                z = np.dot(xi, self.weights) + self.intercept
                p = self.sigmoid(z)
                error = p - yi

                grad = error * xi
                self.weights -= self.learning_rate * (grad + self.l2 * self.weights + self.l1 * self.sign(self.weights))
                self.intercept -= self.intercept_lr * error

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            z = np.dot(X[i], self.weights) + self.intercept
            p = self.sigmoid(z)
            predictions[i] = p >= 0.5
        return predictions

# Example usage with mock data:
# Define some example data
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0]
])

y = np.array([0, 0, 1, 1, 1])

# Create and train the model
model = LogisticRegression(learning_rate=0.01)
model.fit(X, y, epochs=100)

X_pred=np.array([
    [5.0, 4.0],
    [3.0, 3.0],
    [1.0, 3.0]
])
# Make predictions
predictions = model.predict(X_pred)
print(predictions)
