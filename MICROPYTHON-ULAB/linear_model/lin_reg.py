from ulab import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, l2=0.0, l1=0.0):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.l1 = l1
        self.weights = None
        self.intercept = 0.0

    def fit(self, X, y, epochs=100):
        num_samples, num_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(num_features)
        
        for _ in range(epochs):
            predictions = self.predict(X)
            errors = y - predictions
            gradient = -2 / num_samples * np.dot(X.T, errors)
            
            # Apply L1 and L2 regularization
            if self.l1 > 0:
                gradient += self.l1 * np.sign(self.weights)
            if self.l2 > 0:
                gradient += self.l2 * self.weights
            
            # Update weights
            self.weights -= self.learning_rate * gradient

            # Update intercept
            self.intercept -= self.learning_rate * np.mean(errors)

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def debug_one(self, x):
        prediction = self.predict(np.array([x]))
        contributions = [xi * self.weights[i] for i, xi in enumerate(x)] + [self.intercept]
        return {
            "Prediction": prediction[0],
            "Weights": self.weights.tolist(),
            "Intercept": self.intercept,
            "Contributions": contributions
        }

# Example Usage
if __name__ == "__main__":
    # Example data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    y = np.array([3, 7, 11])

    # Initialize and fit the model
    model = LinearRegression(learning_rate=0.01, l2=0.1)
    model.fit(X, y, epochs=1000)

    # Predict
    prediction = model.predict(np.array([[3, 4]]))
    print("Prediction:", prediction)

    # Debug one example
    debug_info = model.debug_one([2, 3])
    print("Debug Info:", debug_info)
