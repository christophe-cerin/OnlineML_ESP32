from __future__ import annotations
from ulab import numpy as np
import random

def outer_product(a, b):
    """Compute the outer product of vectors a and b."""
    result = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            result[i, j] = a[i] * b[j]
    return result

class BayesianLinearRegression:
    """Bayesian linear regression.

    Parameters
    ----------
    alpha
        Prior parameter.
    beta
        Noise parameter.
    smoothing
        Smoothing allows the model to gradually "forget" the past.

    """

    def __init__(self, alpha=1, beta=1, smoothing: float = None):
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self._ss = {}
        self._ss_inv = {}
        self._m = {}
        self._n = 1

    def _get_arrays(self, features, m=True, ss=True, ss_inv=True):
        m_arr = np.array([self._m.get(i, 0.0) for i in features]) if m else None
        ss_arr = (
            np.array(
                [
                    [
                        self._ss.get(
                            min((i, j), (j, i)),
                            1.0 / self.alpha if i == j else 0.0,
                        )
                        for j in features
                    ]
                    for i in features
                ]
            )
            if ss
            else None
        )
        ss_inv_arr = (
            np.array(
                [
                    [
                        self._ss_inv.get(
                            min((i, j), (j, i)),
                            1.0 / self.alpha if i == j else 0.0,
                        )
                        for j in features
                    ]
                    for i in features
                ]
            )
            if ss_inv
            else None
        )
        return m_arr, ss_arr, ss_inv_arr

    def _set_arrays(self, features, m_arr, ss_arr, ss_inv_arr):
        for i, fi in enumerate(features):
            self._m[fi] = m_arr[i]
            ss_row = ss_arr[i]
            ss_inv_row = ss_inv_arr[i]
            for j, fj in enumerate(features):
                self._ss[min((fi, fj), (fj, fi))] = ss_row[j]
                self._ss_inv[min((fi, fj), (fj, fi))] = ss_inv_row[j]

    def _matrix_inverse(self, matrix):
        """ Simple matrix inversion for 2x2 matrices as an example. """
        if matrix.shape[0] != 2 or matrix.shape[1] != 2:
            raise NotImplementedError("Only 2x2 matrices are supported for inversion.")
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        inv_matrix = np.array([
            [matrix[1, 1] / det, -matrix[0, 1] / det],
            [-matrix[1, 0] / det, matrix[0, 0] / det]
        ])
        return inv_matrix

    def learn_one(self, x, y):
        x_arr = np.array(list(x.values()))
        m_arr, ss_arr, ss_inv_arr = self._get_arrays(x.keys())

        bx = self.beta * x_arr

        if self.smoothing is None:
            # Sherman-Morrison approximation
            A = np.array(ss_inv_arr)
            u = bx
            v = x_arr
            A_inv = self._matrix_inverse(A)
            A_inv += outer_product(u, v) / (1.0 + np.dot(v, np.dot(A_inv, u)))
            m_arr = np.dot(A_inv, np.dot(ss_arr, m_arr) + bx * y)
            ss_arr += outer_product(bx, x_arr)
        else:
            new_ss_arr = self.smoothing * ss_arr + (1 - self.smoothing) * outer_product(bx, x_arr)
            ss_inv_arr = self._matrix_inverse(new_ss_arr)
            m_arr = np.dot(ss_inv_arr, self.smoothing * np.dot(ss_arr, m_arr) + (1 - self.smoothing) * bx * y)
            ss_arr = new_ss_arr

        self._set_arrays(x.keys(), m_arr, ss_arr, ss_inv_arr)

    def predict_one(self, x, with_dist=False):
        x_arr = np.array(list(x.values()))
        y_pred_mean = np.dot(x_arr, np.array([self._m.get(i, 0.0) for i in x.keys()]))

        if not with_dist:
            return y_pred_mean

        ss_inv_arr = np.array(self._get_arrays(x.keys(), m=False, ss=False, ss_inv=True)[2])
        y_pred_var = 1 / self.beta + np.dot(x_arr, np.dot(ss_inv_arr, x_arr))
        
        return {"mean": y_pred_mean, "std_dev": y_pred_var**0.5}

    def predict_many(self, X):
        m = np.array([self._m.get(i, 0.0) for i in range(X.shape[1])])
        return np.dot(X, m)

# Helper function to generate random data
def random_data(coefs, n, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = {i: random.random() for i, c in enumerate(coefs)}
        y = sum(c * xi for c, xi in zip(coefs, x.values()))
        data.append((x, y))
    return data

"""def main():
    # Create a dataset with and without concept drift
    data1 = random_data([0.1, 3], 100)
    data2 = random_data([10, -2], 100)
    dataset = data1 + data2

    # Initialize model
    model = BayesianLinearRegression(alpha=1, beta=1, smoothing=0.8)

    # Train model
    for x, y in dataset:
        model.learn_one(x, y)

    # Predict
    x_test = {0: 0.5, 1: 0.5}
    print("Prediction without distribution:", model.predict_one(x_test))
    print("Prediction with distribution:", model.predict_one(x_test, with_dist=True))

if __name__ == "__main__":
    main()
"""
def test_model(model, data):
    for x, y in data:
        model.learn_one(x, y)

    # Test predictions
    x_test = {0: 0.5, 1: 0.5}
    pred = model.predict_one(x_test)
    dist_pred = model.predict_one(x_test, with_dist=True)

    print("Prediction without distribution:", pred)
    print("Prediction with distribution:", dist_pred)

    # Optionally compare to some known or expected values
    # known_mean = ...
    # known_std_dev = ...
    # print(f"Known mean: {known_mean}, Known std_dev: {known_std_dev}")

if __name__ == "__main__":
    model = BayesianLinearRegression(alpha=1, beta=1, smoothing=0.8)
    data = random_data([0.1, 3], 100) + random_data([10, -2], 100)
    test_model(model, data)



