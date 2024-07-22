from __future__ import annotations
import contextlib
from ulab import numpy as np

# Dummy classes to replace river's optim and utils; these should be adapted for actual use
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.n_iterations = 0

    def look_ahead(self, w):
        pass

    def step(self, w, g):
        for key in w:
            if key in g:
                w[key] -= self.learning_rate * g[key]
        self.n_iterations += 1

class Loss:
    def gradient(self, y_true, y_pred):
        return y_pred - y_true

class VectorDict(dict):
    def __init__(self, data=None, initializer=None, mask=None):
        super().__init__(data or {})
        self.initializer = initializer
        self.mask = mask
        if initializer:
            for key in self.keys():
                if key not in self:
                    self[key] = initializer()

    def to_dict(self):
        return dict(self)

    def to_numpy(self, columns):
        return np.array([self.get(col, 0) for col in columns])

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

class GLM:
    """Generalized Linear Model."""

    def __init__(
        self,
        optimizer,
        loss,
        l2,
        l1,
        intercept_init,
        intercept_lr,
        clip_gradient,
        initializer,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.l2 = l2
        self.l1 = l1
        self.intercept_init = intercept_init
        self.intercept = intercept_init
        self.intercept_lr = intercept_lr
        self.clip_gradient = clip_gradient
        self.initializer = initializer
        self._weights = VectorDict(initializer=initializer)

        self._y_name = None

        if l1 != 0 and l2 != 0:
            raise NotImplementedError("L1 and L2 penalties cannot be used together!")

        if l1 != 0:
            self.max_cum_l1 = 0
            self.cum_l1 = VectorDict(initializer=lambda: 0)

    @property
    def _mutable_attributes(self):
        return {"optimizer", "l2", "l1", "loss", "intercept_lr", "clip_gradient", "initializer"}

    @property
    def weights(self):
        return self._weights.to_dict()

    @contextlib.contextmanager
    def _learn_mode(self, mask=None):
        weights = self._weights
        try:
            self._weights = VectorDict(weights, self.initializer, mask)
            yield
        finally:
            self._weights = weights

    def _get_intercept_update(self, loss_gradient):
        return self.intercept_lr * loss_gradient

    def _fit(self, x, y, w, get_grad):
        self.optimizer.look_ahead(w=self._weights)

        gradient, loss_gradient = get_grad(x, y, w)

        # Debugging statements
        print(f"Gradient: {gradient}")
        print(f"Loss Gradient: {loss_gradient}")

        self.intercept -= self._get_intercept_update(loss_gradient)

        # Debugging statement
        print(f"Updated Intercept: {self.intercept}")

        self.optimizer.step(w=self._weights, g=gradient)

        # Debugging statement
        print(f"Updated Weights: {self._weights.to_dict()}")

        if self.l1 != 0.0:
            self.max_cum_l1 += self.l1 * self.optimizer.learning_rate
            self._update_weights(x)

        return self

    def _update_weights(self, x):
        for j, xj in x.items():
            wj_temp = self._weights.get(j, 0)
            if wj_temp > 0:
                self._weights[j] = max(0, wj_temp - (self.max_cum_l1 + self.cum_l1.get(j, 0)))
            elif wj_temp < 0:
                self._weights[j] = min(0, wj_temp + (self.max_cum_l1 - self.cum_l1.get(j, 0)))
            else:
                self._weights[j] = wj_temp
            self.cum_l1[j] = self.cum_l1.get(j, 0) + (self._weights[j] - wj_temp)

        # Debugging statement
        print(f"Updated Weights after L1 Penalty: {self._weights.to_dict()}")

    def _raw_dot_one(self, x: dict) -> float:
        return sum(self._weights.get(i, 0) * xi for i, xi in x.items()) + self.intercept

    def _eval_gradient_one(self, x: dict, y: float, w: float) -> tuple[dict, float]:
        y_pred = self._raw_dot_one(x)
        loss_gradient = self.loss.gradient(y_true=y, y_pred=y_pred)
        loss_gradient *= w
        loss_gradient = clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        # Debugging statement
        print(f"Evaluated Gradient (One): {loss_gradient}")

        if self.l2:
            gradient = {i: loss_gradient * xi + self.l2 * self._weights.get(i, 0) for i, xi in x.items()}
            return gradient, loss_gradient
        else:
            gradient = {i: loss_gradient * xi for i, xi in x.items()}
            return gradient, loss_gradient

    def learn_one(self, x, y, w=1.0):
        with self._learn_mode(x):
            self._fit(x, y, w, get_grad=self._eval_gradient_one)

    def _raw_dot_many(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._weights.to_numpy(range(X.shape[1]))) + self.intercept

    def _eval_gradient_many(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[dict, float]:
        y_pred = self._raw_dot_many(X)
        loss_gradient = self.loss.gradient(y_true=y, y_pred=y_pred)
        loss_gradient *= w
        loss_gradient = np.clip(loss_gradient, -self.clip_gradient, self.clip_gradient)

        # Debugging statements
        print(f"Loss Gradient (Many): {loss_gradient}")

        gradient = np.dot(X.T, loss_gradient) / X.shape[0]
        if self.l2:
            gradient += self.l2 * self._weights.to_numpy(range(X.shape[1]))

        return dict(enumerate(gradient)), np.mean(loss_gradient)

    def learn_many(self, X: np.ndarray, y: np.ndarray, w: np.ndarray = 1):
        self._y_name = 'target'
        self._fit(X, y, w, get_grad=self._eval_gradient_many)

# Example usage
optimizer = Optimizer(learning_rate=0.01)
loss = Loss()
glm = GLM(
    optimizer=optimizer,
    loss=loss,
    l2=0.01,
    l1=0.0,
    intercept_init=0.0,
    intercept_lr=0.01,
    clip_gradient=1.0,
    initializer=lambda: 0.0,
)

# Dummy data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 0, 1])
w = np.array([1, 1, 1])

print("Before Learning:")
print(f"Weights: {glm.weights}")
print(f"Intercept: {glm.intercept}")

glm.learn_many(X, y, w)

print("After Learning:")
print(f"Weights: {glm.weights}")
print(f"Intercept: {glm.intercept}")
