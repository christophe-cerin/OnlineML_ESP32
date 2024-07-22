from ulab import numpy as np
from math import sqrt, exp

class SimpleDefaultDict(dict):
    def __init__(self, default_factory):
        self.default_factory = default_factory
        super().__init__()

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.default_factory()
        return super().__getitem__(key)

class ALMAClassifier:
    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2**0.5):
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.w = SimpleDefaultDict(float)
        self.k = 1

    def _raw_dot(self, x):
        return sum(xi * self.w.get(i, 0) for i, xi in x.items())

    def sigmoid(self, z):
        return 1 / (1 + exp(-z))

    def predict_proba_one(self, x):
        yp = self.sigmoid(self._raw_dot(x))
        return {False: 1 - yp, True: yp}

    def learn_one(self, x, y):
        y = int(y or -1)
        gamma = self.B * sqrt(self.p - 1) / sqrt(self.k)
        if y * self._raw_dot(x) < (1 - self.alpha) * gamma:
            eta = self.C / (sqrt(self.p - 1) * sqrt(self.k))
            for i, xi in x.items():
                self.w[i] += eta * y * xi
            norm = self.norm(self.w, order=self.p)
            for i in x:
                self.w[i] /= max(1, norm)
            self.k += 1
        print(f"Updated weights: {self.w}")

    def norm(self, w, order):
        return sum(abs(wi)**order for wi in w.values())**(1 / order)

class PhishingDataset:
    def __init__(self):
        self.data = [
            ({"feature1": 0.11, "feature2": 0.4}, 1),
            ({"feature1": 0.2, "feature2": 0.3}, 0),
            ({"feature1": 0.4, "feature2": 0.1}, 1),
            ({"feature1": 0.3, "feature2": 0.22}, 1),
            ({"feature1": 0.15, "feature2": 0.5}, 1),
            ({"feature1": 0.65, "feature2": 0.6}, 0),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            instance = self.data[self.index]
            self.index += 1
            return instance
        else:
            raise StopIteration

class StandardScaler:
    def __init__(self):
        self.mean = SimpleDefaultDict(float)
        self.var = SimpleDefaultDict(float)
        self.count = 0

    def learn_one(self, x):
        self.count += 1
        for i, xi in x.items():
            delta = xi - self.mean[i]
            self.mean[i] += delta / self.count
            self.var[i] += delta * (xi - self.mean[i])
        return self

    def transform_one(self, x):
        return {i: (xi - self.mean[i]) / (sqrt(self.var[i] / (self.count - 1)) if self.count > 1 else 1)
                for i, xi in x.items()}

    def __or__(self, other):
        self.model = other
        return self

    def learn_one(self, x, y):
        x = self.transform_one(x)
        self.model.learn_one(x, y)

    def predict_proba_one(self, x):
        x = self.transform_one(x)
        return self.model.predict_proba_one(x)

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, y_true, y_pred):
        self.total += 1
        if y_true == y_pred:
            self.correct += 1
        print(f"True: {y_true}, Predicted: {y_pred}, Correct: {self.correct}, Total: {self.total}")

    def get(self):
        return self.correct / self.total if self.total else 0

def progressive_val_score(dataset, model, metric):
    for x, y in dataset:
        y_pred_proba = model.predict_proba_one(x)
        y_pred = max(y_pred_proba, key=y_pred_proba.get)
        print(f"Input: {x}, True label: {y}, Predicted proba: {y_pred_proba}, Predicted label: {y_pred}")
        metric.update(y, y_pred)
        model.learn_one(x, y)
    return metric.get()

dataset = PhishingDataset()
model = StandardScaler() | ALMAClassifier()
metric = Accuracy()

accuracy = progressive_val_score(dataset, model, metric)
print(f"Accuracy: {accuracy * 100:.2f}%")
