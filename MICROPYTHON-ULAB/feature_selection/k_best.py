import copy
import ulab as np
import math

class PearsonCorr:
    """Compute Pearson correlation between two variables."""

    def __init__(self):
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_xx = 0.0
        self.sum_yy = 0.0
        self.n = 0

    def update(self, x, y):
        self.sum_x += x
        self.sum_y += y
        self.sum_xy += x * y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.n += 1

    def get(self):
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den = math.sqrt((self.n * self.sum_xx - self.sum_x**2) * (self.n * self.sum_yy - self.sum_y**2))
        return num / den if den else 0.0


class SelectKBest:
    """Removes all but the $k$ highest scoring features."""

    def __init__(self, similarity, k=10):
        self.k = k
        self.similarity = similarity
        self.similarities = {}
        self.leaderboard = {}

    def learn_one(self, x, y):
        for i, xi in x.items():
            if i not in self.similarities:
                self.similarities[i] = copy.deepcopy(self.similarity)
            self.similarities[i].update(xi, y)
            self.leaderboard[i] = self.similarities[i].get()

    def transform_one(self, x):
        best_features = sorted(self.leaderboard.items(), key=lambda item: item[1], reverse=True)[:self.k]
        best_features = {i for i, _ in best_features}
        return {i: xi for i, xi in x.items() if i in best_features}


# Example usage

def example():
    # Simulated dataset
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
    # Output the transformed last instance
    print(selector.transform_one(X[-1]))

example()
