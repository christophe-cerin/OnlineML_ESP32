import ulab as np
import micropython_random
import math

class RBFSampler:
    def __init__(self, gamma=1.0, n_components=10, seed=None):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed
        self.rng = micropython_random.Random(seed)
        self.weights = [[0] * n_components for _ in range(n_components)]
        self.offsets = [self.rng.uniform(0, 2 * math.pi) for _ in range(n_components)]
        self._initialize_weights()

    def _manual_sqrt(self, x):
        return x ** 0.5

    def _initialize_weights(self):
        for j in range(self.n_components):
            self.weights[j] = [
                self._manual_sqrt(2 * self.gamma) * self.rng.gauss(0, 1)
                for _ in range(self.n_components)
            ]

    def transform_one(self, x):
        transformed = {}
        for i, xi in enumerate(x):
            for j in range(self.n_components):
                transformed[(i, j)] = math.cos(xi * self.weights[j][i] + self.offsets[j])
        return transformed

class LogisticRegression:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None

    def predict_one(self, x):
        if self.weights is None:
            return 0
        linear_combination = sum(x[i] * self.weights[i] for i in range(len(self.weights)))
        return 1 / (1 + math.exp(-linear_combination))

    def learn_one(self, x, y):
        if self.weights is None:
            self.weights = [0] * len(x)
        y_pred = self.predict_one(x)
        error = y - y_pred
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * x[i]


#Example Usage
            
# Sample data
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
Y = [0, 0, 1, 1]

# Create model
model = RBFSampler(seed=3)
log_reg = LogisticRegression()

# Train and predict
for x, y in zip(X, Y):
    transformed_x = model.transform_one(x)
    log_reg.learn_one(list(transformed_x.values()), y)
    y_pred = log_reg.predict_one(list(transformed_x.values()))
    print(y, int(y_pred))
