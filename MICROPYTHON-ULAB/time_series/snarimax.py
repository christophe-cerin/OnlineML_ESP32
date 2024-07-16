from collections import deque
import math

class Differencer:
    def __init__(self, d, m=1):
        self.d = d
        self.m = m
        self.coeffs = {0: 1}
        self._calculate_coeffs()

    def _calculate_coeffs(self):
        for k in range(1, self.d + 1):
            t = k * self.m
            coeff = (-1 if k % 2 else 1) * self._n_choose_k(self.d, k)
            self.coeffs[t] = coeff

    def _n_choose_k(self, n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def diff(self, p, Y):
        total = p
        for t, c in self.coeffs.items():
            if t == 0:
                continue
            total += c * Y[t - 1]
        return total

    def undiff(self, p, Y):
        total = p
        for t, c in self.coeffs.items():
            if t == 0:
                continue
            total -= c * Y[t - 1]
        return total


class SNARIMAX:
    def __init__(self, p, d, q, m=1):
        self.p = p
        self.d = d
        self.q = q
        self.m = m
        self.differencer = Differencer(d=d)
        self.y_hist = []
        self.y_diff = []
        self.errors = []

    def _add_lag_features(self, x=None):
        if x is None:
            x = {}
        for t in range(self.p):
            if t < len(self.y_diff):
                x[f"y-{t+1}"] = self.y_diff[t]
        for t in range(self.q):
            if t < len(self.errors):
                x[f"e-{t+1}"] = self.errors[t]
        return x

    def learn_one(self, y, x=None):
        if len(self.y_hist) >= self.differencer.d:
            x = self._add_lag_features(x)
            y_diff = self.differencer.diff(y, self.y_hist)
            self.y_diff.insert(0, y_diff)  # Add to the front
            self.errors.insert(0, y_diff)   # Simplified error update
        self.y_hist.insert(0, y)  # Add to the front

        # Keep only the latest d + m values
        if len(self.y_hist) > self.d + self.m:
            self.y_hist.pop()
        if len(self.y_diff) > max(self.p, self.m):
            self.y_diff.pop()
        if len(self.errors) > self.q:
            self.errors.pop()

    def forecast(self, horizon, xs=None):
        forecasts = []
        for _ in range(horizon):
            x = self._add_lag_features(xs)  # Use features from previous steps
            y_pred = self.differencer.undiff(self.y_diff[0], self.y_hist)
            forecasts.append(y_pred)
            self.y_hist.insert(0, y_pred)
            self.errors.insert(0, 0)  # Simplified for illustration

            # Maintain lengths
            if len(self.y_hist) > self.d + self.m:
                self.y_hist.pop()
            if len(self.y_diff) > max(self.p, self.m):
                self.y_diff.pop()
            if len(self.errors) > self.q:
                self.errors.pop()

        return forecasts


# Example Usage


# Simulated Airline Passengers dataset
airline_passengers = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119,
    104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
    158, 133, 114, 140, 145, 150, 178, 163, 172, 178,
    199, 199, 197, 180, 162, 190, 193, 196, 215, 230,
    204, 202, 218, 224, 230, 242, 246, 237, 220, 215,
    197, 205, 222, 230, 245, 247, 239, 242, 250, 255,
    240, 250, 257, 260, 270, 270, 290, 305, 310, 340,
]

period = 12
model = SNARIMAX(
    p=period,
    d=1,
    q=period,
    m=period
)

# Learning from the dataset
for y in airline_passengers:
    model.learn_one(y)

# Forecasting for the next horizon
horizon = 12
future = [f'1961-{m:02d}-01' for m in range(1, horizon + 1)]
forecast = model.forecast(horizon=horizon)

# Printing the forecasts
for month, y_pred in zip(future, forecast):
    print(month, f'{y_pred:.3f}')
