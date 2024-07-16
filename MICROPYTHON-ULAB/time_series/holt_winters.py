from ucollections import deque
import math

class Component:
    def __init__(self, maxlen):
        self.deque = deque((), maxlen)
        self.maxlen = maxlen

    def append(self, value):
        if len(self.deque) == self.maxlen:
            self.deque.popleft()
        self.deque.append(value)

    def __getitem__(self, index):
        if index < 0:
            index += len(self.deque)
        for i, item in enumerate(self.deque):
            if i == index:
                return item
        raise IndexError("deque index out of range")

    def extend(self, values):
        for value in values:
            self.append(value)

class AdditiveLevel(Component):
    def __init__(self, alpha):
        super().__init__(2)
        self.alpha = alpha

    def update(self, y, trend, season):
        season_value = season[-season.seasonality] if season else 0
        trend_value = trend[-1] if trend else 0
        self.append(self.alpha * (y - season_value) + (1 - self.alpha) * (self[-1] + trend_value))

class MultiplicativeLevel(Component):
    def __init__(self, alpha):
        super().__init__(2)
        self.alpha = alpha

    def update(self, y, trend, season):
        season_value = season[-season.seasonality] if season else 1
        trend_value = trend[-1] if trend else 0
        self.append(self.alpha * (y / season_value) + (1 - self.alpha) * (self[-1] + trend_value))

class Trend(Component):
    def __init__(self, beta):
        super().__init__(2)
        self.beta = beta

    def update(self, y, level):
        self.append(self.beta * (level[-1] - level[-2]) + (1 - self.beta) * self[-1])

class AdditiveSeason(Component):
    def __init__(self, gamma, seasonality):
        super().__init__(seasonality + 1)
        self.gamma = gamma
        self.seasonality = seasonality

    def update(self, y, level, trend):
        self.append(self.gamma * (y - level[-2] - trend[-2]) + (1 - self.gamma) * self[-self.seasonality])

class MultiplicativeSeason(Component):
    def __init__(self, gamma, seasonality):
        super().__init__(seasonality + 1)
        self.gamma = gamma
        self.seasonality = seasonality

    def update(self, y, level, trend):
        self.append(self.gamma * y / (level[-2] + trend[-2]) + (1 - self.gamma) * self[-self.seasonality])

class HoltWinters:
    def __init__(self, alpha, beta=None, gamma=None, seasonality=0, multiplicative=False):
        if seasonality and gamma is None:
            raise ValueError("gamma must be set if seasonality is set")
        if gamma and beta is None:
            raise ValueError("beta must be set if gamma is set")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonality = seasonality
        self.multiplicative = multiplicative
        self.level = MultiplicativeLevel(alpha) if multiplicative else AdditiveLevel(alpha)
        self.trend = Trend(beta) if beta else None
        self.season = (MultiplicativeSeason(gamma, seasonality) if multiplicative else AdditiveSeason(gamma, seasonality)) if seasonality else None
        self._first_values = []
        self._initialized = False

    def learn_one(self, y, x=None):
        if self._initialized:
            self.level.update(y, self.trend, self.season)
            if self.trend is not None:
                self.trend.update(y, self.level)
            if self.season is not None:
                self.season.update(y, self.level, self.trend)
            return

        self._first_values.append(y)
        if len(self._first_values) < max(2, self.seasonality):
            return

        # Initialize components
        level_init = sum(self._first_values) / len(self._first_values)
        self.level.append(level_init)

        if self.trend is not None:
            diffs = [self._first_values[i + 1] - self._first_values[i] for i in range(len(self._first_values) - 1)]
            trend_init = sum(diffs) / len(diffs)
            self.trend.append(trend_init)

        if self.season is not None:
            season_init = [y / level_init for y in self._first_values]
            self.season.extend(season_init)

        self._initialized = True

    def forecast(self, horizon, xs=None):
        forecasts = []
        for h in range(horizon):
            if self.trend:
                trend_forecast = self.level[-1] + (h + 1) * self.trend[-1]
            else:
                trend_forecast = self.level[-1]

            if self.season:
                season_forecast = self.season[-self.seasonality + h % self.seasonality]
            else:
                season_forecast = 1 if self.multiplicative else 0

            if self.multiplicative:
                forecasts.append(trend_forecast * season_forecast)
            else:
                forecasts.append(trend_forecast + season_forecast)
        
        return forecasts


# Example time series data (e.g., monthly sales data with a seasonal pattern)
time_series_data = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 242,
    233, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227
]

# Initialize the Holt-Winters model
model = HoltWinters(alpha=0.3, beta=0.1, gamma=0.6, seasonality=12, multiplicative=False)

# Learn from the time series data
for i, value in enumerate(time_series_data):
    model.learn_one(value)
    print(f"Learned value {i + 1}: {value}")

# Forecast the next 12 periods
forecast_horizon = 12
forecasts = model.forecast(horizon=forecast_horizon)

# Print the forecasted values
print("\nForecasted values for the next 12 periods:")
for i, forecast in enumerate(forecasts):
    print(f"Period {i + 1}: {forecast}")

# Optional: Simulate real-time learning and forecasting
import time
for value in [250, 260, 270]:  # Simulate new incoming data
    model.learn_one(value)
    print(f"New incoming value: {value}")
    time.sleep(1)  # Simulate a delay (e.g., 1 second) between new data points

# Forecast again after learning from new data
forecasts = model.forecast(horizon=forecast_horizon)
print("\nUpdated forecasted values for the next 12 periods after new data:")
for i, forecast in enumerate(forecasts):
    print(f"Period {i + 1}: {forecast}")
