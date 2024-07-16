class FixedQueue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.queue = []

    def append(self, item):
        if len(self.queue) >= self.maxlen:
            self.queue.pop(0)
        self.queue.append(item)

    def popleft(self):
        if self.queue:
            return self.queue.pop(0)
        return None

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def __repr__(self):
        return repr(self.queue)

# Define the abstract base class for forecasters
class Forecaster:
    def learn_one(self, y, x=None):
        """Updates the model with a single observation.
        
        Parameters:
        y -- The target variable (numeric)
        x -- Optional additional features (dictionary)
        """
        raise NotImplementedError

    def forecast(self, horizon, xs=None):
        """Generates forecasts for a given horizon.
        
        Parameters:
        horizon -- Number of steps ahead to forecast
        xs -- Optional additional features for each step in the horizon (list of dictionaries)
        """
        raise NotImplementedError

# Define a simple forecaster implementation
class SimpleForecaster(Forecaster):
    def __init__(self):
        self.data = []

    def learn_one(self, y, x=None):
        self.data.append(y)
        return self

    def forecast(self, horizon, xs=None):
        if not self.data:
            return [0] * horizon
        mean_value = sum(self.data) / len(self.data)
        return [mean_value] * horizon

# Dataset iterator with horizon
def _iter_with_horizon(dataset, horizon):
    x_horizon = FixedQueue(maxlen=horizon)
    y_horizon = FixedQueue(maxlen=horizon)

    stream = iter(dataset)
    for _ in range(horizon):
        x, y = next(stream)
        x_horizon.append(x)
        y_horizon.append(y)

    for x, y in stream:
        x_now = x_horizon.popleft()
        y_now = y_horizon.popleft()
        x_horizon.append(x)
        y_horizon.append(y)
        yield x_now, y_now, x_horizon, y_horizon

# Horizon metric classes
class HorizonMetric:
    def __init__(self, metric):
        self.metric = metric
        self.total_error = 0
        self.count = 0

    def update(self, y_true, y_pred):
        error = self.metric(y_true, y_pred)
        self.total_error += error
        self.count += 1

    def get(self):
        return self.total_error / self.count if self.count else float('inf')

def mean_absolute_error(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Iterative evaluation function
def iter_evaluate(dataset, model, metric, horizon, grace_period=None):
    horizon_metric = HorizonMetric(metric)
    steps = _iter_with_horizon(dataset, horizon)

    grace_period = horizon if grace_period is None else grace_period
    for _ in range(grace_period):
        x, y, x_horizon, y_horizon = next(steps)
        model.learn_one(y=y, x=x)

    for x, y, x_horizon, y_horizon in steps:
        y_pred = model.forecast(horizon, xs=x_horizon)
        horizon_metric.update(y_horizon, y_pred)
        model.learn_one(y=y, x=x)
        yield x, y, y_pred, horizon_metric

# Complete evaluation function
def evaluate(dataset, model, metric, horizon, grace_period=None):
    horizon_metric = None
    steps = iter_evaluate(dataset, model, metric, horizon, grace_period)
    for *_, horizon_metric in steps:
        pass
    return horizon_metric.get()

# Example usage
def airline_passengers():
    data = [
        ({'month': (1949, 1, 1)}, 112),
        ({'month': (1949, 2, 1)}, 118),
        ({'month': (1949, 3, 1)}, 132),
        ({'month': (1949, 4, 1)}, 129),
        ({'month': (1949, 5, 1)}, 121),
        ({'month': (1949, 6, 1)}, 135),
        ({'month': (1949, 7, 1)}, 148),
        ({'month': (1949, 8, 1)}, 148),
        # Add more data as needed
    ]
    for item in data:
        yield item

# Instantiate the forecaster, metric, and dataset
model = SimpleForecaster()
metric = mean_absolute_error
dataset = airline_passengers()
horizon = 3

# Perform the evaluation
result = evaluate(dataset, model, metric, horizon)
print('Final Metric:', result)
