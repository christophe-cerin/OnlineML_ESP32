from abc import ABC, abstractmethod

class ForecastingMetric(ABC):
    @abstractmethod
    def update(self, y_true, y_pred):
        """Update the metric at each step along the horizon."""

    @abstractmethod
    def get(self):
        """Return the current performance along the horizon."""

class HorizonMetric(ForecastingMetric):
    def __init__(self, metric):
        self.metric = metric
        self.metrics = []

    def update(self, y_true, y_pred):
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if t >= len(self.metrics):
                self.metrics.append(self.metric.clone())
            self.metrics[t].update(yt, yp)
        return self

    def get(self):
        return [metric.get() for metric in self.metrics]

    def __repr__(self):
        result = ""
        for t, metric in enumerate(self.metrics):
            result += "+{} {:.6f}\n".format(t + 1, metric.get())
        return result.strip()

class HorizonAggMetric(HorizonMetric):
    def __init__(self, metric, agg_func):
        super().__init__(metric)
        self.agg_func = agg_func

    def get(self):
        return self.agg_func(super().get())

    def __repr__(self):
        name = "{}({})".format(self.agg_func.__name__, self.metric.__class__.__name__)
        return "{}: {:.6f}".format(name, self.get())

# Custom mean function
def mean(values):
    return sum(values) / len(values) if values else 0

# Dummy regression metric for testing purposes
class DummyMetric:
    def __init__(self):
        self.values = []

    def update(self, y_true, y_pred):
        self.values.append(abs(y_true - y_pred))
        return self

    def get(self):
        return sum(self.values) / len(self.values) if self.values else 0

    def clone(self):
        return DummyMetric()

    def __repr__(self):
        return "{:.6f}".format(self.get())  # Show the computed value

# Example usage
y_true = [3, 5, 7, 9]
y_pred = [2.5, 5.5, 6.5, 8.5]

metric = DummyMetric()
horizon_metric = HorizonMetric(metric)
horizon_metric.update(y_true, y_pred)
print(horizon_metric)

agg_metric = HorizonAggMetric(metric, mean)
agg_metric.update(y_true, y_pred)
print(agg_metric)
