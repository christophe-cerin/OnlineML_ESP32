class ThresholdFilter:
    def __init__(self, anomaly_detector, threshold, protect_anomaly_detector=True):
        self.anomaly_detector = anomaly_detector
        self.threshold = threshold
        self.protect_anomaly_detector = protect_anomaly_detector

    def classify(self, score):
        return score >= self.threshold

    def learn_one(self, x):
        score = self.anomaly_detector.score_one(x)
        if not self.protect_anomaly_detector or not self.classify(score):
            self.anomaly_detector.learn_one(x)
        return score


class QuantileFilter:
    def __init__(self, anomaly_detector, q, protect_anomaly_detector=True):
        self.anomaly_detector = anomaly_detector
        self.q = q
        self.protect_anomaly_detector = protect_anomaly_detector
        self.scores = []

    def classify(self, score):
        if len(self.scores) == 0:
            return False
        self.scores.sort()
        threshold = self.scores[int(len(self.scores) * self.q)]
        return score >= threshold

    def learn_one(self, x):
        score = self.anomaly_detector.score_one(x)
        if not self.protect_anomaly_detector or not self.classify(score):
            self.anomaly_detector.learn_one(x)
        self.scores.append(score)
        return score


class SimpleAnomalyDetector:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.n = 0

    def score_one(self, x):
        return abs(x - self.mean)

    def learn_one(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.var += delta * (x - self.mean)


import random

def test_threshold_filter():
    data = [random.uniform(0, 1) for _ in range(1000)] + [random.uniform(1, 2) for _ in range(10)]
    anomaly_detector = SimpleAnomalyDetector()
    threshold_filter = ThresholdFilter(anomaly_detector, threshold=0.5)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, x in enumerate(data):
        score = threshold_filter.learn_one(x)
        is_anomaly = threshold_filter.classify(score)
        is_actual_anomaly = i >= 1000

        if is_anomaly and is_actual_anomaly:
            true_positives += 1
        elif is_anomaly and not is_actual_anomaly:
            false_positives += 1
        elif not is_anomaly and not is_actual_anomaly:
            true_negatives += 1
        elif not is_anomaly and is_actual_anomaly:
            false_negatives += 1

    print('ThresholdFilter Results:')
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'True Negatives: {true_negatives}')
    print(f'False Negatives: {false_negatives}')


def test_quantile_filter():
    data = [random.uniform(0, 1) for _ in range(1000)] + [random.uniform(1, 2) for _ in range(10)]
    anomaly_detector = SimpleAnomalyDetector()
    quantile_filter = QuantileFilter(anomaly_detector, q=0.95)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i, x in enumerate(data):
        score = quantile_filter.learn_one(x)
        is_anomaly = quantile_filter.classify(score)
        is_actual_anomaly = i >= 1000

        if is_anomaly and is_actual_anomaly:
            true_positives += 1
        elif is_anomaly and not is_actual_anomaly:
            false_positives += 1
        elif not is_anomaly and not is_actual_anomaly:
            true_negatives += 1
        elif not is_anomaly and is_actual_anomaly:
            false_negatives += 1

    print('QuantileFilter Results:')
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'True Negatives: {true_negatives}')
    print(f'False Negatives: {false_negatives}')


# Run the tests
test_threshold_filter()
test_quantile_filter()
