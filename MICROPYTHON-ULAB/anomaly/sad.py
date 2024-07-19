# Standard Absolute Deviation (SAD) anomaly detector for MicroPython

import micropython_random

class StandardAbsoluteDeviation:
    def __init__(self):
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0

    def update(self, value):
        self.count += 1
        prev_mean = self.mean
        self.mean = prev_mean + (value - prev_mean) / self.count
        self.variance = self.variance + (value - prev_mean) * (value - self.mean)

    def score(self, value):
        if self.count == 0:
            return 0.0
        else:
            stddev = (self.variance / self.count) ** 0.5
            score = abs((value - self.mean) / (stddev + 1e-10))
            return score

# Example usage:
def example_usage():
    seed = 42
    _rng = micropython_random.Random(seed)
        #print("RRRGGGGG :", _rng.gauss)
 
    model = StandardAbsoluteDeviation()
    
    for _ in range(150):
        y = _rng.gauss(0, 1)  # Generate random data
        model.update(y)  # Update model with data

    # Calculate scores for different values
    score1 = model.score(2)
    score2 = model.score(0)
    score3 = model.score(1)

    # Print scores
    print("Score 1:", score1)
    print("Score 2:", score2)
    print("Score 3:", score3)

# Run example usage
example_usage()
