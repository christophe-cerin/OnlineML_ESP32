import ulab as np
import random
import time

class TimeRollingAgg:
    def __init__(self, on, by, window):
        self.on = on
        self.by = by
        self.window = window
        self.groups = {}

    def _make_key(self, x):
        return tuple(x[k] for k in self.by)

    def learn_one(self, x, t):
        key = self._make_key(x)
        current_time = t  # Using time directly instead of datetime

        if key not in self.groups:
            self.groups[key] = {'sum': 0, 'count': 0, 'timestamps': []}

        # Remove old timestamps
        self.groups[key]['timestamps'] = [
            ts for ts in self.groups[key]['timestamps'] if current_time - ts < self.window
        ]

        # Update sum and count
        self.groups[key]['timestamps'].append(current_time)
        self.groups[key]['sum'] += x[self.on]
        self.groups[key]['count'] += 1

    def transform_one(self, x):
        key = self._make_key(x)
        if key in self.groups and self.groups[key]['count'] > 0:
            mean = self.groups[key]['sum'] / self.groups[key]['count']
            return {f"mean_{'_'.join(self.by)}": mean}
        else:
            return {f"mean_{'_'.join(self.by)}": None}

    @property
    def state(self):
        return {key: {'mean': grp['sum'] / grp['count'] if grp['count'] > 0 else None} for key, grp in self.groups.items()}


# Initialize the TimeRollingAgg transformer
agg = TimeRollingAgg(on="value", by=["group"], window=7 * 24 * 3600)  # 7 days in seconds

# Simulate daily data for one year
start_time = time.mktime((2023, 1, 1, 0, 0, 0, 0, 0, -1))  # Starting timestamp

for day in range(366):
    g = chr(random.randint(97, 122))  # Generate a random lowercase letter
    x = {
        "group": g,
        "value": ord(g) - 97 + random.random(),  # Generate a value based on the group
    }
    t = start_time + day * 24 * 3600  # Incrementing day by day
    agg.learn_one(x, t=t)

# Output the current state of the aggregator
print("Number of unique groups:", len(agg.state))
print("Current state:", agg.state)
