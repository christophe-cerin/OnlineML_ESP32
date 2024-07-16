import ulab as np

class PolynomialExtender:
    def __init__(self, degree=2, interaction_only=False, include_bias=False, bias_name="bias"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.bias_name = bias_name

    def _powerset(self, iterable):
        """Generate the powerset of an iterable."""
        items = list(iterable)
        powerset = []
        n = len(items)
        
        for size in range(1, self.degree + 1):
            for i in range(n):
                if size == 1:
                    powerset.append((items[i],))
                else:
                    for combo in self._powerset(items[i + 1:]):
                        powerset.append((items[i],) + combo)
        
        return powerset

    def _product(self, values):
        """Calculate the product of a list of values."""
        result = 1
        for value in values:
            result *= value
        return result

    def transform_one(self, x):
        """Transform a single input dictionary into polynomial features."""
        features = {}
        
        for combo in self._powerset(x.keys()):
            key = "*".join(sorted(combo))
            product = self._product([x[c] for c in combo])  # Calculate product of the selected features
            features[key] = product

        if self.include_bias:
            features[self.bias_name] = 1

        return features

# Example usage
poly = PolynomialExtender(degree=2, include_bias=True)
X = [{'x': 0, 'y': 1}, {'x': 2, 'y': 3}, {'x': 4, 'y': 5}]

for x in X:
    transformed = poly.transform_one(x)
    print(transformed)
