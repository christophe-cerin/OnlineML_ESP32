import random

class HSTNode:
    def __init__(self):
        self.l_mass = 0
        self.r_mass = 0

class HSTBranch(HSTNode):
    def __init__(self, left, right, feature, threshold):
        super().__init__()
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold

    def next(self, x):
        try:
            value = x[self.feature]
        except KeyError:
            if self.left.l_mass < self.right.l_mass:
                return self.right
            return self.left
        if value < self.threshold:
            return self.left
        return self.right

class HSTLeaf(HSTNode):
    pass

def make_padded_tree(limits, height, padding):
    if height == 0:
        return HSTLeaf()

    feature = random.choice(list(limits.keys()))
    a, b = limits[feature]
    threshold = random.uniform(a + padding * (b - a), b - padding * (b - a))

    left_limits = limits.copy()
    left_limits[feature] = (a, threshold)
    left = make_padded_tree(left_limits, height - 1, padding)

    right_limits = limits.copy()
    right_limits[feature] = (threshold, b)
    right = make_padded_tree(right_limits, height - 1, padding)

    return HSTBranch(left, right, feature, threshold)

class HalfSpaceTrees:
    def __init__(self, n_trees=10, height=8, window_size=250, limits=None, seed=None):
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.limits = limits if limits else {'x': (0, 1)}
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.trees = []
        self.counter = 0
        self._first_window = True
        self._max_score = self.n_trees * self.window_size * (2 ** (self.height + 1) - 1)

    def learn_one(self, x):
        if not self.trees:
            self.trees = [make_padded_tree(self.limits, self.height, 0.15) for _ in range(self.n_trees)]

        for t in self.trees:
            node = t
            while isinstance(node, HSTBranch):
                node.l_mass += 1
                node = node.next(x)
            node.l_mass += 1

        self.counter += 1
        if self.counter == self.window_size:
            for t in self.trees:
                nodes = [t]
                while nodes:
                    node = nodes.pop()
                    node.r_mass = node.l_mass
                    node.l_mass = 0
                    if isinstance(node, HSTBranch):
                        nodes.append(node.left)
                        nodes.append(node.right)
            self._first_window = False
            self.counter = 0

    def score_one(self, x):
        if self._first_window:
            return 0

        score = 0.0
        for t in self.trees:
            node = t
            depth = 0
            while isinstance(node, HSTBranch):
                score += node.r_mass * (2 ** depth)
                if node.r_mass < 0.1 * self.window_size:
                    break
                node = node.next(x)
                depth += 1

        score /= self._max_score
        return 1 - score

# Example usage
values = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
hst = HalfSpaceTrees(n_trees=5, height=3, window_size=3, seed=42)

for x in values[:3]:
    hst.learn_one({'x': x})

for x in values:
    features = {'x': x}
    hst.learn_one(features)
    print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')
