import math
import random
#from typing import Callable, List

class Node(list):
    def add(self, other):
        return Node([x + y for x, y in zip(self, other)])

    def sub(self, other):
        return Node([x - y for x, y in zip(self, other)])

    def mul(self, scalar):
        return Node([x * scalar for x in self])

class Cluster:
    def __init__(self, center: Node, count: int = 0):
        self.center = center
        self.count = count

class Kmeans:
    def __init__(self, k: int, distance: Callable[[Node, Node], float], alpha: float):
        self.clusters = [Cluster(Node([])) for _ in range(k)]
        self.k = k
        self.distance = distance
        self.alpha = alpha

    def add(self, node: Node):
        self._add(node, self.clusters, self.distance)

    def addf(self, node: Node):
        self._addf(node, self.clusters, self.distance, self.alpha)

    def sequential(self, nodes: List[Node]):
        if not self.seed(nodes):
            raise ValueError("There is not enough data")
        for node in nodes:
            self.add(node)

    def seed(self, nodes: List[Node]):
        if len(nodes) < self.k:
            return False
        self._seed(nodes, self.clusters, self.distance, self.k)
        return True

def near(node: Node, clusters: List[Cluster], distance_func: Callable[[Node, Node], float]):
    min_distance = math.inf
    index = -1
    for j, cluster in enumerate(clusters):
        dist = distance_func(node, cluster.center)
        if dist < min_distance:
            min_distance = dist
            index = j
    return index, min_distance

def _seed(nodes: List[Node], clusters: List[Cluster], distance_func: Callable[[Node, Node], float], k: int):
    length = len(nodes)
    clusters[0].center = nodes[random.randint(0, length - 1)]
    for i in range(1, k):
        d2 = [0] * length
        total = 0
        for j, node in enumerate(nodes):
            _, dist = near(node, clusters[:i], distance_func)
            d2[j] = dist ** 2
            total += d2[j]
        target = random.random() * total
        j = 0
        for running_sum in itertools.accumulate(d2):
            if running_sum >= target:
                break
            j += 1
        clusters[i].center = nodes[j]

def _add(node: Node, clusters: List[Cluster], distance_func: Callable[[Node, Node], float]):
    index, _ = near(node, clusters, distance_func)
    cluster = clusters[index]
    cluster.count += 1
    cluster.center = cluster.center.add(node.sub(cluster.center).mul(1 / cluster.count))

def _addf(node: Node, clusters: List[Cluster], distance_func: Callable[[Node, Node], float], alpha: float):
    index, _ = near(node, clusters, distance_func)
    cluster = clusters[index]
    cluster.count += 1
    cluster.center = cluster.center.add(node.sub(cluster.center).mul(alpha))

