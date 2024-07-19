import ulab
import ulab as np
import math
import copy

def check_equal(x_list, y_list):
    #result = [x for x in x_list if not any(np.array_equal(x, y) for y in y_list)]
    #return result, len(x_list) - len(result)
    result = [x for x in x_list if not any(x == y for y in y_list)]
    return result, len(x_list) - len(result)

def expand_objects(new_particles, x_list, neighborhoods, rev_neighborhoods, k_dist, reach_dist, dist_dict, local_reach, lof):
    n = len(x_list)
    m = len(new_particles)
    x_list.extend(new_particles)
    neighborhoods.update({i: [] for i in range(n + m)})
    rev_neighborhoods.update({i: [] for i in range(n + m)})
    k_dist.update({i: float("inf") for i in range(n + m)})
    reach_dist.update({i + n: {} for i in range(m)})
    dist_dict.update({i + n: {} for i in range(m)})
    local_reach.update({i + n: 0 for i in range(m)})
    lof.update({i + n: 0 for i in range(m)})
    return (n, m), x_list, neighborhoods, rev_neighborhoods, k_dist, reach_dist, dist_dict, local_reach, lof

def define_sets(nm, neighborhoods, rev_neighborhoods):
    set_new_points = set(range(nm[0], nm[0] + nm[1]))
    set_neighbors = set()
    set_rev_neighbors = set()

    for i in set_new_points:
        set_neighbors.update(neighborhoods[i])
        set_rev_neighbors.update(rev_neighborhoods[i])

    set_upd_lrd = set_rev_neighbors.copy()
    for j in set_rev_neighbors:
        set_upd_lrd.update(rev_neighborhoods[j])
    set_upd_lrd.update(set_new_points)

    set_upd_lof = set_upd_lrd.copy()
    for m in set_upd_lrd:
        set_upd_lof.update(rev_neighborhoods[m])

    return set_new_points, set_neighbors, set_rev_neighbors, set_upd_lrd, set_upd_lof

def calc_reach_dist_new_points(set_index, neighborhoods, rev_neighborhoods, reach_dist, dist_dict, k_dist):
    for c in set_index:
        for j in neighborhoods[c]:
            reach_dist[c][j] = max(dist_dict[c][j], k_dist[j])
        for j in rev_neighborhoods[c]:
            reach_dist[j][c] = max(dist_dict[j][c], k_dist[c])
    return reach_dist

def calc_reach_dist_other_points(set_index, rev_neighborhoods, reach_dist, dist_dict, k_dist):
    for j in set_index:
        for i in rev_neighborhoods[j]:
            reach_dist[i][j] = max(dist_dict[i][j], k_dist[j])
    return reach_dist

def calc_local_reach_dist(set_index, neighborhoods, reach_dist, local_reach_dist):
    for i in set_index:
        denominator = sum(reach_dist[i][j] for j in neighborhoods[i])
        local_reach_dist[i] = len(neighborhoods[i]) / denominator if denominator else 0
    return local_reach_dist

def calc_lof(set_index, neighborhoods, local_reach, lof):
    for i in set_index:
        denominator = len(neighborhoods[i]) * local_reach[i]
        lof[i] = sum(local_reach[j] for j in neighborhoods[i]) / denominator if denominator else 0
    return lof

class LocalOutlierFactor:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.x_list = []
        self.x_batch = []
        self.x_scores = []
        self.dist_dict = {}
        self.neighborhoods = {}
        self.rev_neighborhoods = {}
        self.k_dist = {}
        self.reach_dist = {}
        self.lof = {}
        self.local_reach = {}

    def learn_many(self, x):
        x = x[0].tolist()
        self.learn(x)

    def learn_one(self, x):
        self.x_batch.append(x)
        if len(self.x_list) or len(self.x_batch) > 1:
            self.learn(self.x_batch)
            self.x_batch = []

    def learn(self, x_batch):
        x_batch, equal = check_equal(x_batch, self.x_list)

        (nm, self.x_list, self.neighborhoods, self.rev_neighborhoods, self.k_dist, self.reach_dist, self.dist_dict, self.local_reach, self.lof) = expand_objects(x_batch, self.x_list, self.neighborhoods, self.rev_neighborhoods, self.k_dist, self.reach_dist, self.dist_dict, self.local_reach, self.lof)

        (self.neighborhoods, self.rev_neighborhoods, self.k_dist, self.dist_dict) = self._initial_calculations(self.x_list, nm, self.neighborhoods, self.rev_neighborhoods, self.k_dist, self.dist_dict)

        (set_new_points, set_neighbors, set_rev_neighbors, set_upd_lrd, set_upd_lof) = define_sets(nm, self.neighborhoods, self.rev_neighborhoods)

        self.reach_dist = calc_reach_dist_new_points(set_new_points, self.neighborhoods, self.rev_neighborhoods, self.reach_dist, self.dist_dict, self.k_dist)
        self.reach_dist = calc_reach_dist_other_points(set_rev_neighbors, self.rev_neighborhoods, self.reach_dist, self.dist_dict, self.k_dist)

        self.local_reach = calc_local_reach_dist(set_upd_lrd, self.neighborhoods, self.reach_dist, self.local_reach)

        self.lof = calc_lof(set_upd_lof, self.neighborhoods, self.local_reach, self.lof)

    def score_one(self, x):
        self.x_scores.append(x)
        self.x_scores, equal = check_equal(self.x_scores, self.x_list)

        if len(self.x_scores) == 0 or len(self.x_list) == 0:
            return 0.0

        x_list_copy = self.x_list[:]

        (nm, x_list_copy, neighborhoods, rev_neighborhoods, k_dist, reach_dist, dist_dict, local_reach, lof) = expand_objects(
            self.x_scores, x_list_copy, self.neighborhoods.copy(), self.rev_neighborhoods.copy(), self.k_dist.copy(),
            copy.deepcopy(self.reach_dist), copy.deepcopy(self.dist_dict), self.local_reach.copy(), self.lof.copy()
        )

        neighborhoods, rev_neighborhoods, k_dist, dist_dict = self._initial_calculations(
            x_list_copy, nm, neighborhoods, rev_neighborhoods, k_dist, dist_dict
        )
        (set_new_points, set_neighbors, set_rev_neighbors, set_upd_lrd, set_upd_lof) = define_sets(nm, neighborhoods, rev_neighborhoods)
        reach_dist = calc_reach_dist_new_points(set_new_points, neighborhoods, rev_neighborhoods, reach_dist, dist_dict, k_dist)
        reach_dist = calc_reach_dist_other_points(set_rev_neighbors, rev_neighborhoods, reach_dist, dist_dict, k_dist)
        local_reach = calc_local_reach_dist(set_upd_lrd, neighborhoods, reach_dist, local_reach)
        lof = calc_lof(set_upd_lof, neighborhoods, local_reach, lof)
        self.x_scores = []

        return lof[nm[0]]

    def _initial_calculations(self, x_list, nm, neighborhoods, rev_neighborhoods, k_distances, dist_dict):
        n = nm[0]
        m = nm[1]
        k = self.n_neighbors

        new_distances = [
            [i, j, self.distance(x_list[i], x_list[j])]
            for i in range(n + m)
            for j in range(i)
            if i >= n
        ]

        for i in range(len(new_distances)):
            dist_dict[new_distances[i][0]][new_distances[i][1]] = new_distances[i][2]
            dist_dict[new_distances[i][1]][new_distances[i][0]] = new_distances[i][2]

        for i, inner_dict in enumerate(dist_dict.values()):
            k_distances[i] = sorted(list(inner_dict.values()))[min(k, len(list(inner_dict.values()))) - 1]

        dist_dict = {
            k: {k2: v2 for k2, v2 in v.items() if v2 <= k_distances[k]}
            for k, v in dist_dict.items()
        }

        for key, value in dist_dict.items():
            neighborhoods[key] = [index for index in value]

        for particle_id, neighbor_ids in neighborhoods.items():
            for neighbor_id in neighbor_ids:
                rev_neighborhoods[neighbor_id].append(particle_id)

        return neighborhoods, rev_neighborhoods, k_distances, dist_dict

    def sqrt(self, x):
        """Compute the square root of a number using the Newton-Raphson method."""
        if x == 0:
            return 0
        guess = x / 2.0
        for _ in range(20):  # Iterate to improve the guess
            guess = (guess + x / guess) / 2.0
        return guess

    def distance(self, point_a, point_b):
        #return math.sqrt(utils.math.minkowski_distance(point_a, point_b, 2)) #Kayani Comment with below code without using Utils
          # Ensure the points have the same dimensions
        assert len(point_a) == len(point_b), "Points must have the same dimensions"
        
        sum_of_squares = (point_a['x'] - point_b['x']) ** 2 
        # Calculate the sum of the squared differences
        #sum_of_squares = sum((a - b) ** 2 for a, b in zip(point_a, point_b))
        
        # Return the square root of the sum of the squares
        return math.sqrt(sum_of_squares)

X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
lof = LocalOutlierFactor()

for x in X[:3]:
    lof.learn_one({'x': x})  # Warming up

for x in X:
    features = {'x': x}
    print(f'Anomaly score for x={x:.3f}: {lof.score_one(features):.3f}')
    lof.learn_one(features)


