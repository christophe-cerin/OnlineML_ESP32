from __future__ import annotations

import collections
import functools
#import random
import micropython_random

# Cerin: initial code next line
#from river import base
from clusterer import *

__all__ = ["KMeans"]

#-----------------------------------------------------------------------
# gauss.py
#-----------------------------------------------------------------------

#import stdio
#import sys
import math

#-----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean 0.0
# and standard deviation 1.0 at the given x value.

def phi(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)

#-----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean mu
# and standard deviation sigma at the given x value.

def pdf(x, mu=0.0, sigma=1.0):
    return phi((x - mu) / sigma) / sigma

#-----------------------------------------------------------------------

# Return the value of the cumulative Gaussian distribution function
# with mean 0.0 and standard deviation 1.0 at the given z value.

def Phi(z):
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z * z / float(i)
        i += 2
    return 0.5 + phi(z) * total

#-----------------------------------------------------------------------

# Return the value of the cumulative Gaussian distribution function
# with mean mu and standard deviation sigma at the given z value.

def cdf(z, mu=0.0, sigma=1.0):
    return Phi((z - mu) / sigma)

#-----------------------------------------------------------------------

# Accept floats z, mu, and sigma as command-line arguments. Use them
# to test the cdf() and pdf() functions. Write the
# results to standard output.

#z = float(sys.argv[1])
#mu = float(sys.argv[2])
#sigma = float(sys.argv[3])

#stdio.writeln(cdf(z, mu, sigma))

#-----------------------------------------------------------------------

# python gauss.py 820 1019 209
# 0.17050966869132106

# python gauss.py 1500 1019 209
# 0.9893164837383885

# python gauss.py 1500 1025 231
# 0.9801220907365491


# Cerin: initial code next line
#class KMeans(base.Clusterer):
class KMeans(Clusterer):
    """Incremental k-means.

    The most common way to implement batch k-means is to use Lloyd's algorithm, which consists in
    assigning all the data points to a set of cluster centers and then moving the centers
    accordingly. This requires multiple passes over the data and thus isn't applicable in a
    streaming setting.

    In this implementation we start by finding the cluster that is closest to the current
    observation. We then move the cluster's central position towards the new observation. The
    `halflife` parameter determines by how much to move the cluster toward the new observation.
    You will get better results if you scale your data appropriately.

    Parameters
    ----------
    n_clusters
        Maximum number of clusters to assign.
    halflife
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
    mu
        Mean of the normal distribution used to instantiate cluster positions.
    sigma
        Standard deviation of the normal distribution used to instantiate cluster positions.
    p
        Power parameter for the Minkowski metric. When `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.
    seed
        Random seed used for generating initial centroid positions.

    Attributes
    ----------
    centers : dict
        Central positions of each cluster.

    Examples
    --------
h
    In the following example the cluster assignments are exactly the same as when using
    `sklearn`'s batch implementation. However changing the `halflife` parameter will
    produce different outputs.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [-4, 2],
    ...     [-4, 4],
    ...     [-4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.1, sigma=3, seed=42)

    >>> for i, (x, _) in enumerate(stream.iter_array(X)):
    ...     k_means.learn_one(x)
    ...     print(f'{X[i]} is assigned to cluster {k_means.predict_one(x)}')
    [1, 2] is assigned to cluster 1
    [1, 4] is assigned to cluster 1
    [1, 0] is assigned to cluster 0
    [-4, 2] is assigned to cluster 1
    [-4, 4] is assigned to cluster 1
    [-4, 0] is assigned to cluster 0

    >>> k_means.predict_one({0: 0, 1: 0})
    0

    >>> k_means.predict_one({0: 4, 1: 4})
    1

    References
    ----------
    [^1]: [Sequential k-Means Clustering](http://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm)
    [^2]: [Sculley, D., 2010, April. Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web (pp. 1177-1178](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)

    """

    def __init__(self, n_clusters=5, halflife=0.5, mu=0, sigma=1, p=2, seed: int | None = None):
        self.n_clusters = n_clusters
        self.halflife = halflife
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.seed = seed
        # Cerin: initial code next line
        #self._rng = random.Random(seed)
        _rng = micropython_random.Random(seed)
        rand_gauss = functools.partial(_rng.gauss, mu, sigma)

        #self._rng = micropython_random.Random(seed)
        #print(type(self._rng),self._rng.seed)
        #self._rng = random.seed(seed)
        # Cerin: initial code next line
        #rand_gauss = functools.partial(self._rng, self.mu, self.sigma)
        #print(type(rand_gauss),rand_gauss())
        #print('===========')
        #jj = collections.defaultdict(rand_gauss)
        #rand_gauss = cdf(float(seed), float(self.mu), float(self.sigma))
        # cerin initial code next 3 lines
        #self.centers: dict[int, collections.defaultdict] = {
        #    i: collections.defaultdict(rand_gauss) for i in range(n_clusters)
        #}
        self.centers: dict[int, collections.OrderedDict] = {
            #i: collections.OrderedDict({i,rand_gauss()}) for i in range(self.n_clusters)
            i: collections.OrderedDict({}) for i in range(self.n_clusters)
        }
        #self.centers[0] = -0.4322709887337851
        #self.centers[1] = -0.5187108009945579
        for key in self.centers:
            for i in range(self.n_clusters):
                self.centers[key][i] = rand_gauss()
        print('============')
        #print(type(self.centers))
        #for key,val in self.centers.items():
        #    print('key:',key,' ; val[0]:',val[0],' ; val[1]:',val[1])
        #print('============')

    @property
    def _mutable_attributes(self):
        return {"halflife"}

    def learn_predict_one(self, x):
        """Equivalent to `k_means.learn_one(x).predict_one(x)`, but faster."""
        # Find the cluster with the closest center
        closest = self.predict_one(x)
        #            print('=========')
        #print(self.centers[closest][i],'==',self.halflife * (xi - self.centers[closest][i]))
        #print('=========') Move the cluster's center
        for i, xi in x.items():
            #print('closest: ',closest,'***',i, xi,self.centers[closest][i])
            #print('=========')
            #print('Centers:',self.centers)
            #print(self.centers[closest][i],'==',self.halflife * (xi - self.centers[closest][i]))
            #print('=========')
            self.centers[closest][i] += self.halflife * (xi - self.centers[closest][i])
            
        return closest

    def learn_one(self, x):
        self.learn_predict_one(x)

    def predict_one(self, x):
        def get_distance(c):
            center = self.centers[c]
            #print('=== ',c,' ',center,'  ',x,'    self.p: ',self.p)
            # Cerin: initial code next line
            #return sum((abs(center[k] - x.get(k, 0))) ** self.p for k in {*center.keys(), *x.keys()})
            #return sum((abs(self.centers[k] - x)) ** self.p for k in self.centers)
            my_list = []
            for k in center.keys():
                my_list.append(k) if k not in my_list else my_list
            for k in x.keys():
                my_list.append(k) if k not in my_list else my_list
            my_list = sorted(my_list)
            #print('center_k:',my_list,'x:',x)
            s = 0
            for k in my_list:
                if k in center.keys():
                    #print('k in:',k,center.keys())
                    s += abs(center[k] - x.get(k, 0)) ** self.p
                else:
                    #print('k not in',k,center.keys())
                    continue
            return s
            #return sum((abs(center[k] - x.get(k, 0))) ** self.p for k in my_list)

        return min(self.centers, key=get_distance)

    @classmethod
    def _unit_test_params(cls):
        yield {"n_clusters": 5}

#
# Main
#         
import iter_array
from ulab import numpy as np
#import array

X = np.array([[1, 2],[1, 4],[1, 0],[-4, 2],[-4, 4],[-4, 0]], dtype=np.int16)

#print('Input array:')
#print(X)

k_means = KMeans(n_clusters=2, halflife=0.1, sigma=3, seed=42)

my_enum = enumerate(iter_array.iter_array(X))
for i, (x,_) in my_enum:
    #print('Before learn_one:',x,' Type:',type(x))
    k_means.learn_one(x)
    print(f'{X[i]} is assigned to cluster {k_means.predict_one(x)}')
