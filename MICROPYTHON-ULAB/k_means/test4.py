import micropython_random
#import random
import functools
import collections

seed=42
mu=0
sigma=1

_rng = micropython_random.Random(seed)
rand_gauss = functools.partial(_rng.gauss, mu, sigma)

print(type(rand_gauss),'    ',rand_gauss())

bar = collections.OrderedDict([("z", 1), ("a", 2)])

print(type(bar),'    ',bar)
