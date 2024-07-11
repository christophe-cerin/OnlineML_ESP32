from math import sqrt, exp, pi

class Gaussian:
    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.n_samples = 0

    def update(self, x):
        self.n_samples += 1
        if self.n_samples == 1:
            self.mean = x
        else:
            old_mean = self.mean
            self.mean += (x - self.mean) / self.n_samples
            self.variance = ((self.n_samples - 2) / (self.n_samples - 1)) * self.variance + (x - self.mean) * (x - old_mean)
    
    def cdf(self, x):
        if self.n_samples < 2:
            return 0.5
        std = sqrt(self.variance)
        return 0.5 * (1 + erf((x - self.mean) / (std * sqrt(2))))

    def pdf(self, x):
        std = sqrt(self.variance)
        return (1 / (std * sqrt(2 * pi))) * exp(-0.5 * ((x - self.mean) / std) ** 2)

class GaussianScorer:
    def __init__(self, window_size=None, grace_period=100):
        self.window_size = window_size
        self.grace_period = grace_period
        self.data = []
        self.gaussian = Gaussian()

    def learn_one(self, x, y):
        self.gaussian.update(y)
        if self.window_size:
            self.data.append(y)
            if len(self.data) > self.window_size:
                self.data.pop(0)
                self.recompute_stats()

    def recompute_stats(self):
        self.gaussian = Gaussian()
        for value in self.data:
            self.gaussian.update(value)

    def score_one(self, x, y):
        if self.gaussian.n_samples < self.grace_period:
            return 0
        return 2 * abs(self.gaussian.cdf(y) - 0.5)

# Helper function for computing the error function
def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)

    return sign * y

# Test the GaussianScorer
# List of pre-defined Gaussian-distributed values
values = [
    -0.14409032957792836, 0.11737966009083385, -0.1729036003315193, -0.2725784581147548, 0.8181263048121569,
    0.6866231836653894, -0.40566272402074026, -0.15072392361937865, 0.7194846770441458, -1.1636224691524142,
    -0.2308127678016577, -0.8618611049208959, -0.20066954893851103, 0.4762216630620553, 1.3420435285339358,
    -0.8277944850709707, -0.9867575272701108, -1.1073052571162372, -1.296735073050282, -1.4691865402527732,
    -1.1968530553461949, 0.4941446230368065, -0.03984261282716686, 0.2463878456942844, -0.04385700724222363,
    0.8492264177932402, 0.043334433128898626, -1.407313878299949, -0.2876062325412928, -0.978617011263443,
    -0.47294566932515356, 0.7329807692099164, -0.003274351180956523, -0.4709705371879879, 0.1304340171706219,
    0.04521257208220028, 1.1900363728251824, -1.2533447063306337, 1.631090585085477, 0.6650952647402653,
    -1.4569954805685552, -0.22713454725804233, 0.572926411448601, 0.22641877450242083, -0.1325507170586148,
    -1.3415733552942277, 0.8016946660537623, 0.02005485853092355, -0.293006784479667, -0.2691149611312983,
    -1.5641440589579425, 0.7319394278359875, 0.4672389483387075, -0.8998186653649652, 0.3820395752872284,
    -0.6336351716100085, -1.3276705149136696, 0.6262106431718283, -0.3270745814039179, -0.16364405112178964,
    0.4489453019512347, -0.5182718900480242, -0.24934301063308552, 0.10149684809622784, -0.6570219744908673,
    -0.22576613483180437, 1.3946954577738276, 0.36569065859026164, -0.2343844240704171, -1.0468353082474625,
    -0.6970919482138127, 0.37385684939213756, -0.16964890258783096, 1.179568101049046, 0.2367748676129413,
    0.7031097049656364, -1.3886739617752042, -0.026282078155389707, 0.10269908432707474, 0.6811356762022383,
    -0.43077375272666425, -0.6080659321731015, -0.6191676346391586, -0.08496787988362522, 1.0441911400669965,
    0.3326868587592107, -1.0812129848557056, -0.6434043996218998, 0.4088705394798912, -0.05074380088802967,
    0.13841249761807742, -1.4362291178105677, 0.04965326263577952, -0.7870253059470116, -0.030708683307139582,
    0.7323524386379657, -0.4240192301257688, -0.3286456301832923, 0.20041229076052623, 0.37435127309978535
]

detector = GaussianScorer()

for y in values:
    detector.learn_one(None, y)

print(detector.score_one(None, 3))
