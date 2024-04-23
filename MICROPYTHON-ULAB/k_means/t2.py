import math

def l1(a, b):
    distance = 0.
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance

def l2(a, b):
    return math.sqrt(l2s(a, b))

def l2s(a, b):
    distance = 0.
    for i in range(len(a)):
        distance += (a[i] - b[i])**2
    return distance

def lp(p):
    def distance(a, b):
        dist = 0.
        for i in range(len(a)):
            dist += abs(a[i] - b[i])**p
        return dist**(1/p)
    return distance

def lpw(w, p):
    def distance(a, b):
        dist = 0.
        for i in range(len(a)):
            dist += w[i] * abs(a[i] - b[i])**p
        return dist**(1/p)
    return distance

def chebyshev_distance(a, b):
    distance = 0.
    for i in range(len(a)):
        if abs(a[i] - b[i]) >= distance:
            distance = abs(a[i] - b[i])
    return distance

def hamming_distance(a, b):
    distance = 0.
    for i in range(len(a)):
        if a[i] != b[i]:
            distance += 1
    return distance

def bray_curtis_distance(a, b):
    n, d = 0., 0.
    for i in range(len(a)):
        n += abs(a[i] - b[i])
        d += abs(a[i] + b[i])
    return n / d

def canberra_distance(a, b):
    distance = 0.
    for i in range(len(a)):
        distance += abs(a[i] - b[i]) / (abs(a[i]) + abs(b[i]))
    return distance

