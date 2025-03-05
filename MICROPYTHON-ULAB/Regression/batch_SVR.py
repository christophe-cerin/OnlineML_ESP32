# import matplotlib.pyplot as plt
import numpy as np
import math
import random

#
# SVR from scratch. Inspired by:
# https://github.com/Sohaib1424/Support-Vector-Machine-from-scratch/blob/main/SVM.ipynb
#
# file name below: batch_SVR.py
#


class Point:
    def __init__(self, x=0.0, y=0.0, z=0):
        self._x = x
        self._y = y
        self._z = z


def make_array(my_points):
    a = []
    b = []
    for p in my_points:
        a.append(p._x)
        b.append(p._y)
    return np.array(a), np.array(b)


import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

# from sklearn import svm
from sklearn.datasets import make_circles
from scipy import optimize


def plotLine(ax, xRange, w, x0, label, color="grey", linestyle="-", alpha=1.0):
    """Plot a (separating) line given the normal vector (weights) and point of intercept"""
    if type(x0) == int or type(x0) == float or type(x0) == np.float64:
        x0 = [0, -x0 / w[1]]
    yy = -(w[0] / w[1]) * (xRange - x0[0]) + x0[1]
    ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)


def plotSvm(
    X,
    y,
    supportVectors=None,
    w=None,
    intercept=0.0,
    label="Data",
    separatorLabel="Separator",
    ax=None,
    bound=[[-1.0, 1.0], [-1.0, 1.0]],
):
    """Plot the SVM separation, and margin"""
    if ax is None:
        fig, ax = plt.subplots(1)

    im = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.5, label=label)

    if supportVectors is not None:
        ax.scatter(
            supportVectors[:, 0],
            supportVectors[:, 1],
            label="SupportVector",
            s=80,
            facecolors="none",
            edgecolors="gold",
            color="gold",
        )
        print(f"Number of support vectors = {len(supportVectors)}")

    if w is not None:
        xx = np.array(bound[0])
        plotLine(ax, xx, w, intercept, separatorLabel)  # Plotting the optimal margin
        # Plotting the positive and negative margins
        if supportVectors is not None:
            signedDist = np.matmul(supportVectors, w)
            margin = 2.0 / np.linalg.norm(w)  # M = 2. / ||w||

            supportMaxNeg = supportVectors[np.argmin(signedDist)]
            supportMaxPos = supportVectors[np.argmax(signedDist)]

            plotLine(ax, xx, w, supportMaxNeg, "Margin -", linestyle="-.", alpha=0.8)
            plotLine(ax, xx, w, supportMaxPos, "Margin +", linestyle="--", alpha=0.8)
            ax.set_title("Margin = %.3f" % (margin))

    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    loc = np.arange(-1, 1, 1)


def generateBatchBipolar(n, mu=0.5, sigma=0.2):
    """Two gaussian clouds on each side of the origin"""
    X = np.random.normal(mu, sigma, (n, nFeatures))
    # yB in {0, 1}
    yB = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y = 2.0 * yB - 1
    X *= y[:, np.newaxis]
    X -= X.mean(axis=0)
    return X, y


def TransformData(XX, n):
    """Reconstruct data"""
    XXX = []
    for i in XX:
        XXX.append([i._x, i._y])
    X = np.array(XXX)
    # yB in {0, 1}
    yB = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y = 2.0 * yB - 1
    X *= y[:, np.newaxis]
    X -= X.mean(axis=0)
    return X, y


# The classifier is built on the scipy.optimize.minimum solver. For
# more info visit:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize


class MaxMarginClassifier:

    def __init__(self):
        self.alpha = None
        self.w = None
        self.supportVectors = None

    def fit(self, X, y):
        N = len(y)
        yX = y[:, np.newaxis] * X  # yX

        # Gram matrix of (yX)
        GramyX = np.matmul(yX, yX.T)

        # Lagrange dual problem
        def Ld(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))  # equation 8

        # Partial derivate of Ld on alpha
        def LdAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints on alpha:
        # 1.sum (alpha_i*y_i) = 0 for i in {1,...,n}
        # 2.aplha_i >= 0 for i in {1,...,n}
        I = np.eye(N)
        constraints = (
            {"type": "eq", "fun": lambda a: np.dot(a, y), "jac": lambda a: y},
            {"type": "ineq", "fun": lambda a: np.dot(I, a), "jac": lambda a: I},
        )

        # Maximizing by 'Minimizing' the opposite
        optRes = optimize.minimize(
            fun=lambda a: -Ld(GramyX, a),
            x0=np.zeros(N),
            method="SLSQP",
            jac=lambda a: -LdAlpha(GramyX, a),
            constraints=constraints,
        )
        self.alpha = optRes.x
        self.w = np.sum((self.alpha[:, np.newaxis] * yX), axis=0)  # equation 6
        epsilon = 1e-6
        self.supportVectors = X[self.alpha > epsilon]

        # Support vectors are at a distance <= 1 to the separation hyperplane
        negSupportVec = max(
            np.dot(self.supportVectors[np.dot(self.supportVectors, self.w) < 0], self.w)
        )
        posSupportVec = min(
            np.dot(
                self.supportVectors[np.dot(self.supportVectors, self.w) >= 0], self.w
            )
        )
        self.intercept = -(negSupportVec + posSupportVec) / 2  # equation 9

    def predict(self, X):
        """Predict y value in {-1, 1}"""
        assert self.w is not None
        assert self.w.shape[0] == X.shape[1]
        return 2 * (np.matmul(X, self.w) > 0) - 1


class SoftMarginClassifier:

    def __init__(self, C):
        self.C = C
        self.alpha = None
        self.w = None
        self.supportVectors = None

    def fit(self, X, y):
        N = len(y)
        # Gram matrix of (y.X)
        yX = X * y[:, np.newaxis]
        GramyX = np.matmul(yX, yX.T)

        # Lagrange dual problem
        def Ld(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))  # equation 8

        # Partial derivate of Ld on alpha
        def LdAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints on alpha:

        ##### this code is equivalent to the code below.
        #         C = self.C * np.ones(N)
        #         I = np.eye(N)
        #         constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
        #                        {'type': 'ineq', 'fun': lambda a: np.dot(I, a), 'jac': lambda a: I},
        #                        {'type': 'ineq', 'fun': lambda a: C - np.dot(I, a), 'jac': lambda a: -I})
        #####

        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        constraints = (
            {"type": "eq", "fun": lambda a: np.dot(a, y), "jac": lambda a: y},
            {"type": "ineq", "fun": lambda a: b - np.dot(A, a), "jac": lambda a: -A},
        )

        # Maximize by minimizing the opposite
        optRes = optimize.minimize(
            fun=lambda a: -Ld(GramyX, a),
            x0=np.ones(N),
            method="SLSQP",
            jac=lambda a: -LdAlpha(GramyX, a),
            constraints=constraints,
        )
        self.alpha = optRes.x
        self.w = np.sum((self.alpha[:, np.newaxis] * yX), axis=0)  # equation 6
        epsilon = 1e-6
        self.supportVectors = X[self.alpha > epsilon]

        # Support vectors are at a distance <= 1 to the separation hyperplane
        negSupportVec = max(
            np.dot(self.supportVectors[np.dot(self.supportVectors, self.w) < 0], self.w)
        )
        posSupportVec = min(
            np.dot(
                self.supportVectors[np.dot(self.supportVectors, self.w) >= 0], self.w
            )
        )
        self.intercept = -(negSupportVec + posSupportVec) / 2  # equation 9

    def predict(self, X):
        """Predict y value in {-1, 1}"""
        assert self.w is not None
        assert self.w.shape[0] == X.shape[1]
        return 2 * (np.matmul(X, self.w) > 0) - 1


class KernelSvmClassifier:

    def __init__(self, C, kernel, epsilon=1e-3):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.supportVectors = None
        self.epsilon = epsilon

    def fit(self, X, y):
        N = len(y)
        # Gram matrix of y.h(x)
        hX = np.apply_along_axis(
            lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X
        )
        yp = y.reshape(-1, 1)
        GramyHX = np.matmul(yp, yp.T) * hX

        # Lagrange dual problem
        def Ld(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))  # equation 14

        # Partial derivate of Ld on alpha
        def LdAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        ########
        # although something like L2 regularization was tested on this problem to
        # reduce the number of support vectors needed, but it wasn't useful.
        # the implementation that was done is shown below:

        # def Ld(G, alpha, r=10.):
        # return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G)) - r/2. * sum(alpha**2)

        # def LdAlpha(G, alpha, r=10.):
        # return np.ones_like(alpha) - alpha.dot(G) - r*alpha
        ########

        # Constraints on alpha:
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))
        constraints = (
            {"type": "eq", "fun": lambda a: np.dot(a, y), "jac": lambda a: y},
            {"type": "ineq", "fun": lambda a: b - np.dot(A, a), "jac": lambda a: -A},
        )

        # Maximize by minimizing the opposite
        optRes = optimize.minimize(
            fun=lambda a: -Ld(GramyHX, a),
            x0=np.ones(N),
            method="SLSQP",
            jac=lambda a: -LdAlpha(GramyHX, a),
            constraints=constraints,
        )
        self.alpha = optRes.x
        epsilon = self.epsilon
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]

    def predict(self, X):
        """Predict y values in {-1, 1}"""

        def predict1(x):
            x1 = np.apply_along_axis(
                lambda s: self.kernel(s, x), 1, self.supportVectors
            )
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)

        d = np.apply_along_axis(predict1, 1, X)
        return 2 * (d > 0) - 1


def GRBF(x1, x2, sigma=0.1):
    diff = x1 - x2
    return np.exp(-np.linalg.norm(diff) ** 2 / (2.0 * np.linalg.norm(sigma) ** 2))


#
# train_test_split from https://www.kaggle.com/code/marwanahmed1911/train-test-split-function-from-scratch
#
def shuffle_data(X, y):

    Data_num = np.arange(X.shape[0])
    np.random.shuffle(Data_num)

    return X[Data_num], y[Data_num]


def train_test_split_scratch(X, y, test_size=0.5, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)
    if test_size < 1:
        train_ratio = len(y) - int(len(y) * test_size)
        X_train, X_test = X[:train_ratio], X[train_ratio:]
        y_train, y_test = y[:train_ratio], y[train_ratio:]
        return X_train, X_test, y_train, y_test
    elif test_size in range(1, len(y)):
        X_train, X_test = X[test_size:], X[:test_size]
        y_train, y_test = y[test_size:], y[:test_size]
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    W = 256
    N = 10
    sqrt_W = int(math.sqrt(W))
    colors = ["blue", "red"]
    cmap = pltcolors.ListedColormap(colors)
    nFeatures = 2
    N = 100

    #
    # Initialization
    #
    points = [Point(0, 0) for _ in range(W * N)]
    my_points = [Point(0, 0) for _ in range(W)]

    #
    # Buffer points receives random data points and
    # Buffer my_points receives the first block of data
    #
    for i in range(W * N):
        p = Point(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
        points[i] = p
        if i < W:
            my_points[i] = p
    # print_arr(my_points,W)

    # Splitting dataset
    my_X_train, my_Y_train = TransformData(my_points, W)

    #
    # random Points have been generated and captured in
    # variable my_X_train, my_Y_train
    #

    model_random = MaxMarginClassifier()
    model_random.fit(my_X_train, my_Y_train)
    # print(model_random.w, model_random.intercept)

    # Fig 1
    fig, ax = plt.subplots(1, figsize=(12, 7))
    plotSvm(
        my_X_train,
        my_Y_train,
        model_random.supportVectors,
        model_random.w,
        model_random.intercept,
        ax=ax,
    )

    X, y = generateBatchBipolar(N, mu=0.5, sigma=0.22)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)

    model = MaxMarginClassifier()
    model.fit(X, y)
    model.w, model.intercept

    # Fig 2
    fig, ax = plt.subplots(1, figsize=(12, 7))
    plotSvm(X, y, model.supportVectors, model.w, model.intercept, ax=ax)

    # Fig 3
    X, y = generateBatchBipolar(N, mu=0.3, sigma=0.3)
    plotSvm(X, y, label="Training")

    model = SoftMarginClassifier(C=1)
    model.fit(X, y)
    model.w, model.intercept

    # Fig 4
    fig, ax = plt.subplots(1, figsize=(11, 7))
    plotSvm(
        X, y, model.supportVectors, model.w, model.intercept, label="Training", ax=ax
    )

    X, y = make_circles(N, factor=0.2, noise=0.1)
    y = 2.0 * y - 1

    # Fig 5
    plotSvm(X, y, label="Training")

    modelK = KernelSvmClassifier(C=1.0, kernel=GRBF)
    modelK.fit(X, y)

    # Fig 6
    fig, ax = plt.subplots(1, figsize=(11, 7))
    plotSvm(X, y, modelK.supportVectors, label="Training", ax=ax)

    # Estimate and plot decision boundary
    xx = np.linspace(-1, 1, 50)
    X0, X1 = np.meshgrid(xx, xx)
    xy = np.vstack([X0.ravel(), X1.ravel()]).T
    Y30 = modelK.predict(xy).reshape(X0.shape)
    ax.contour(
        X0, X1, Y30, colors="k", levels=[-1, 0], alpha=0.3, linestyles=["-.", "-"]
    )

    #
    # The most important to plot the results
    #
    plt.show()
