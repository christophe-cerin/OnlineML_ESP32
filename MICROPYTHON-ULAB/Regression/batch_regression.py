import random
import math


class Point:
    def __init__(self, x=0.0, y=0.0, z=0):
        self._x = x
        self._y = y
        self._z = z


def less_points(lhs, rhs):
    return (lhs._x < rhs._x) or ((lhs._x == rhs._x) and (lhs._y < rhs._y))


def less_equal_points(lhs, rhs):
    return (lhs._x <= rhs._x) or ((lhs._x == rhs._x) and (lhs._y <= rhs._y))


def greater_points(lhs, rhs):
    return (lhs._x > rhs._x) or ((lhs._x == rhs._x) and (lhs._y > rhs._y))


def greater_equal_points(lhs, rhs):
    return (lhs._x >= rhs._x) or ((lhs._x == rhs._x) and (lhs._y >= rhs._y))


def equal_points(lhs, rhs):
    return (lhs._x == rhs._x) and (lhs._y == rhs._y)


def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and less_points(key, arr[j]):
            # print(f"{i} -> ({arr[j+1]._x}, {arr[j+1]._y}) ; ({arr[j]._x}, {arr[j]._y})")
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def qsort(xs, fst, lst):
    """
    Sort the range xs[fst, lst] in-place with vanilla QuickSort

    :param xs:  the list of numbers to sort
    :param fst: the first index from xs to begin sorting from,
                must be in the range [0, len(xs))
    :param lst: the last index from xs to stop sorting at
                must be in the range [fst, len(xs))
    :return:    nothing, the side effect is that xs[fst, lst] is sorted
    """
    if fst >= lst:
        return

    i, j = fst, lst
    pivot = xs[random.randint(fst, lst)]

    while i <= j:
        while less_points(xs[i], pivot):
            i += 1
        while greater_points(xs[j], pivot):
            j -= 1

        if i <= j:
            xs[i], xs[j] = xs[j], xs[i]
            i, j = i + 1, j - 1
    qsort(xs, fst, j)
    qsort(xs, i, lst)


def part(a, p, r):
    k = a[r]  # pivot
    j, q = p, p
    if p < r:  # if the length of the subarray is greater than 0
        for i in range(p, r + 1):
            if less_equal_points(a[i], k):
                t = a[q]
                a[q] = a[j]
                a[j] = t
                if i != r:
                    q += 1
                j += 1
            else:
                j += 1
        part(a, p, q - 1)  # sort the subarray to the left of the pivot
        part(a, q + 1, r)  # sort the subarray to the right of the pivot
    return a


def quicksort(a):
    if len(a) > 1:
        return part(a, 0, len(a) - 1)
    else:
        return a


def partition3(a, l, r):
    # write your code here
    pivot = a[r]
    i = l
    j = l - 1
    iterable_length = r

    while i <= iterable_length:
        if less_points(a[i], pivot):
            j += 1
            a[i], a[j] = a[j], a[i]

        elif greater_points(a[i], pivot):
            a[i], a[iterable_length] = a[iterable_length], a[i]
            iterable_length -= 1
            i -= 1

        i += 1

    return j, iterable_length + 1


def randomized_quick_sort(a, l, r):
    if l >= r:
        return
    k = random.randint(l, r)
    a[l], a[k] = a[k], a[l]
    # use partition3
    m1, m2 = partition3(a, l, r)
    randomized_quick_sort(a, l, m1)
    randomized_quick_sort(a, m2, r)


"""
This function partitions a[] in three parts
a) a[first..start] contains all elements smaller than pivot
b) a[start+1..mid-1] contains all occurrences of pivot
c) a[mid..last] contains all elements greater than pivot
	
"""


def partition_3way(arr, first, last, start, mid):

    pivot = arr[last]
    end = last

    # Iterate while mid is not greater than end.
    while mid[0] <= end:
        # Inter Change position of element at the starting if it's value is less than pivot.
        if less_points(arr[mid[0]], pivot):
            arr[mid[0]], arr[start[0]] = arr[start[0]], arr[mid[0]]
            mid[0] = mid[0] + 1
            start[0] = start[0] + 1
        # Inter Change position of element at the end if it's value is greater than pivot.
        elif greater_points(arr[mid[0]], pivot):
            arr[mid[0]], arr[end] = arr[end], arr[mid[0]]
            end = end - 1
        else:
            mid[0] = mid[0] + 1


# Function to sort the array elements in 3 cases
def quicksort_3way(arr, first, last):
    # First case when an array contain only 1 element
    if first >= last:
        return

    # Second case when an array contain only 2 elements
    if last == first + 1:
        if greater_points(arr[first], arr[last]):
            arr[first], arr[last] = arr[last], arr[first]
            return

    # Third case when an array contain more than 2 elements
    start = [first]
    mid = [first]

    # Function to partition the array.
    partition_3way(arr, first, last, start, mid)

    # Recursively sort sublist containing elements that are less than the pivot.
    quicksort_3way(arr, first, start[0] - 1)

    # Recursively sort sublist containing elements that are more than the pivot
    quicksort_3way(arr, mid[0], last)


#
# Prediction algorithms
#

import numpy as np


def linear_regression(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


#
# Ridge regression from https://gist.github.com/diogojc/1519756
#
class RidgeRegressor(object):
    """
    Linear Least Squares Regression with Tikhonov regularization.
    More simply called Ridge Regression.

    We wish to fit our model so both the least squares residuals and L2 norm
    of the parameters are minimized.
    argmin Theta ||X*Theta - y||^2 + alpha * ||Theta||^2

    A closed form solution is available.
    Theta = (X'X + G'G)^-1 X'y

    Where X contains the independent variables, y the dependent variable and G
    is matrix alpha * I, where alpha is called the regularization parameter.
    When alpha=0 the regression is equivalent to ordinary least squares.

    http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)
    http://en.wikipedia.org/wiki/Tikhonov_regularization
    http://en.wikipedia.org/wiki/Ordinary_least_squares
    """

    def fit(self, X, y, alpha=0):
        """
        Fits our model to our training data.

        Arguments
        ----------
        X: mxn matrix of m examples with n independent variables
        y: dependent variable vector for m examples
        alpha: regularization parameter. A value of 0 will model using the
        ordinary least squares regression.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = alpha * np.eye(X.shape[1])
        G[0, 0] = 0  # Don't regularize bias
        self.params = np.dot(
            np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)), np.dot(X.T, y)
        )

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.
        The assumption here is that the new data is iid to the training data.

        Arguments
        ----------
        X: mxn matrix of m examples with n independent variables
        alpha: regularization parameter. Default of 0.

        Returns
        ----------
        Dependent variable vector for m examples
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.params)


#
# Lasso Regression from https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
#
class LassoRegression:
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # calculate gradients
        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (
                    -2 * (self.X[:, j]).dot(self.Y - Y_pred) + self.l1_penalty
                ) / self.m
            else:
                dW[j] = (
                    -2 * (self.X[:, j]).dot(self.Y - Y_pred) - self.l1_penalty
                ) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function h(x)
    def predict(self, X):
        return X.dot(self.W) + self.b


#
# Random Forests From Scratch
# (https://carbonati.github.io/posts/random-forests-from-scratch/)
#
# Note: impractical on embedded devices because the run time is
# too large. if W=128: ok ; but if W=4095: not ok!
#
# file name below: rf.py
#
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import random

# class Point:
#    def __init__(self, x=0.0, y=0.0, z=0):
#        self._x = x
#        self._y = y
#        self._z = z

# def make_array(my_points):
#    a=[]
#    b=[]
#    for p in my_points:
#        a.append(p._x)
#        b.append(p._y)
#    return np.array(a), np.array(b)


class RandomforestRegressor:

    def __init__(self):
        pass

    # def __init__(self, X, Y,n_estimators=100, max_features=3, max_depth=10, min_samples_split=2):
    #    self.X_train = X
    #    self.Y_train = Y
    #    self.n_estimators = n_estimators
    #    self.max_features = max_features
    #    self.max_depth = max_depth
    #    self.min_samples_split = min_samples_split

    def entropy(self, p):
        if p == 0:
            return 0
        elif p == 1:
            return 0
        else:
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def information_gain(self, left_child, right_child):
        parent = left_child + right_child
        p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
        IG_p = self.entropy(p_parent)
        IG_l = self.entropy(p_left)
        IG_r = self.entropy(p_right)
        return (
            IG_p
            - len(left_child) / len(parent) * IG_l
            - len(right_child) / len(parent) * IG_r
        )

    def draw_bootstrap(self, X_train, y_train):
        bootstrap_indices = list(
            np.random.choice(range(len(X_train)), len(X_train), replace=True)
        )
        oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
        # X_bootstrap = X_train.iloc[bootstrap_indices].values
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        # X_oob = X_train.iloc[oob_indices].values
        X_oob = X_train[oob_indices]
        y_oob = y_train[oob_indices]
        return X_bootstrap, y_bootstrap, X_oob, y_oob

    def oob_score(self, tree, X_test, y_test):
        mis_label = 0
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            if pred != y_test[i]:
                mis_label += 1
        return mis_label / len(X_test)

    def find_split_point(self, X_bootstrap, y_bootstrap, max_features):
        feature_ls = list()
        num_features = len(X_bootstrap[0])

        while len(feature_ls) <= max_features:
            feature_idx = random.sample(range(num_features), 1)
            if feature_idx not in feature_ls:
                feature_ls.extend(feature_idx)

        best_info_gain = -999
        node = None
        for feature_idx in feature_ls:
            for split_point in X_bootstrap[:, feature_idx]:
                left_child = {"X_bootstrap": [], "y_bootstrap": []}
                right_child = {"X_bootstrap": [], "y_bootstrap": []}

                # split children for continuous variables
                if type(split_point) in [int, float]:
                    for i, value in enumerate(X_bootstrap[:, feature_idx]):
                        if value <= split_point:
                            left_child["X_bootstrap"].append(X_bootstrap[i])
                            left_child["y_bootstrap"].append(y_bootstrap[i])
                        else:
                            right_child["X_bootstrap"].append(X_bootstrap[i])
                            right_child["y_bootstrap"].append(y_bootstrap[i])
                # split children for categoric variables
                else:
                    for i, value in enumerate(X_bootstrap[:, feature_idx]):
                        if value == split_point:
                            left_child["X_bootstrap"].append(X_bootstrap[i])
                            left_child["y_bootstrap"].append(y_bootstrap[i])
                        else:
                            right_child["X_bootstrap"].append(X_bootstrap[i])
                            right_child["y_bootstrap"].append(y_bootstrap[i])

                split_info_gain = self.information_gain(
                    left_child["y_bootstrap"], right_child["y_bootstrap"]
                )
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    left_child["X_bootstrap"] = np.array(left_child["X_bootstrap"])
                    right_child["X_bootstrap"] = np.array(right_child["X_bootstrap"])
                    node = {
                        "information_gain": split_info_gain,
                        "left_child": left_child,
                        "right_child": right_child,
                        "split_point": split_point,
                        "feature_idx": feature_idx,
                    }

        return node

    def terminal_node(self, node):
        y_bootstrap = node["y_bootstrap"]
        pred = max(y_bootstrap, key=y_bootstrap.count)
        return pred

    def split_node(self, node, max_features, min_samples_split, max_depth, depth):
        left_child = node["left_child"]
        right_child = node["right_child"]

        del node["left_child"]
        del node["right_child"]

        if len(left_child["y_bootstrap"]) == 0 or len(right_child["y_bootstrap"]) == 0:
            empty_child = {
                "y_bootstrap": left_child["y_bootstrap"] + right_child["y_bootstrap"]
            }
            node["left_split"] = self.terminal_node(empty_child)
            node["right_split"] = self.terminal_node(empty_child)
            return

        if depth >= max_depth:
            node["left_split"] = self.terminal_node(left_child)
            node["right_split"] = self.terminal_node(right_child)
            return node

        if len(left_child["X_bootstrap"]) <= min_samples_split:
            node["left_split"] = node["right_split"] = self.terminal_node(left_child)
        else:
            node["left_split"] = find_split_point(
                left_child["X_bootstrap"],
                left_child["y_bootstrap"],
                max_features,
            )
            self.split_node(
                node["left_split"],
                max_depth,
                min_samples_split,
                max_depth,
                depth + 1,
            )
        if len(right_child["X_bootstrap"]) <= min_samples_split:
            node["right_split"] = node["left_split"] = self.terminal_node(right_child)
        else:
            node["right_split"] = self.find_split_point(
                right_child["X_bootstrap"],
                right_child["y_bootstrap"],
                max_features,
            )
            self.split_node(
                node["right_split"],
                max_features,
                min_samples_split,
                max_depth,
                depth + 1,
            )

    def build_tree(
        self, X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features
    ):
        root_node = self.find_split_point(X_bootstrap, y_bootstrap, max_features)
        self.split_node(root_node, max_features, min_samples_split, max_depth, 1)
        return root_node

    def random_forest(
        self,
        X_train,
        y_train,
        n_estimators,
        max_features,
        max_depth,
        min_samples_split,
    ):
        tree_ls = list()
        oob_ls = list()
        for i in range(n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.draw_bootstrap(
                X_train, y_train
            )
            tree = self.build_tree(
                X_bootstrap,
                y_bootstrap,
                max_features,
                max_depth,
                min_samples_split,
            )
            tree_ls.append(tree)
            oob_error = self.oob_score(tree, X_oob, y_oob)
            oob_ls.append(oob_error)
        print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
        return tree_ls

    def predict_tree(self, tree, X_test):
        feature_idx = tree["feature_idx"]

        if X_test[feature_idx] <= tree["split_point"]:
            if type(tree["left_split"]) == dict:
                return self.predict_tree(tree["left_split"], X_test)
            else:
                value = tree["left_split"]
                return value
        else:
            if type(tree["right_split"]) == dict:
                return self.predict_tree(tree["right_split"], X_test)
            else:
                return tree["right_split"]

    def predict_rf(self, tree_ls, X_test):
        pred_ls = list()
        for i in range(len(X_test)):
            ensemble_preds = [self.predict_tree(tree, X_test[i]) for tree in tree_ls]
            final_pred = max(ensemble_preds, key=ensemble_preds.count)
            pred_ls.append(final_pred)
        return np.array(pred_ls)


#
# train_test_split from https://www.kaggle.com/code/marwanahmed1911/train-test-split-function-from-scratch
#
# def shuffle_data(X, y):
#    Data_num = np.arange(X.shape[0])
#    np.random.shuffle(Data_num)
#
#    return X[Data_num], y[Data_num]

# def train_test_split_scratch(X, y, test_size=0.5, shuffle=True):
#    if shuffle:
#        X, y = shuffle_data(X, y)
#    if test_size <1 :
#        train_ratio = len(y) - int(len(y) *test_size)
#        X_train, X_test = X[:train_ratio], X[train_ratio:]
#        y_train, y_test = y[:train_ratio], y[train_ratio:]
#        return X_train, X_test, y_train, y_test
#    elif test_size in range(1,len(y)):
#        X_train, X_test = X[test_size:], X[:test_size]
#        y_train, y_test = y[test_size:], y[:test_size]
#        return X_train, X_test, y_train, y_test

# if __name__ == "__main__":
#
#    W      = 128
#    N      = 10
#    sqrt_W = int(math.sqrt(W))
#
#    #
#    # Initialization
#    #
#    points = [Point(0, 0) for _ in range(W * N)]
#    my_points = [Point(0, 0) for _ in range(W)]
#
#    #
#    # Buffer points receives random data points and
#    # Buffer my_points receives the first block of data
#    #
#    for i in range(W * N):
#        p = Point(random.uniform(0.0, 100.0), random.uniform(0.0, 100.0))
#        points[i] = p
#        if i < W:
#            my_points[i] = p
#    #print_arr(my_points,W)
#
#    a, b = make_array(my_points)
#    ta = np.array([a]).T
#
#    # Splitting dataset into train and test set
#    X_train, X_test, Y_train, Y_test = train_test_split_scratch(ta, b,test_size=1/3)
#
#    # Compute and Plot regressors
#    rf=RandomforestRegressor()
#    #print(type(rf.random_forest))
#    model = rf.random_forest(X_train, Y_train, 300, 1, 50, 2)
#
#    preds = rf.predict_rf(model, X_test)
#    acc = sum(preds == Y_test) / len(Y_test)
#    print("Testing accuracy: {}".format(np.round(acc,3)))
#
#    #print("X_test",X_test)
#    preds = rf.predict_rf(model, [[50.0]])
#    print("Predict by our RandomForest implementation:",preds)
#
#    #
#    # scikit-learn
#    #
#    from sklearn.ensemble import RandomForestRegressor
#    regr = RandomForestRegressor(max_depth=2, random_state=0)
#    regr.fit(X_train, Y_train)
#    print("predict by Scikit-learn RandomForest implementation:",regr.predict([[50.0]]))

# End file rf.py


#
# Utilitty functions
#


def print_arr(a, n):
    for i in range(n):
        print(f"{i} -> {a[i]._x}, {a[i]._y}")
    print()


def is_sorted(a, n):
    for i in range(n - 1):
        # print(f"{i} -> {a[i]._x}, {a[i]._y}")
        if greater_points(a[i], a[i + 1]):
            return False
    return True


def make_array(my_points):
    a = []
    b = []
    for p in my_points:
        a.append(p._x)
        b.append(p._y)
    return np.array(a), np.array(b)


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


#
# Scalers
#


def normalize(values):
    return (values - values.min()) / (values.max() - values.min())


#
# cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Normalize the feature columns
#
# df[cols] = df[cols].apply(normalize)


def standardize(values):
    return (values - values.mean()) / values.std()


# cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Standardize the feature columns; Dataframe needs to be recreated for
# the following command to work properly.
#
# df[cols] = df[cols].apply(standardize)


#
# Performance metrics
# MSE: Measures the average squared difference between actual and
# predicted values. Lower is better.
# RMSE: A more interpretable version of MSE (it’s in the same units as
# the target variable).
# MAE: The average absolute difference between the predicted and
# actual values.
# R-squared: Indicates how well the model explains the variance in the
# data. A value close to 1 means the model is doing well.
#
# TODO
#

#
# Main is comming!
#

if __name__ == "__main__":

    W = 4096
    N = 10
    sqrt_W = int(math.sqrt(W))

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
        p = Point(random.uniform(0.0, 100.0), random.uniform(0.0, 100.0))
        points[i] = p
        if i < W:
            my_points[i] = p
    # print_arr(my_points,W)

    #
    # We prepare indexes for the regular sampling technique
    #
    mes_indices = [i for i in range(sqrt_W - 1, W - sqrt_W, sqrt_W)]
    print("Size:", len(mes_indices), "Indexes:", mes_indices)

    # Uncomment the following block for choosing one sorting method
    print("Sorting starts")
    # insertion_sort(my_points)
    # qsort(my_points,0,W - 1)
    # quicksort(my_points)
    # randomized_quick_sort(my_points,0,W - 1)
    quicksort_3way(my_points, 0, W - 1)
    print("End sorting")
    # print_arr('Sorted data:',my_points,W)
    # print('is_sorted: ',is_sorted(my_points,W))

    #
    # We lanch the dedicated calculation on the W data sored in my_points
    #

    # a, b = make_array(my_points)
    # c1, c2 = linear_regression(a,b)
    # print(c1, c2)

    # polyfit takes two, or three arguments. The last one is the
    # degree of the polynomial that will be fitted, the last but
    # one is an array or iterable with the y (dependent) values,
    # and the first one, an array or iterable with the x
    # (independent) values, can be dropped. If that is the case, x
    # will be generated in the function as range(len(y)).
    # polyfit is available with ulab. See:
    # https://micropython-ulab.readthedocs.io/en/latest/numpy-functions.html#polyfit
    # a, b = make_array(my_points)
    # print('fitted values:\t\t', np.polyfit(a, b, 2))

    #
    # Second, third, fourth... round
    #
    for i in range(N - 1):
        #
        # First we replace some data by a regular sampling technique
        #
        K = 0
        for j in mes_indices:
            # print(i+1,j,K,(i+1)*W + K)
            my_points[j] = points[(i + 1) * W + K]
            K = K + 1

        #
        # Now we start the decicated calculation: linear regression, polyfit,
        # ridge regression and so on
        #

        # if i == N-2:
        #    import matplotlib.pyplot as plt
        #
        #    a, b = make_array(my_points)
        #
        #    c1, c2 = linear_regression(a,b)
        #    #print(c1, c2)
        #
        #    x = np.linspace(0, 100, W)
        #    y = c2 * x + c1
        #    # Create the plot
        #    plt.plot(x, y,label=f'$y = {c2:.4f}x {c1:+.4f}$')
        #    plt.legend()
        #    plt.show()

        # polyfit takes two, or three arguments. The last one is the
        # degree of the polynomial that will be fitted, the last but
        # one is an array or iterable with the y (dependent) values,
        # and the first one, an array or iterable with the x
        # (independent) values, can be dropped. If that is the case, x
        # will be generated in the function as range(len(y)).
        # polyfit is available with ulab. See:
        # https://micropython-ulab.readthedocs.io/en/latest/numpy-functions.html#polyfit
        #
        # a, b = make_array(my_points)
        # print('fitted values:\t\t', np.polyfit(a, b, 2))

        #
        # Ridge regression
        #
        # if i == N-2:
        #    import matplotlib.pyplot as plt
        #
        #    a, b = make_array(my_points)
        #    X = np.linspace(0, 100, W)
        #    #X = a
        #
        #    # Create feature matrix
        #    tX = np.array([X]).T
        #    tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))
        #
        #    # Compute and Plot regressors
        #    r = RidgeRegressor()
        #    r.fit(tX, b)
        #    plt.plot(X, r.predict(tX), 'b', label=u'ŷ (alpha=0.0)')
        #    #alpha = 3.0
        #    #r.fit(tX, b, alpha)
        #    #plt.plot(X, r.predict(tX), 'y', label=u'ŷ (alpha=%.1f)' % alpha)
        #
        #    plt.legend()
        #    plt.show()

        #
        # Lasso regression
        #
        if i == N - 2:
            import matplotlib.pyplot as plt

            a, b = make_array(my_points)
            # X = np.linspace(0, 100, W)
            # X = a

            # Create feature matrix
            # tX = np.array([X]).T
            # tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))

            # Standardize data
            # a = normalize(a)
            # b = standardize(b)
            a = standardize(a)
            b = standardize(b)
            # print(a)
            ta = np.array([a]).T

            # Splitting dataset into train and test set
            X_train, X_test, Y_train, Y_test = train_test_split_scratch(
                ta, b, test_size=1 / 3
            )

            # Compute and Plot regressors
            r = LassoRegression(iterations=5, learning_rate=0.01, l1_penalty=1)
            # r.fit(tX, b)
            r.fit(X_train, Y_train)

            # Prediction on test set
            Y_pred = r.predict(X_test)

            print("Predicted values: ", np.round(Y_pred[:3], 2))
            print("Real values:      ", Y_test[:3])
            print("Trained W:        ", round(r.W[0], 2))
            print("Trained b:        ", round(r.b, 2))

            # plt.plot(X, r.predict(tX), 'b', label=u'ŷ (iteration=1000, learning_rate=0.01, l1_penalty=500)')
            plt.scatter(X_test, Y_test, color="blue", label="Actual Data")
            plt.plot(X_test, Y_pred, color="orange", label="Lasso Regression Line")
            plt.legend()
            plt.show()

        #
        # Then we sort
        #
        quicksort_3way(my_points, 0, W - 1)

    #
    # Final check
    #
    print("Is the final buffer sorted? -->", is_sorted(my_points, W))
