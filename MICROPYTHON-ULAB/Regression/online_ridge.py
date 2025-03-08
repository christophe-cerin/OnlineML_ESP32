import random
import math
import sys


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

    W = 512  # 4096
    N = 10
    sqrt_W = int(math.sqrt(W))

    #
    # Initialization
    #
    from numpy import genfromtxt

    #
    # Humidity is at position 3, and temperature at position 5
    #
    data = genfromtxt(
        "TourPerret.csv", delimiter=";", comments="#", usecols=(3, 5), skip_header=1
    )

    points = [Point(0, 0) for _ in range(W * N)]
    my_points = [Point(0, 0) for _ in range(W)]

    #
    # Buffer points receives data points and
    # Buffer my_points receives the first block of data
    #
    for i in range(W * N):
        p = Point(data[i][0], data[i][1])
        points[i] = p
        if i < W:
            my_points[i] = p
    # print_arr(my_points,W)
    # sys.exit(0)

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
        # Ridge regression
        #
        if i == N - 2:
            import matplotlib.pyplot as plt

            a, b = make_array(my_points)

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
            r = RidgeRegressor()
            # r.fit(tX, b)
            r.fit(X_train, Y_train)

            # Prediction on test set
            Y_pred = r.predict(X_test)

            #
            # Scikit-learn Ridge
            from sklearn import linear_model

            clf = linear_model.Ridge(alpha=0.1)
            clf.fit(X_train, Y_train)
            Y_pred_scikit = clf.predict(X_test)

            print("Predicted values: ", np.round(Y_pred[:3], 2))
            print("Real values:      ", Y_test[:3])

            # plt.plot(X, r.predict(tX), 'b', label=u'ŷ (iteration=1000, learning_rate=0.01, l1_penalty=500)')
            plt.scatter(X_test, Y_test, color="blue", label="Actual Data")
            plt.plot(X_test, Y_pred_scikit, color="green", label="Scikit-learn Ridge")
            plt.plot(X_test, Y_pred, color="red", label="Ridge Regression Line")

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
