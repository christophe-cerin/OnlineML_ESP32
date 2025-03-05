# import matplotlib.pyplot as plt
import numpy as np
import math
import random

#
# Random Forests From Scratch
# (https://carbonati.github.io/posts/random-forests-from-scratch/)
#
# Note: impractical on embedded devices because the run time is
# too large. if W=128: ok ; but if W=4095: not ok!
#
# file name below: rf.py
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

    a, b = make_array(my_points)
    ta = np.array([a]).T

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split_scratch(ta, b, test_size=1 / 3)

    # Compute and print predictions
    rf = RandomforestRegressor()
    model = rf.random_forest(X_train, Y_train, 300, 1, 50, 2)

    preds = rf.predict_rf(model, X_test)
    acc = sum(preds == Y_test) / len(Y_test)
    print("Testing accuracy: {}".format(np.round(acc, 3)))

    # print("X_test",X_test)
    preds = rf.predict_rf(model, [[50.0]])
    print("Predict by our RandomForest implementation:", preds)

    #
    # scikit-learn
    #
    from sklearn.ensemble import RandomForestRegressor

    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, Y_train)
    print(
        "predict by Scikit-learn RandomForest implementation:", regr.predict([[50.0]])
    )
