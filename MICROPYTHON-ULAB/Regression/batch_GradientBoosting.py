#
# Gradient Boosting Regressor in pure Python
#
# Explanations for method A:
#
# DecisionTreeRegressorA: This is a simple decision tree regressor that
# splits the data based on the feature that minimizes the mean squared
# error (MSE) of the target variable. It builds a tree recursively
# until it reaches the maximum depth or the minimum number of samples
# per split is reached.
#
# GradientBoostingRegressorA: This class implements the gradient
# boosting algorithm. It iteratively fits decision trees to the
# residuals (differences between the actual target values and the
# current predictions) of the previous models. Each tree's prediction
# is weighted by the learning rate and added to the final prediction.
#
# Key Parameters:
# n_estimators: The number of trees in the ensemble.
# learning_rate: A shrinkage parameter that scales the contribution
# of each tree. Smaller values require more trees but can lead to
# better performance.
# max_depth: The maximum depth of each tree.
# min_samples_split: The minimum number of samples required to split
# an internal node.
#
# This code provides a basic implementation and can be extended with
# additional features such as handling categorical variables,
# regularization, and more efficient tree building algorithms.
#
#
# Explanations for method B
#
# SimpleDecisionTreeB: This is a basic decision tree implementation
# that splits on the feature that maximizes variance reduction. It
# grows to a maximum depth of 1, making it a weak learner.

# GradientBoostingRegressorB: This class uses the SimpleDecisionTreeB
# as a weak learner and builds a sequence of such trees, each trying
# to correct the errors of the previous ones. The predictions of all
# trees are combined with a learning rate to produce the final
# prediction.
#
# This implementation is quite basic and does not include many
# optimizations or features found in scikit-learn's Gradient Boosting,
# such as handling categorical data, pruning, or more sophisticated
# stopping criteria. However, it should give you a good starting point
# for understanding how gradient boosting works.

import numpy as np

####################################
#
# Method A
#
####################################


class DecisionTreeRegressorA:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, float("inf")
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                left_y, right_y = y[left_mask], y[right_mask]
                if (
                    len(left_y) < self.min_samples_split
                    or len(right_y) < self.min_samples_split
                ):
                    continue

                score = self._mse(left_y, right_y)
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _mse(self, left_y, right_y):
        n_left, n_right = len(left_y), len(right_y)
        total_n = n_left + n_right
        left_mean, right_mean = np.mean(left_y), np.mean(right_y)
        return (n_left / total_n) * np.sum((left_y - left_mean) ** 2) + (
            n_right / total_n
        ) * np.sum((right_y - right_mean) ** 2)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return np.mean(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (feature, threshold, left_tree, right_tree)

    def _predict_tree(self, row, tree):
        if isinstance(tree, float):
            return tree

        feature, threshold, left_tree, right_tree = tree

        if row[feature] <= threshold:
            return self._predict_tree(row, left_tree)
        else:
            return self._predict_tree(row, right_tree)

    def predict(self, X):
        return np.array([self._predict_tree(row, self.tree) for row in X])


class GradientBoostingRegressorA:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        y_pred = np.zeros_like(y)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressorA(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            y_pred = y_pred + (self.learning_rate * tree.predict(X))
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred = y_pred + (self.learning_rate * tree.predict(X))
        return y_pred


####################################
#
# Method B
#
####################################


class SimpleDecisionTreeB:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < 2:
            leaf_value = np.mean(y)
            return leaf_value

        best_feature, best_threshold = self._best_criteria(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (best_feature, best_threshold, left_tree, right_tree)

    def _best_criteria(self, X, y):
        best_feature = None
        best_threshold = None
        best_variance_reduction = -1
        num_samples, num_features = X.shape

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                variance_left = np.var(y[left_indices])
                variance_right = np.var(y[right_indices])
                variance_reduction = np.var(y) - (
                    len(y[left_indices]) / num_samples * variance_left
                    + len(y[right_indices]) / num_samples * variance_right
                )

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left_tree, right_tree = node
            if inputs[feature] < threshold:
                node = left_tree
            else:
                node = right_tree
        return node


class GradientBoostingRegressorB:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        y_pred = np.zeros_like(y)
        for _ in range(self.n_estimators):
            tree = SimpleDecisionTreeB(max_depth=self.max_depth)
            tree.fit(X, y - y_pred)
            self.trees.append(tree)
            y_pred = y_pred + (self.learning_rate * tree.predict(X))

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# Examples usage:
if __name__ == "__main__":
    # Sample data for method B
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([2, 3, 4, 5, 6])

    # Create and train the model
    gb = GradientBoostingRegressorB(n_estimators=100, learning_rate=0.1, max_depth=1)
    gb.fit(X, y)

    # Predict
    X_test = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_pred = gb.predict(X_test)
    print("Predictions:", y_pred)

    # Sample data for method A
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Initialize and train the Gradient Boosting model
    gb = GradientBoostingRegressorA(
        n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2
    )
    gb.fit(X, y)

    # Predict
    predictions = gb.predict(X)
    print("Predictions:", predictions)

    #
    # scikit-learn dataset exploration
    #
    from sklearn import datasets
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=13
    )

    gb = GradientBoostingRegressorA(
        n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2
    )
    gb.fit(X_train, y_train)

    # Predict
    predictions = gb.predict(X_test)
    # print("Predictions:", predictions)

    mse = mean_squared_error(y_test, predictions)
    print(
        "GradientBoostingRegressorA - The mean squared error (MSE) on test set: {:.4f}".format(
            mse
        )
    )
    print(
        "Scikit-learn says that, with the same dataset and its Gradient Boosting regression:"
    )
    print("             The mean squared error (MSE) on test set: 3010.2061")
    print(
        "See: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html"
    )

    gb = GradientBoostingRegressorB(n_estimators=100, learning_rate=0.1, max_depth=1)
    gb.fit(X_train, y_train)

    # Predict
    predictions = gb.predict(X_test)
    # print("Predictions:", predictions)

    mse = mean_squared_error(y_test, predictions)
    print(
        "GradientBoostingRegressorB - The mean squared error (MSE) on test set: {:.4f}".format(
            mse
        )
    )
    print(
        "Scikit-learn says that, with the same dataset and its Gradient Boosting regression:"
    )
    print("             The mean squared error (MSE) on test set: 3010.2061")
    print(
        "See: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html"
    )
