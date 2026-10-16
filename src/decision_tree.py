"""NumPy decision tree for classification and regression."""

from __future__ import annotations

import numpy as np


class Node:
    def __init__(self, value, proba, n_samples, impurity):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value
        self.proba = proba
        self.n_samples = n_samples
        self.impurity = impurity

    def is_leaf(self):
        return self.left is None


class DecisionTree:
    def __init__(
        self,
        task="classification",
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        random_state=42,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        allowed = {"classification": {"gini", "entropy"}, "regression": {"mse"}}
        if criterion not in allowed[task]:
            raise ValueError(f"criterion {criterion!r} is invalid for {task}")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if not np.isfinite(min_impurity_decrease) or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be finite and nonnegative")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state
        self.root = None

    def _impurity(self, y):
        """Return Gini, base-2 entropy, or population variance."""
        if len(y) == 0:
            raise ValueError("impurity is undefined for an empty node")
        if self.task == "regression":
            return float(np.var(y))
        counts = np.unique(y, return_counts=True)[1]
        p = counts / len(y)
        if self.criterion == "gini":
            return float(1 - np.sum(p * p))
        return float(-np.sum(p * np.log2(p)))

    def _best_split(self, X, y):
        """Find the largest impurity decrease among adjacent-value midpoints."""
        n, n_features = X.shape
        if n < 2 * self.min_samples_leaf:
            return None
        parent = self._impurity(y)
        best_gain = 1e-12
        best = None
        positions = np.arange(1, n)
        valid_size = (positions >= self.min_samples_leaf) & (n - positions >= self.min_samples_leaf)

        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="mergesort")
            values = X[order, feature]
            valid = valid_size & (values[:-1] < values[1:])
            if not np.any(valid):
                continue
            if self.task == "regression":
                sorted_y = y[order]
                sums = np.cumsum(sorted_y, dtype=float)[:-1]
                squares = np.cumsum(sorted_y * sorted_y, dtype=float)[:-1]
                left_var = np.maximum(squares / positions - (sums / positions) ** 2, 0)
                right_n = n - positions
                right_sum = sums[-1] + sorted_y[-1] - sums
                right_sq = squares[-1] + sorted_y[-1] ** 2 - squares
                right_var = np.maximum(right_sq / right_n - (right_sum / right_n) ** 2, 0)
                children = (positions * left_var + right_n * right_var) / n
            else:
                encoded = y[order]
                # Prefix counts score all thresholds after a single sort.
                left_counts = np.cumsum(
                    np.eye(len(self.classes_), dtype=np.int64)[encoded], axis=0
                )[:-1]
                right_counts = np.bincount(y, minlength=len(self.classes_)) - left_counts
                left_p = left_counts / positions[:, None]
                right_p = right_counts / (n - positions)[:, None]
                if self.criterion == "gini":
                    left_impurity = 1 - np.sum(left_p * left_p, axis=1)
                    right_impurity = 1 - np.sum(right_p * right_p, axis=1)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        left_impurity = -np.sum(np.where(left_p > 0, left_p * np.log2(left_p), 0), axis=1)
                        right_impurity = -np.sum(np.where(right_p > 0, right_p * np.log2(right_p), 0), axis=1)
                children = (positions * left_impurity + (n - positions) * right_impurity) / n

            gains = parent - children
            gains[~valid] = -np.inf
            index = int(np.argmax(gains))
            gain = float(gains[index])
            if gain > best_gain + 1e-12:
                threshold = float(values[index] + (values[index + 1] - values[index]) / 2)
                if threshold >= values[index + 1]:
                    threshold = float(values[index])
                best_gain = gain
                best = (feature, threshold, gain)
        return best

    def _grow(self, X, y, depth):
        n = len(y)
        impurity = self._impurity(y)
        if self.task == "classification":
            counts = np.bincount(y, minlength=len(self.classes_))
            proba = counts / n
            value = self.classes_[np.argmax(counts)]
        else:
            proba = None
            value = float(np.mean(y))
        node = Node(value, proba, n, impurity)
        self.max_depth_ = max(self.max_depth_, depth)

        if ((self.max_depth is not None and depth >= self.max_depth)
                or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf
                or impurity <= 1e-12):
            self.n_leaves_ += 1
            return node

        split = self._best_split(X, y)
        if split is None:
            self.n_leaves_ += 1
            return node
        feature, threshold, gain = split
        # Weight the node gain by its share of training rows, as sklearn does.
        weighted_gain = n / self.n_samples_ * gain
        if weighted_gain + 1e-12 < self.min_impurity_decrease:
            self.n_leaves_ += 1
            return node
        left = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        self.feature_importances_[feature] += weighted_gain
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match X")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinity; impute missing values before fit")
        if self.task == "regression":
            y = np.asarray(y, dtype=float)
            if not np.all(np.isfinite(y)):
                raise ValueError("y must be finite")
        else:
            self.classes_, y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = len(X)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_leaves_ = 0
        self.max_depth_ = 0
        self.root = self._grow(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _check_predict_input(self, X):
        if self.root is None:
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.all(np.isfinite(X)):
            raise ValueError("X has invalid shape or contains nonfinite values")
        return X

    def _predict_one(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        X = self._check_predict_input(X)
        return np.asarray([self._predict_one(x, self.root).value for x in X])

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        X = self._check_predict_input(X)
        return np.asarray([self._predict_one(x, self.root).proba for x in X])
