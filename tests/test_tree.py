import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classification_matches_sklearn_on_unambiguous_data(criterion):
    X = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    ours = DecisionTree(criterion=criterion, max_depth=3).fit(X, y)
    reference = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42).fit(X, y)
    assert ours.root.feature == reference.tree_.feature[0]
    assert ours.root.threshold == pytest.approx(reference.tree_.threshold[0])
    assert ours.root.right.feature == reference.tree_.feature[reference.tree_.children_right[0]]
    assert ours.root.right.threshold == pytest.approx(
        reference.tree_.threshold[reference.tree_.children_right[0]]
    )
    np.testing.assert_array_equal(ours.predict(X), reference.predict(X))
    np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(X))
    np.testing.assert_allclose(ours.feature_importances_, reference.feature_importances_)


def test_regression_matches_sklearn_on_unambiguous_data():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([10., 10., 10., 10., 30., 30., 30., 30.])
    ours = DecisionTree(task="regression", criterion="mse", max_depth=2).fit(X, y)
    reference = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
    assert ours.root.feature == reference.tree_.feature[0]
    assert ours.root.threshold == pytest.approx(reference.tree_.threshold[0])
    np.testing.assert_allclose(ours.predict(X), reference.predict(X))
    np.testing.assert_allclose(ours.feature_importances_, reference.feature_importances_)


def test_impurities_and_leaf_probabilities():
    tree = DecisionTree(criterion="gini", max_depth=1)
    assert tree._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    entropy = DecisionTree(criterion="entropy", max_depth=1)
    assert entropy._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    mse = DecisionTree(task="regression", criterion="mse")
    assert mse._impurity(np.array([1., 3.])) == pytest.approx(1.0)
    X = np.array([[0.], [1.], [2.], [3.], [4.]])
    y = np.array(["low", "low", "high", "high", "high"])
    tree.fit(X, y)
    assert tree.classes_.tolist() == ["high", "low"]
    np.testing.assert_allclose(tree.predict_proba([[0.]]).sum(axis=1), [1.])
    assert tree.predict([[0.]])[0] == "low"
    with pytest.raises(ValueError):
        mse.fit(X, np.arange(5)).predict_proba(X)


def test_each_stopping_parameter_changes_structure():
    X = np.arange(16, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1] * 4)
    base = DecisionTree().fit(X, y)
    assert base.n_leaves_ > 2
    assert DecisionTree(max_depth=1).fit(X, y).n_leaves_ == 2
    assert DecisionTree(min_samples_split=17).fit(X, y).n_leaves_ == 1
    assert DecisionTree(min_samples_leaf=5).fit(X, y).n_leaves_ < base.n_leaves_
    assert DecisionTree(min_impurity_decrease=0.5).fit(X, y).n_leaves_ == 1


def test_fit_rejects_missing_values_and_predict_requires_fit():
    with pytest.raises(ValueError, match="fit"):
        DecisionTree().predict([[0.]])
    with pytest.raises(ValueError, match="impute"):
        DecisionTree().fit([[np.nan], [1.]], [0, 1])
