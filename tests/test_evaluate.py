from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src import evaluate as ev

def test_regression_metrics():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    
    assert np.isclose(ev.rmse(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)))
    assert np.isclose(ev.mae(y_true, y_pred), mean_absolute_error(y_true, y_pred))
    assert np.isclose(ev.r2(y_true, y_pred), r2_score(y_true, y_pred))
    print("Regression testləri uğurla keçdi!")

def test_classification_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    cm = ev.confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    
    p, r, f1 = ev.precision_recall_f1(y_true, y_pred)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
    print("Classification testləri uğurla keçdi!")