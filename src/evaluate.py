import numpy as np

# ==========================================
# Task A: Reqressiya Metrikləri (Price Prediction)
# ==========================================

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def evaluate_regression(y_true, y_pred):
    """
    Reqressiya üçün bütün ana metrikləri hesablayır və lüğət (dict) şəklində qaytarır.
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# ==========================================
# Task B: Təsnifat Metrikləri (Tier Classification)
# ==========================================

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn

def precision_score(y_true, y_pred):
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    tp, _, fn, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def evaluate_classification(y_true, y_pred):
    """
    Təsnifat üçün bütün ana metrikləri hesablayır və lüğət (dict) şəklində qaytarır.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

# Test üçün kiçik nümunə
if __name__ == "__main__":
    y_test_reg = np.array([100, 200, 300])
    y_pred_reg = np.array([110, 190, 310])
    print("Reqressiya Metrikləri Testi:")
    print(evaluate_regression(y_test_reg, y_pred_reg))

    y_test_cls = np.array([1, 0, 1, 1, 0])
    y_pred_cls = np.array([1, 0, 1, 0, 0])
    print("\nTəsnifat Metrikləri Testi:")
    print(evaluate_classification(y_test_cls, y_pred_cls))