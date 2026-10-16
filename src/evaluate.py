from __future__ import annotations
import numpy as np
import pandas as pd

# -- Task A: Reqressiya metrikləri
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else 0.0

# -- Task B: Təsnifat metrikləri
def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def precision_recall_f1(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f)

def roc_auc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def compare(name_to_metrics, path=None):
    cols = []
    for d in name_to_metrics.values():
        cols += [k for k in d.keys() if k not in cols]
    
    head = f"{'model':<24}" + "".join(f"{c:>12}" for c in cols)
    print(head)
    print("-" * len(head))
    
    for name, d in name_to_metrics.items():
        cells = "".join(f"{d[c]:>12.4f}" if isinstance(d.get(c), (int, float))
                        else f"{'-':>12}" for c in cols)
        print(f"{name:<24}{cells}")
        
    if path:
        pd.DataFrame(name_to_metrics).T.to_csv(path)