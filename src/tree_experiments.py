"""Reproduce the Decision Tree tuning, baseline, and diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep as dp
from . import evaluate as ev
from .decision_tree import DecisionTree


SEED = 42


def regression_scores(y, prediction):
    return {"RMSE_AZN": ev.rmse(y, prediction), "MAE_AZN": ev.mae(y, prediction), "R2": ev.r2(y, prediction)}


def classification_scores(y, prediction, probability):
    precision, recall, f1 = ev.precision_recall_f1(y, prediction)
    return {"Precision": precision, "Recall": recall, "F1": f1, "AUC": ev.roc_auc(y, probability)}


def timed_fit_predict(model, X_train, y_train, X_test):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = perf_counter() - start
    return prediction, fit_seconds, predict_seconds


def plot_depth(frame, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for key, group in frame.groupby("criterion"):
        ax.plot(group["max_depth"], group[f"train_{metric}"], linestyle="--", label=f"{key} train")
        ax.plot(group["max_depth"], group[f"validation_{metric}"], label=f"{key} validation")
    ax.set(xlabel="Maximum depth", ylabel=ylabel, title=title)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_leaf(frame, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    group = frame.sort_values("min_samples_leaf")
    ax.plot(group.min_samples_leaf, group[f"train_{metric}"], marker="o", label="train")
    ax.plot(group.min_samples_leaf, group[f"validation_{metric}"], marker="o", label="validation")
    ax.set(xlabel="Minimum samples per leaf", ylabel=ylabel, title=title)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tune_regression(X_train, y_train, X_validation, y_validation, output):
    rows = []
    log_train = np.log(y_train)
    for depth in range(1, 21):
        model = DecisionTree(task="regression", criterion="mse", max_depth=depth, min_samples_leaf=5)
        model.fit(X_train, log_train)
        rows.append({"criterion": "mse", "max_depth": depth, "min_samples_leaf": 5,
                     "train_RMSE": ev.rmse(y_train, np.exp(model.predict(X_train))),
                     "validation_RMSE": ev.rmse(y_validation, np.exp(model.predict(X_validation))),
                     "leaves": model.n_leaves_})
    best_depth = min(rows, key=lambda row: row["validation_RMSE"])["max_depth"]
    for leaf in (1, 20):
        model = DecisionTree(task="regression", criterion="mse", max_depth=best_depth, min_samples_leaf=leaf)
        model.fit(X_train, log_train)
        rows.append({"criterion": "mse", "max_depth": best_depth, "min_samples_leaf": leaf,
                     "train_RMSE": ev.rmse(y_train, np.exp(model.predict(X_train))),
                     "validation_RMSE": ev.rmse(y_validation, np.exp(model.predict(X_validation))),
                     "leaves": model.n_leaves_})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tree_regression_sweep.csv", index=False)
    plot_depth(frame[frame.min_samples_leaf == 5], "RMSE", "RMSE (AZN)",
               "Regression: depth versus RMSE", output.parent / "report/figures/tree_regression_depth.png")
    plot_leaf(frame[frame.max_depth == best_depth], "RMSE", "RMSE (AZN)",
              "Regression: leaf size at selected depth", output.parent / "report/figures/tree_regression_leaf.png")
    return min(rows, key=lambda row: (row["validation_RMSE"], row["leaves"]))


def tune_classification(X_train, y_train, X_validation, y_validation, output):
    rows = []
    for criterion in ("gini", "entropy"):
        for depth in range(1, 21):
            model = DecisionTree(criterion=criterion, max_depth=depth, min_samples_leaf=5)
            model.fit(X_train, y_train)
            rows.append({"criterion": criterion, "max_depth": depth, "min_samples_leaf": 5,
                         "train_F1": ev.precision_recall_f1(y_train, model.predict(X_train))[2],
                         "validation_F1": ev.precision_recall_f1(y_validation, model.predict(X_validation))[2],
                         "validation_AUC": ev.roc_auc(y_validation, model.predict_proba(X_validation)[:, 1]),
                         "leaves": model.n_leaves_})
    best = max(rows, key=lambda row: (row["validation_F1"], row["validation_AUC"], -row["leaves"]))
    for leaf in (1, 20):
        model = DecisionTree(criterion=best["criterion"], max_depth=best["max_depth"], min_samples_leaf=leaf)
        model.fit(X_train, y_train)
        rows.append({"criterion": best["criterion"], "max_depth": best["max_depth"], "min_samples_leaf": leaf,
                     "train_F1": ev.precision_recall_f1(y_train, model.predict(X_train))[2],
                     "validation_F1": ev.precision_recall_f1(y_validation, model.predict(X_validation))[2],
                     "validation_AUC": ev.roc_auc(y_validation, model.predict_proba(X_validation)[:, 1]),
                     "leaves": model.n_leaves_})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tree_classification_sweep.csv", index=False)
    plot_depth(frame[frame.min_samples_leaf == 5], "F1", "Premium-class F1",
               "Classification: depth versus F1", output.parent / "report/figures/tree_classification_depth.png")
    plot_leaf(frame[(frame.max_depth == best["max_depth"]) & (frame.criterion == best["criterion"])],
              "F1", "Premium-class F1", "Classification: leaf size at selected depth",
              output.parent / "report/figures/tree_classification_leaf.png")
    return max(rows, key=lambda row: (row["validation_F1"], row["validation_AUC"], -row["leaves"]))


def plot_importance(model, names, path):
    order = np.argsort(model.feature_importances_)[-15:]
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    ax.barh(np.asarray(names)[order], model.feature_importances_[order])
    ax.set(xlabel="Normalized impurity decrease", title="Decision Tree feature importance")
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(d, train_cap=5_000, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    output = root / "results"
    figures = root / "report" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data, counts = d["data"], {k: d["info"][k] for k in ("raw_rows", "unique_listing_rows", "clean_rows", "usd_rows")}
    train, validation, test = d["train"], d["val"], d["test"]
    names = d["names"]
    X_train, X_validation, X_test = d["X_tr"], d["X_val"], d["X_te"]
    y_train, y_validation, y_test = d["y_tr"], d["y_val"], d["y_te"]
    threshold = d["thr"]
    tier_train, tier_validation, tier_test = d["t_tr"], d["t_val"], d["t_te"]

    rng = np.random.default_rng(SEED)
    cap = len(train) if not train_cap else min(train_cap, len(train))
    chosen = np.sort(rng.choice(len(train), cap, replace=False))
    X_fit, price_fit, tier_fit = X_train[chosen], y_train[chosen], tier_train[chosen]
    selected_regression = tune_regression(X_fit, price_fit, X_validation, y_validation, output)
    selected_classification = tune_classification(X_fit, tier_fit, X_validation, tier_validation, output)

    regression_args = {"max_depth": int(selected_regression["max_depth"]),
                       "min_samples_leaf": int(selected_regression["min_samples_leaf"]),
                       "min_samples_split": 2, "min_impurity_decrease": 0.0}
    classification_args = {"criterion": selected_classification["criterion"],
                           "max_depth": int(selected_classification["max_depth"]),
                           "min_samples_leaf": int(selected_classification["min_samples_leaf"]),
                           "min_samples_split": 2, "min_impurity_decrease": 0.0}
    ours_reg = DecisionTree(task="regression", criterion="mse", **regression_args)
    reference_reg = DecisionTreeRegressor(criterion="squared_error", random_state=SEED, **regression_args)
    ours_clf = DecisionTree(**classification_args)
    reference_clf = DecisionTreeClassifier(random_state=SEED, **classification_args)
    rows = []

    for name, model in (("Our Tree", ours_reg), ("sklearn Tree", reference_reg)):
        log_prediction, fit_time, predict_time = timed_fit_predict(model, X_fit, np.log(price_fit), X_test)
        rows.append({"task": "A", "model": name, **regression_scores(y_test, np.exp(log_prediction)),
                     "fit_seconds": fit_time, "predict_seconds": predict_time})
    for name, model in (("Our Tree", ours_clf), ("sklearn Tree", reference_clf)):
        prediction, fit_time, predict_time = timed_fit_predict(model, X_fit, tier_fit, X_test)
        rows.append({"task": "B", "model": name,
                     **classification_scores(tier_test, prediction, model.predict_proba(X_test)[:, 1]),
                     "fit_seconds": fit_time, "predict_seconds": predict_time})

    pd.DataFrame(rows).to_csv(output / "tree_test_metrics.csv", index=False)
    plot_importance(ours_reg, names, figures / "tree_regression_importance.png")
    plot_importance(ours_clf, names, figures / "tree_classification_importance.png")
    price_prediction = np.exp(ours_reg.predict(X_test))
    classification_prediction = ours_clf.predict(X_test)
    train_locations = data.iloc[train].location.fillna("unknown").value_counts()
    test_locations = data.iloc[test].location.fillna("unknown")
    rare = test_locations.map(train_locations).fillna(0).to_numpy() < 30
    luxury = y_test > np.quantile(y_train, 0.9)
    near = np.abs(y_test - threshold) <= 0.1 * threshold
    test_categories = data.iloc[test].category.fillna("unknown")
    category_errors = {}
    for category, indices in test_categories.groupby(test_categories).groups.items():
        positions = test_categories.index.get_indexer(indices)
        if len(positions) >= 50:
            category_errors[str(category)] = {
                "count": len(positions),
                "regression_mae_AZN": ev.mae(y_test[positions], price_prediction[positions]),
                "classification_error": float(np.mean(tier_test[positions] != classification_prediction[positions])),
            }
    diagnostics = {
        "regression_mae_all": ev.mae(y_test, price_prediction),
        "regression_mae_luxury": ev.mae(y_test[luxury], price_prediction[luxury]),
        "regression_mae_rare_location": ev.mae(y_test[rare], price_prediction[rare]) if rare.any() else None,
        "classification_error_all": float(np.mean(tier_test != classification_prediction)),
        "classification_error_near_threshold": float(np.mean(tier_test[near] != classification_prediction[near])) if near.any() else None,
        "classification_error_rare_location": float(np.mean(tier_test[rare] != classification_prediction[rare])) if rare.any() else None,
        "luxury_test_rows": int(luxury.sum()),
        "rare_location_test_rows": int(rare.sum()),
        "near_threshold_test_rows": int(near.sum()),
        "category_errors": category_errors,
        "confusion_matrix": ev.confusion_matrix(tier_test, classification_prediction).tolist(),
        "top_regression_features": [(names[i], float(ours_reg.feature_importances_[i]))
                                    for i in np.argsort(ours_reg.feature_importances_)[-10:][::-1]],
        "top_classification_features": [(names[i], float(ours_clf.feature_importances_[i]))
                                        for i in np.argsort(ours_clf.feature_importances_)[-10:][::-1]],
    }
    summary = {**counts, "train_rows": len(train), "validation_rows": len(validation),
               "test_rows": len(test), "fit_rows": len(chosen), "feature_count": len(names),
               "tier_threshold_AZN": threshold, "selected_regression": selected_regression,
               "class_balance": {"train": int(tier_train.sum()), "validation": int(tier_validation.sum()),
                                 "test": int(tier_test.sum())},
               "selected_classification": selected_classification, "diagnostics": diagnostics}
    (output / "tree_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string(index=False))
    return {"rows": rows, "summary": summary,
            "regression_args": regression_args, "classification_args": classification_args}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=dp.DATA_PATH)
    parser.add_argument("--train-cap", type=int, default=5_000, help="0 = use the full training split")
    args = parser.parse_args()
    if 0 < args.train_cap < 100:
        parser.error("--train-cap must be 0 (full train) or at least 100")
    root = Path(__file__).resolve().parents[1]
    run(dp.prepare(root / args.data), args.train_cap, root)



if __name__ == "__main__":
    main()
