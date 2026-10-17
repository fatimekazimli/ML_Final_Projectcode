from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import evaluate as ev
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RFFPegasosSVM

SEED = 42
FIG_DIR = "report/figures"
RES_DIR = "results"
LAMBDAS = np.logspace(-6, -1, 11)
GAMMA_MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]
RFF_LAMBDAS = [1e-5, 1e-4, 1e-3]
RFF_DIMS = [100, 300, 1000]
SWEEP_MAX_N = 20_000
RFF_F1_TOL = 0.005
SVC_MAX_N = 5_000
EPOCHS = 20
BATCH = 64
TREE_DEPTH = 8
CV_FOLDS = 5


def scale(X_fit, *others):
    mu = X_fit.mean(axis=0)
    sd = X_fit.std(axis=0)
    sd[sd == 0] = 1.0
    return [(A - mu) / sd for A in (X_fit,) + others]


def scores_of(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    p = np.asarray(model.predict_proba(X))
    return p[:, 1] if p.ndim == 2 else p


def clf_metrics(y, pred, score):
    p, r, f = ev.precision_recall_f1(y, pred)
    return {"Precision": p, "Recall": r, "F1": f, "AUC": ev.roc_auc(y, score)}


def reg_metrics(y, pred):
    return {"RMSE": ev.rmse(y, pred), "MAE": ev.mae(y, pred), "R2": ev.r2(y, pred)}


def subsample(X, y, max_n, seed=SEED):
    if len(y) <= max_n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), max_n, replace=False)
    return X[idx], y[idx]


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def toy_margin_plot():
    print("\n[svm] 2d toy: how lambda changes the margin")
    X, y = make_blobs(n_samples=200, centers=[[-1.5, -1], [1.5, 1]], cluster_std=1.1, random_state=SEED)
    X = scale(X)[0]
    lo, hi = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, lam in zip(axes, [1e-3, 1e-2, 1e-1]):
        svm = PegasosSVM(lambda_=lam, n_epochs=300, batch_size=None).fit(X, y)
        info = svm.margin_info(X, y)
        print(f"  lambda={lam:g}  ||w||={info['w_norm']:.3f}  margin={info['margin_width']:.3f}  n_sv={info['n_sv']}")
        zz = svm.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-100, 0, 100], colors=["#dbe7f5", "#f7e3d6"], alpha=0.8)
        ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, c="#1f5fad", label="standard (0)")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, c="#c4511f", label="premium (1)")
        sv = info["sv_idx"]
        ax.scatter(X[sv, 0], X[sv, 1], s=60, facecolors="none", edgecolors="k", linewidths=0.8, label="support vectors")
        ax.set_title(f"λ={lam:g}   margin=2/||w||={info['margin_width']:.2f}   SVs={info['n_sv']}")
        ax.set_xlabel("x1 (standardized)")
        ax.set_ylabel("x2 (standardized)")
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "fig_svm_margin_2d.png")


def objective_plot(X, y, lam):
    print(f"\n[svm] objective vs epoch for the chosen lambda={lam:.0e} (batch size + lr schedule)")
    Xs, ys = subsample(X, y, SWEEP_MAX_N)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for bs, name in [(None, "full batch"), (256, "mini-batch 256"), (32, "mini-batch 32")]:
        svm = PegasosSVM(lambda_=lam, n_epochs=40, batch_size=bs).fit(Xs, ys)
        h = svm.objective_history_
        print(f"  {name:<16} J: {h[0]:.4f} -> {h[-1]:.4f}")
        axes[0].plot(range(1, len(h) + 1), h, label=name)
    axes[0].set_title(f"Batch size (pegasos lr, λ={lam:g})")

    for lr, eta0 in [("pegasos", None), ("invscaling", 0.1), ("constant", 0.01)]:
        svm = PegasosSVM(lambda_=lam, n_epochs=40, batch_size=BATCH, lr=lr, eta0=eta0 or 0.01).fit(Xs, ys)
        h = svm.objective_history_
        label = "1/(λt)" if lr == "pegasos" else f"{lr} η0={eta0}"
        print(f"  {label:<16} J: {h[0]:.4f} -> {h[-1]:.4f}")
        axes[1].plot(range(1, len(h) + 1), h, label=label)
    axes[1].set_title(f"Learning-rate schedule (batch {BATCH})")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("objective J(w,b)")
        ax.set_yscale("log")
        ax.legend()
    save(fig, "fig_svm_objective.png")


def lambda_sweep(X_tr, y_tr, X_val, y_val):
    print("\n[svm] lambda sweep (linear pegasos)")
    Xs, ys = subsample(X_tr, y_tr, SWEEP_MAX_N)
    rows = []
    for lam in LAMBDAS:
        svm = PegasosSVM(lambda_=lam, n_epochs=EPOCHS, batch_size=BATCH).fit(Xs, ys)
        info = svm.margin_info(Xs, ys)
        tr = clf_metrics(ys, svm.predict(Xs), svm.decision_function(Xs))
        va = clf_metrics(y_val, svm.predict(X_val), svm.decision_function(X_val))
        rows.append((lam, info["margin_width"], info["sv_frac"], tr["F1"], va["F1"], va["AUC"]))
        print(f"  λ={lam:.0e}  margin={info['margin_width']:8.3f}  sv%={info['sv_frac']*100:5.1f}"
              f"  train F1={tr['F1']:.4f}  val F1={va['F1']:.4f}  val AUC={va['AUC']:.4f}")
    rows = np.array(rows)

    best = rows[np.argmax(rows[:, 4])]
    print(f"  -> best lambda by val F1: {best[0]:.0e}  (val F1={best[4]:.4f}, val AUC={best[5]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rows[:, 0], rows[:, 3], "o-", label="train F1")
    axes[0].plot(rows[:, 0], rows[:, 4], "o-", label="val F1")
    axes[0].plot(rows[:, 0], rows[:, 5], "s--", label="val AUC")
    axes[0].axvline(best[0], color="gray", ls=":", label="chosen λ")
    axes[0].set_ylabel("score")
    axes[0].set_title("Validation metric vs λ")
    axes[1].plot(rows[:, 0], rows[:, 1], "o-", color="#c4511f", label="margin width 2/||w||")
    ax2 = axes[1].twinx()
    ax2.plot(rows[:, 0], rows[:, 2] * 100, "s--", color="#1f5fad", label="support vectors %")
    axes[1].set_ylabel("margin width")
    ax2.set_ylabel("support vectors (% of train)")
    axes[1].set_yscale("log")
    axes[1].set_title("Margin vs λ")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="lower right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("λ (regularization)")
    axes[0].legend()
    save(fig, "fig_svm_lambda_sweep.png")
    np.savetxt(os.path.join(RES_DIR, "svm_lambda_sweep.csv"), rows, delimiter=",",
               header="lambda,margin_width,sv_frac,train_f1,val_f1,val_auc", comments="")
    return float(best[0])


def rff_sweep(X_tr, y_tr, X_val, y_val):
    print("\n[svm] rff sweep (gamma x lambda, then n_components)")
    Xs, ys = subsample(X_tr, y_tr, SWEEP_MAX_N)
    d = X_tr.shape[1]
    grid = []
    for mult in GAMMA_MULTS:
        for lam in RFF_LAMBDAS:
            m = RFFPegasosSVM(n_components=300, gamma=mult / d, lambda_=lam, n_epochs=EPOCHS, batch_size=BATCH).fit(Xs, ys)
            va = clf_metrics(y_val, m.predict(X_val), m.decision_function(X_val))
            grid.append((mult / d, lam, va["F1"], va["AUC"]))
            print(f"  γ={mult}/d={mult/d:.4f}  λ={lam:.0e}  val F1={va['F1']:.4f}  val AUC={va['AUC']:.4f}")
    grid = np.array(grid)
    g_best, l_best = grid[np.argmax(grid[:, 2]), :2]
    print(f"  -> best gamma={g_best:.4f}, lambda={l_best:.0e}")

    dims = []
    for D in RFF_DIMS:
        t0 = time.perf_counter()
        m = RFFPegasosSVM(n_components=D, gamma=g_best, lambda_=l_best, n_epochs=EPOCHS, batch_size=BATCH).fit(Xs, ys)
        sec = time.perf_counter() - t0
        va = clf_metrics(y_val, m.predict(X_val), m.decision_function(X_val))
        dims.append((D, va["F1"], va["AUC"], sec))
        print(f"  D={D:5d}  val F1={va['F1']:.4f}  val AUC={va['AUC']:.4f}  fit={sec:.1f}s")
    dims = np.array(dims)
    ok = dims[:, 1] >= dims[:, 1].max() - RFF_F1_TOL
    D_best = int(dims[ok, 0].min())
    print(f"  -> n_components={D_best}  (smallest D within {RFF_F1_TOL} F1 of the best, rff is memory heavy)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lam in RFF_LAMBDAS:
        sel = grid[:, 1] == lam
        axes[0].plot(grid[sel, 0], grid[sel, 2], "o-", label=f"λ={lam:.0e}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("γ (RBF bandwidth)")
    axes[0].set_ylabel("val F1")
    axes[0].set_title("RFF: γ and λ (D=300)")
    axes[0].legend()
    axes[1].plot(dims[:, 0], dims[:, 1], "o-", label="val F1")
    axes[1].plot(dims[:, 0], dims[:, 2], "s--", label="val AUC")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of random features D")
    axes[1].set_title(f"RFF: D (γ={g_best:.4f}, λ={l_best:.0e})")
    axes[1].legend()
    save(fig, "fig_svm_rff_sweep.png")
    return float(g_best), float(l_best), D_best


def weights_plot(svm, feat_names, top=10):
    print(f"\n[svm] top {top} weights (+ pushes to premium, - to standard)")
    order = np.argsort(svm.w)
    idx = np.r_[order[:top], order[-top:]]
    for i in idx[::-1]:
        print(f"  {svm.w[i]:+.4f}  {feat_names[i]}")
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f5fad" if svm.w[i] < 0 else "#c4511f" for i in idx]
    ax.barh(range(len(idx)), svm.w[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feat_names[i] for i in idx], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("weight on standardized feature")
    ax.set_title("Linear Pegasos SVM: largest weights")
    save(fig, "fig_svm_weights.png")


def error_analysis(svm, X_val_s, X_val_raw, y_val_t, price_val, thr, feat_names):
    print("\n[svm] error analysis on validation")
    wrong = svm.predict(X_val_s) != y_val_t
    print(f"  overall val error: {wrong.mean():.4f}  ({wrong.sum()} of {len(wrong)})")

    dist = np.abs(np.log(price_val / thr))
    edges = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, np.inf]
    labels, rates, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (dist >= lo) & (dist < hi)
        if sel.sum() == 0:
            continue
        labels.append(f"{lo}-{hi}" if np.isfinite(hi) else f">{lo}")
        rates.append(wrong[sel].mean())
        counts.append(int(sel.sum()))
        print(f"  |log(price/thr)| in [{lo}, {hi}):  n={sel.sum():6d}  error={wrong[sel].mean():.3f}")

    loc_rows = []
    for j, name in enumerate(feat_names):
        if not name.startswith("location_"):
            continue
        sel = X_val_raw[:, j] == 1
        if sel.sum() >= 30:
            loc_rows.append((name.replace("location_", ""), int(sel.sum()), wrong[sel].mean()))
    loc_rows.sort(key=lambda r: -r[2])
    if loc_rows:
        print("  worst districts (n>=30):")
        for name, n, e in loc_rows[:8]:
            print(f"    {name:<25} n={n:5d}  error={e:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, rates, color="#1f5fad")
    for i, (r, c) in enumerate(zip(rates, counts)):
        ax.text(i, r, f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("distance from tier threshold  |log(price / threshold)|")
    ax.set_ylabel("validation error rate")
    ax.set_title("SVM errors concentrate near the price threshold")
    save(fig, "fig_svm_errors.png")


def stratified_folds(y, k, seed=SEED):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cross_validate(factories, X, y, task, k=CV_FOLDS, max_fit=None):
    max_fit = max_fit or {}
    folds = stratified_folds(y if task == "clf" else np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8])), k)
    table = {}
    for name, make in factories.items():
        scores, fit_s, pred_s = [], [], []
        for f in range(k):
            tr, te = folds != f, folds == f
            X_fit, X_te = scale(X[tr], X[te])
            y_fit = y[tr]
            if name in max_fit:
                X_fit, y_fit = subsample(X_fit, y_fit, max_fit[name], seed=SEED + f)
            model = make(len(y_fit))
            t0 = time.perf_counter()
            model.fit(X_fit, y_fit)
            t1 = time.perf_counter()
            pred = model.predict(X_te)
            m = clf_metrics(y[te], pred, scores_of(model, X_te)) if task == "clf" else reg_metrics(y[te], pred)
            t2 = time.perf_counter()
            scores.append(m)
            fit_s.append(t1 - t0)
            pred_s.append(t2 - t1)
        row = {}
        for key in scores[0]:
            vals = np.array([s[key] for s in scores])
            row[key] = float(vals.mean())
            row[key + "_std"] = float(vals.std())
        row["fit_s"] = float(np.mean(fit_s))
        row["pred_s"] = float(np.mean(pred_s))
        table[name] = row
        main = "F1" if task == "clf" else "RMSE"
        note = f"  (fit on {max_fit[name]} rows)" if name in max_fit else ""
        print(f"  {name:<20} {main}={row[main]:.4f} ± {row[main + '_std']:.4f}   fit={row['fit_s']:.2f}s{note}")
    return table


def print_cv(table, keys, path):
    head = f"{'model':<20}" + "".join(f"{k:>20}" for k in keys) + f"{'fit s':>9}{'pred s':>9}"
    print(head)
    print("-" * len(head))
    for name, row in table.items():
        cells = "".join(f"{row[k]:>11.4f} ± {row[k + '_std']:.4f}" for k in keys)
        print(f"{name:<20}{cells}{row['fit_s']:>9.2f}{row['pred_s']:>9.2f}")
    pd.DataFrame(table).T.to_csv(path)


def run(X_tr, X_val, X_te, y_tr, y_val, y_te, thr, feat_names, tree_clf_args=None, tree_reg_args=None):
    os.makedirs(RES_DIR, exist_ok=True)
    yt_tr, yt_val, yt_te = (y_tr > thr).astype(int), (y_val > thr).astype(int), (y_te > thr).astype(int)
    X_tr_s, X_val_s, X_te_s = scale(X_tr, X_val, X_te)
    n, d = X_tr_s.shape
    print(f"\n========== SVM + benchmark  (train n={n}, d={d}, premium share={yt_tr.mean():.3f}) ==========")

    toy_margin_plot()
    lam = lambda_sweep(X_tr_s, yt_tr, X_val_s, yt_val)
    objective_plot(X_tr_s, yt_tr, lam)
    g_rff, l_rff, D_rff = rff_sweep(X_tr_s, yt_tr, X_val_s, yt_val)

    print("\n[svm] final configs (chosen on validation only)")
    print(f"  linear: lambda={lam:.0e}, epochs={EPOCHS}, batch={BATCH}, lr=1/(λt)")
    print(f"  rff:    gamma={g_rff:.4f}, lambda={l_rff:.0e}, D={D_rff}")
    svm = PegasosSVM(lambda_=lam, n_epochs=EPOCHS, batch_size=BATCH).fit(X_tr_s, yt_tr)
    info = svm.margin_info(X_tr_s, yt_tr)
    print(f"  linear on full train: ||w||={info['w_norm']:.4f}  margin={info['margin_width']:.4f}"
          f"  support vectors={info['n_sv']} ({info['sv_frac']*100:.1f}%)  final J={svm.objective_history_[-1]:.4f}")
    weights_plot(svm, feat_names)
    error_analysis(svm, X_val_s, X_val, yt_val, y_val, thr, feat_names)

    # C = 1/(lambda * n_fit) so sklearn minimizes the same objective as ours
    clf_models = {
        "Our SVM linear": lambda n_fit: PegasosSVM(lambda_=lam, n_epochs=EPOCHS, batch_size=BATCH),
        "Our SVM RFF": lambda n_fit: RFFPegasosSVM(n_components=D_rff, gamma=g_rff, lambda_=l_rff, n_epochs=EPOCHS, batch_size=BATCH),
        "sk LinearSVC hinge": lambda n_fit: make_pipeline(StandardScaler(), LinearSVC(C=1.0 / (lam * n_fit), loss="hinge", dual=True, max_iter=20_000, random_state=SEED)),
        "sk SGD hinge": lambda n_fit: make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=lam, max_iter=EPOCHS, tol=None, random_state=SEED)),
        "sk SVC RBF": lambda n_fit: make_pipeline(StandardScaler(), SVC(C=1.0 / (l_rff * n_fit), gamma=g_rff, kernel="rbf")),
    }
    # trees use the hyperparameters person 2 picked on validation, same for ours and sklearn
    ca = tree_clf_args or {"max_depth": TREE_DEPTH}
    ra = tree_reg_args or {"max_depth": TREE_DEPTH}
    clf_models["Our Tree clf"] = lambda n_fit: DecisionTree(task="classification", **ca)
    clf_models["sk Tree clf"] = lambda n_fit: DecisionTreeClassifier(random_state=SEED, **ca)
    reg_models = {
        "Our Tree reg": lambda n_fit: DecisionTree(task="regression", criterion="mse", **ra),
        "sk Tree reg": lambda n_fit: DecisionTreeRegressor(criterion="squared_error", random_state=SEED, **ra),
    }
    print(f"\n  tree args  clf={ca}  reg={ra}")
    svc_cap = {"sk SVC RBF": SVC_MAX_N}

    print(f"\n[benchmark] Task B: {CV_FOLDS}-fold stratified CV on train+val (scaling fit inside each fold)")
    X_cv = np.vstack([X_tr, X_val])
    cv_b = cross_validate(clf_models, X_cv, np.r_[yt_tr, yt_val], "clf", max_fit=svc_cap)
    print(f"\n[benchmark] Task A: {CV_FOLDS}-fold CV on log(price)")
    cv_a = cross_validate(reg_models, X_cv, np.log(np.r_[y_tr, y_val]), "reg")

    print("\n[benchmark] Task B: test set (fit on train, scaled with train stats)")
    test_b = {}
    for name, make in clf_models.items():
        X_fit, y_fit = X_tr_s, yt_tr
        if name in svc_cap:
            X_fit, y_fit = subsample(X_fit, y_fit, svc_cap[name])
        model = make(len(y_fit)).fit(X_fit, y_fit)
        pred = model.predict(X_te_s)
        test_b[name] = clf_metrics(yt_te, pred, scores_of(model, X_te_s))
        cm = ev.confusion_matrix(yt_te, pred)
        print(f"  {name:<20} F1={test_b[name]['F1']:.4f}  AUC={test_b[name]['AUC']:.4f}  cm [[TN,FP],[FN,TP]]={cm.tolist()}")

    print(f"\n=== Task B: {CV_FOLDS}-fold CV mean ± std (train+val) ===")
    print_cv(cv_b, ["F1", "AUC", "Precision", "Recall"], os.path.join(RES_DIR, "cv_task_b.csv"))
    print(f"\n=== Task A: {CV_FOLDS}-fold CV mean ± std on log(price) (train+val) ===")
    print_cv(cv_a, ["RMSE", "MAE", "R2"], os.path.join(RES_DIR, "cv_task_a.csv"))
    print("\n=== Task B: test set ===")
    ev.compare(test_b, os.path.join(RES_DIR, "test_task_b.csv"))
    return test_b, cv_b, cv_a
