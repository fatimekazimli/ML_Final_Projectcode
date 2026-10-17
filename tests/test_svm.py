import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import LinearSVC

from src.svm import PegasosSVM, RandomFourierFeatures, RFFPegasosSVM


def scale(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def blobs(spread, n=400, seed=0):
    X, y = make_blobs(n_samples=n, centers=[[-2, -1], [2, 1]], cluster_std=spread, random_state=seed)
    return scale(X), y


def test_labels_are_mapped_back():
    X, y = blobs(0.6)
    y = np.where(y == 1, 7, 3)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=30).fit(X, y)
    pred = svm.predict(X)
    print("classes:", svm.classes_, "pred values:", np.unique(pred))
    assert set(np.unique(pred)) <= {3, 7}


def test_separable_data():
    X, y = blobs(0.4)
    svm = PegasosSVM(lambda_=1e-3, n_epochs=50).fit(X, y)
    acc = np.mean(svm.predict(X) == y)
    print("separable train acc:", acc)
    assert acc == 1.0


def test_objective_decreases():
    X, y = blobs(1.5)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=100, batch_size=None).fit(X, y)
    h = svm.objective_history_
    print(f"objective  first={h[0]:.4f}  last={h[-1]:.4f}")
    assert h[-1] < h[0]
    assert h[-1] <= min(h[:10]) + 1e-9


def test_matches_sklearn_linearsvc():
    X, y = blobs(1.5, n=600)
    lam = 1e-2
    ours = PegasosSVM(lambda_=lam, n_epochs=300, batch_size=None).fit(X, y)
    # same objective as ours: C = 1/(lambda*n), hinge loss, intercept regularized like our folded bias
    sk = LinearSVC(C=1.0 / (lam * len(y)), loss="hinge", dual=True, max_iter=100_000, random_state=0).fit(X, y)

    y_pm = np.where(y == 1, 1.0, -1.0)
    j_ours = ours.objective(X, y_pm)
    ref = PegasosSVM(lambda_=lam)
    ref.w, ref.b = sk.coef_.ravel(), float(sk.intercept_[0])
    j_sk = ref.objective(X, y_pm)
    agree = np.mean(ours.predict(X) == sk.predict(X))
    print(f"objective ours={j_ours:.5f}  sklearn={j_sk:.5f}  prediction agreement={agree:.3f}")
    assert abs(j_ours - j_sk) / j_sk < 0.02
    assert agree >= 0.97


def test_lambda_controls_margin():
    X, y = blobs(1.2)
    widths = []
    for lam in [1e-3, 1e-2, 1e-1]:
        svm = PegasosSVM(lambda_=lam, n_epochs=200, batch_size=None).fit(X, y)
        info = svm.margin_info(X, y)
        widths.append(info["margin_width"])
        print(f"lambda={lam:g}  ||w||={info['w_norm']:.3f}  margin={info['margin_width']:.3f}  n_sv={info['n_sv']}")
    assert widths[0] < widths[1] < widths[2]


def test_support_vectors_definition():
    X, y = blobs(1.2)
    svm = PegasosSVM(lambda_=1e-2, n_epochs=100, batch_size=None).fit(X, y)
    info = svm.margin_info(X, y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    m = y_pm * svm.decision_function(X)
    print("n_sv:", info["n_sv"], "of", len(y))
    assert info["n_sv"] > 0
    assert np.all(m[info["sv_idx"]] <= 1.0)
    assert np.all(np.delete(m, info["sv_idx"]) > 1.0)
    assert np.isclose(info["margin_width"], 2.0 / np.linalg.norm(svm.w))


def test_rff_approximates_rbf_kernel():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    gamma = 0.3
    phi = RandomFourierFeatures(n_components=4000, gamma=gamma).fit_transform(X)
    approx = phi @ phi.T
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    exact = np.exp(-gamma * sq)
    err = np.abs(approx - exact).mean()
    print("mean |K_rff - K_rbf|:", round(err, 4))
    assert err < 0.03


def test_rff_solves_nonlinear_data():
    X, y = make_circles(n_samples=500, factor=0.4, noise=0.08, random_state=0)
    X = scale(X)
    lin = PegasosSVM(lambda_=1e-3, n_epochs=50).fit(X, y)
    rff = RFFPegasosSVM(n_components=300, gamma=1.0, lambda_=1e-4, n_epochs=50).fit(X, y)
    acc_lin = np.mean(lin.predict(X) == y)
    acc_rff = np.mean(rff.predict(X) == y)
    print(f"circles  linear acc={acc_lin:.3f}  rff acc={acc_rff:.3f}")
    assert acc_lin < 0.7
    assert acc_rff > 0.95


def test_same_seed_same_model():
    X, y = blobs(1.0)
    a = PegasosSVM(lambda_=1e-3, n_epochs=5, random_state=1).fit(X, y)
    b = PegasosSVM(lambda_=1e-3, n_epochs=5, random_state=1).fit(X, y)
    assert np.allclose(a.w, b.w) and a.b == b.b


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n--- {fn.__name__}")
        fn()
        print("ok")
    print(f"\nall {len(tests)} svm tests passed")
