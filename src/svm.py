from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,
        n_epochs: int = 20,
        batch_size: int | None = 64,
        lr: str = "pegasos",
        eta0: float = 0.01,
        project: bool = True,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eta0 = eta0
        self.project = project
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.classes_ = None
        self.objective_history_ = []

    def _to_pm1(self, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"need exactly 2 classes, got {self.classes_}")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _eta(self, t):
        if self.lr == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr == "invscaling":
            return self.eta0 / np.sqrt(t)
        if self.lr == "constant":
            return self.eta0
        raise ValueError(f"unknown lr schedule: {self.lr}")

    def objective(self, X, y_pm):
        margins = y_pm * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        # b gets regularized too (constant 1 column trick), tiny difference from the pdf formula
        return 0.5 * self.lambda_ * (self.w @ self.w + self.b ** 2) + hinge

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_pm = self._to_pm1(y)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        rng = np.random.default_rng(self.random_state)

        k = n if self.batch_size is None else min(self.batch_size, n)
        steps_per_epoch = int(np.ceil(n / k))
        radius = 1.0 / np.sqrt(self.lambda_)
        self.objective_history_ = []
        t = 0

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            w_sum = np.zeros(d)
            b_sum = 0.0
            for s in range(steps_per_epoch):
                t += 1
                idx = order[s * k:(s + 1) * k]
                eta = self._eta(t)
                Xb, yb = X[idx], y_pm[idx]
                active = yb * (Xb @ w + b) < 1
                shrink = 1.0 - eta * self.lambda_
                # same as adding a constant 1 column to x, so b is updated like the other weights
                w = shrink * w + eta * (yb[active, None] * Xb[active]).sum(axis=0) / len(idx)
                b = shrink * b + eta * yb[active].sum() / len(idx)
                if self.project:
                    norm = np.sqrt(w @ w + b * b)
                    if norm > radius:
                        w = w * (radius / norm)
                        b = b * (radius / norm)
                w_sum += w
                b_sum += b
            # keep the average of this epoch, last iterate jumps a lot when lambda is tiny (pegasos paper averages too)
            self.w, self.b = w_sum / steps_per_epoch, float(b_sum / steps_per_epoch)
            self.objective_history_.append(float(self.objective(X, y_pm)))

        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def margin_info(self, X, y):
        y_pm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        margins = y_pm * self.decision_function(X)
        w_norm = float(np.linalg.norm(self.w))
        sv_idx = np.where(margins <= 1.0)[0]
        return {
            "w_norm": w_norm,
            "margin_width": 2.0 / w_norm if w_norm > 0 else np.inf,
            "n_sv": int(len(sv_idx)),
            "sv_frac": len(sv_idx) / len(y_pm),
            "sv_idx": sv_idx,
        }


class RandomFourierFeatures:
    def __init__(self, n_components: int = 300, gamma: float | None = None, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W = None
        self.offset = None

    def fit(self, X):
        d = np.asarray(X).shape[1]
        gamma = self.gamma if self.gamma is not None else 1.0 / d
        rng = np.random.default_rng(self.random_state)
        # K(x,z)=exp(-gamma*||x-z||^2)  ->  W ~ N(0, 2*gamma), offset ~ U(0, 2pi)
        self.W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, self.n_components))
        self.offset = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(np.asarray(X, dtype=float) @ self.W + self.offset)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RFFPegasosSVM:
    def __init__(self, n_components: int = 300, gamma: float | None = None, random_state: int = 42, **svm_kwargs):
        self.rff = RandomFourierFeatures(n_components, gamma, random_state)
        self.svm = PegasosSVM(random_state=random_state, **svm_kwargs)

    def fit(self, X, y):
        self.svm.fit(self.rff.fit_transform(X), y)
        return self

    def decision_function(self, X):
        return self.svm.decision_function(self.rff.transform(X))

    def predict(self, X):
        return self.svm.predict(self.rff.transform(X))
