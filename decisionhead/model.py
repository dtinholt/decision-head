"""Pure-Python multinomial logistic regression with L2 regularization.

This is the classifier :mod:`decisionhead.train` fits and
:mod:`decisionhead.learned` serves. It has exactly one job: map a fixed-
length numeric feature vector (built by :mod:`decisionhead.features`) to
a probability distribution over a fixed, ordered set of class names.

Zero-dependency guarantee: :func:`MultinomialLogisticRegression.predict_proba`
(the only method the serving path calls) is pure standard-library Python --
a handful of multiplications and one softmax. Training
(:meth:`MultinomialLogisticRegression.fit`) is the one place numpy is used
*if installed* (the ``decisionhead[train]`` extra), purely for speed on
larger trace sets; the pure-Python path (used automatically when numpy is
absent) implements the identical algorithm and both are covered by tests.
Neither path is used at serve time, so a serving-only install never needs
numpy.

Determinism: :meth:`MultinomialLogisticRegression.fit` is full-batch
gradient descent (every example contributes to every step, so there is no
minibatch order to fix) seeded only for its initial-weight jitter (a small
uniform perturbation that breaks the perfect symmetry of an all-zero
start without introducing any other source of randomness). Two calls with
the same data, hyperparameters, and seed produce bit-for-bit identical
weights.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, cast

from ._numpy_opt import HAVE_NUMPY, np

JSONDict = dict[str, Any]

_EPS = 1e-12
_INIT_JITTER = 1e-3


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a plain list of logits."""
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    if total <= 0:
        n = len(logits) or 1
        return [1.0 / n] * n
    return [v / total for v in exps]


@dataclass
class StandardScaler:
    """Per-feature zero-mean, unit-variance scaling, fit on the training
    set only and reapplied unchanged at serve time (its parameters are
    saved in the model file). A feature with zero variance in the
    training data (e.g. a flag that never fired) is left unscaled (std
    treated as 1.0) rather than dividing by zero."""

    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, rows: list[list[float]]) -> StandardScaler:
        """Fits per-feature means and population standard deviations on
        ``rows``. Raises ``ValueError`` on an empty list."""
        if not rows:
            raise ValueError("cannot fit a scaler on zero rows")
        n_features = len(rows[0])
        means = [0.0] * n_features
        for row in rows:
            for i, v in enumerate(row):
                means[i] += v
        n = len(rows)
        means = [m / n for m in means]
        variances = [0.0] * n_features
        for row in rows:
            for i, v in enumerate(row):
                d = v - means[i]
                variances[i] += d * d
        stds = [math.sqrt(v / n) if v > 0 else 1.0 for v in variances]
        return cls(means=means, stds=stds)

    def transform(self, row: list[float]) -> list[float]:
        """Scales one feature vector. Raises ``ValueError`` if its length
        differs from the fitted width."""
        return [(v - m) / s for v, m, s in zip(row, self.means, self.stds, strict=True)]

    def transform_batch(self, rows: list[list[float]]) -> list[list[float]]:
        """:meth:`transform` applied to each row."""
        return [self.transform(row) for row in rows]

    def to_dict(self) -> JSONDict:
        """The model-file form read back by :meth:`from_dict`."""
        return {"means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, data: JSONDict) -> StandardScaler:
        """Rebuilds a scaler from :meth:`to_dict` output. Finiteness is
        checked by :class:`decisionhead.learned.LearnedModel` at load."""
        return cls(means=list(data["means"]), stds=list(data["stds"]))


@dataclass
class MultinomialLogisticRegression:
    """A dense ``n_classes x n_features`` weight matrix plus a per-class
    bias, trained by full-batch gradient descent on the multinomial
    (softmax) cross-entropy loss with L2 regularization on the weights
    (never the bias)."""

    n_classes: int
    n_features: int
    weights: list[list[float]]
    bias: list[float]

    @classmethod
    def initialize(cls, n_classes: int, n_features: int, seed: int = 0) -> MultinomialLogisticRegression:
        """All-zero weights plus a tiny seeded jitter, just enough to
        break symmetry between classes so gradient descent does not stay
        stuck at the fully-symmetric zero point when two classes start
        with identical gradients."""
        rng = random.Random(seed)
        weights = [[rng.uniform(-_INIT_JITTER, _INIT_JITTER) for _ in range(n_features)] for _ in range(n_classes)]
        bias = [0.0 for _ in range(n_classes)]
        return cls(n_classes=n_classes, n_features=n_features, weights=weights, bias=bias)

    def predict_proba(self, x: list[float]) -> list[float]:
        """Pure standard-library inference: the only method the serving
        path (:mod:`decisionhead.learned`) calls."""
        if len(x) != self.n_features:
            raise ValueError(f"expected a {self.n_features}-feature vector, got {len(x)}")
        logits = [
            sum(w * xi for w, xi in zip(self.weights[k], x, strict=True)) + self.bias[k]
            for k in range(self.n_classes)
        ]
        return softmax(logits)

    def predict_proba_batch(self, rows: list[list[float]]) -> list[list[float]]:
        """:meth:`predict_proba` applied to each row."""
        return [self.predict_proba(row) for row in rows]

    def fit(
        self,
        rows: list[list[float]],
        labels: list[int],
        *,
        l2: float = 0.1,
        lr: float = 0.5,
        epochs: int = 300,
        tol: float = 1e-7,
    ) -> list[float]:
        """Full-batch gradient descent in place. Returns the mean training
        loss (cross-entropy + L2 penalty) after each epoch, so a caller
        can inspect convergence; stops early once the loss stops moving by
        more than ``tol`` between epochs. Uses numpy internally when
        available (:data:`decisionhead._numpy_opt.HAVE_NUMPY`); the result
        is the same algorithm either way, not necessarily bit-identical
        across the two implementations (only within one implementation,
        given the same seed)."""
        if len(rows) != len(labels):
            raise ValueError("rows and labels must have the same length")
        if not rows:
            raise ValueError("cannot fit on zero rows")
        if HAVE_NUMPY:
            return self._fit_numpy(rows, labels, l2=l2, lr=lr, epochs=epochs, tol=tol)
        return self._fit_pure_python(rows, labels, l2=l2, lr=lr, epochs=epochs, tol=tol)

    # ------------------------------------------------------------- pure python

    def _fit_pure_python(
        self, rows: list[list[float]], labels: list[int], *, l2: float, lr: float, epochs: int, tol: float
    ) -> list[float]:
        n = len(rows)
        k_range = range(self.n_classes)
        d_range = range(self.n_features)
        history: list[float] = []
        prev_loss = float("inf")
        for _epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in k_range]
            grad_b = [0.0] * self.n_classes
            total_ce = 0.0
            for x, y in zip(rows, labels, strict=True):
                logits = [sum(self.weights[k][d] * x[d] for d in d_range) + self.bias[k] for k in k_range]
                probs = softmax(logits)
                total_ce += -math.log(max(probs[y], _EPS))
                for k in k_range:
                    g = probs[k] - (1.0 if k == y else 0.0)
                    grad_b[k] += g
                    row_grad = grad_w[k]
                    for d in d_range:
                        row_grad[d] += g * x[d]
            l2_penalty = 0.0
            for k in k_range:
                grad_b[k] /= n
                for d in d_range:
                    grad_w[k][d] = grad_w[k][d] / n + l2 * self.weights[k][d]
                    l2_penalty += self.weights[k][d] ** 2
            loss = total_ce / n + 0.5 * l2 * l2_penalty
            for k in k_range:
                self.bias[k] -= lr * grad_b[k]
                for d in d_range:
                    self.weights[k][d] -= lr * grad_w[k][d]
            history.append(loss)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        return history

    # ------------------------------------------------------------- numpy

    def _fit_numpy(
        self, rows: list[list[float]], labels: list[int], *, l2: float, lr: float, epochs: int, tol: float
    ) -> list[float]:
        x_mat = np.asarray(rows, dtype=np.float64)
        y_idx = np.asarray(labels, dtype=np.int64)
        n = x_mat.shape[0]
        w_mat = np.asarray(self.weights, dtype=np.float64)
        b_vec = np.asarray(self.bias, dtype=np.float64)
        y_onehot = np.zeros((n, self.n_classes), dtype=np.float64)
        y_onehot[np.arange(n), y_idx] = 1.0

        history: list[float] = []
        prev_loss = float("inf")
        for _epoch in range(epochs):
            logits = x_mat @ w_mat.T + b_vec
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            true_probs = probs[np.arange(n), y_idx]
            ce = float(np.mean(-np.log(np.clip(true_probs, _EPS, None))))
            l2_penalty = float(np.sum(w_mat**2))
            loss = ce + 0.5 * l2 * l2_penalty

            diff = probs - y_onehot
            grad_w = (diff.T @ x_mat) / n + l2 * w_mat
            grad_b = diff.mean(axis=0)

            w_mat = w_mat - lr * grad_w
            b_vec = b_vec - lr * grad_b

            history.append(loss)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        self.weights = cast(list[list[float]], w_mat.tolist())
        self.bias = cast(list[float], b_vec.tolist())
        return history

    # ------------------------------------------------------------- (de)serialization

    def to_dict(self) -> JSONDict:
        """The model-file form read back by :meth:`from_dict`."""
        return {
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> MultinomialLogisticRegression:
        """Rebuilds a classifier from :meth:`to_dict` output. Finiteness
        is checked by :class:`decisionhead.learned.LearnedModel` at load."""
        return cls(
            n_classes=int(data["n_classes"]),
            n_features=int(data["n_features"]),
            weights=[list(row) for row in data["weights"]],
            bias=list(data["bias"]),
        )


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed
    (mirrors the ``run_selftest`` convention every other module in this
    package follows). The equivalent, more thorough assertions live in
    ``tests/test_model.py``."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    probs = softmax([1.0, 1.0, 1.0])
    check("softmax is uniform for equal logits", all(abs(p - 1 / 3) < 1e-9 for p in probs))
    check("softmax sums to 1", abs(sum(softmax([3.0, -1.0, 0.5])) - 1.0) < 1e-9)

    scaler = StandardScaler.fit([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    check("scaler mean is centered", abs(scaler.means[0] - 2.0) < 1e-9)
    check("scaler handles zero-variance feature without dividing by zero", scaler.stds[1] == 1.0)
    scaled = scaler.transform([2.0, 5.0])
    check("scaling the mean gives zero", abs(scaled[0]) < 1e-9)

    # A trivially separable 2-class, 1-feature problem: label 1 iff x > 0.
    rng = random.Random(0)
    rows = []
    labels = []
    for _ in range(200):
        x = rng.uniform(-3, 3)
        rows.append([x])
        labels.append(1 if x > 0 else 0)
    model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
    history = model.fit(rows, labels, l2=0.01, lr=0.5, epochs=200)
    check("training loss decreases", history[-1] < history[0])
    check("fitted model separates the classes", model.predict_proba([2.0])[1] > model.predict_proba([-2.0])[1])

    d = model.to_dict()
    restored = MultinomialLogisticRegression.from_dict(d)
    check(
        "round-tripping through to_dict/from_dict preserves predictions",
        restored.predict_proba([1.5]) == model.predict_proba([1.5]),
    )

    model_a = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=7)
    model_b = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=7)
    model_a.fit(rows, labels, l2=0.01, lr=0.5, epochs=50)
    model_b.fit(rows, labels, l2=0.01, lr=0.5, epochs=50)
    check("fit is deterministic given a seed", model_a.to_dict() == model_b.to_dict())

    try:
        model.predict_proba([1.0, 2.0])
        check("predict_proba rejects a mismatched feature vector length", False)
    except ValueError:
        check("predict_proba rejects a mismatched feature vector length", True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
