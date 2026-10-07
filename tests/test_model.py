from __future__ import annotations

import random

import pytest

from decisionhead import _numpy_opt
from decisionhead.model import MultinomialLogisticRegression, StandardScaler, softmax


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        probs = softmax([2.0, -1.0, 0.5, 3.3])
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_uniform_for_equal_logits(self) -> None:
        probs = softmax([1.0, 1.0, 1.0])
        assert all(abs(p - 1 / 3) < 1e-9 for p in probs)

    def test_numerically_stable_for_large_logits(self) -> None:
        probs = softmax([1000.0, 1000.0, 999.0])
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6


class TestStandardScaler:
    def test_fit_computes_mean_and_std(self) -> None:
        scaler = StandardScaler.fit([[0.0], [2.0], [4.0]])
        assert abs(scaler.means[0] - 2.0) < 1e-9
        assert scaler.stds[0] > 0

    def test_zero_variance_feature_does_not_divide_by_zero(self) -> None:
        scaler = StandardScaler.fit([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        assert scaler.stds[1] == 1.0
        transformed = scaler.transform([9.0, 5.0])
        assert transformed[1] == 0.0

    def test_transform_centers_and_scales(self) -> None:
        scaler = StandardScaler.fit([[0.0], [10.0]])
        assert abs(scaler.transform([5.0])[0]) < 1e-9

    def test_fit_on_zero_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="zero rows"):
            StandardScaler.fit([])

    def test_round_trips_through_dict(self) -> None:
        scaler = StandardScaler.fit([[1.0, 2.0], [3.0, 4.0]])
        restored = StandardScaler.from_dict(scaler.to_dict())
        assert restored.means == scaler.means
        assert restored.stds == scaler.stds


class TestMultinomialLogisticRegression:
    def _separable_dataset(self, n: int = 200, seed: int = 0) -> tuple[list[list[float]], list[int]]:
        rng = random.Random(seed)
        rows, labels = [], []
        for _ in range(n):
            x = rng.uniform(-3, 3)
            rows.append([x])
            labels.append(1 if x > 0 else 0)
        return rows, labels

    def test_predict_proba_sums_to_one(self) -> None:
        model = MultinomialLogisticRegression.initialize(n_classes=3, n_features=2, seed=0)
        probs = model.predict_proba([1.0, -1.0])
        assert abs(sum(probs) - 1.0) < 1e-9
        assert len(probs) == 3

    def test_predict_proba_rejects_wrong_length(self) -> None:
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=3, seed=0)
        with pytest.raises(ValueError, match="3-feature"):
            model.predict_proba([1.0, 2.0])

    def test_fit_reduces_loss(self) -> None:
        rows, labels = self._separable_dataset()
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
        history = model.fit(rows, labels, l2=0.01, lr=0.5, epochs=200)
        assert history[-1] < history[0]

    def test_fit_learns_a_separable_boundary(self) -> None:
        rows, labels = self._separable_dataset()
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
        model.fit(rows, labels, l2=0.01, lr=0.5, epochs=300)
        assert model.predict_proba([3.0])[1] > 0.9
        assert model.predict_proba([-3.0])[1] < 0.1

    def test_fit_is_deterministic_given_a_seed(self) -> None:
        rows, labels = self._separable_dataset()
        m1 = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=7)
        m2 = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=7)
        m1.fit(rows, labels, l2=0.05, lr=0.3, epochs=50)
        m2.fit(rows, labels, l2=0.05, lr=0.3, epochs=50)
        assert m1.to_dict() == m2.to_dict()

    def test_different_seeds_give_different_initial_weights(self) -> None:
        m1 = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=1)
        m2 = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=2)
        assert m1.weights != m2.weights

    def test_fit_rejects_mismatched_lengths(self) -> None:
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
        with pytest.raises(ValueError, match="same length"):
            model.fit([[1.0], [2.0]], [0], epochs=1)

    def test_fit_rejects_zero_rows(self) -> None:
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
        with pytest.raises(ValueError, match="zero rows"):
            model.fit([], [], epochs=1)

    def test_round_trips_through_dict(self) -> None:
        rows, labels = self._separable_dataset()
        model = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=0)
        model.fit(rows, labels, epochs=50)
        restored = MultinomialLogisticRegression.from_dict(model.to_dict())
        assert restored.predict_proba([1.5]) == model.predict_proba([1.5])

    @pytest.mark.skipif(not _numpy_opt.HAVE_NUMPY, reason="numpy is not installed")
    def test_numpy_and_pure_python_paths_agree_closely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows, labels = self._separable_dataset()

        model_np = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=3)
        model_np.fit(rows, labels, l2=0.05, lr=0.3, epochs=100)

        monkeypatch.setattr("decisionhead.model.HAVE_NUMPY", False)
        model_py = MultinomialLogisticRegression.initialize(n_classes=2, n_features=1, seed=3)
        model_py.fit(rows, labels, l2=0.05, lr=0.3, epochs=100)

        for a, b in zip(model_np.predict_proba([2.0]), model_py.predict_proba([2.0]), strict=True):
            assert abs(a - b) < 1e-3
