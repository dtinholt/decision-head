from __future__ import annotations

import json
import math
import random

import pytest

from decisionhead.backends import FakeBackend
from decisionhead.calibrate import (
    MIN_ITEMS_PER_TYPE,
    _ece_and_reliability,
    _nll_noul,
    brier_and_ece_multi,
    brier_and_ece_noul,
    calibrate,
    fit_temperature,
    main,
    read_labelled_jsonl,
)


@pytest.fixture
def overconfident_noul_pairs() -> list[tuple[float, int]]:
    rng = random.Random(42)
    pairs = []
    for _ in range(200):
        q = rng.random()
        y = 1 if rng.random() < q else 0
        p_over = min(max(0.5 + 2 * (q - 0.5), 0.02), 0.98)
        pairs.append((p_over, y))
    return pairs


class TestFitTemperature:
    def test_finds_t_greater_than_1_for_overconfident_predictor(self, overconfident_noul_pairs) -> None:
        t_fit = fit_temperature(overconfident_noul_pairs, "noul")
        assert t_fit > 1.0

    def test_fitted_t_reduces_nll(self, overconfident_noul_pairs) -> None:
        t_fit = fit_temperature(overconfident_noul_pairs, "noul")
        assert _nll_noul(overconfident_noul_pairs, t_fit) <= _nll_noul(overconfident_noul_pairs, 1.0)

    def test_multi_option_temperature_is_positive(self) -> None:
        rng = random.Random(1)
        options = ["A", "B", "C"]
        pairs = []
        for _ in range(200):
            truth = rng.choice(options)
            base = {o: rng.random() * 0.2 for o in options}
            base[truth] += 0.6
            total = sum(base.values())
            probs = {o: v / total for o, v in base.items()}
            pairs.append((probs, truth))
        t_multi = fit_temperature(pairs, "choice")
        assert t_multi > 0


class TestMetrics:
    def test_ece_and_reliability_has_requested_bin_count(self, overconfident_noul_pairs) -> None:
        result = brier_and_ece_noul(overconfident_noul_pairs, 1.0, bins=10)
        assert len(result["reliability"]) == 10

    def test_reliability_bins_with_data_report_stats(self, overconfident_noul_pairs) -> None:
        result = brier_and_ece_noul(overconfident_noul_pairs, 1.0)
        for b in result["reliability"]:
            if b["count"] > 0:
                assert b["mean_predicted"] is not None
                assert b["observed_frequency"] is not None

    def test_empty_scored_list_does_not_crash(self) -> None:
        ece, table = _ece_and_reliability([], bins=10)
        assert ece == 0.0
        assert len(table) == 10

    def test_multi_brier_is_finite(self) -> None:
        pairs = [({"A": 0.9, "B": 0.1}, "A"), ({"A": 0.2, "B": 0.8}, "B")]
        result = brier_and_ece_multi(pairs, 1.0)
        assert math.isfinite(result["brier"])


class TestCalibrateEndToEnd:
    def test_fewer_than_minimum_items_is_skipped(self) -> None:
        tiny_items = [
            {
                "state": "s",
                "questions": {"q1": {"type": "noul", "instructions": "is it true"}},
                "answers_true": {"q1": 1},
            }
            for _ in range(5)
        ]
        fb = FakeBackend(seed=9)
        result = calibrate(tiny_items, fb, max_workers=2)
        assert "noul" not in result["per_type"]
        assert any(qtype == "noul" and n == 5 for qtype, n in result["skipped"])

    def test_enough_items_fits_a_temperature(self) -> None:
        big_items = [
            {
                "state": f"synthetic state {i}",
                "questions": {"q1": {"type": "noul", "instructions": f"is condition {i} true"}},
                "answers_true": {"q1": i % 2},
            }
            for i in range(MIN_ITEMS_PER_TYPE + 10)
        ]
        fb = FakeBackend(seed=9)
        result = calibrate(big_items, fb, max_workers=4)
        assert "noul" in result["per_type"]
        assert result["per_type"]["noul"]["n"] == MIN_ITEMS_PER_TYPE + 10
        assert result["per_type"]["noul"]["T"] > 0

    def test_result_is_json_serializable(self) -> None:
        big_items = [
            {
                "state": f"s{i}",
                "questions": {"q1": {"type": "noul", "instructions": f"cond {i}"}},
                "answers_true": {"q1": i % 2},
            }
            for i in range(MIN_ITEMS_PER_TYPE)
        ]
        fb = FakeBackend(seed=3)
        result = calibrate(big_items, fb, max_workers=2)
        json.dumps(result)  # must not raise

    def test_question_without_truth_label_is_ignored(self) -> None:
        items = [
            {
                "state": "s",
                "questions": {
                    "q1": {"type": "noul", "instructions": "x"},
                    "q2": {"type": "noul", "instructions": "y"},
                },
                "answers_true": {"q1": 1},  # q2 has no truth
            }
            for _ in range(MIN_ITEMS_PER_TYPE)
        ]
        fb = FakeBackend(seed=4)
        result = calibrate(items, fb, max_workers=2)
        assert result["per_type"]["noul"]["n"] == MIN_ITEMS_PER_TYPE


class TestReadLabelledJsonl:
    def test_reads_one_object_per_line(self, tmp_path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n')
        items = read_labelled_jsonl(str(path))
        assert items == [{"a": 1}, {"a": 2}]

    def test_skips_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n')
        items = read_labelled_jsonl(str(path))
        assert len(items) == 2

    def test_invalid_json_line_raises_with_line_number(self, tmp_path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text('{"a": 1}\nnot json\n')
        with pytest.raises(ValueError, match="2"):
            read_labelled_jsonl(str(path))


class TestCLI:
    def test_dry_run_against_the_bundled_example_file(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        import pathlib

        example = pathlib.Path(__file__).resolve().parent.parent / "examples" / "calibration_example.jsonl"
        out_path = tmp_path / "calibration.json"
        code = main(["--labelled", str(example), "--dry", "--out", str(out_path)])
        assert code == 0
        assert out_path.exists()
        written = json.loads(out_path.read_text())
        assert "noul" in written["per_type"]

    def test_missing_labelled_flag_errors(self) -> None:
        with pytest.raises(SystemExit):
            main(["--dry"])

    def test_missing_base_and_model_without_dry_errors(self, tmp_path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text('{"state": "s", "questions": {}, "answers_true": {}}\n')
        with pytest.raises(SystemExit):
            main(["--labelled", str(path)])
