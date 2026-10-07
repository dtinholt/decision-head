from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from decisionhead.features import FeatureConfig
from decisionhead.learned import LearnedModel
from decisionhead.train import (
    _cluster_ids,
    _kfold_case_clusters,
    build_arg_parser,
    main,
    train_and_evaluate,
)


def _synthetic_traces(n_cases: int = 20, seed: int = 0) -> list[dict]:
    """A trivially learnable synthetic trace set: label depends only on
    the number of open items, so any reasonable classifier should clear
    the majority baseline comfortably. Two checkpoints per case (same
    case_id), so case-clustered CV has something to cluster."""
    rng = random.Random(seed)
    traces = []
    for case_num in range(n_cases):
        n_open = rng.randint(0, 3)
        label = "DONE" if n_open == 0 else ("INFO" if n_open % 2 == 0 else "REASON")
        for step in range(2):
            traces.append(
                {
                    "state": {
                        "requirements": [{"item_id": f"r{i}"} for i in range(3)],
                        "open_items": [f"r{i}" for i in range(n_open)],
                        "history": [],
                        "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
                        "steps_used": step,
                    },
                    "label": label,
                    "meta": {"case_id": f"case-{case_num}"},
                }
            )
    return traces


class TestClusterIds:
    def test_uses_meta_case_id(self) -> None:
        traces = [
            {"state": {}, "label": "A", "meta": {"case_id": "c1"}},
            {"state": {}, "label": "A", "meta": {"case_id": "c1"}},
        ]
        ids = _cluster_ids(traces)
        assert ids[0] == ids[1] == "c1"

    def test_missing_case_id_is_its_own_singleton_cluster(self) -> None:
        traces = [{"state": {}, "label": "A"}, {"state": {}, "label": "A"}]
        ids = _cluster_ids(traces)
        assert ids[0] != ids[1]


class TestKfoldCaseClusters:
    def test_every_index_appears_exactly_once(self) -> None:
        ids = [f"c{i}" for i in range(10)]
        folds = _kfold_case_clusters(ids, k=5, seed=0)
        flattened = sorted(i for fold in folds for i in fold)
        assert flattened == list(range(10))

    def test_same_cluster_never_split_across_folds(self) -> None:
        ids = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]
        folds = _kfold_case_clusters(ids, k=2, seed=0)
        for fold in folds:
            clusters_in_fold = {ids[i] for i in fold}
            for cluster in clusters_in_fold:
                all_positions = {i for i, cid in enumerate(ids) if cid == cluster}
                assert all_positions <= set(fold)

    def test_deterministic_given_a_seed(self) -> None:
        ids = [f"c{i}" for i in range(12)]
        assert _kfold_case_clusters(ids, k=4, seed=3) == _kfold_case_clusters(ids, k=4, seed=3)


class TestTrainAndEvaluate:
    def test_beats_majority_baseline_on_learnable_signal(self) -> None:
        traces = _synthetic_traces()
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=150)
        assert result.report["decision_class_accuracy_cv"] >= result.report["majority_baseline_accuracy"]

    def test_report_is_marked_dev_only(self) -> None:
        traces = _synthetic_traces()
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        assert result.report["dev_only"] is True
        assert "dev" in result.report["note"].lower() or "eval" in result.report["note"].lower()

    def test_case_clustering_prevents_leakage_report_fields(self) -> None:
        traces = _synthetic_traces(n_cases=20)
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        assert result.report["n_cases"] == 20
        assert result.report["n_traces"] == 40
        assert result.report["cv_folds_used"] == 5

    def test_per_class_recall_covers_every_declared_class(self) -> None:
        traces = _synthetic_traces()
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        assert set(result.report["per_class_recall_cv"]) == set(result.report["classes"])

    def test_saved_model_reproduces_report_feature_config(self) -> None:
        traces = _synthetic_traces()
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        assert result.model.feature_config == cfg
        assert result.model.classes == result.report["classes"]

    def test_deterministic_given_a_seed(self) -> None:
        traces = _synthetic_traces()
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        r1 = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        r2 = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=50)
        assert r1.model.classifier.to_dict() == r2.model.classifier.to_dict()
        assert r1.report["decision_class_accuracy_cv"] == r2.report["decision_class_accuracy_cv"]

    def test_zero_traces_rejected(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        with pytest.raises(ValueError, match="zero traces"):
            train_and_evaluate([], feature_config=cfg)

    def test_single_label_rejected(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        with pytest.raises(ValueError, match="2 distinct labels"):
            train_and_evaluate([{"state": {}, "label": "ONLY"}], feature_config=cfg)

    def test_single_cluster_skips_cross_validation_gracefully(self) -> None:
        traces = [
            {"state": {"open_items": []}, "label": "DONE", "meta": {"case_id": "c1"}},
            {"state": {"open_items": ["a"]}, "label": "INFO", "meta": {"case_id": "c1"}},
        ]
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=20)
        assert result.report["decision_class_accuracy_cv"] is None
        assert result.report["cv_folds_used"] == 1


class TestTrainCLI:
    def test_end_to_end_writes_model_and_report(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        model_path = tmp_path / "model.json"
        report_path = tmp_path / "report.json"

        code = main(
            [
                "--traces",
                str(traces_path),
                "--cv",
                "5",
                "--seed",
                "0",
                "--epochs",
                "50",
                "--out",
                str(model_path),
                "--report",
                str(report_path),
            ]
        )
        assert code == 0
        assert model_path.exists()
        assert report_path.exists()

        loaded_model = LearnedModel.load(str(model_path))
        report = json.loads(report_path.read_text())
        assert report["dev_only"] is True
        assert loaded_model.classes == report["classes"]

    def test_missing_traces_file_is_a_clean_cli_error(self) -> None:
        with pytest.raises(SystemExit):
            main(["--traces", "/does/not/exist.jsonl"])

    def test_feature_group_flags_are_wired(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        report_path = tmp_path / "report.json"
        code = main(
            ["--traces", str(traces_path), "--no-prior-answers", "--epochs", "20", "--report", str(report_path)]
        )
        assert code == 0
        report = json.loads(report_path.read_text())
        assert report["feature_config"]["use_prior_answers"] is False
        assert report["feature_config"]["use_structured"] is True

    def test_no_prior_answers_records_uses_prior_features_false_in_model_metadata(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        model_path = tmp_path / "model.json"
        code = main(
            ["--traces", str(traces_path), "--no-prior-answers", "--epochs", "20", "--out", str(model_path)]
        )
        assert code == 0
        loaded = LearnedModel.load(str(model_path))
        assert loaded.metadata["uses_prior_features"] is False
        assert loaded.feature_config.use_prior_answers is False

    def test_no_prior_features_is_an_alias_for_no_prior_answers(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        model_path = tmp_path / "model.json"
        code = main(
            ["--traces", str(traces_path), "--no-prior-features", "--epochs", "20", "--out", str(model_path)]
        )
        assert code == 0
        loaded = LearnedModel.load(str(model_path))
        assert loaded.metadata["uses_prior_features"] is False

    def test_default_training_records_uses_prior_features_true(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        model_path = tmp_path / "model.json"
        code = main(["--traces", str(traces_path), "--epochs", "20", "--out", str(model_path)])
        assert code == 0
        loaded = LearnedModel.load(str(model_path))
        assert loaded.metadata["uses_prior_features"] is True

    def test_use_embedding_without_dim_is_a_clean_cli_error(self, tmp_path: Path) -> None:
        traces = _synthetic_traces()
        traces_path = tmp_path / "traces.jsonl"
        traces_path.write_text("\n".join(json.dumps(t) for t in traces))
        with pytest.raises(SystemExit):
            main(["--traces", str(traces_path), "--use-embedding"])


class TestBuildArgParser:
    def test_traces_is_required(self) -> None:
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--traces", "x.jsonl"])
        assert args.cv == 5
        assert args.seed == 0
        assert args.use_embedding is False
