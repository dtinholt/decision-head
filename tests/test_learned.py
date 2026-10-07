from __future__ import annotations

from pathlib import Path

import pytest

from decisionhead.core import QuestionError
from decisionhead.features import DECISION_CLASSES, FeatureConfig, feature_names
from decisionhead.learned import (
    LearnedBackendAdapter,
    LearnedModel,
    answer_questions_learned,
)
from decisionhead.model import MultinomialLogisticRegression, StandardScaler

STATE = {
    "requirements": [{"item_id": "a"}],
    "open_items": ["a"],
    "history": [],
    "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
    "steps_used": 0,
}


def _model(temperature: float = 1.0) -> LearnedModel:
    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    names = feature_names(cfg)
    classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
    scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
    return LearnedModel(
        classifier=classifier,
        scaler=scaler,
        feature_config=cfg,
        classes=["BLOCKED", "DONE", "INFO", "REASON"],
        temperature=temperature,
        metadata={"name": "test-model"},
    )


class TestLearnedModel:
    def test_class_probabilities_sums_to_one(self) -> None:
        model = _model()
        probs = model.class_probabilities(STATE)
        assert set(probs) == {"BLOCKED", "DONE", "INFO", "REASON"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_inconsistent_feature_count_is_rejected_at_construction(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=3, seed=0)
        scaler = StandardScaler(means=[0.0] * 3, stds=[1.0] * 3)
        with pytest.raises(ValueError, match="features"):
            LearnedModel(classifier=classifier, scaler=scaler, feature_config=cfg, classes=list(DECISION_CLASSES))

    def test_class_count_mismatch_is_rejected(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        names = feature_names(cfg)
        classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
        scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
        with pytest.raises(ValueError, match="classes"):
            LearnedModel(classifier=classifier, scaler=scaler, feature_config=cfg, classes=["A", "B"])

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        model = _model(temperature=1.7)
        path = tmp_path / "model.json"
        model.save(str(path))
        restored = LearnedModel.load(str(path))
        assert restored.class_probabilities(STATE) == model.class_probabilities(STATE)
        assert restored.temperature == 1.7
        assert restored.classes == model.classes

    def test_uses_prior_features_false_is_accepted_when_config_agrees(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        names = feature_names(cfg)
        classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
        scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
        model = LearnedModel(
            classifier=classifier,
            scaler=scaler,
            feature_config=cfg,
            classes=list(DECISION_CLASSES),
            metadata={"uses_prior_features": False},
        )
        assert model.feature_config.use_prior_answers is False

    def test_uses_prior_features_mismatch_is_rejected(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        names = feature_names(cfg)
        classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
        scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
        with pytest.raises(ValueError, match="uses_prior_features"):
            LearnedModel(
                classifier=classifier,
                scaler=scaler,
                feature_config=cfg,
                classes=list(DECISION_CLASSES),
                # claims prior features are used, but feature_config says otherwise
                metadata={"uses_prior_features": True},
            )

    def test_serving_a_no_prior_features_model_ignores_prior_answers_in_state(self) -> None:
        """A model trained with use_prior_answers=False must produce the
        SAME answer whether or not the request's state happens to carry a
        prior_answers block -- it never reads it."""
        model = _model()  # use_prior_answers=False
        state_with_prior = {**STATE, "prior_answers": {"cert::met": {"type": "noul", "noul": 0.99}}}
        assert model.class_probabilities(STATE) == model.class_probabilities(state_with_prior)

    def test_serving_never_calls_a_backend(self) -> None:
        """answer_questions_learned takes no backend argument at all --
        there is no fast/zero-shot head it could call."""
        import inspect

        params = inspect.signature(answer_questions_learned).parameters
        assert "backend" not in params


class TestActionProbabilities:
    def test_direct_class_options(self) -> None:
        model = _model()
        result = answer_questions_learned(
            STATE,
            {"q": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)}},
            model,
        )
        probs = result["answers"]["q"]["probabilities"]
        assert set(probs) == set(DECISION_CLASSES)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_action_subset_options_are_mapped_by_class(self) -> None:
        model = _model()
        result = answer_questions_learned(
            STATE,
            {"q": {"type": "choice", "instructions": "x", "options": ["RETRIEVE", "REQUEST"]}},
            model,
        )
        probs = result["answers"]["q"]["probabilities"]
        assert set(probs) == {"RETRIEVE", "REQUEST"}
        # both actions belong to the INFO class, so they split its mass evenly
        assert abs(probs["RETRIEVE"] - probs["REQUEST"]) < 1e-9
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_unrecognized_option_set_raises(self) -> None:
        model = _model()
        with pytest.raises(QuestionError, match="options"):
            answer_questions_learned(
                STATE, {"q": {"type": "choice", "instructions": "x", "options": ["FOO", "BAR"]}}, model
            )

    def test_mixed_class_and_action_names_raises(self) -> None:
        model = _model()
        with pytest.raises(QuestionError):
            answer_questions_learned(
                STATE, {"q": {"type": "choice", "instructions": "x", "options": ["INFO", "RETRIEVE"]}}, model
            )


class TestAnswerQuestionsLearned:
    def test_noul_question_is_rejected(self) -> None:
        model = _model()
        with pytest.raises(QuestionError, match="choice"):
            answer_questions_learned(STATE, {"q": {"type": "noul", "instructions": "x"}}, model)

    def test_score_question_is_rejected(self) -> None:
        model = _model()
        with pytest.raises(QuestionError, match="choice"):
            answer_questions_learned(
                STATE, {"q": {"type": "score", "instructions": "x", "levels": ["low", "high"]}}, model
            )

    def test_empty_questions_rejected(self) -> None:
        model = _model()
        with pytest.raises(QuestionError):
            answer_questions_learned(STATE, {}, model)

    def test_usage_carries_no_llm_tokens(self) -> None:
        model = _model()
        result = answer_questions_learned(
            STATE, {"q": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)}}, model
        )
        assert result["usage"] == {"input_tokens": None, "output_tokens": None}

    def test_audit_row_shape_matches_core_answer_questions(self) -> None:
        model = _model()
        result = answer_questions_learned(
            STATE, {"q": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)}}, model
        )
        row = result["audit"][0]
        assert {"request_hash", "question_id", "model", "backend", "unparsed_mass", "calibrated", "latency_s"} <= set(
            row.keys()
        )

    def test_calibrated_flag_reflects_temperature(self) -> None:
        model_t1 = _model(temperature=1.0)
        model_t2 = _model(temperature=2.0)
        q = {"q": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)}}
        assert answer_questions_learned(STATE, q, model_t1)["audit"][0]["calibrated"] is False
        assert answer_questions_learned(STATE, q, model_t2)["audit"][0]["calibrated"] is True

    def test_multiple_questions_share_one_request_hash(self) -> None:
        model = _model()
        q = {
            "q1": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)},
            "q2": {"type": "choice", "instructions": "y", "options": list(DECISION_CLASSES)},
        }
        result = answer_questions_learned(STATE, q, model)
        hashes = {row["request_hash"] for row in result["audit"]}
        assert len(hashes) == 1


class TestLearnedBackendAdapter:
    def test_exposes_model_name(self) -> None:
        adapter = LearnedBackendAdapter(_model())
        assert adapter.model == "test-model"

    def test_token_distribution_raises(self) -> None:
        adapter = LearnedBackendAdapter(_model())
        with pytest.raises(NotImplementedError):
            adapter.token_distribution([], "x", ["A", "B"])

    def test_sample_completions_raises(self) -> None:
        adapter = LearnedBackendAdapter(_model())
        with pytest.raises(NotImplementedError):
            adapter.sample_completions([], "x", ["A", "B"], n=1, temperature=1.0, max_tokens=8)
