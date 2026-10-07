from __future__ import annotations

import pytest

from decisionhead.embeddings import FakeEmbeddingsBackend
from decisionhead.features import (
    DECISION_CLASSES,
    FeatureConfig,
    extract_features,
    feature_names,
)

STATE_TWO_OPEN: dict = {
    "requirements": [
        {"item_id": "a", "category": "incorporation"},
        {"item_id": "b", "category": "tax"},
    ],
    "open_items": ["a", "b"],
    "findings": [{"item_id": "a", "status": "unmet"}],
    "history": [{"action": "CONTINUE"}, {"action": "CONTINUE"}],
    "budget": {"steps": 8, "expert_calls": 1, "requests": 2},
    "steps_used": 4,
}


class TestFeatureConfig:
    def test_defaults_enable_structured_and_prior_answers(self) -> None:
        cfg = FeatureConfig()
        assert cfg.use_structured is True
        assert cfg.use_prior_answers is True
        assert cfg.use_embedding is False

    def test_embedding_without_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_dim"):
            FeatureConfig(use_embedding=True)

    def test_at_least_one_group_required(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=False)

    def test_round_trips_through_dict(self) -> None:
        cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=True, embedding_dim=16)
        restored = FeatureConfig.from_dict(cfg.to_dict())
        assert restored == cfg


class TestStructuredFeatures:
    def test_vector_length_matches_names(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        vector = extract_features(STATE_TWO_OPEN, cfg)
        assert len(vector) == len(feature_names(cfg))

    def test_open_and_closed_counts(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        vector = extract_features(STATE_TWO_OPEN, cfg)
        names = feature_names(cfg)
        assert vector[names.index("n_open")] == 2.0
        assert vector[names.index("n_closed")] == 0.0

    def test_empty_state_is_all_defaults(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        vector = extract_features({}, cfg)
        assert len(vector) == len(feature_names(cfg))
        assert all(isinstance(v, float) for v in vector)

    def test_trailing_continue_count(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        vector = extract_features(STATE_TWO_OPEN, cfg)
        names = feature_names(cfg)
        assert vector[names.index("trailing_continue_count")] == 2.0

    def test_flags_are_counted_not_enumerated(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {**STATE_TWO_OPEN, "flags": {"urgent": True, "vip": False, "escalated": True}}
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("n_flags")] == 3.0
        assert vector[names.index("n_true_flags")] == 2.0


class TestBookkeepingFeatures:
    def test_retrieve_tried_for_focus_item(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "a", "category": "tax"}],
            "open_items": ["a"],
            "history": [{"action": "RETRIEVE", "args": {"category": "tax"}}],
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("retrieve_tried_for_focus_item")] == 1.0

    def test_retrieve_for_a_different_category_does_not_count(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "a", "category": "tax"}],
            "open_items": ["a"],
            "history": [{"action": "RETRIEVE", "args": {"category": "insurance"}}],
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("retrieve_tried_for_focus_item")] == 0.0

    def test_request_tried_for_focus_item(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "a"}],
            "open_items": ["a"],
            "history": [{"action": "REQUEST", "args": {"item_id": "a"}}],
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("request_tried_for_focus_item")] == 1.0

    def test_focus_finding_status_one_hot(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "a"}],
            "open_items": ["a"],
            "findings": [{"item_id": "a", "status": "uncertain"}],
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("focus_finding_status_uncertain")] == 1.0
        assert vector[names.index("focus_finding_status_met")] == 0.0
        assert vector[names.index("focus_finding_status_missing")] == 0.0

    def test_focus_finding_status_missing_when_no_finding(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {"requirements": [{"item_id": "a"}], "open_items": ["a"], "findings": []}
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("focus_finding_status_missing")] == 1.0

    def test_n_uncertain_findings_counts_across_all_findings(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "a"}, {"item_id": "b"}],
            "open_items": ["a", "b"],
            "findings": [{"item_id": "a", "status": "uncertain"}, {"item_id": "b", "status": "uncertain"}],
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("n_uncertain_findings")] == 2.0

    def test_budget_exhaustion_flags(self) -> None:
        cfg = FeatureConfig(use_prior_answers=False, use_embedding=False)
        state = {"budget": {"steps": 0, "expert_calls": 0, "requests": 1}}
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("step_budget_exhausted")] == 1.0
        assert vector[names.index("expert_call_budget_exhausted")] == 1.0
        assert vector[names.index("request_budget_exhausted")] == 0.0


class TestPriorAnswersFeatures:
    def test_absent_prior_answers_gives_neutral_defaults(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        vector = extract_features({}, cfg)
        names = feature_names(cfg)
        assert vector[names.index("prior_answers_present")] == 0.0
        assert vector[names.index("prior_p_onfile")] == 0.5

    def test_reads_item_scoped_wire_shape(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "cert"}],
            "open_items": ["cert"],
            "prior_answers": {
                "cert::q_onfile": {"type": "noul", "noul": 0.83},
                "cert::q_supply": {"type": "noul", "noul": 0.1},
                "cert::q_interp": {"type": "noul", "noul": 0.05},
            },
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("prior_answers_present")] == 1.0
        assert vector[names.index("prior_p_onfile")] == 0.83
        assert vector[names.index("prior_p_supply")] == 0.1
        assert vector[names.index("prior_p_interp")] == 0.05

    def test_zero_probability_is_not_treated_as_missing(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "cert"}],
            "open_items": ["cert"],
            "prior_answers": {"cert::q_onfile": {"type": "noul", "noul": 0.0}},
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("prior_p_onfile")] == 0.0

    def test_legacy_pre_rename_keys_are_accepted(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        state = {
            "requirements": [{"item_id": "cert"}],
            "open_items": ["cert"],
            "prior_answers": {"cert::met": {"type": "noul", "noul": 0.42}},
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("prior_p_onfile")] == 0.42

    def test_next_step_probabilities_aggregate_by_class(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        state = {
            "prior_answers": {
                "next_step": {
                    "type": "choice",
                    "confidence": 0.6,
                    "probabilities": {"RETRIEVE": 0.5, "REQUEST": 0.1, "CONTINUE": 0.4},
                }
            }
        }
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert abs(vector[names.index("prior_class_prob_INFO")] - 0.6) < 1e-9
        assert abs(vector[names.index("prior_class_prob_REASON")] - 0.4) < 1e-9
        assert set(DECISION_CLASSES) == {"INFO", "REASON", "DONE", "BLOCKED"}

    def test_no_open_items_means_no_focus_item_lookup(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_embedding=False)
        state = {"requirements": [{"item_id": "a"}], "open_items": [], "prior_answers": {}}
        vector = extract_features(state, cfg)
        names = feature_names(cfg)
        assert vector[names.index("prior_p_onfile")] == 0.5


class TestEmbeddingFeatures:
    def test_uses_the_given_embedder(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=True, embedding_dim=6)
        vector = extract_features({"note": "hello"}, cfg, embedder=FakeEmbeddingsBackend(dim=6))
        assert len(vector) == 6

    def test_missing_embedder_raises(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=True, embedding_dim=6)
        with pytest.raises(ValueError, match="embedder"):
            extract_features({"note": "hello"}, cfg, embedder=None)

    def test_dimension_mismatch_raises(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=True, embedding_dim=4)
        with pytest.raises(ValueError, match="dim"):
            extract_features({"note": "hello"}, cfg, embedder=FakeEmbeddingsBackend(dim=9))

    def test_embedding_is_deterministic_for_the_same_state(self) -> None:
        cfg = FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=True, embedding_dim=6)
        embedder = FakeEmbeddingsBackend(dim=6)
        v1 = extract_features({"note": "same text"}, cfg, embedder=embedder)
        v2 = extract_features({"note": "same text"}, cfg, embedder=embedder)
        assert v1 == v2


class TestAllGroupsTogether:
    def test_combined_vector_is_the_concatenation(self) -> None:
        cfg_all = FeatureConfig(use_structured=True, use_prior_answers=True, use_embedding=True, embedding_dim=3)
        cfg_struct = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
        cfg_prior = FeatureConfig(use_structured=False, use_prior_answers=True, use_embedding=False)
        embedder = FakeEmbeddingsBackend(dim=3)

        combined = extract_features(STATE_TWO_OPEN, cfg_all, embedder=embedder)
        struct_only = extract_features(STATE_TWO_OPEN, cfg_struct)
        prior_only = extract_features(STATE_TWO_OPEN, cfg_prior)
        embed_only = extract_features(STATE_TWO_OPEN, FeatureConfig(False, False, True, 3), embedder=embedder)

        assert combined == [*struct_only, *prior_only, *embed_only]
