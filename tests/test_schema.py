from __future__ import annotations

import pytest

from decisionhead.core import QuestionError
from decisionhead.schema import build_openapi_document, validate_envelope

VALID_BODY = {
    "state": "some state",
    "model": "jev-latest",
    "questions": {"q1": {"type": "noul", "instructions": "is it true"}},
}


class TestValidateEnvelope:
    def test_valid_body_passes(self) -> None:
        validate_envelope(VALID_BODY, max_questions=10)

    def test_non_dict_body_rejected(self) -> None:
        with pytest.raises(QuestionError):
            validate_envelope("not a dict", max_questions=10)

    def test_missing_state_rejected(self) -> None:
        body = {"model": "m", "questions": VALID_BODY["questions"]}
        with pytest.raises(QuestionError, match="state"):
            validate_envelope(body, max_questions=10)

    def test_empty_string_state_rejected(self) -> None:
        body = {**VALID_BODY, "state": ""}
        with pytest.raises(QuestionError):
            validate_envelope(body, max_questions=10)

    def test_state_wrong_type_rejected(self) -> None:
        body = {**VALID_BODY, "state": 42}
        with pytest.raises(QuestionError):
            validate_envelope(body, max_questions=10)

    def test_state_as_object_accepted(self) -> None:
        body = {**VALID_BODY, "state": {"a": 1}}
        validate_envelope(body, max_questions=10)

    def test_state_as_list_accepted(self) -> None:
        body = {**VALID_BODY, "state": [1, 2, 3]}
        validate_envelope(body, max_questions=10)

    def test_missing_model_rejected(self) -> None:
        body = {"state": "s", "questions": VALID_BODY["questions"]}
        with pytest.raises(QuestionError, match="model"):
            validate_envelope(body, max_questions=10)

    def test_blank_model_rejected(self) -> None:
        body = {**VALID_BODY, "model": "   "}
        with pytest.raises(QuestionError):
            validate_envelope(body, max_questions=10)

    def test_missing_questions_rejected(self) -> None:
        body = {"state": "s", "model": "m"}
        with pytest.raises(QuestionError, match="questions"):
            validate_envelope(body, max_questions=10)

    def test_empty_questions_rejected(self) -> None:
        body = {**VALID_BODY, "questions": {}}
        with pytest.raises(QuestionError):
            validate_envelope(body, max_questions=10)

    def test_questions_not_a_dict_rejected(self) -> None:
        body = {**VALID_BODY, "questions": ["nope"]}
        with pytest.raises(QuestionError):
            validate_envelope(body, max_questions=10)

    def test_too_many_questions_rejected(self) -> None:
        body = {**VALID_BODY, "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(5)}}
        with pytest.raises(QuestionError, match="limit"):
            validate_envelope(body, max_questions=3)

    def test_exactly_at_limit_is_accepted(self) -> None:
        body = {**VALID_BODY, "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(3)}}
        validate_envelope(body, max_questions=3)


class TestOpenAPIDocument:
    def test_has_core_paths(self) -> None:
        doc = build_openapi_document(version="0.1.0")
        for path in ("/v1/systemone", "/health", "/ready", "/metrics", "/openapi.json"):
            assert path in doc["paths"]

    def test_reports_the_given_version(self) -> None:
        doc = build_openapi_document(version="9.9.9")
        assert doc["info"]["version"] == "9.9.9"

    def test_systemone_post_documents_error_statuses(self) -> None:
        doc = build_openapi_document(version="0.1.0")
        responses = doc["paths"]["/v1/systemone"]["post"]["responses"]
        for status in ("200", "400", "413", "415", "429", "502", "504"):
            assert status in responses

    def test_is_json_serializable(self) -> None:
        import json

        doc = build_openapi_document(version="0.1.0")
        json.dumps(doc)  # must not raise

    def test_openapi_version_field(self) -> None:
        doc = build_openapi_document(version="0.1.0")
        assert doc["openapi"].startswith("3.0")
