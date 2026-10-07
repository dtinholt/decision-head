from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from decisionhead.config import ServerConfig
from decisionhead.features import DECISION_CLASSES, FeatureConfig
from decisionhead.learned import LearnedModel
from decisionhead.model import MultinomialLogisticRegression, StandardScaler
from decisionhead.server import resolve_effective_mode, run_server

STATE = {
    "requirements": [{"item_id": "a"}],
    "open_items": ["a"],
    "history": [],
    "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
    "steps_used": 0,
}


def _write_model(path: Path) -> None:
    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    from decisionhead.features import feature_names

    names = feature_names(cfg)
    classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
    scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
    model = LearnedModel(
        classifier=classifier,
        scaler=scaler,
        feature_config=cfg,
        classes=["BLOCKED", "DONE", "INFO", "REASON"],
        metadata={"name": "test-learned-model"},
    )
    model.save(str(path))


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(base: str, path: str) -> tuple[int, str]:
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


@pytest.fixture
def learned_server(tmp_path: Path):
    model_path = tmp_path / "model.json"
    _write_model(model_path)
    cfg = ServerConfig(mode="learned", head_model_path=str(model_path), port=0)
    httpd: ThreadingHTTPServer = run_server(cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class TestRunServerLearnedMode:
    def test_requires_head_model_path(self) -> None:
        cfg = ServerConfig(mode="learned", port=0)
        with pytest.raises(ValueError, match="head-model-path"):
            run_server(cfg)

    def test_does_not_require_base_or_model(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.json"
        _write_model(model_path)
        cfg = ServerConfig(mode="learned", head_model_path=str(model_path), port=0)
        httpd = run_server(cfg)
        httpd.server_close()

    def test_missing_model_file_raises(self, tmp_path: Path) -> None:
        cfg = ServerConfig(mode="learned", head_model_path=str(tmp_path / "nope.json"), port=0)
        with pytest.raises(OSError):
            run_server(cfg)


class TestLearnedModeEndToEnd:
    def test_health_reports_the_model_name(self, learned_server: str) -> None:
        status, body = _get(learned_server, "/health")
        assert status == 200

    def test_models_endpoint_reports_learned_model_name(self, learned_server: str) -> None:
        status, body = _get(learned_server, "/v1/models")
        parsed = json.loads(body)
        assert parsed["models"][0]["id"] == "test-learned-model"

    def test_ready_is_always_true_for_a_learned_backend(self, learned_server: str) -> None:
        status, body = _get(learned_server, "/ready")
        parsed = json.loads(body)
        assert parsed["ready"] is True

    def test_answers_a_choice_question_over_decision_classes(self, learned_server: str) -> None:
        req = {
            "state": STATE,
            "model": "jev-latest",
            "questions": {
                "next_step": {"type": "choice", "instructions": "which class?", "options": list(DECISION_CLASSES)}
            },
        }
        status, body = _post(learned_server, "/v1/systemone", req)
        assert status == 200
        probs = body["answers"]["next_step"]["probabilities"]
        assert set(probs) == set(DECISION_CLASSES)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert body["usage"] == {"input_tokens": None, "output_tokens": None}

    def test_rejects_a_noul_question(self, learned_server: str) -> None:
        req = {
            "state": STATE,
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "is it ready?"}},
        }
        status, body = _post(learned_server, "/v1/systemone", req)
        assert status == 400
        assert "choice" in body["error"]["message"]

    def test_model_field_in_response_reports_learned_model_name(self, learned_server: str) -> None:
        req = {
            "state": STATE,
            "model": "jev-latest",
            "questions": {"q": {"type": "choice", "instructions": "x", "options": list(DECISION_CLASSES)}},
        }
        status, body = _post(learned_server, "/v1/systemone", req)
        assert body["model"] == "test-learned-model"


class TestResolveEffectiveMode:
    def test_learned_is_an_allowed_mode(self) -> None:
        cfg = ServerConfig(mode="fast", mode_override_allowed=True)
        assert resolve_effective_mode(cfg, {"mode": "learned"}) == "learned"

    def test_configured_learned_mode_is_the_default(self) -> None:
        cfg = ServerConfig(mode="learned")
        assert resolve_effective_mode(cfg, {}) == "learned"
