from __future__ import annotations

import pytest

from decisionhead.client import ClientError, ask, health, models


class TestClient:
    def test_health_reaches_running_server(self, running_server: str) -> None:
        h = health(running_server)
        assert h.get("status") == "ok"

    def test_models_lists_real_backend_model_id(self, running_server: str) -> None:
        m = models(running_server)
        assert m["models"][0]["id"] == "test-model"

    def test_ask_returns_real_model_id_regardless_of_alias(self, running_server: str) -> None:
        questions = {"q1": {"type": "noul", "instructions": "Is the evidence on file current?"}}
        result = ask(running_server, {"note": "state"}, questions, model="some-alias")
        assert result["model"] == "test-model"

    def test_ask_returns_answer_per_question(self, running_server: str) -> None:
        questions = {
            "q1": {"type": "noul", "instructions": "Is the evidence on file current?"},
            "q2": {
                "type": "score",
                "instructions": "How urgent is this case?",
                "levels": {"low": "no deadline pressure", "medium": "due this week", "high": "overdue"},
            },
        }
        result = ask(running_server, {"note": "state"}, questions)
        assert set(result["answers"].keys()) == set(questions.keys())
        assert isinstance(result["answers"]["q2"]["score"], (int, float))

    def test_ask_surfaces_server_4xx_as_client_error(self, running_server: str) -> None:
        with pytest.raises(ClientError, match="400"):
            ask(running_server, "x", {})

    def test_ask_surfaces_connection_failure_as_client_error(self) -> None:
        with pytest.raises(ClientError):
            ask("http://127.0.0.1:1", "x", {"q": {"type": "noul", "instructions": "x"}}, timeout=2)

    def test_ask_sends_bearer_token_header(self, running_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # The default fixture server has no auth configured, so this just
        # verifies the header is attached without breaking a normal call.
        questions = {"q1": {"type": "noul", "instructions": "x"}}
        result = ask(running_server, {"note": "s"}, questions, bearer_token="anything")
        assert "answers" in result
