"""Tests for the reference integration loop in ``examples/workflow_loop.py``
-- the DecisionHeadClient, the deterministic-bookkeeping engine, and the
end-to-end ``--dry`` CLI path. FakeBackend only; no network.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import examples.workflow_loop as workflow_loop
from decisionhead.traces import validate_trace


class TestDecisionHeadClient:
    def test_health_reaches_running_server(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        assert client.health().get("status") == "ok"

    def test_ready_reaches_running_server(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        assert client.ready().get("ready") is True

    def test_ask_returns_real_model_id_regardless_of_alias(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server, model="some-alias")
        questions = {"q1": {"type": "noul", "instructions": "Is the evidence on file current?"}}
        result = client.ask({"note": "state"}, questions)
        assert result["model"] == "test-model"

    def test_ask_returns_answer_per_question(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        questions = {
            "q1": {"type": "noul", "instructions": "Is the evidence on file current?"},
            "next_step": {
                "type": "choice",
                "instructions": "What next?",
                "criteria": {"RETRIEVE": "pull it", "CONTINUE": "keep going"},
            },
        }
        result = client.ask({"note": "state"}, questions)
        assert set(result["answers"].keys()) == set(questions.keys())

    def test_ask_surfaces_server_4xx_as_decisionhead_error(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        with pytest.raises(workflow_loop.DecisionHeadError, match="400"):
            client.ask("x", {})

    def test_ask_surfaces_connection_failure_as_decisionhead_error(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://127.0.0.1:1", timeout=2)
        with pytest.raises(workflow_loop.DecisionHeadError):
            client.ask("x", {"q": {"type": "noul", "instructions": "x"}})

    def test_bearer_token_sent_as_authorization_header(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://example.invalid", bearer_token="FAKEKEY-test-bearer")  # noqa: S106
        headers = client._headers()
        assert headers["Authorization"] == "Bearer FAKEKEY-test-bearer"

    def test_no_bearer_token_omits_authorization_header(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://example.invalid")
        assert "Authorization" not in client._headers()


class TestMakeSyntheticCase:
    def test_two_requirements_both_open(self) -> None:
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(0))
        assert ledger.open_items == ["coi", "w9"]
        assert set(hidden_truth.keys()) == {"coi", "w9"}

    def test_deterministic_given_a_seed(self) -> None:
        ledger1, truth1 = workflow_loop.make_synthetic_case("case-x", random.Random(5))
        ledger2, truth2 = workflow_loop.make_synthetic_case("case-x", random.Random(5))
        assert {k: v.resolution for k, v in truth1.items()} == {k: v.resolution for k, v in truth2.items()}
        assert ledger1.requirements == ledger2.requirements


class TestApplyGuardrail:
    def test_claim_complete_with_open_items_is_downgraded_to_continue(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://unused.invalid")
        engine = workflow_loop.WorkflowEngine(client)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(0))
        assert ledger.open_items  # still has open requirements
        applied = engine._apply(ledger, "CLAIM_COMPLETE", ledger.open_items[0], hidden_truth)
        assert applied == "CONTINUE"
        assert ledger.status == "open"

    def test_claim_complete_with_no_open_items_closes_the_case(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://unused.invalid")
        engine = workflow_loop.WorkflowEngine(client)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(0))
        ledger.open_items = []
        applied = engine._apply(ledger, "CLAIM_COMPLETE", None, hidden_truth)
        assert applied == "CLAIM_COMPLETE"
        assert ledger.status == "done"

    def test_handoff_escalates_regardless_of_open_items(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://unused.invalid")
        engine = workflow_loop.WorkflowEngine(client)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(0))
        applied = engine._apply(ledger, "HANDOFF", ledger.open_items[0], hidden_truth)
        assert applied == "HANDOFF"
        assert ledger.status == "escalated"

    def test_exhausting_request_budget_forces_handoff(self) -> None:
        client = workflow_loop.DecisionHeadClient("http://unused.invalid")
        engine = workflow_loop.WorkflowEngine(client)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(0))
        ledger.budget["requests"] = 0
        applied = engine._apply(ledger, "REQUEST", ledger.open_items[0], hidden_truth)
        assert applied == "HANDOFF"
        assert ledger.status == "escalated"


class TestWorkflowEngineRunCase:
    def test_run_case_against_running_server_reaches_a_terminal_status(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        engine = workflow_loop.WorkflowEngine(client, max_steps=12)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(1))
        trace_rows: list[dict] = []
        result = engine.run_case(ledger, hidden_truth, trace_rows)
        assert result.status in ("done", "escalated")
        assert trace_rows  # at least one checkpoint was logged

    def test_every_trace_row_validates(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        engine = workflow_loop.WorkflowEngine(client, max_steps=12)
        trace_rows: list[dict] = []
        for i in range(5):
            ledger, hidden_truth = workflow_loop.make_synthetic_case(f"case-{i}", random.Random(i))
            engine.run_case(ledger, hidden_truth, trace_rows)
        assert trace_rows
        for row in trace_rows:
            validate_trace(row)  # raises TraceError on a malformed row

    def test_trace_rows_carry_unknown_label_sentinel(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        engine = workflow_loop.WorkflowEngine(client, max_steps=12)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(2))
        trace_rows: list[dict] = []
        engine.run_case(ledger, hidden_truth, trace_rows)
        assert all(row["label"] == "UNKNOWN" for row in trace_rows)

    def test_trace_state_carries_prior_answers_in_features_wire_shape(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        engine = workflow_loop.WorkflowEngine(client, max_steps=12)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(2))
        trace_rows: list[dict] = []
        engine.run_case(ledger, hidden_truth, trace_rows)
        first = trace_rows[0]
        assert "prior_answers" in first["state"]
        assert "next_step" in first["state"]["prior_answers"]

    def test_run_case_never_exceeds_max_steps(self, running_server: str) -> None:
        client = workflow_loop.DecisionHeadClient(running_server)
        engine = workflow_loop.WorkflowEngine(client, max_steps=3)
        ledger, hidden_truth = workflow_loop.make_synthetic_case("case-x", random.Random(9))
        result = engine.run_case(ledger, hidden_truth, [])
        assert result.steps_used <= 3
        assert result.status in ("done", "escalated")


class TestMainCLI:
    def test_dry_run_end_to_end_writes_valid_traces(self, tmp_path: Path) -> None:
        trace_out = tmp_path / "traces.jsonl"
        exit_code = workflow_loop.main(["--dry", "--cases", "4", "--seed", "1", "--trace-out", str(trace_out)])
        assert exit_code == 0
        assert trace_out.exists()
        rows = [json.loads(line) for line in trace_out.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            validate_trace(row)

    def test_dry_run_is_deterministic_given_a_seed(self, tmp_path: Path) -> None:
        out1, out2 = tmp_path / "t1.jsonl", tmp_path / "t2.jsonl"
        workflow_loop.main(["--dry", "--cases", "3", "--seed", "7", "--trace-out", str(out1)])
        workflow_loop.main(["--dry", "--cases", "3", "--seed", "7", "--trace-out", str(out2)])
        assert out1.read_text() == out2.read_text()

    def test_missing_base_and_dry_is_a_clean_argparse_error(self) -> None:
        with pytest.raises(SystemExit):
            workflow_loop.main([])

    def test_default_trace_out_writes_a_discoverable_temp_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = workflow_loop.main(["--dry", "--cases", "2", "--seed", "0"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "trace rows written to" in out
        written_path = out.split("trace rows written to", 1)[1].splitlines()[0].strip()
        Path(written_path).unlink(missing_ok=True)
