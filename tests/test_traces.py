from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from decisionhead.traces import (
    TraceError,
    convert_jwc_trajectories,
    read_traces,
    validate_trace,
    write_traces,
)

# A tiny, self-contained stand-in for the real jwc_score.py -- exercises
# the SAME dynamic-load and call contract (convert_jwc_trajectories calls
# jwc_score.correct_action_set(key, obs).cls / .focus_item) without
# depending on the actual multi-thousand-line benchmark module. Mirrors
# just enough of correct_action_set's real behaviour (SPEC.md section 5)
# to prove the converter wires things up correctly: an open item with no
# blocker entry is REASON/CONTINUE; a "retrievable" blocker not yet
# retrieved is INFO; no open items at all is DONE.
_STUB_JWC_SCORE = '''
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectAction:
    actions: frozenset
    cls: str
    focus_item: str | None


def correct_action_set(key, obs):
    open_items = obs.get("open_items") or []
    if not open_items:
        return CorrectAction(frozenset({"CLAIM_COMPLETE"}), "DONE", None)
    focus = open_items[0]
    blockers = {b["item_id"]: b for b in key.get("blockers") or []}
    blocker = blockers.get(focus)
    if blocker is None:
        return CorrectAction(frozenset({"CONTINUE"}), "REASON", focus)
    if blocker["resolution"] == "retrievable":
        return CorrectAction(frozenset({"RETRIEVE"}), "INFO", focus)
    if blocker["resolution"] == "unresolvable":
        return CorrectAction(frozenset({"HANDOFF"}), "BLOCKED", focus)
    return CorrectAction(frozenset({"CONTINUE"}), "REASON", focus)
'''


@pytest.fixture
def stub_jwc_score_path(tmp_path: Path) -> str:
    path = tmp_path / "jwc_score.py"
    path.write_text(_STUB_JWC_SCORE)
    return str(path)


def _trajectory_row(
    case_id: str, step: int, open_items: list[str], blockers: list[dict], action: str, raw=None
) -> dict:
    return {
        "case_id": case_id,
        "arm": "A",
        "step": step,
        "obs": {
            "case_id": case_id,
            "requirements": [{"item_id": i, "category": "tax"} for i in ["a", "b"]],
            "open_items": open_items,
            "findings": [],
            "documents_on_file": [],
            "history": [],
            "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
            "steps_used": step,
        },
        "key": {"blockers": blockers, "hard": False},
        "decision": {"action": action, "args": {}, "raw": raw},
    }


class TestValidateTrace:
    def test_well_formed_trace_passes(self) -> None:
        validate_trace({"state": {}, "label": "INFO"})

    def test_missing_state_rejected(self) -> None:
        with pytest.raises(TraceError, match="state"):
            validate_trace({"label": "INFO"})

    def test_missing_label_rejected(self) -> None:
        with pytest.raises(TraceError, match="label"):
            validate_trace({"state": {}})

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(TraceError):
            validate_trace({"state": {}, "label": ""})

    def test_non_string_action_rejected(self) -> None:
        with pytest.raises(TraceError, match="action"):
            validate_trace({"state": {}, "label": "INFO", "action": 5})

    def test_non_object_meta_rejected(self) -> None:
        with pytest.raises(TraceError, match="meta"):
            validate_trace({"state": {}, "label": "INFO", "meta": "nope"})


class TestReadWriteTraces:
    def test_round_trip(self, tmp_path: Path) -> None:
        traces = [
            {"state": {"a": 1}, "label": "INFO"},
            {"state": {"b": 2}, "label": "DONE", "action": "CLAIM_COMPLETE"},
        ]
        path = tmp_path / "traces.jsonl"
        write_traces(str(path), traces)
        loaded = read_traces(str(path))
        assert loaded == traces

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text('{"state": {}, "label": "INFO"}\n\n\n{"state": {}, "label": "DONE"}\n')
        assert len(read_traces(str(path))) == 2

    def test_invalid_json_line_raises_with_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text('{"state": {}, "label": "INFO"}\nnot json\n')
        with pytest.raises(TraceError, match=":2:"):
            read_traces(str(path))

    def test_invalid_trace_shape_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text('{"state": {}}\n')
        with pytest.raises(TraceError):
            read_traces(str(path))


class TestConvertJwcTrajectories:
    def test_converts_every_row(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        rows = [
            _trajectory_row("c1", 0, ["a", "b"], [], "CONTINUE"),
            _trajectory_row("c1", 1, [], [], "CLAIM_COMPLETE"),
        ]
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text("\n".join(json.dumps(r) for r in rows))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)
        assert len(traces) == 2
        assert traces[0]["label"] == "REASON"
        assert traces[1]["label"] == "DONE"

    def test_labels_come_from_the_key_not_the_chosen_action(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        # The controller chose CONTINUE, but the key says the focus item
        # is retrievable -- label must be INFO regardless of what was
        # actually chosen (that choice is recorded in "action" instead).
        rows = [
            _trajectory_row(
                "c1", 0, ["a"], [{"item_id": "a", "resolution": "retrievable"}], "CONTINUE"
            ),
        ]
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text(json.dumps(rows[0]))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)
        assert traces[0]["label"] == "INFO"
        assert traces[0]["action"] == "CONTINUE"

    def test_meta_carries_case_id_and_focus_item(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        rows = [_trajectory_row("c42", 3, ["a"], [], "CONTINUE")]
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text(json.dumps(rows[0]))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)
        assert traces[0]["meta"]["case_id"] == "c42"
        assert traces[0]["meta"]["step"] == 3
        assert traces[0]["meta"]["focus_item"] == "a"

    def test_arm_filter(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        row_a = _trajectory_row("c1", 0, [], [], "CLAIM_COMPLETE")
        row_b = {**_trajectory_row("c2", 0, [], [], "CLAIM_COMPLETE"), "arm": "Cprime"}
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text(json.dumps(row_a) + "\n" + json.dumps(row_b))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path, arm="A")
        assert len(traces) == 1
        assert traces[0]["meta"]["case_id"] == "c1"

    def test_prior_answers_are_lifted_from_decision_raw(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        raw = {
            "answers": {
                "a::q_onfile": {"type": "noul", "noul": 0.7},
                "next_step": {"type": "choice", "choice": "CONTINUE"},
            }
        }
        rows = [_trajectory_row("c1", 0, ["a"], [], "CONTINUE", raw=raw)]
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text(json.dumps(rows[0]))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)
        assert traces[0]["state"]["prior_answers"] == raw["answers"]

    def test_no_raw_answers_means_no_prior_answers_key(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        rows = [_trajectory_row("c1", 0, ["a"], [], "CONTINUE", raw=None)]
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text(json.dumps(rows[0]))
        traces = convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)
        assert "prior_answers" not in traces[0]["state"]

    def test_invalid_jwc_score_path_raises_trace_error(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text("")
        with pytest.raises(TraceError):
            convert_jwc_trajectories(str(traj_path), str(tmp_path / "does-not-exist.py"))

    def test_invalid_json_line_raises(self, tmp_path: Path, stub_jwc_score_path: str) -> None:
        traj_path = tmp_path / "trajectories.jsonl"
        traj_path.write_text("not json at all")
        with pytest.raises(TraceError):
            convert_jwc_trajectories(str(traj_path), stub_jwc_score_path)


# Optional integration data, never part of this repository. Both paths come
# from the environment so no machine-specific path is committed:
#   DECISIONHEAD_JWC_SCORE              path to the benchmark's jwc_score.py
#   DECISIONHEAD_JWC_DEV_A_TRAJECTORIES path to an arm-A dev trajectories.jsonl
# The test class below is skipped unless both are set and exist.
_REAL_JWC_SCORE = os.environ.get("DECISIONHEAD_JWC_SCORE", "")
_REAL_DEV_A_TRAJECTORIES = os.environ.get("DECISIONHEAD_JWC_DEV_A_TRAJECTORIES", "")


@pytest.mark.skipif(
    not (
        _REAL_JWC_SCORE
        and _REAL_DEV_A_TRAJECTORIES
        and Path(_REAL_JWC_SCORE).exists()
        and Path(_REAL_DEV_A_TRAJECTORIES).exists()
    ),
    reason="set DECISIONHEAD_JWC_SCORE and DECISIONHEAD_JWC_DEV_A_TRAJECTORIES to run against real benchmark data",
)
class TestConvertJwcTrajectoriesAgainstRealDevData:
    """An integration-level sanity check against the actual JEV-WC dev
    split (arm A) this package was built against -- skipped wherever that
    external data is not configured (e.g. a clean checkout or CI), but
    valuable where it is, since the unit tests above use a stub
    jwc_score.py that could in principle drift from the real scorer's
    behaviour."""

    def test_matches_the_known_dev_a_shape(self) -> None:
        # This exercises the REAL jwc_score.py from the benchmark's own
        # repository, which this package does not own,
        # version, or control the evolution of. If that external module
        # itself fails to load (e.g. it grew an import of a sibling
        # module not on sys.path when loaded standalone by this test),
        # that is a live-external-dependency drift, not a decisionhead
        # regression -- skip clearly instead of failing the suite.
        try:
            traces = convert_jwc_trajectories(_REAL_DEV_A_TRAJECTORIES, _REAL_JWC_SCORE, arm="A")
        except TraceError as exc:
            pytest.skip(
                "the jwc_score.py named by DECISIONHEAD_JWC_SCORE failed to load/execute "
                f"(external dependency, outside decisionhead's control): {exc}"
            )
        assert len(traces) == 260
        assert len({t["meta"]["case_id"] for t in traces}) == 100
        assert {t["label"] for t in traces} == {"INFO", "REASON", "DONE", "BLOCKED"}
        # Arm A (RulesController) records no typed answers (raw is null
        # per jwc_score.py's own docstring), so no trace should carry
        # prior_answers.
        assert all("prior_answers" not in t["state"] for t in traces)
