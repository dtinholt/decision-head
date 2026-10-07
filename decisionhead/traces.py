"""The trainable-mode trace format, and a converter from the JEV-WC
benchmark's own artifacts into it.

Trace format (one JSON object per line, UTF-8, ``\\n``-terminated)::

    {"state": <the same state object /v1/systemone receives>,
     "label": <decision class string>,
     "action": <action string, optional>,
     "meta": {...}, <optional>}

``state`` is opaque to this module -- :mod:`decisionhead.features` reads
whatever shape it has. ``label`` is the ground-truth 4-way decision class
this state should have produced; :mod:`decisionhead.train` trains against
it directly. ``action`` is the actual action recorded at this checkpoint
(what really happened -- may or may not agree with ``label``, e.g. a
controller's mistake), kept for reference and future finer-grained
supervision; it is never read by :mod:`decisionhead.train`. ``meta`` is
free-form provenance (case id, step, arm, ...); :mod:`decisionhead.train`
reads only ``meta.case_id`` when present, to cluster cross-validation
folds by case rather than by individual checkpoint (see its module
docstring).

JEV-WC converter
-----------------

Source: the JSONL files in one JEV-WC run directory (``cases.jsonl``,
``checkpoints.jsonl``, ``trajectories.jsonl`` -- see
the benchmark's own specification).
:func:`convert_jwc_trajectories` reads ``trajectories.jsonl`` only: each of
its rows already embeds ``obs`` (the environment's own observation at that
step -- verbatim the ``state`` shape this package's ``/v1/systemone``
expects), ``key`` (already merged with the case's ``hard`` flag by
``jwc_run.py``'s own ``_key_with_hard``), and ``decision`` (the
controller's actual choice and, for a typed controller, its raw per-item
question probabilities -- see :func:`_prior_answers_from_decision`).
``cases_path``/``checkpoints_path`` are accepted only for provenance
(recorded in each trace's ``meta``); nothing in them is needed to build a
trace once ``trajectories.jsonl`` is available.

Ground truth comes from ``jwc_score.correct_action_set`` (the JEV-WC
harness's own scorer), loaded dynamically from a path the caller supplies
(``--jwc-score-path``) -- this package deliberately does NOT import or
vendor that module, since it belongs to the benchmark, not to this
package; the PRIMARY registered metric for the whole JEV-WC
campaign (``decision_class_accuracy``) is defined by that one function, so
re-deriving the label logic here (rather than calling it) would risk a
silent, undetected disagreement with the benchmark itself.

Only the ``dev`` split should ever be handed to this converter for
building traces used to inform design or hyperparameters; ``eval`` and
``challenge`` are held out for scoring and must never influence anything
upstream of them. This module does not check which split a case belongs
to (it trusts the caller's input paths), but a caller MUST NOT point it
at eval/challenge data for anything other than one-shot, no-parameter-tuning
final scoring.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType
from typing import Any

JSONDict = dict[str, Any]


class TraceError(ValueError):
    """Raised for a malformed trace line or an unusable JEV-WC input."""


def read_traces(path: str) -> list[JSONDict]:
    """Reads a trace JSONL file. Blank lines are skipped; a malformed
    line raises :class:`TraceError` naming its 1-based line number."""
    traces: list[JSONDict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TraceError(f"{path}:{line_no}: invalid JSON ({exc})") from None
            validate_trace(row, context=f"{path}:{line_no}")
            traces.append(row)
    return traces


def validate_trace(row: JSONDict, *, context: str = "<trace>") -> None:
    """Raises :class:`TraceError` if ``row`` is not a well-formed trace."""
    if not isinstance(row, dict):
        raise TraceError(f"{context}: a trace row must be a JSON object")
    if "state" not in row:
        raise TraceError(f"{context}: a trace row requires 'state'")
    if not isinstance(row.get("label"), str) or not row["label"].strip():
        raise TraceError(f"{context}: a trace row requires a non-empty string 'label'")
    if "action" in row and row["action"] is not None and not isinstance(row["action"], str):
        raise TraceError(f"{context}: 'action', if present, must be a string")
    if "meta" in row and row["meta"] is not None and not isinstance(row["meta"], dict):
        raise TraceError(f"{context}: 'meta', if present, must be an object")


def write_traces(path: str, traces: list[JSONDict]) -> None:
    """Writes ``traces`` as JSONL, one compact JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _load_jwc_score(path: str) -> ModuleType:
    """Dynamically loads ``jwc_score.py`` from ``path`` as a standalone
    module (it is not part of this package and is not installed -- see
    the module docstring). Registers
    it in ``sys.modules`` under its own name before executing it, which
    ``jwc_score.py`` itself requires (it decorates a class with
    ``@dataclass(frozen=True)``, and Python's dataclass machinery looks the
    defining module up in ``sys.modules`` while doing so)."""
    spec = importlib.util.spec_from_file_location("jwc_score", path)
    if spec is None or spec.loader is None:
        raise TraceError(f"could not load a Python module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- surfaced with the path, not a bare traceback
        raise TraceError(f"failed to load jwc_score.py from {path!r}: {exc}") from exc
    return module


def _prior_answers_from_decision(decision: JSONDict | None) -> JSONDict | None:
    """Lifts the per-item Noul answers and the ``next_step`` choice answer
    out of a JEV-WC trajectory row's ``decision["raw"]["answers"]`` --
    exactly the shape :mod:`decisionhead.features`'s ``state["prior_answers"]``
    convention expects (see its ``_NOUL_SUFFIXES``), because that
    convention IS this wire shape, not a separate one. ``None`` for a
    controller that recorded no typed answers at all (arm A/RulesController
    records ``raw: null``; arm B/FreeLLMController records free text, not
    typed Nouls -- see ``jwc_score.py``'s own docstring on which arms
    "carry probabilities"), so those traces simply omit
    ``state["prior_answers"]`` and feature group (b) falls back to its
    documented neutral defaults for them."""
    if not decision:
        return None
    raw = decision.get("raw")
    if not isinstance(raw, dict):
        return None
    answers = raw.get("answers")
    return dict(answers) if isinstance(answers, dict) else None


def convert_jwc_trajectories(
    trajectories_path: str,
    jwc_score_path: str,
    *,
    arm: str | None = None,
    cases_path: str | None = None,
    checkpoints_path: str | None = None,
) -> list[JSONDict]:
    """Converts one JEV-WC ``trajectories.jsonl`` file into trace rows.

    Each trajectory row already carries everything one trace needs in
    one place: ``obs`` (the environment's own ``Env.observe()`` output --
    verbatim the ``state`` shape :mod:`decisionhead.features` expects and
    a real ``/v1/systemone`` caller would send), ``key`` (already merged
    with ``hard`` by ``jwc_run.py``'s own ``_key_with_hard``, exactly what
    ``jwc_score.correct_action_set`` expects), and ``decision`` (the
    controller's actual choice, and -- for a typed controller (B'/C/Cprime)
    -- its raw per-question answers, lifted into ``state["prior_answers"]``
    by :func:`_prior_answers_from_decision` so feature group (b) can train
    on real banked question probabilities, not just structured state).
    ``cases_path``/``checkpoints_path`` are accepted only for provenance
    (recorded in each trace's ``meta``) -- everything this converter
    actually reads lives in ``trajectories.jsonl`` itself.

    ``jwc_score_path`` is the path to the benchmark's own ``jwc_score.py``
    (read, never modified, never vendored -- see the module docstring);
    its ``correct_action_set(key, obs)`` is the single oracle used for
    every trace's ``label``. ``arm`` restricts conversion to rows recorded
    under that controller arm (e.g. ``"A"``, ``"Cprime"``); ``None``
    converts every row regardless of arm.

    Each output trace's ``meta`` carries ``case_id``, ``step``, ``arm``,
    and ``focus_item`` (the open item ``correct_action_set`` judged this
    checkpoint about); this ``meta.case_id`` is exactly what
    :mod:`decisionhead.train`'s case-clustered cross-validation groups by.
    """
    jwc_score = _load_jwc_score(jwc_score_path)
    traces: list[JSONDict] = []
    with open(trajectories_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TraceError(f"{trajectories_path}:{line_no}: invalid JSON ({exc})") from None
            if arm is not None and row.get("arm") != arm:
                continue
            obs = row["obs"]
            key = row["key"]
            decision = row.get("decision")
            correct = jwc_score.correct_action_set(key, obs)

            state = dict(obs)
            prior_answers = _prior_answers_from_decision(decision)
            if prior_answers is not None:
                state["prior_answers"] = prior_answers

            traces.append(
                {
                    "state": state,
                    "label": correct.cls,
                    "action": (decision or {}).get("action"),
                    "meta": {
                        "case_id": row.get("case_id"),
                        "step": row.get("step"),
                        "arm": row.get("arm"),
                        "focus_item": correct.focus_item,
                        "source": "jev-wc",
                        "source_line": line_no,
                        "trajectories_path": trajectories_path,
                        "cases_path": cases_path,
                        "checkpoints_path": checkpoints_path,
                    },
                }
            )
    return traces


def run_selftest() -> bool:
    """Standalone smoke check exercising the generic trace I/O only (the
    JEV-WC converter needs a real ``jwc_score.py`` on disk and is
    exercised in ``tests/test_traces.py`` with a small stub module
    instead). See ``tests/test_traces.py`` for the thorough version."""
    import tempfile

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    good = {"state": {"a": 1}, "label": "INFO", "action": "RETRIEVE", "meta": {"case_id": "c1"}}
    try:
        validate_trace(good)
        check("a well-formed trace validates", True)
    except TraceError:
        check("a well-formed trace validates", False)

    try:
        validate_trace({"state": {}})
        check("a trace missing 'label' is rejected", False)
    except TraceError:
        check("a trace missing 'label' is rejected", True)

    try:
        validate_trace({"label": "INFO"})
        check("a trace missing 'state' is rejected", False)
    except TraceError:
        check("a trace missing 'state' is rejected", True)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/traces.jsonl"
        write_traces(path, [good, {"state": {"b": 2}, "label": "DONE"}])
        loaded = read_traces(path)
        check("write_traces/read_traces round-trips", len(loaded) == 2 and loaded[0]["label"] == "INFO")

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
