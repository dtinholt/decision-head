"""Reference integration loop: a workflow engine that JUDGES through
decisionhead and keeps its OWN books.

This is the division of labour the product is built around, as the
pre-registration of the workflow-completion benchmark puts it: "The
[workflow] controller does the bookkeeping, playing the customer's
workflow engine; the [decision head] service does the judging. That
division is the product's division." Concretely:

* The ENGINE (:class:`WorkflowEngine`/:class:`CaseLedger` below) owns
  every mutation to the case's own state: which requirements are still
  open, how much budget remains, what history has accumulated, and any
  guardrail that overrides a judgement the ledger itself contradicts
  (see ``CLAIM_COMPLETE`` in :meth:`WorkflowEngine._apply`). None of this
  is delegated to the model.
* The HEAD (:class:`DecisionHeadClient`, a running ``decisionhead
  serve``) only ever answers the one typed question it is asked about
  the state it is shown: is this requirement on file, and what should
  happen next. It never sees or touches the ledger directly.

:class:`DecisionHeadClient` is deliberately dependency-free (standard
library ``urllib`` only, no import of the ``decisionhead`` package
itself) so it can be copied verbatim into a workflow engine's own
codebase -- which, in a real pilot, usually is not a Python project that
wants a new package dependency at all, just an HTTP client for one
endpoint. Everything below the client class is this repository's own
worked example and DOES import ``decisionhead`` (for
:func:`decisionhead.server.run_server`'s ``--dry`` fake backend in
``--dry`` mode, and :mod:`decisionhead.traces` to guarantee the logged
rows are exactly the format ``decisionhead.train`` expects) -- neither of
those imports are needed by a real workflow engine talking to an
already-running server over HTTP.

Every decision this loop makes is logged as one JSONL row in
``decisionhead.traces`` format (see ``docs/traces.md``). Its ``label`` is
always the sentinel ``"UNKNOWN"``: this loop has no ground truth of its
own (it is not told, after the fact, whether an action was actually
correct) -- in a real pilot, a human reviewer fills in the real decision
class before the file is used to train ``mode: "learned"``. See
``docs/pilot.md`` for how that labelling step fits into a two-week pilot.

Usage::

    python examples/workflow_loop.py --dry                       # against an in-process FakeBackend
    python examples/workflow_loop.py --base http://127.0.0.1:8787  # against an already-running server

Run standalone (only the standard library plus, for ``--dry``/trace
writing, this repository's own ``decisionhead`` package -- no third-party
dependency either way).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import tempfile
import threading
import urllib.error
import urllib.request
from typing import Any

JSONDict = dict[str, Any]

DEFAULT_BUDGET: dict[str, int] = {"steps": 12, "requests": 3}
LABEL_UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------- client
#
# Copy everything from here down to the "end DecisionHeadClient" marker
# into a workflow engine's own codebase to talk to a running decisionhead
# server -- it has no dependency on the decisionhead package or on
# anything else in this file.


class DecisionHeadError(Exception):
    """Wraps any transport or server-side error into one exception type,
    carrying the server's own ``{"error": {...}}`` body when there is
    one, never a raw traceback from ``urllib``."""


class DecisionHeadClient:
    """Minimal client for a running ``decisionhead serve`` instance:
    ``POST /v1/systemone`` plus the two health endpoints. Standard
    library only."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "jev-latest",
        timeout: float = 60.0,
        bearer_token: str | None = None,
    ) -> None:
        """``model`` is an alias only -- the server always answers with
        its real backend model id regardless of what is sent here (see
        the API docs). ``bearer_token``, if given, is sent as an
        ``Authorization: Bearer`` header; read it from your own
        environment variable before constructing this client, never
        hardcode one."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.bearer_token = bearer_token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def ask(self, state: str | JSONDict, questions: JSONDict) -> JSONDict:
        """POSTs one Jev-shaped request to ``/v1/systemone`` and returns
        the parsed response (``{"model", "answers", "usage", "mode"}``).
        Raises :class:`DecisionHeadError` on any transport or server-side
        failure -- never lets a raw ``urllib`` exception escape."""
        url = f"{self.base_url}/v1/systemone"
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result: JSONDict = json.loads(resp.read().decode("utf-8"))
                return result
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise DecisionHeadError(f"HTTP {exc.code} from {url}: {raw}") from None
        except urllib.error.URLError as exc:
            raise DecisionHeadError(f"connection failed to {url}: {exc.reason}") from None

    def ready(self) -> JSONDict:
        """GETs ``/ready``: ``{"status", "ready", "detail"}``."""
        with urllib.request.urlopen(f"{self.base_url}/ready", timeout=self.timeout) as resp:
            result: JSONDict = json.loads(resp.read().decode("utf-8"))
            return result

    def health(self) -> JSONDict:
        """GETs ``/health``: ``{"status": "ok"}`` as soon as the process is up."""
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as resp:
            result: JSONDict = json.loads(resp.read().decode("utf-8"))
            return result


# --------------------------------------------------------------------- end DecisionHeadClient


# --------------------------------------------------------------------- worked example: a workflow engine
#
# A small supplier-onboarding case: two requirements ("coi", "w9") must
# each be closed out before the case can be marked complete. The state
# shape below (requirements/open_items/history/findings/budget/
# steps_used) is exactly the shape decisionhead.features already knows
# how to read (see decisionhead/features.py's module docstring), so a
# trace log built from this loop needs no reshaping to train mode:
# "learned" later -- and the "next_step" choice options below are exactly
# decisionhead.features.KNOWN_ACTIONS, the taxonomy that mode's
# classifier is trained against.

RETRIEVABLE = "retrievable"
NEEDS_REQUEST = "needs_request"
UNRESOLVABLE = "unresolvable"


@dataclasses.dataclass
class HiddenTruth:
    """The world's actual answer for one requirement -- never sent to the
    head, never read by the engine directly; only :meth:`WorkflowEngine._apply`
    consults it, standing in for "what actually happens if you retrieve
    it / ask the supplier for it" in a real integration."""

    resolution: str


@dataclasses.dataclass
class CaseLedger:
    """Everything the ENGINE tracks for one case. The head never sees
    this object -- only the plain-dict :meth:`state` snapshot it derives
    from it."""

    case_id: str
    requirements: list[dict[str, str]]
    open_items: list[str]
    findings: list[dict[str, str]] = dataclasses.field(default_factory=list)
    history: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    budget: dict[str, int] = dataclasses.field(default_factory=lambda: dict(DEFAULT_BUDGET))
    steps_used: int = 0
    status: str = "open"  # "open" | "done" | "escalated"

    def state(self) -> JSONDict:
        """The state a real ``/v1/systemone`` request would carry for
        this case right now."""
        return {
            "requirements": [dict(r) for r in self.requirements],
            "open_items": list(self.open_items),
            "findings": [dict(f) for f in self.findings],
            "history": [dict(h) for h in self.history],
            "budget": dict(self.budget),
            "steps_used": self.steps_used,
        }


def make_synthetic_case(case_id: str, rng: random.Random) -> tuple[CaseLedger, dict[str, HiddenTruth]]:
    """One synthetic supplier-onboarding case: two requirements, each
    with a hidden resolution the engine discovers only by acting.
    Deterministic given ``rng``."""
    requirements = [
        {"item_id": "coi", "category": "insurance"},
        {"item_id": "w9", "category": "tax"},
    ]
    hidden_truth = {
        r["item_id"]: HiddenTruth(rng.choices([RETRIEVABLE, NEEDS_REQUEST, UNRESOLVABLE], weights=[0.5, 0.35, 0.15])[0])
        for r in requirements
    }
    ledger = CaseLedger(
        case_id=case_id,
        requirements=requirements,
        open_items=[r["item_id"] for r in requirements],
    )
    return ledger, hidden_truth


def _questions_for(ledger: CaseLedger, focus: str | None) -> JSONDict:
    """The typed questions asked at one checkpoint. With an open focus
    item, offers RETRIEVE/REQUEST/CONTINUE/HANDOFF; with none left open,
    offers only the terminal judgement (CLAIM_COMPLETE/CONTINUE/HANDOFF)
    -- the single highest-stakes judgement in this loop, since a wrong
    "done" claim is exactly the failure mode ``docs/pilot.md``'s
    wrong-claim-rate criterion is watching for.

    The instructions text folds in a short summary of the ledger itself
    (prior attempts, remaining budget) -- a real backend already sees the
    full ``state`` object (sent separately, not just this instructions
    string), so this is redundant against a real model; it matters here
    because the bundled ``FakeBackend`` used by ``--dry`` derives its
    deterministic answer from the instructions text alone (see
    ``decisionhead.backends.FakeBackend.token_distribution``), so without
    this every case with the same focus item would get the identical
    canned answer regardless of what actually happened in that case."""
    tried = sorted({h["action"] for h in ledger.history if h.get("item_id") == focus})
    progress = (
        f"case {ledger.case_id}, step {ledger.steps_used}, already tried {tried or 'nothing'} on this item, "
        f"requests remaining {ledger.budget['requests']}"
    )
    if focus is None:
        return {
            "next_step": {
                "type": "choice",
                "instructions": (
                    f"Every listed requirement is already resolved in this case's ledger ({progress}). "
                    "What should happen to this case?"
                ),
                "criteria": {
                    "CLAIM_COMPLETE": "close the case; nothing remains open",
                    "CONTINUE": "double check before closing; not confident yet",
                    "HANDOFF": "something here looks wrong; a human should look before closing",
                },
            }
        }
    return {
        "q_onfile": {
            "type": "noul",
            "instructions": f"The '{focus}' requirement is on file, current, and needs no further action ({progress}).",
        },
        "next_step": {
            "type": "choice",
            "instructions": f"This case still has '{focus}' open ({progress}). What should happen next?",
            "criteria": {
                "RETRIEVE": "pull it from the system of record; try this before asking the supplier",
                "REQUEST": "ask the supplier directly for it",
                "CONTINUE": "keep evaluating; nothing new to fetch on this item yet",
                "HANDOFF": "something here needs a human -- stop and escalate",
            },
        },
    }


class WorkflowEngine:
    """Drives one case to completion: at each checkpoint, asks the head a
    judgement question about the current ledger state, then applies
    DETERMINISTIC bookkeeping around the answer. The head is never
    trusted blindly -- see the ``CLAIM_COMPLETE`` guardrail in
    :meth:`_apply`, the one place this engine overrides a judgement
    rather than applying it as-is."""

    def __init__(self, client: DecisionHeadClient, *, max_steps: int = 12) -> None:
        self.client = client
        self.max_steps = max_steps

    def run_case(
        self,
        ledger: CaseLedger,
        hidden_truth: dict[str, HiddenTruth],
        trace_sink: list[JSONDict],
    ) -> CaseLedger:
        """Runs ``ledger`` to completion (done, escalated, or budget
        exhausted), appending one trace row per checkpoint to
        ``trace_sink``."""
        while ledger.status == "open" and ledger.steps_used < self.max_steps:
            if ledger.budget["steps"] <= 0:
                self._force_escalate(ledger, "step budget exhausted")
                break

            focus = ledger.open_items[0] if ledger.open_items else None
            state_before = ledger.state()
            questions = _questions_for(ledger, focus)

            result = self.client.ask(state_before, questions)
            answers = result["answers"]
            head_choice = str(answers["next_step"]["choice"])
            head_probabilities = dict(answers["next_step"]["probabilities"])
            p_onfile = answers.get("q_onfile", {}).get("noul") if focus is not None else None

            applied_action = self._apply(ledger, head_choice, focus, hidden_truth)

            trace_sink.append(
                _build_trace_row(
                    state_before=state_before,
                    action=applied_action,
                    ledger=ledger,
                    focus=focus,
                    head_choice=head_choice,
                    head_probabilities=head_probabilities,
                    p_onfile=p_onfile,
                    overridden=applied_action != head_choice,
                )
            )

        if ledger.status == "open":
            # Ran out of steps without the head ever claiming completion
            # or being handed off -- the engine's own safety rule, not a
            # head judgement.
            self._force_escalate(ledger, "max_steps reached without a terminal judgement")
        return ledger

    def _force_escalate(self, ledger: CaseLedger, reason: str) -> None:
        ledger.status = "escalated"
        ledger.history.append({"action": "HANDOFF", "item_id": None, "args": {"reason": reason}})

    def _apply(
        self,
        ledger: CaseLedger,
        action: str,
        focus: str | None,
        hidden_truth: dict[str, HiddenTruth],
    ) -> str:
        """The BOOKKEEPING. Mutates ``ledger`` (budget, findings,
        open_items, history) according to ``action``, and returns the
        action actually applied -- which is ``action`` itself, UNLESS the
        engine's own guardrail overrides it: a ``CLAIM_COMPLETE`` while
        the ledger still shows open items is never trusted at face value
        (downgraded to ``CONTINUE`` instead), because the ledger, not the
        head, is this engine's source of truth about what is actually
        still open."""
        ledger.steps_used += 1
        ledger.budget["steps"] -= 1

        if action == "CLAIM_COMPLETE" and ledger.open_items:
            action = "CONTINUE"

        if action == "RETRIEVE" and focus is not None:
            truth = hidden_truth[focus]
            if truth.resolution == RETRIEVABLE:
                ledger.findings.append({"item_id": focus, "status": "met"})
                ledger.open_items.remove(focus)
            else:
                ledger.findings.append({"item_id": focus, "status": "uncertain"})
        elif action == "REQUEST" and focus is not None:
            ledger.budget["requests"] -= 1
            truth = hidden_truth[focus]
            if truth.resolution in (RETRIEVABLE, NEEDS_REQUEST):
                ledger.findings.append({"item_id": focus, "status": "met"})
                ledger.open_items.remove(focus)
            else:
                ledger.findings.append({"item_id": focus, "status": "uncertain"})
        elif action == "HANDOFF":
            ledger.status = "escalated"
        elif action == "CLAIM_COMPLETE":
            ledger.status = "done"
        # else: "CONTINUE" -- a no-op on the ledger by design.

        ledger.history.append({"action": action, "item_id": focus, "args": {}})

        if ledger.budget["requests"] < 0:
            ledger.status = "escalated"
            action = "HANDOFF"

        return action


def _build_trace_row(
    *,
    state_before: JSONDict,
    action: str,
    ledger: CaseLedger,
    focus: str | None,
    head_choice: str,
    head_probabilities: dict[str, float],
    p_onfile: float | None,
    overridden: bool,
) -> JSONDict:
    """One decisionhead.traces-shaped row for this checkpoint. ``label``
    is always the ``"UNKNOWN"`` sentinel (see the module docstring and
    ``docs/traces.md``): this loop has no ground truth of its own. The
    ``state`` recorded here additionally carries ``prior_answers`` in
    exactly the wire shape ``decisionhead.features`` reads (see its
    module docstring's feature group (b)) -- the zero-shot answer this
    checkpoint actually got, banked for a future ``mode: "learned"``
    training run over this same file."""
    state_with_prior = dict(state_before)
    prior_answers: JSONDict = {"next_step": {"probabilities": head_probabilities}}
    if focus is not None and p_onfile is not None:
        prior_answers[f"{focus}::q_onfile"] = {"noul": p_onfile}
    state_with_prior["prior_answers"] = prior_answers

    return {
        "state": state_with_prior,
        "label": LABEL_UNKNOWN,
        "action": action,
        "meta": {
            "case_id": ledger.case_id,
            "step": ledger.steps_used,
            "focus_item": focus,
            "head_choice": head_choice,
            "head_probabilities": head_probabilities,
            "engine_overrode_head": overridden,
            "source": "examples/workflow_loop.py",
            "note": (
                "label is the 'UNKNOWN' placeholder; replace it with the human-reviewed "
                "ground-truth decision class before training on this file -- see docs/traces.md"
            ),
        },
    }


# --------------------------------------------------------------------- CLI


def _run_dry_server(seed: int) -> tuple[Any, str]:
    """Starts a real ``decisionhead serve --dry`` instance in-process
    (the deterministic FakeBackend -- no network, no model) on an
    ephemeral port, and returns ``(httpd, base_url)``. The caller is
    responsible for ``httpd.shutdown()``/``httpd.server_close()``."""
    from decisionhead.config import ServerConfig
    from decisionhead.server import run_server

    cfg = ServerConfig(model="workflow-loop-dry-model", port=0, dry=True, dry_seed=seed, max_workers=2)
    httpd = run_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{port}"


def build_arg_parser() -> argparse.ArgumentParser:
    """The command-line flags for this example (``--dry`` or ``--base``
    is required)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=None, help="base URL of an already-running decisionhead server")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="start an in-process decisionhead server against the deterministic FakeBackend instead of --base",
    )
    parser.add_argument("--cases", type=int, default=5, help="number of synthetic cases to run (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for synthetic cases (default: 0)")
    parser.add_argument(
        "--trace-out", default=None, help="path to write the trace JSONL to (default: a temp file, path printed)"
    )
    parser.add_argument(
        "--bearer-token-env", default=None, help="env var name holding a bearer token, if the server requires one"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs ``--cases`` synthetic cases through the engine, validates
    every logged row against :mod:`decisionhead.traces`, writes the trace
    file, and prints a per-status summary. Returns the exit code."""
    import os

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dry and not args.base:
        parser.error("one of --dry or --base is required")

    httpd = None
    base = args.base
    try:
        if args.dry:
            httpd, base = _run_dry_server(args.seed)

        bearer_token = os.environ.get(args.bearer_token_env) if args.bearer_token_env else None
        client = DecisionHeadClient(base, bearer_token=bearer_token)
        ready = client.ready()
        print(f"decisionhead at {base}: ready={ready.get('ready')}")

        engine = WorkflowEngine(client)
        rng = random.Random(args.seed)
        trace_rows: list[JSONDict] = []
        status_counts: dict[str, int] = {}

        for i in range(args.cases):
            ledger, hidden_truth = make_synthetic_case(f"case-{i:03d}", rng)
            ledger = engine.run_case(ledger, hidden_truth, trace_rows)
            status_counts[ledger.status] = status_counts.get(ledger.status, 0) + 1
            print(f"  {ledger.case_id}: status={ledger.status} steps_used={ledger.steps_used}")

        from decisionhead.traces import validate_trace, write_traces

        for row in trace_rows:
            validate_trace(row)

        trace_out = args.trace_out
        if trace_out is None:
            fd, trace_out = tempfile.mkstemp(prefix="workflow_loop_traces_", suffix=".jsonl")
            os.close(fd)
        write_traces(trace_out, trace_rows)

        print(f"\n{args.cases} cases run: {status_counts}")
        print(f"{len(trace_rows)} trace rows written to {trace_out}")
        print("Every row's label is the 'UNKNOWN' placeholder -- see docs/traces.md before training on this file.")
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
