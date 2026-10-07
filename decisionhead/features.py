"""Turns one trace's ``state`` (the same JSON object the ``/v1/systemone``
service receives) into a fixed-length numeric feature vector for
:mod:`decisionhead.model`.

Three independently switchable feature groups (:class:`FeatureConfig`),
matching the product spec:

(a) **structured** -- counts and flags read directly off the state: how
    many ledger items are open/closed, remaining budget, history length
    and "trailing run" features (how many of the same action were just
    taken in a row), any boolean ``flags`` the state exposes, a
    checkpoint index, and the rule-table-shaped BOOKKEEPING signals a
    hand-written rule table effectively hard-codes and a zero-shot head
    has no direct read on -- has RETRIEVE/REQUEST already been tried for
    the focus item, which per-resource budgets are exhausted, and the
    focus item's own finding status (including ``"uncertain"``, which a
    plain met/unmet read misses). See :func:`_bookkeeping_features`.
    Always available; on by default.

(b) **prior_answers** -- P(yes)-shaped values from a previous zero-shot
    pass over the SAME state, when the caller attaches them, in the exact
    wire shape JEV-WC's own typed controllers bank
    (``"<item_id>::q_onfile"`` etc -- see :data:`_NOUL_SUFFIXES`).
    OPTIONAL -- a real deployment may call this package cold, with no
    prior answers at all, so every value in this group has a documented
    neutral default and an explicit presence flag rather than making the
    feature vector's length depend on whether the field was present.

(c) **embedding** -- a text embedding of the state's rendered prefix
    (:func:`decisionhead.core.render_state`), via an
    :class:`~decisionhead.embeddings.EmbeddingsBackend`. OFF by default;
    requires the caller to pass an embedder and an expected dimension
    (stored in :class:`FeatureConfig` and in the saved model file, since
    serving must reproduce the exact same feature layout training used).

The state shape assumed for groups (a)/(b) is the JEV-WC benchmark's
observation (``Obs``) shape --
``requirements``, ``open_items``, ``history``, ``budget``, ``steps_used``
-- because that is the only labelled data this package ships a converter
for (:mod:`decisionhead.traces`). Every read here is defensive
(``dict.get`` with a default), so a state missing some or all of these
keys still produces a valid, fixed-length vector (mostly zeros/neutral
defaults) rather than raising -- a company adapting this to its own
workflow trace shape gets a working baseline immediately and can enrich
``flags``/``prior_answers`` incrementally.

Feature ORDER is significant and is exactly :func:`feature_names`'s
output for a given :class:`FeatureConfig`; :mod:`decisionhead.train`
saves this order in the model file and :mod:`decisionhead.learned`
rebuilds features in the same order at serve time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .core import QuestionError, render_state
from .embeddings import EmbeddingsBackend

JSONDict = dict[str, Any]

# The six-action taxonomy this package understands by convention (the same
# one JEV-WC's environment exposes -- SPEC.md section 3) and its collapse
# into four decision classes (SPEC.md section 5). Used here to aggregate a
# prior "next_step" choice answer's per-action probabilities into per-class
# sums, and reused by decisionhead.learned for the reverse mapping when
# serving a "choice" question over these same action names.
KNOWN_ACTIONS: tuple[str, ...] = ("RETRIEVE", "REQUEST", "CONTINUE", "CONSULT_EXPERT", "HANDOFF", "CLAIM_COMPLETE")
ACTION_CLASS: dict[str, str] = {
    "RETRIEVE": "INFO",
    "REQUEST": "INFO",
    "CONTINUE": "REASON",
    "CONSULT_EXPERT": "REASON",
    "CLAIM_COMPLETE": "DONE",
    "HANDOFF": "BLOCKED",
}
DECISION_CLASSES: tuple[str, ...] = ("INFO", "REASON", "DONE", "BLOCKED")

_NEUTRAL_NOUL = 0.5
_UNIFORM_CLASS_PROB = 1.0 / len(DECISION_CLASSES)

# The default budget caps a state's "budget" field is a REMAINING count
# against (JEV-WC SPEC.md section 3: 12 steps, 2 expert calls, 3
# requests). Used only to normalize the remaining-budget features into
# [0, 1] so they sit on a comparable scale to the other structured
# features; a state from a different workflow with different caps still
# produces a valid (if less meaningfully-scaled) feature, since these are
# defaults, not requirements.
DEFAULT_BUDGET_CAPS: dict[str, float] = {"steps": 12.0, "expert_calls": 2.0, "requests": 3.0}


@dataclass(frozen=True)
class FeatureConfig:
    """Which feature groups to compute, and the embedding settings needed
    to compute group (c) when it is on. Serialized verbatim into the
    saved model file so serving reproduces training's exact feature
    layout."""

    use_structured: bool = True
    use_prior_answers: bool = True
    use_embedding: bool = False
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        if self.use_embedding and not self.embedding_dim:
            raise ValueError("use_embedding=True requires a positive embedding_dim")
        if not (self.use_structured or self.use_prior_answers or self.use_embedding):
            raise ValueError("at least one feature group must be enabled")

    def to_dict(self) -> JSONDict:
        """The form saved in a model file's ``feature_config``."""
        return {
            "use_structured": self.use_structured,
            "use_prior_answers": self.use_prior_answers,
            "use_embedding": self.use_embedding,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> FeatureConfig:
        """Rebuilds a config from :meth:`to_dict` output; a missing key
        takes the dataclass default."""
        return cls(
            use_structured=bool(data.get("use_structured", True)),
            use_prior_answers=bool(data.get("use_prior_answers", True)),
            use_embedding=bool(data.get("use_embedding", False)),
            embedding_dim=data.get("embedding_dim"),
        )


def _ordered_open_items(state: JSONDict) -> list[str]:
    """The focus item, deterministically: the first open item in
    ``requirements`` order, falling back to ``open_items`` order for any
    item not found there. Mirrors ``jwc_score._ordered_open_items`` (the
    JEV-WC scorer this package's converter derives labels from) so
    feature group (b) reads the prior answer for the SAME item the label
    is about; reimplemented here (rather than imported) because this
    module must not depend on the benchmark harness at runtime -- only
    :mod:`decisionhead.traces`'s converter touches that code, and only at
    conversion time."""
    requirements = state.get("requirements") or []
    open_items = list(state.get("open_items") or [])
    ordered = [r["item_id"] for r in requirements if r.get("item_id") in open_items]
    for oid in open_items:
        if oid not in ordered:
            ordered.append(oid)
    return ordered


def _trailing_run_length(history: list[JSONDict]) -> int:
    """How many entries at the END of ``history`` share the last entry's
    action name. Zero for empty history."""
    if not history:
        return 0
    last_action = history[-1].get("action")
    n = 0
    for entry in reversed(history):
        if entry.get("action") != last_action:
            break
        n += 1
    return n


def _trailing_action_count(history: list[JSONDict], action: str) -> int:
    """How many entries at the END of ``history`` are ``action``
    specifically (0 if the last action is not ``action`` at all)."""
    n = 0
    for entry in reversed(history):
        if entry.get("action") != action:
            break
        n += 1
    return n


def _history_has(history: list[JSONDict], action: str, arg_key: str | None = None, arg_val: Any = None) -> bool:
    """Whether ``history`` contains ``action`` at all (``arg_key`` is
    ``None``) or ``action`` with ``args[arg_key] == arg_val``. Mirrors
    ``jwc_score._history_has`` (the JEV-WC scorer's own remaining-work
    bookkeeping check) -- reimplemented here, generically, because this
    module must not depend on the benchmark harness at runtime (see
    :func:`_ordered_open_items`'s docstring for the same reasoning)."""
    for entry in history or ():
        if entry.get("action") != action:
            continue
        if arg_key is None:
            return True
        if (entry.get("args") or {}).get(arg_key) == arg_val:
            return True
    return False


def _focus_item_category(state: JSONDict, focus_item: str | None) -> str | None:
    if focus_item is None:
        return None
    for r in state.get("requirements") or []:
        if r.get("item_id") == focus_item:
            category = r.get("category")
            return str(category) if category is not None else None
    return None


def _focus_finding_status(state: JSONDict, focus_item: str | None) -> str | None:
    if focus_item is None:
        return None
    for finding in state.get("findings") or []:
        if finding.get("item_id") == focus_item:
            status = finding.get("status")
            return str(status) if status is not None else None
    return None


def _bookkeeping_features(state: JSONDict) -> list[float]:
    """The rule-table-shaped bookkeeping signals the zero-shot head has
    no direct read on but a hand-written rule table (arm A) effectively
    hard-codes: whether the focus item's evidence has already been
    RETRIEVEd/REQUESTed once (so the same action should not fire again),
    remaining per-resource budget flags, and the focus item's own finding
    status, including the ``"uncertain"`` state a plain met/unmet read
    misses entirely. Added per the 2026-09-24 eval diagnosis (dev-only):
    the head's own confidence signal (``q_interp``) is the strongest
    single feature for the REASON class, but it has no equivalent read on
    this bookkeeping, which is exactly what the rule table wins on."""
    history = state.get("history") or []
    open_items = _ordered_open_items(state)
    focus = open_items[0] if open_items else None
    category = _focus_item_category(state, focus)
    focus_status = _focus_finding_status(state, focus)

    retrieve_tried_for_focus = _history_has(history, "RETRIEVE", "category", category) if category else False
    request_tried_for_focus = _history_has(history, "REQUEST", "item_id", focus) if focus else False
    n_uncertain_findings = float(sum(1 for f in (state.get("findings") or []) if f.get("status") == "uncertain"))

    budget = state.get("budget") or {}
    steps_exhausted = float(budget.get("steps", DEFAULT_BUDGET_CAPS["steps"])) <= 0
    requests_exhausted = float(budget.get("requests", DEFAULT_BUDGET_CAPS["requests"])) <= 0
    expert_calls_exhausted = float(budget.get("expert_calls", DEFAULT_BUDGET_CAPS["expert_calls"])) <= 0

    return [
        1.0 if retrieve_tried_for_focus else 0.0,
        1.0 if request_tried_for_focus else 0.0,
        n_uncertain_findings,
        1.0 if steps_exhausted else 0.0,
        1.0 if requests_exhausted else 0.0,
        1.0 if expert_calls_exhausted else 0.0,
        1.0 if focus_status == "met" else 0.0,
        1.0 if focus_status == "unmet" else 0.0,
        1.0 if focus_status == "uncertain" else 0.0,
        1.0 if focus_status is None else 0.0,
    ]


def _bookkeeping_feature_names() -> list[str]:
    return [
        "retrieve_tried_for_focus_item",
        "request_tried_for_focus_item",
        "n_uncertain_findings",
        "step_budget_exhausted",
        "request_budget_exhausted",
        "expert_call_budget_exhausted",
        "focus_finding_status_met",
        "focus_finding_status_unmet",
        "focus_finding_status_uncertain",
        "focus_finding_status_missing",
    ]


def _structured_features(state: JSONDict) -> list[float]:
    requirements = state.get("requirements") or []
    open_items = state.get("open_items") or []
    history = state.get("history") or []
    budget = state.get("budget") or {}
    flags = state.get("flags") or {}

    n_requirements = float(len(requirements))
    n_open = float(len(open_items))
    n_closed = max(n_requirements - n_open, 0.0)
    frac_open = (n_open / n_requirements) if n_requirements > 0 else 0.0

    steps_used = float(state.get("steps_used", 0) or 0)
    checkpoint_index = float(state.get("checkpoint_index", steps_used) or 0)

    budget_steps = float(budget.get("steps", DEFAULT_BUDGET_CAPS["steps"]))
    budget_expert_calls = float(budget.get("expert_calls", DEFAULT_BUDGET_CAPS["expert_calls"]))
    budget_requests = float(budget.get("requests", DEFAULT_BUDGET_CAPS["requests"]))
    budget_steps_frac = budget_steps / DEFAULT_BUDGET_CAPS["steps"]
    budget_expert_calls_frac = budget_expert_calls / DEFAULT_BUDGET_CAPS["expert_calls"]
    budget_requests_frac = budget_requests / DEFAULT_BUDGET_CAPS["requests"]

    history_length = float(len(history))
    trailing_run_length = float(_trailing_run_length(history))
    trailing_continue_count = float(_trailing_action_count(history, "CONTINUE"))
    action_counts = [
        float(sum(1 for h in history if h.get("action") == action)) for action in KNOWN_ACTIONS
    ]

    flag_values = [1.0 if v else 0.0 for v in flags.values()]
    n_flags = float(len(flag_values))
    n_true_flags = float(sum(flag_values))

    return [
        n_requirements,
        n_open,
        n_closed,
        frac_open,
        steps_used,
        checkpoint_index,
        budget_steps,
        budget_expert_calls,
        budget_requests,
        budget_steps_frac,
        budget_expert_calls_frac,
        budget_requests_frac,
        history_length,
        trailing_run_length,
        trailing_continue_count,
        *action_counts,
        n_flags,
        n_true_flags,
        *_bookkeeping_features(state),
    ]


def _structured_feature_names() -> list[str]:
    return [
        "n_requirements",
        "n_open",
        "n_closed",
        "frac_open",
        "steps_used",
        "checkpoint_index",
        "budget_steps",
        "budget_expert_calls",
        "budget_requests",
        "budget_steps_frac",
        "budget_expert_calls_frac",
        "budget_requests_frac",
        "history_length",
        "trailing_run_length",
        "trailing_continue_count",
        *(f"history_count_{a}" for a in KNOWN_ACTIONS),
        "n_flags",
        "n_true_flags",
        *_bookkeeping_feature_names(),
    ]


# The exact wire shape JEV-WC's typed controllers (B'/C/Cprime) bank a
# checkpoint's per-item Noul answers under (jwc_score.py's own
# "_NOUL_WIRE_IDS"/"_noul()" convention, questions_v1.json v1.2): three
# neutrally-named questions about the FOCUS item -- "is the evidence on
# file", "can the supplier supply it", "does it need interpretation" --
# plus a pre-v1.2 fallback name for each, for a banked file that predates
# the rename. Reused verbatim here (rather than re-invented) so this
# feature group can be trained directly on real banked typed-controller
# output (see decisionhead.traces' JEV-WC converter), not just on a
# hypothetical future caller's own convention.
_NOUL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("q_onfile", "met"),
    ("q_supply", "requestable"),
    ("q_interp", "interpretable"),
)


def _item_noul(prior: JSONDict, item_id: str | None, current_name: str, legacy_name: str) -> float | None:
    if item_id is None:
        return None
    entry = prior.get(f"{item_id}::{current_name}") or prior.get(f"{item_id}::{legacy_name}")
    if not entry:
        return None
    noul = entry.get("noul")
    return float(noul) if noul is not None else None


def _prior_answers_features(state: JSONDict) -> list[float]:
    """Reads ``state["prior_answers"]``: the SAME wire shape a JEV-WC
    typed controller's raw answers carry (see :data:`_NOUL_SUFFIXES`),
    keyed to the state's own focus item
    (:func:`_ordered_open_items`'s first entry) plus a ``"next_step"``
    choice answer over the six known actions. A caller outside the
    JEV-WC ecosystem attaching its own prior zero-shot pass is expected
    to use this same ``"<item_id>::q_onfile"`` keying. Missing entirely,
    or missing just the focus item's entries (e.g. a DONE state with no
    open items, which JEV-WC's own typed controllers never ask an
    item-scoped question about either): every value takes its documented
    neutral default; ``prior_answers_present`` reflects only whether the
    ``prior_answers`` object itself was given, not whether the focus
    item happened to have entries in it."""
    prior = state.get("prior_answers")
    present = 1.0 if isinstance(prior, dict) else 0.0
    prior = prior if isinstance(prior, dict) else {}

    focus_items = _ordered_open_items(state)
    focus = focus_items[0] if focus_items else None

    def _with_default(value: float | None) -> float:
        return value if value is not None else _NEUTRAL_NOUL

    p_onfile, p_supply, p_interp = (
        _with_default(_item_noul(prior, focus, current, legacy)) for current, legacy in _NOUL_SUFFIXES
    )

    next_step = prior.get("next_step") or {}
    confidence = float(next_step.get("confidence", 0.0))
    action_probs: JSONDict = next_step.get("probabilities") or {}
    class_probs = {c: 0.0 for c in DECISION_CLASSES}
    total_mass = 0.0
    for action, p in action_probs.items():
        cls = ACTION_CLASS.get(action)
        if cls is not None:
            class_probs[cls] += float(p)
            total_mass += float(p)
    if total_mass <= 0.0:
        class_probs = {c: _UNIFORM_CLASS_PROB for c in DECISION_CLASSES}

    return [present, p_onfile, p_supply, p_interp, confidence, *(class_probs[c] for c in DECISION_CLASSES)]


def _prior_answers_feature_names() -> list[str]:
    return [
        "prior_answers_present",
        "prior_p_onfile",
        "prior_p_supply",
        "prior_p_interp",
        "prior_next_step_confidence",
        *(f"prior_class_prob_{c}" for c in DECISION_CLASSES),
    ]


def extract_features(
    state: JSONDict,
    config: FeatureConfig,
    embedder: EmbeddingsBackend | None = None,
) -> list[float]:
    """Builds the feature vector for one state, in the exact order
    :func:`feature_names` documents for the same ``config``.

    ``embedder`` is required (and used) only when ``config.use_embedding``
    is True; it is never called otherwise, so a training or serving run
    with embeddings off makes no network call regardless of whether an
    embedder object was constructed by the caller.
    """
    vector: list[float] = []
    try:
        if config.use_structured:
            vector.extend(_structured_features(state))
        if config.use_prior_answers:
            vector.extend(_prior_answers_features(state))
    except (TypeError, ValueError, AttributeError) as exc:
        raise QuestionError(f"learned-mode state has a malformed numeric field ({type(exc).__name__})") from None
    # float("inf")/float("nan") accept the strings "inf"/"nan", so a
    # string-typed budget or confidence field can smuggle a non-finite
    # value past the strict request parser; it would otherwise reach the
    # classifier and come back as a NaN probability (2026-10-06 security
    # review, finding 4).
    if not all(math.isfinite(v) for v in vector):
        raise QuestionError("learned-mode state has a non-finite numeric field (NaN or infinity)")
    if config.use_embedding:
        if embedder is None:
            raise ValueError("config.use_embedding is True but no embedder was provided")
        text = render_state(state)
        embedding = embedder.embed(text)
        if config.embedding_dim is not None and len(embedding) != config.embedding_dim:
            raise ValueError(
                f"embedder returned a {len(embedding)}-dim vector but config.embedding_dim is "
                f"{config.embedding_dim}; retrain with a consistent embedder"
            )
        vector.extend(float(v) for v in embedding)
    return vector


def feature_names(config: FeatureConfig) -> list[str]:
    """The name of every entry :func:`extract_features` returns for
    ``config``, in the same order. Saved in the model file for
    interpretability and for a fast dimension-mismatch check at load
    time."""
    names: list[str] = []
    if config.use_structured:
        names.extend(_structured_feature_names())
    if config.use_prior_answers:
        names.extend(_prior_answers_feature_names())
    if config.use_embedding:
        dim = config.embedding_dim or 0
        names.extend(f"embedding_{i}" for i in range(dim))
    return names


def run_selftest() -> bool:
    """Standalone smoke check; see ``tests/test_features.py`` for the
    thorough version."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    state: JSONDict = {
        "requirements": [{"item_id": "a"}, {"item_id": "b"}],
        "open_items": ["a"],
        "history": [{"action": "CONTINUE"}, {"action": "CONTINUE"}, {"action": "RETRIEVE"}],
        "budget": {"steps": 10, "expert_calls": 2, "requests": 3},
        "steps_used": 2,
    }
    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    vec = extract_features(state, cfg)
    names = feature_names(cfg)
    check("feature vector length matches feature_names length", len(vec) == len(names))
    check("n_open reads open_items", vec[names.index("n_open")] == 1.0)
    check("n_closed derives from requirements minus open", vec[names.index("n_closed")] == 1.0)
    check(
        "trailing_run_length is 1 for a history ending in a different action",
        vec[names.index("trailing_run_length")] == 1.0,
    )

    state_empty: JSONDict = {}
    vec_empty = extract_features(state_empty, cfg)
    check("an empty state produces a full-length, all-default vector", len(vec_empty) == len(names))

    cfg_prior = FeatureConfig(use_structured=False, use_prior_answers=True, use_embedding=False)
    vec_no_prior = extract_features({}, cfg_prior)
    check(
        "missing prior_answers gives prior_answers_present=0",
        vec_no_prior[feature_names(cfg_prior).index("prior_answers_present")] == 0.0,
    )
    state_with_prior: JSONDict = {
        "requirements": [{"item_id": "cert"}],
        "open_items": ["cert"],
        "prior_answers": {
            "cert::q_onfile": {"type": "noul", "noul": 0.9},
            "next_step": {"type": "choice", "confidence": 0.4, "probabilities": {"RETRIEVE": 0.7, "CONTINUE": 0.3}},
        },
    }
    vec_prior = extract_features(state_with_prior, cfg_prior)
    prior_names = feature_names(cfg_prior)
    check("prior_answers_present is 1 when present", vec_prior[prior_names.index("prior_answers_present")] == 1.0)
    check("p_onfile is read from prior_answers", vec_prior[prior_names.index("prior_p_onfile")] == 0.9)
    check(
        "next_step probabilities aggregate into the INFO class (RETRIEVE)",
        abs(vec_prior[prior_names.index("prior_class_prob_INFO")] - 0.7) < 1e-9,
    )

    state_legacy_name: JSONDict = {
        "requirements": [{"item_id": "cert"}],
        "open_items": ["cert"],
        "prior_answers": {"cert::met": {"type": "noul", "noul": 0.0}},
    }
    vec_legacy = extract_features(state_legacy_name, cfg_prior)
    check(
        "a real P(yes)=0.0 is read as 0.0, not defaulted (falsy-zero pitfall)",
        vec_legacy[prior_names.index("prior_p_onfile")] == 0.0,
    )
    check(
        "the pre-rename 'met' key name is accepted as a fallback for 'q_onfile'",
        vec_legacy[prior_names.index("prior_p_onfile")] == 0.0,
    )

    state_bookkeeping: JSONDict = {
        "requirements": [{"item_id": "a", "category": "incorporation"}],
        "open_items": ["a"],
        "findings": [{"item_id": "a", "status": "uncertain"}],
        "history": [{"action": "RETRIEVE", "args": {"category": "incorporation"}}],
        "budget": {"steps": 0, "expert_calls": 2, "requests": 3},
    }
    vec_book = extract_features(state_bookkeeping, cfg)
    book_names = feature_names(cfg)
    check(
        "retrieve_tried_for_focus_item is set when history has a matching RETRIEVE",
        vec_book[book_names.index("retrieve_tried_for_focus_item")] == 1.0,
    )
    check(
        "focus_finding_status_uncertain is set from the focus item's own finding",
        vec_book[book_names.index("focus_finding_status_uncertain")] == 1.0,
    )
    check(
        "step_budget_exhausted is set when the remaining step budget is zero",
        vec_book[book_names.index("step_budget_exhausted")] == 1.0,
    )

    from .embeddings import FakeEmbeddingsBackend

    cfg_emb = FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=True, embedding_dim=4)
    vec_emb = extract_features({"note": "x"}, cfg_emb, embedder=FakeEmbeddingsBackend(dim=4))
    check("embedding feature group has the configured dimension", len(vec_emb) == 4)
    try:
        extract_features({"note": "x"}, cfg_emb, embedder=None)
        check("embedding group without an embedder raises", False)
    except ValueError:
        check("embedding group without an embedder raises", True)

    try:
        FeatureConfig(use_structured=False, use_prior_answers=False, use_embedding=False)
        check("at least one feature group must be enabled", False)
    except ValueError:
        check("at least one feature group must be enabled", True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
