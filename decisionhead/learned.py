"""Serving integration for a trained decision head (``mode =
"learned"`` in :mod:`decisionhead.config`/:mod:`decisionhead.server`).

:mod:`decisionhead.core`'s ``answer_questions`` answers every question by
calling an LLM backend's ``token_distribution``. A learned head answers
differently -- it extracts numeric features from the raw ``state`` object
(:mod:`decisionhead.features`) and reads a probability distribution off a
small trained classifier (:mod:`decisionhead.model`), never touching a
model backend at all. :class:`LearnedModel` is the loaded artifact
(classifier weights, feature scaler, feature config, class order,
calibration temperature); :func:`answer_questions_learned` is this
module's equivalent of ``core.answer_questions`` -- same return shape
(``{"answers", "audit", "usage", "model"}``), same audit-row field names,
so :mod:`decisionhead.server` can write either path's result through the
one shared :class:`~decisionhead.server.AuditLog`.

Only ``choice`` questions are answered in learned mode. The classifier is
trained on the JEV-WC 4-way decision class (``INFO``/``REASON``/``DONE``/
``BLOCKED`` -- :mod:`decisionhead.features`'s ``DECISION_CLASSES``); a
``noul`` or ``score`` question has no defined meaning against that output,
so both raise a clear :class:`~decisionhead.core.QuestionError` rather
than silently guessing. A ``choice`` question's own options must be
either exactly the four decision classes, or a subset of the six known
action names (:mod:`decisionhead.features`'s ``KNOWN_ACTIONS`` -- the
JEV-WC action taxonomy those classes collapse); see
:func:`_action_probabilities` for the mapping between the two.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import deadline as _deadline
from .core import (
    JSONDict,
    QuestionError,
    apply_calibration_distribution,
    normalize_question,
    request_hash,
    top_two_gap,
)
from .embeddings import EmbeddingsBackend
from .features import ACTION_CLASS, DECISION_CLASSES, KNOWN_ACTIONS, FeatureConfig, extract_features, feature_names
from .model import MultinomialLogisticRegression, StandardScaler

BACKEND_NAME = "learned"


@dataclass
class LearnedModel:
    """A fully-loaded trained decision head: everything
    :func:`answer_questions_learned` needs, with nothing left to infer
    from the environment (no ambient config, no hidden defaults beyond
    the ones baked in at training time by :mod:`decisionhead.train`)."""

    classifier: MultinomialLogisticRegression
    scaler: StandardScaler
    feature_config: FeatureConfig
    classes: list[str]
    temperature: float = 1.0
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_names = feature_names(self.feature_config)
        if self.classifier.n_features != len(expected_names):
            raise ValueError(
                f"model expects {self.classifier.n_features} features but its feature_config "
                f"implies {len(expected_names)} ({expected_names}); the model file is inconsistent"
            )
        if len(self.classes) != self.classifier.n_classes:
            raise ValueError(
                f"model has {self.classifier.n_classes} output classes but {len(self.classes)} class names"
            )
        # A NaN/inf weight, scaler entry or temperature turns every answer
        # into NaN; refuse the file at load time instead (2026-10-06
        # security review, findings 4 and 15).
        numbers = [x for row in self.classifier.weights for x in row]
        numbers += list(self.classifier.bias) + list(self.scaler.means) + list(self.scaler.stds)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in numbers):
            raise ValueError("model file has a non-numeric or non-finite weight, bias or scaler value")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("model file temperature must be a finite number > 0")
        # A model file's metadata["uses_prior_features"] (written by
        # decisionhead.train) is a claim about this model's serving
        # behaviour -- callers (e.g. an eval harness deciding whether it
        # is safe to withhold prior_answers) may rely on it without
        # loading and inspecting feature_config themselves. If present,
        # it must agree with feature_config.use_prior_answers, the field
        # that actually governs whether extract_features reads
        # state["prior_answers"] at serve time; a mismatch means the file
        # is lying about its own behaviour, which is worse than a load
        # failure.
        uses_prior_features = self.metadata.get("uses_prior_features")
        if uses_prior_features is not None and bool(uses_prior_features) != self.feature_config.use_prior_answers:
            raise ValueError(
                f"model file is inconsistent: metadata['uses_prior_features'] = {uses_prior_features!r} "
                f"but feature_config.use_prior_answers = {self.feature_config.use_prior_answers!r}"
            )

    def class_probabilities(self, state: JSONDict, embedder: EmbeddingsBackend | None = None) -> dict[str, float]:
        """The calibrated probability of each of ``self.classes`` for
        ``state``: extract features in training's exact order, scale with
        training's fitted scaler, read the classifier's softmax output,
        then apply this model's own temperature scaling
        (:func:`decisionhead.core.apply_calibration_distribution`, the
        same elementwise ``sigmoid(logit(p)/T)`` transform fast-mode
        calibration uses)."""
        vector = extract_features(state, self.feature_config, embedder=embedder)
        scaled = self.scaler.transform(vector)
        raw_probs = self.classifier.predict_proba(scaled)
        probs = dict(zip(self.classes, raw_probs, strict=True))
        calibrated, _ = apply_calibration_distribution(probs, self.temperature)
        return calibrated

    def to_dict(self) -> JSONDict:
        """The JSON-serializable model-file form read back by
        :meth:`from_dict`."""
        return {
            "classifier": self.classifier.to_dict(),
            "scaler": self.scaler.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "classes": list(self.classes),
            "temperature": self.temperature,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> LearnedModel:
        """Rebuilds a model from :meth:`to_dict` output. Raises
        ``KeyError`` for a missing section and ``ValueError`` (from
        ``__post_init__``) for an inconsistent file or any non-finite
        weight, scaler value or temperature."""
        return cls(
            classifier=MultinomialLogisticRegression.from_dict(data["classifier"]),
            scaler=StandardScaler.from_dict(data["scaler"]),
            feature_config=FeatureConfig.from_dict(data["feature_config"]),
            classes=list(data["classes"]),
            temperature=float(data.get("temperature", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str) -> None:
        """Writes the model file as indented, key-sorted JSON."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> LearnedModel:
        """Reads a model file written by :meth:`save`. Validation is
        :meth:`from_dict`'s; the server calls this once at start-up, so a
        bad file stops the server rather than failing requests."""
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _action_probabilities(class_probs: dict[str, float], options: dict[str, str], question_id: str) -> dict[str, float]:
    """Maps a class-probability distribution onto the option set a
    ``choice`` question actually asked for.

    Two supported shapes for ``options`` (its keys, ignoring the
    descriptions):

    * exactly :data:`decisionhead.features.DECISION_CLASSES` -- used
      directly, no mapping needed.
    * a (non-empty) subset of :data:`decisionhead.features.KNOWN_ACTIONS`
      -- each class's probability mass is split EQUALLY among the
      requested actions that belong to it (the classifier has no
      finer-grained opinion within a class than this), then the whole
      thing is renormalized over the requested actions.

    Anything else raises :class:`QuestionError` naming both taxonomies,
    rather than guessing at an unfamiliar option set.
    """
    option_names = set(options.keys())
    if option_names == set(DECISION_CLASSES):
        return {name: class_probs[name] for name in options}

    if option_names and option_names <= set(KNOWN_ACTIONS):
        action_probs: dict[str, float] = {}
        members_by_class: dict[str, list[str]] = {c: [] for c in DECISION_CLASSES}
        for name in options:
            members_by_class[ACTION_CLASS[name]].append(name)
        for cls_name, members in members_by_class.items():
            if not members:
                continue
            share = class_probs.get(cls_name, 0.0) / len(members)
            for name in members:
                action_probs[name] = share
        total = sum(action_probs.values())
        if total <= 0.0:
            n = len(options) or 1
            return dict.fromkeys(options, 1.0 / n)
        return {name: p / total for name, p in action_probs.items()}

    raise QuestionError(
        f"question '{question_id}': learned mode's options must be exactly the four decision classes "
        f"{DECISION_CLASSES} or a subset of the known actions {KNOWN_ACTIONS}; got {sorted(option_names)}"
    )


def answer_questions_learned(
    state: JSONDict,
    questions: JSONDict,
    model: LearnedModel,
    embedder: EmbeddingsBackend | None = None,
    # kept for signature parity with core.answer_questions; inference
    # here is cheap enough that this module never needs multiple workers.
    max_workers: int = 4,
    deadline: _deadline.Deadline | None = None,
) -> JSONDict:
    """The learned-mode equivalent of
    :func:`decisionhead.core.answer_questions`: same return shape
    (``{"answers", "audit", "usage", "model"}``) and the same audit-row
    field names, so both modes can be written through one
    :class:`~decisionhead.server.AuditLog`. Only ``choice`` questions are
    supported (see the module docstring); a ``noul`` or ``score`` question
    raises :class:`~decisionhead.core.QuestionError`.
    """
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("'questions' must be a non-empty object")

    model_name = str(model.metadata.get("name", "decisionhead-learned"))
    req_hash = request_hash(state, questions, model_name)
    if deadline is not None:
        deadline.check()  # the embedder call below may reach the network
    class_probs = model.class_probabilities(state, embedder=embedder)

    answers: JSONDict = {}
    audit: list[JSONDict] = []
    for qid, qdef in questions.items():
        qnorm = normalize_question(qid, qdef)
        if qnorm["type"] != "choice":
            raise QuestionError(
                f"question '{qid}': head.mode='learned' only answers 'choice' questions (got {qnorm['type']!r})"
            )
        t0 = time.time()
        action_probs = _action_probabilities(class_probs, qnorm["options"], qid)
        latency_s = time.time() - t0
        winner, confidence = top_two_gap(action_probs)
        answers[qid] = {
            "type": "choice",
            "choice": winner,
            "probabilities": action_probs,
            "confidence": confidence,
        }
        audit.append(
            {
                "request_hash": req_hash,
                "question_id": qid,
                "model": model_name,
                "backend": BACKEND_NAME,
                "unparsed_mass": 0.0,
                "calibrated": model.temperature != 1.0,
                "latency_s": latency_s,
                "raw_distribution": dict(class_probs),
            }
        )

    return {
        "answers": answers,
        "audit": audit,
        "usage": {"input_tokens": None, "output_tokens": None},
        "model": model_name,
    }


class LearnedBackendAdapter:
    """A minimal stand-in satisfying :class:`decisionhead.backends.Backend`'s
    shape (a ``model`` attribute plus ``token_distribution`` and
    ``sample_completions``), so :mod:`decisionhead.server` can use its
    existing ``/health``, ``/v1/models``, and readiness-check plumbing
    unchanged when ``mode = "learned"`` -- none of that plumbing calls
    either method on the object it holds, so this never actually needs to
    answer one; it exists purely so those endpoints have something with a
    ``.model`` name to report. Calling either method directly (which
    nothing in this codebase does for a learned-mode server) raises
    rather than fabricating an answer."""

    def __init__(self, model: LearnedModel) -> None:
        self.model = str(model.metadata.get("name", "decisionhead-learned"))

    def _unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "LearnedBackendAdapter does not serve LLM calls: mode='learned' answers requests "
            "via decisionhead.learned.answer_questions_learned directly, never through a backend call"
        )

    def token_distribution(
        self, prefix_messages: list[JSONDict], question_text: str, allowed_tokens: Sequence[str]
    ) -> JSONDict:
        """Always raises ``NotImplementedError``: a learned-mode server
        has no LLM behind it. The server refuses a request that
        overrides its mode to ``"fast"`` or ``"deliberate"`` with a
        ``400`` before this is ever reached (see
        :func:`decisionhead.server.resolve_effective_mode`)."""
        raise self._unsupported()

    def sample_completions(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_answers: Sequence[str],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> tuple[list[str], JSONDict]:
        """Always raises ``NotImplementedError``; see
        :meth:`token_distribution`."""
        raise self._unsupported()


def run_selftest() -> bool:
    """Standalone smoke check; see ``tests/test_learned.py`` for the
    thorough version."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    names = feature_names(cfg)
    classes = ["BLOCKED", "DONE", "INFO", "REASON"]
    classifier = MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0)
    scaler = StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names))
    model = LearnedModel(classifier=classifier, scaler=scaler, feature_config=cfg, classes=classes, temperature=1.5)

    state: JSONDict = {
        "requirements": [{"item_id": "a"}],
        "open_items": ["a"],
        "history": [],
        "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
        "steps_used": 0,
    }
    probs = model.class_probabilities(state)
    check("class_probabilities covers every class", set(probs.keys()) == set(classes))
    check("class_probabilities sums to 1", abs(sum(probs.values()) - 1.0) < 1e-6)

    d = model.to_dict()
    restored = LearnedModel.from_dict(d)
    check("LearnedModel round-trips through to_dict/from_dict", restored.class_probabilities(state) == probs)

    questions = {"next_step": {"type": "choice", "instructions": "which class?", "options": list(DECISION_CLASSES)}}
    result = answer_questions_learned(state, questions, model)
    check("answer_questions_learned returns every question id", set(result["answers"].keys()) == {"next_step"})
    check(
        "answer_questions_learned's choice sums to 1",
        abs(sum(result["answers"]["next_step"]["probabilities"].values()) - 1.0) < 1e-6,
    )
    check("audit has one row per question", len(result["audit"]) == 1)
    check("audit row is flagged calibrated when temperature != 1", result["audit"][0]["calibrated"] is True)
    check("usage has no LLM tokens", result["usage"] == {"input_tokens": None, "output_tokens": None})

    action_questions = {
        "next_step": {
            "type": "choice",
            "instructions": "which action?",
            "options": ["RETRIEVE", "REQUEST", "CONTINUE"],
        }
    }
    action_result = answer_questions_learned(state, action_questions, model)
    check(
        "a subset-of-actions option set is answered too",
        set(action_result["answers"]["next_step"]["probabilities"].keys()) == {"RETRIEVE", "REQUEST", "CONTINUE"},
    )

    try:
        answer_questions_learned(state, {"q": {"type": "noul", "instructions": "x"}}, model)
        check("a noul question is rejected in learned mode", False)
    except QuestionError:
        check("a noul question is rejected in learned mode", True)

    try:
        answer_questions_learned(
            state, {"q": {"type": "choice", "instructions": "x", "options": ["FOO", "BAR"]}}, model
        )
        check("an unrecognized option set is rejected", False)
    except QuestionError:
        check("an unrecognized option set is rejected", True)

    adapter = LearnedBackendAdapter(model)
    check("LearnedBackendAdapter exposes a model name", isinstance(adapter.model, str))
    try:
        adapter.token_distribution([], "x", ["A"])
        check("LearnedBackendAdapter.token_distribution raises", False)
    except NotImplementedError:
        check("LearnedBackendAdapter.token_distribution raises", True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
