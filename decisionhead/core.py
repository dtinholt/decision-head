"""Turns a Jev-shaped question set into Jev-shaped answers, against any
backend from :mod:`decisionhead.backends`.

Design choices worth stating up front (also covered in the README and in
``docs/design.md``):

* The state is rendered into a chat prefix ONCE per request (a system turn
  plus a user turn holding the state), and every question is asked as a
  separate final user turn appended to that same prefix, in its own model
  call. This is what "evaluated in parallel and in isolation" means here:
  no question's prompt ever contains another question's text, but all
  questions about one state share an identical prefix -- which is also what
  lets a serving engine's prefix cache make question 2..N of a request
  cheaper than question 1 (measured in :mod:`decisionhead.bench`).

* Calibration is temperature scaling on the logit of a probability:
  ``p' = sigmoid(logit(p) / T)``, exactly as specified. For a noul question
  that is applied directly to the single ``P(yes)``. For choice and score
  questions there is more than one probability per question; the same
  elementwise transform is applied to every option's probability and the
  result is renormalized to sum to 1. This is a generalization the spec
  does not spell out for the multi-option case; it is the natural one
  (apply the documented transform to every probability there is) and it
  degenerates correctly to the noul case when there are exactly two
  options.

* Confidence for choice and score answers is the top probability minus the
  second, taken after calibration when calibration is applied (so the
  reported confidence matches the reported probabilities).

* Score's numeric value: if the winning level's own name parses as a
  number (``"1"``, ``"2.5"``), that number is used; otherwise the 1-based
  rank of the level in the order given is used. Levels must be supplied in
  ascending order (documented in the README); nothing here re-sorts them,
  because "low"/"medium"/"high" has no sort order without that assumption.

Invariants (each pinned by tests in ``tests/test_core.py`` or
``tests/test_security_review_2026_10_06.py``):

* no question's prompt contains another question's text;
* every fast-mode answer carries ``unparsed_mass``, ``no_family_match``,
  ``non_finite_logprob_count`` and ``degraded``;
* a reading with no allowed token is ``degraded`` with ``null`` values
  and is never written to the cache;
* calibration never produces a non-finite probability, and a
  calibration file that could is refused by :func:`validate_calibration`;
* no backend call starts after the request's deadline has passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from typing import Any, Protocol

from . import deadline as _deadline

JSONDict = dict[str, Any]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

NOUL_QUESTION_TEMPLATE = "Question: {instructions}\nAnswer with exactly one word: Yes or No."
CHOICE_QUESTION_TEMPLATE = (
    "Question: {instructions}\nOptions:\n{option_lines}\n"
    "Answer with exactly one letter naming your choice."
)
SCORE_QUESTION_TEMPLATE = (
    "Question: {instructions}\nLevels (low to high):\n{option_lines}\n"
    "Answer with exactly one letter naming the level."
)

SYSTEM_PREFIX = (
    "You answer typed questions about the state below, one question per turn, "
    "each in isolation. Respond with exactly the single token the question "
    "asks for and nothing else: no punctuation, no explanation."
)


class QuestionError(ValueError):
    """A question in the request is malformed.

    Callers (:mod:`decisionhead.server`) turn this into a 4xx with the
    message verbatim.
    """


class TokenDistributionBackend(Protocol):
    """Structural type every backend in :mod:`decisionhead.backends`
    satisfies. ``model`` is a plain attribute, not a property, on every
    concrete implementation."""

    model: str

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: list[str],
    ) -> JSONDict:
        """Returns ``{token: prob, ..., "_unparsed": mass, "_raw": {...}}``
        for the single token generated after ``prefix_messages`` plus a
        final user turn holding ``question_text``."""
        ...


# --------------------------------------------------------------------- state/question shaping


def render_state(state: str | JSONDict | list[Any]) -> str:
    """The vendor shape allows state as a string or an object; either way we
    need one string to put in the shared prefix."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, indent=2)


def _options_map(qid: str, qdef: JSONDict, *field_names: str) -> dict[str, str]:
    """Reads an ordered option/level set from whichever of the accepted
    field names is present (the vendor shape says "criteria/options/levels
    as applicable"; we accept any of them so a client does not have to know
    which noun we prefer for which type). Accepts either a dict (name ->
    description) or a plain list of names (description defaults to the
    name). Order is preserved -- it matters for score."""
    for field in field_names:
        if field in qdef:
            raw = qdef[field]
            if isinstance(raw, dict):
                if len(raw) < 2:
                    raise QuestionError(f"question '{qid}': '{field}' needs at least 2 entries")
                return {str(k): str(v) for k, v in raw.items()}
            if isinstance(raw, list):
                if len(raw) < 2:
                    raise QuestionError(f"question '{qid}': '{field}' needs at least 2 entries")
                return {str(v): str(v) for v in raw}
            raise QuestionError(f"question '{qid}': '{field}' must be an object or a list")
    raise QuestionError(f"question '{qid}': expected one of {field_names}")


def normalize_question(qid: str, qdef: JSONDict) -> JSONDict:
    """Validates one question and returns a normalized internal form:
    ``{"type": ..., "instructions": ..., "options": {name: description}}``
    for choice/score, or ``{"type": "noul", "instructions": ...}`` for
    noul."""
    if not isinstance(qdef, dict):
        raise QuestionError(f"question '{qid}': must be an object")
    qtype = qdef.get("type")
    if qtype not in ("noul", "choice", "score"):
        raise QuestionError(
            f"question '{qid}': type must be one of noul, choice, score (got {qtype!r})"
        )
    instructions = qdef.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise QuestionError(
            f"question '{qid}': 'instructions' is required and must be a non-empty string"
        )
    if qtype == "noul":
        return {"type": "noul", "instructions": instructions}
    if qtype == "choice":
        options = _options_map(qid, qdef, "options", "criteria")
    else:
        options = _options_map(qid, qdef, "levels", "criteria", "options")
    if len(options) > 26:
        raise QuestionError(
            f"question '{qid}': more than 26 options is not supported (single-letter reading)"
        )
    _validate_claim_guard_fields(qid, qdef, options)
    return {"type": qtype, "instructions": instructions, "options": options}


def _validate_claim_guard_fields(qid: str, qdef: JSONDict, options: dict[str, str]) -> None:
    """Type-checks the optional claim-guard inputs read by
    :func:`decisionhead.deliberate.apply_claim_guard`, so a malformed one
    is an ordinary 400 instead of a substring match (a string
    ``terminal_answers`` made ``"COMPLETE" in "CLAIM_COMPLETE"`` true) or
    an ``AttributeError`` 500 (a list ``completion_evidence``) -- see
    the 2026-10-06 security review, finding 5."""
    if "terminal_answers" in qdef:
        terminal = qdef["terminal_answers"]
        if not isinstance(terminal, list) or not all(isinstance(t, str) for t in terminal):
            raise QuestionError(f"question '{qid}': 'terminal_answers' must be a list of option names")
        unknown = [t for t in terminal if t not in options]
        if unknown:
            raise QuestionError(f"question '{qid}': 'terminal_answers' names {len(unknown)} unknown option(s)")
    if "completion_evidence" in qdef:
        evidence = qdef["completion_evidence"]
        if not isinstance(evidence, dict):
            raise QuestionError(f"question '{qid}': 'completion_evidence' must be an object")
        count = evidence.get("open_obligations_count", 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise QuestionError(
                f"question '{qid}': 'completion_evidence.open_obligations_count' must be a non-negative integer"
            )
        unmet = evidence.get("unmet_requirements", [])
        if not isinstance(unmet, list):
            raise QuestionError(f"question '{qid}': 'completion_evidence.unmet_requirements' must be a list")


def _letters_for(options: dict[str, str]) -> dict[str, str]:
    return {name: LETTERS[i] for i, name in enumerate(options.keys())}


def build_question_text(
    qtype: str, qnorm: JSONDict, letters: dict[str, str] | None = None
) -> tuple[str, list[str]]:
    """Returns ``(question_text, allowed_tokens)`` for one normalized
    question."""
    if qtype == "noul":
        return NOUL_QUESTION_TEMPLATE.format(instructions=qnorm["instructions"]), ["Yes", "No"]
    options: dict[str, str] = qnorm["options"]
    letters = letters or _letters_for(options)
    lines = "\n".join(f"{letters[name]}) {desc}" for name, desc in options.items())
    allowed = [letters[name] for name in options.keys()]
    template = CHOICE_QUESTION_TEMPLATE if qtype == "choice" else SCORE_QUESTION_TEMPLATE
    return (
        template.format(instructions=qnorm["instructions"], option_lines=lines),
        allowed,
    )


# --------------------------------------------------------------------- calibration


def _logit(p: float) -> float:
    # Clamp before the log: a reading of exactly 0 or 1 (common when the
    # model puts all visible mass on one token) would otherwise be
    # log(0) or a division by zero. 1e-9 keeps the logit within about
    # +/-20.7, so dividing by any T > 0 stays finite.
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    # Two branches so math.exp only ever sees a non-positive argument
    # and cannot overflow for large |x|.
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def calibrate_prob(p: float, temperature: float | None) -> float:
    """``p' = sigmoid(logit(p) / T)``. ``T=1`` is a no-op (up to float
    noise). A temperature that is not a finite positive number is a
    passthrough; ``NaN`` used to slip past the ``<= 0`` guard and return
    ``nan`` (2026-10-06 review, finding 4). The server never gets here
    with one: :func:`validate_calibration` refuses such a file at
    start-up."""
    if temperature is None or not math.isfinite(temperature) or temperature <= 0:
        return p
    return _sigmoid(_logit(p) / temperature)


def _calibration_entries(calibration: JSONDict) -> JSONDict:
    per_type = calibration.get("per_type", calibration)  # tolerate a flat {"noul": {...}} shape too
    if not isinstance(per_type, dict):
        raise ValueError("calibration 'per_type' must be an object")
    return per_type


def _entry_temperature(entry: Any) -> Any:
    """The raw temperature value of one per-type entry, ``None`` if the
    entry names none. Explicit key checks, not ``or``: ``T = 0`` used to
    fall through to ``temperature`` and then to "uncalibrated"."""
    if not isinstance(entry, dict):
        return None
    if "T" in entry:
        return entry["T"]
    return entry.get("temperature")


def validate_calibration(calibration: Any) -> None:
    """Raises ``ValueError`` unless ``calibration`` is a usable
    calibration document: an object whose per-type entries (under
    ``per_type``, or at the top level) are objects whose temperature,
    when present, is a finite number greater than zero. Called by
    :func:`decisionhead.server.load_calibration` so a bad file stops the
    server at start-up instead of failing (or silently skewing) every
    request (2026-10-06 review, findings 4 and 15)."""
    if not isinstance(calibration, dict):
        raise ValueError("calibration file must contain a JSON object")
    for qtype in ("noul", "choice", "score"):
        entry = _calibration_entries(calibration).get(qtype)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"calibration entry for {qtype!r} must be an object")
        value = _entry_temperature(entry)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"calibration temperature for {qtype!r} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"calibration temperature for {qtype!r} must be a finite number > 0")


def _temperature_for(calibration: JSONDict | None, qtype: str) -> float | None:
    if not calibration:
        return None
    value = _entry_temperature(_calibration_entries(calibration).get(qtype))
    return float(value) if value is not None else None


def apply_calibration_noul(p_yes: float, temperature: float | None) -> tuple[float, bool]:
    """Applies :func:`calibrate_prob` to a single noul probability.
    Returns ``(p_yes, calibrated_flag)``; ``temperature is None`` means no
    calibration was configured for this type, and the flag is ``False``."""
    if temperature is None:
        return p_yes, False
    return calibrate_prob(p_yes, temperature), True


def apply_calibration_distribution(
    probs: dict[str, float], temperature: float | None
) -> tuple[dict[str, float], bool]:
    """Applies :func:`calibrate_prob` elementwise to every probability in
    a choice/score distribution, then renormalizes to sum to 1. Returns
    ``(probs, calibrated_flag)``; ``temperature is None`` passes ``probs``
    through unchanged with the flag ``False``."""
    if temperature is None:
        return probs, False
    # Each option is transformed as if it were its own binary question,
    # then renormalized. The transform is monotonic, so the winner never
    # changes. With two options the pair already sums to 1
    # (sigmoid(z) + sigmoid(-z) = 1), so this reduces exactly to the noul
    # transform.
    transformed = {k: calibrate_prob(v, temperature) for k, v in probs.items()}
    total = sum(transformed.values())
    if total <= 0:
        n = len(transformed) or 1
        return {k: 1.0 / n for k in transformed}, True
    return {k: v / total for k, v in transformed.items()}, True


def top_two_gap(probs: dict[str, float]) -> tuple[str, float]:
    """``(winner, confidence)`` where confidence is the top probability
    minus the second, as specified. With only one option confidence is
    that option's own probability."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranked[0][0]
    if len(ranked) < 2:
        return winner, ranked[0][1]
    return winner, ranked[0][1] - ranked[1][1]


def _score_value(level_name: str, options: dict[str, str]) -> float | None:
    """The numeric score for a winning level: the level name itself if it
    parses as a FINITE number, else its 1-based rank. A level named
    ``"nan"``, ``"inf"`` or ``"1e999"`` used to come back as a
    non-finite score and reach the response as a bare ``Infinity`` token
    (2026-10-06 review, finding 4); it now falls back to its rank.
    ``None`` only for a name that is not one of ``options`` at all."""
    try:
        value = float(level_name)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    for i, name in enumerate(options.keys()):
        if name == level_name:
            return float(i + 1)
    return None


# --------------------------------------------------------------------- answering


def request_hash(state: str | JSONDict | list[Any], questions: JSONDict, model: str) -> str:
    """A stable SHA-256 hex digest of ``(state, questions, model)``, used
    to tag every audit row from one request with a shared id."""
    payload = json.dumps(
        {"state": state, "questions": questions, "model": model}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _answer_one(
    qid: str,
    qdef: JSONDict,
    prefix_messages: list[JSONDict],
    backend: TokenDistributionBackend,
    calibration: JSONDict | None,
    req_hash: str,
    model: str,
    backend_name: str,
    *,
    cache: Any = None,
    cache_state_json: str = "",
    backend_route: str = "",
    mode: str = "fast",
    calibration_version: str = "",
    max_concurrency_override: int | None = None,
    pending_cache_writes: list[tuple[str, JSONDict]] | None = None,
) -> tuple[JSONDict, JSONDict, JSONDict]:
    """Answers one question: cache lookup, deadline check, one backend
    call, degradation check, calibration, and the audit row. Returns
    ``(answer, audit_row, usage)``. Raises :class:`QuestionError` for a
    malformed question, :class:`~decisionhead.deadline.DeadlineExceeded`
    if the request has timed out, and whatever the backend raises
    (normally :class:`~decisionhead.backends.BackendError`).

    With ``pending_cache_writes`` given, a cacheable answer is appended
    there instead of being written to ``cache``, so the caller can commit
    it only after the whole request succeeds."""
    qnorm = normalize_question(qid, qdef)
    qtype = qnorm["type"]
    letters = _letters_for(qnorm["options"]) if qtype != "noul" else None
    question_text, allowed_tokens = build_question_text(qtype, qnorm, letters)

    cache_key_str: str | None = None
    if cache is not None:
        from .cache import cache_key  # local import: keeps core.py's only optional dependency lazy

        cache_key_str = cache_key(
            model=model,
            backend_route=backend_route,
            state_json=cache_state_json,
            question_id=qid,
            question_text=question_text,
            mode=mode,
            calibration_version=calibration_version,
        )
        cached = cache.get(cache_key_str)
        if cached is not None:
            cached_audit_row: JSONDict = dict(cached["audit_row"])
            cached_audit_row["request_hash"] = req_hash
            cached_audit_row["cache_hit"] = True
            cached_audit_row["original_trace_id"] = cached["original_trace_id"]
            return dict(cached["answer"]), cached_audit_row, dict(cached["usage"])

    # Never start a backend call for a request that has already timed out
    # or been abandoned (2026-10-06 review, finding 1).
    _deadline.check_current()
    t0 = time.time()
    call_kwargs: JSONDict = {}
    if max_concurrency_override is not None and getattr(backend, "accepts_concurrency_override", False):
        call_kwargs["max_concurrency_override"] = max_concurrency_override
    dist = backend.token_distribution(prefix_messages, question_text, allowed_tokens, **call_kwargs)
    latency_s = time.time() - t0

    unparsed = dist.get("_unparsed", 0.0)
    raw = dist.get("_raw")
    token_probs = {t: dist[t] for t in allowed_tokens if t in dist}
    # Degradation signals computed by the backend (see
    # decisionhead.backends.distribution_from_top_logprobs). They used to
    # stop at _raw; SECURITY.md promised them in the audit trail, and a
    # no-family-match reading came back as a plausible 0.5 (2026-10-06
    # review, finding 6). A reading with NO allowed token at all carries
    # no information: it is returned as degraded, with null values.
    raw_meta: JSONDict = raw if isinstance(raw, dict) else {}
    no_family_match = bool(raw_meta.get("no_family_match", False))
    non_finite_count = raw_meta.get("non_finite_logprob_count", 0)
    if not isinstance(non_finite_count, int) or isinstance(non_finite_count, bool):
        non_finite_count = 0
    degraded = no_family_match
    signals: JSONDict = {
        "unparsed_mass": unparsed,
        "no_family_match": no_family_match,
        "non_finite_logprob_count": non_finite_count,
        "degraded": degraded,
    }

    answer: JSONDict
    audit_extra: JSONDict
    calibrated = False
    if qtype == "noul":
        p_yes_raw = token_probs.get("Yes", 0.5)
        if degraded:
            answer = {"type": "noul", "noul": None, **signals}
        else:
            temperature = _temperature_for(calibration, "noul")
            p_yes, calibrated = apply_calibration_noul(p_yes_raw, temperature)
            answer = {"type": "noul", "noul": p_yes, **signals}
        audit_extra = {"raw_distribution": token_probs}
    else:
        by_option_raw = {name: token_probs.get(letters[name], 0.0) for name in qnorm["options"].keys()}  # type: ignore[index]
        value_key = "choice" if qtype == "choice" else "score"
        if degraded:
            answer = {"type": qtype, value_key: None, "probabilities": None, "confidence": None, **signals}
        else:
            temperature = _temperature_for(calibration, qtype)
            by_option, calibrated = apply_calibration_distribution(by_option_raw, temperature)
            winner, confidence = top_two_gap(by_option)
            value: Any = winner if qtype == "choice" else _score_value(winner, qnorm["options"])
            answer = {
                "type": qtype,
                value_key: value,
                "probabilities": by_option,
                "confidence": confidence,
                **signals,
            }
        audit_extra = {"raw_distribution": by_option_raw, "letters": letters}

    audit_row: JSONDict = {
        "request_hash": req_hash,
        "question_id": qid,
        "model": model,
        "backend": backend_name,
        "calibrated": calibrated,
        "latency_s": latency_s,
        "cache_hit": False,
        **signals,
        **audit_extra,
    }
    usage: JSONDict = (raw or {}).get("usage") or {} if isinstance(raw, dict) else {}

    # A degraded reading is never cached: it reflects a transient backend
    # fault (wrong template, thinking enabled), not an answer worth
    # replaying for a day.
    if cache is not None and cache_key_str is not None and not degraded:
        entry: JSONDict = {"answer": answer, "usage": usage, "audit_row": audit_row, "original_trace_id": req_hash}
        if pending_cache_writes is not None:
            pending_cache_writes.append((cache_key_str, entry))
        else:
            cache.set(cache_key_str, entry)

    return answer, audit_row, usage


def answer_questions(
    state: str | JSONDict | list[Any],
    questions: JSONDict,
    backend: TokenDistributionBackend,
    calibration: JSONDict | None = None,
    max_workers: int = 4,
    backend_name: str | None = None,
    *,
    cache: Any = None,
    backend_route: str = "",
    mode: str = "fast",
    calibration_version: str = "",
    max_concurrency_override: int | None = None,
    deadline: _deadline.Deadline | None = None,
    defer_cache_writes: bool = False,
) -> JSONDict:
    """Evaluates every question in ``questions`` about ``state``, each in
    its own model call sharing one state prefix, across up to
    ``max_workers`` threads. Returns::

        {"answers": {qid: <jev-shaped answer>}, "audit": [audit_row, ...],
         "usage": {"input_tokens": n, "output_tokens": n}, "model": ...}

    ``backend_name`` labels the audit rows (e.g. "ollama", "openai",
    "fake"); defaults to the backend class name if not given.

    ``cache`` (a :class:`decisionhead.cache.ResponseCache`, optional):
    when given, each question is looked up before calling the backend and
    stored after, keyed on ``(model, backend_route, canonical state JSON,
    question id + text, mode, calibration_version)`` -- see
    ``decisionhead.cache.cache_key``. A hit skips the backend call
    entirely and its audit row carries ``cache_hit: True`` plus the
    ``original_trace_id`` of the request that first computed it.
    ``backend_route``/``mode``/``calibration_version`` are cache-key
    inputs only and have no effect when ``cache`` is ``None``.

    ``max_concurrency_override``: passed through to the backend's
    ``token_distribution`` call, if (and only if) it declares
    ``accepts_concurrency_override`` (see
    :class:`decisionhead.backends.ConcurrencyLimitedBackend`); a no-op for
    any other backend.

    ``deadline`` (optional): checked before every backend call, in every
    worker thread, so the questions of a request that has timed out are
    abandoned instead of still being sent to the backend (2026-10-06
    review, finding 1).

    ``defer_cache_writes``: when ``True``, nothing is written to
    ``cache`` here; the entries are returned as ``result["cache_writes"]``
    (a list of ``(key, value)``) for the caller to commit once the
    request has actually succeeded. :mod:`decisionhead.server` uses this
    so a request that timed out or failed never populates the cache
    (2026-10-06 review, finding 8).
    """
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("'questions' must be a non-empty object")

    model = getattr(backend, "model", "unknown")
    bname = backend_name or type(backend).__name__
    pending: list[tuple[str, JSONDict]] | None = [] if defer_cache_writes else None
    pending_lock = threading.Lock()
    state_text = render_state(state)
    prefix_messages: list[JSONDict] = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": f"State:\n{state_text}"},
    ]
    req_hash = request_hash(state, questions, model)
    cache_state_json = json.dumps(state, sort_keys=True, default=str)

    qids = list(questions.keys())
    results: list[tuple[str, JSONDict, JSONDict, JSONDict] | None] = [None] * len(qids)
    errors: list[BaseException | None] = [None] * len(qids)

    def worker(indices: list[int]) -> None:
        _deadline.set_current(deadline)
        for i in indices:
            qid = qids[i]
            local_pending: list[tuple[str, JSONDict]] | None = [] if pending is not None else None
            try:
                answer, audit_row, usage = _answer_one(
                    qid,
                    questions[qid],
                    prefix_messages,
                    backend,
                    calibration,
                    req_hash,
                    model,
                    bname,
                    cache=cache,
                    cache_state_json=cache_state_json,
                    backend_route=backend_route,
                    mode=mode,
                    calibration_version=calibration_version,
                    max_concurrency_override=max_concurrency_override,
                    pending_cache_writes=local_pending,
                )
                results[i] = (qid, answer, audit_row, usage)
                if pending is not None and local_pending:
                    with pending_lock:
                        pending.extend(local_pending)
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the caller after join
                errors[i] = exc

    n_workers = max(1, min(max_workers, len(qids)))
    buckets: list[list[int]] = [[] for _ in range(n_workers)]
    for i in range(len(qids)):
        buckets[i % n_workers].append(i)

    threads = [threading.Thread(target=worker, args=(b,)) for b in buckets if b]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for exc in errors:
        if exc is not None:
            raise exc

    answers: JSONDict = {}
    audit: list[JSONDict] = []
    # usage is pulled straight from each backend call's raw response, if the
    # provider reported it (ollama's prompt_eval_count/eval_count, or an
    # OpenAI-compatible usage block); unreported fields are simply excluded
    # from the sum rather than counted as 0, so a provider that never
    # reports usage yields usage: {"input_tokens": null, "output_tokens": null}
    # instead of a fabricated zero.
    in_tokens: list[float] = []
    out_tokens: list[float] = []
    for entry in results:
        assert entry is not None  # every slot is filled or an exception was raised above
        qid, answer, audit_row, usage = entry
        answers[qid] = answer
        audit.append(audit_row)
        if isinstance(usage.get("input_tokens"), (int, float)):
            in_tokens.append(usage["input_tokens"])
        if isinstance(usage.get("output_tokens"), (int, float)):
            out_tokens.append(usage["output_tokens"])
    usage_summary = {
        "input_tokens": sum(in_tokens) if in_tokens else None,
        "output_tokens": sum(out_tokens) if out_tokens else None,
    }
    result: JSONDict = {
        "answers": answers,
        "audit": audit,
        "usage": usage_summary,
        "model": model,
    }
    if pending is not None:
        result["cache_writes"] = pending
    return result


# --------------------------------------------------------------------- low-confidence band


def apply_abstain_band(result: JSONDict, band: tuple[float, float] | None, mode: str = "off") -> JSONDict:
    """Pure post-processing pass over an :func:`answer_questions`-shaped
    ``result`` -- no model call, no mutation of any probability.

    When ``band`` is ``None`` (disabled, the default), returns ``result``
    unchanged. Otherwise, every decisive ``noul`` answer whose probability
    falls inside ``[lo, hi]`` (inclusive) is flagged: ``low_confidence:
    True`` is set on that answer and on its audit row (matched by
    ``question_id``). When ``mode == "handoff"``, that answer additionally
    gets ``recommended_action: "handoff"`` and a ``handoff_reason`` string
    (also copied onto the audit row). Finally, ``result["low_confidence"]``
    is set to ``True`` if any answer was flagged, ``False`` otherwise, so a
    caller can check the whole response at a glance.

    Mutates and returns the same ``result`` (and the answer/audit dicts
    inside it) for simplicity -- there is no other consumer of those
    objects by the time this runs (see ``decisionhead.server``). Always
    sets ``result["low_confidence"]`` (``False`` when ``band`` is
    ``None``), so a caller can rely on the key being present either way.

    A ``degraded`` answer (no usable reading at all; see
    :func:`_answer_one` and the deliberate-mode equivalent) is ALWAYS
    flagged ``low_confidence``, and gets the handoff recommendation in
    ``"handoff"`` mode, whether or not a band is configured.
    ``result["degraded"]`` is set to whether any answer was degraded
    (2026-10-06 review, finding 6).

    A deliberate answer with ``n_missing_samples > 0`` (the backend
    returned fewer than ``k`` samples) is likewise ALWAYS flagged
    ``low_confidence``, with the handoff recommendation in ``"handoff"``
    mode (2026-10-06 confirmation, R4)."""
    audit_by_qid = {row.get("question_id"): row for row in result.get("audit", [])}
    any_low_confidence = False
    any_degraded = False
    for qid, answer in result.get("answers", {}).items():
        if not answer.get("degraded"):
            continue
        any_degraded = any_low_confidence = True
        answer["low_confidence"] = True
        row = audit_by_qid.get(qid)
        if row is not None:
            row["low_confidence"] = True
        if mode == "handoff":
            reason = "no usable reading from the model (degraded answer)"
            answer["recommended_action"] = "handoff"
            answer["handoff_reason"] = reason
            if row is not None:
                row["recommended_action"] = "handoff"
                row["handoff_reason"] = reason
    result["degraded"] = any_degraded
    for qid, answer in result.get("answers", {}).items():
        missing = answer.get("n_missing_samples")
        if answer.get("degraded") or not isinstance(missing, int) or missing <= 0:
            continue
        any_low_confidence = True
        answer["low_confidence"] = True
        row = audit_by_qid.get(qid)
        if row is not None:
            row["low_confidence"] = True
        if mode == "handoff":
            reason = f"the backend returned {missing} fewer sample(s) than requested"
            answer["recommended_action"] = "handoff"
            answer["handoff_reason"] = reason
            if row is not None:
                row["recommended_action"] = "handoff"
                row["handoff_reason"] = reason
    if band is None:
        result["low_confidence"] = any_low_confidence
        return result
    lo, hi = band
    for qid, answer in result.get("answers", {}).items():
        if answer.get("type") != "noul" or answer.get("degraded"):
            continue
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not (lo <= p <= hi):
            continue
        any_low_confidence = True
        answer["low_confidence"] = True
        row = audit_by_qid.get(qid)
        if row is not None:
            row["low_confidence"] = True
        if mode == "handoff":
            reason = f"noul probability {p:.4f} falls within the abstain band [{lo}, {hi}]"
            answer["recommended_action"] = "handoff"
            answer["handoff_reason"] = reason
            if row is not None:
                row["recommended_action"] = "handoff"
                row["handoff_reason"] = reason
    result["low_confidence"] = any_low_confidence
    return result


# --------------------------------------------------------------------- isolation check


def _drift_for(qtype: str, alone: JSONDict, together: JSONDict) -> JSONDict:
    if alone.get("degraded") or together.get("degraded"):
        # No reading to compare; report it rather than inventing a drift.
        changed = bool(alone.get("degraded")) != bool(together.get("degraded"))
        return {"metric": "degraded", "value": 0.0, "decision_changed": changed}
    if qtype == "noul":
        d = abs(alone["noul"] - together["noul"])
        return {
            "metric": "abs_prob_diff",
            "value": d,
            "decision_changed": (alone["noul"] >= 0.5) != (together["noul"] >= 0.5),
        }
    a_key = alone.get("choice", alone.get("score"))
    t_key = together.get("choice", together.get("score"))
    a_probs, t_probs = alone["probabilities"], together["probabilities"]
    l1 = sum(abs(a_probs.get(k, 0.0) - t_probs.get(k, 0.0)) for k in set(a_probs) | set(t_probs))
    return {"metric": "l1_prob_diff", "value": l1, "decision_changed": a_key != t_key}


def isolation_check(
    state: str | JSONDict | list[Any],
    questions: JSONDict,
    backend: TokenDistributionBackend,
    calibration: JSONDict | None = None,
    max_workers: int = 4,
) -> JSONDict:
    """Answers every question alone (the normal :func:`answer_questions`
    path) and again "together" -- sharing a prefix that additionally lists
    every other question's instructions, so the model can see them while
    still being asked for one single-token answer at a time. Isolation, as
    Jev claims it, means these two should not differ; this reports where
    they do. (The single-token generation format is unchanged in both
    cases, so the comparison isolates the effect of context pollution, not
    of a different answer format.)"""
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("'questions' must be a non-empty object")

    alone_result = answer_questions(state, questions, backend, calibration, max_workers)

    state_text = render_state(state)
    normalized = {qid: normalize_question(qid, qdef) for qid, qdef in questions.items()}
    all_texts = [f"- ({qid}) {qn['instructions']}" for qid, qn in normalized.items()]
    together_prefix: list[JSONDict] = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {
            "role": "user",
            "content": f"State:\n{state_text}\n\nAll questions under consideration:\n"
            + "\n".join(all_texts),
        },
    ]
    model = getattr(backend, "model", "unknown")
    bname = type(backend).__name__
    req_hash = request_hash(state, questions, model) + "-together"

    together_answers: JSONDict = {}
    for qid, qdef in questions.items():
        answer, _audit, _usage = _answer_one(
            qid, qdef, together_prefix, backend, calibration, req_hash, model, bname
        )
        together_answers[qid] = answer

    drift: JSONDict = {}
    for qid, qdef in questions.items():
        qtype = qdef["type"]
        drift[qid] = _drift_for(qtype, alone_result["answers"][qid], together_answers[qid])

    any_decision_changed = any(d["decision_changed"] for d in drift.values())
    return {
        "alone": alone_result["answers"],
        "together": together_answers,
        "drift": drift,
        "any_decision_changed": any_decision_changed,
        "max_drift": max((d["value"] for d in drift.values()), default=0.0),
    }


# --------------------------------------------------------------------- selftest


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed
    (used by the ``dh_core.py`` compatibility shim and by
    ``decisionhead selftest``). The equivalent, more thorough assertions
    live in ``tests/test_core.py``."""
    from .backends import FakeBackend

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    n1 = normalize_question("q1", {"type": "noul", "instructions": "Is X true?"})
    check("noul question normalizes", n1 == {"type": "noul", "instructions": "Is X true?"})
    n2 = normalize_question(
        "q2", {"type": "choice", "instructions": "Pick one", "criteria": {"A": "desc a", "B": "desc b"}}
    )
    check("choice question reads 'criteria' as options", n2["options"] == {"A": "desc a", "B": "desc b"})
    n3 = normalize_question("q3", {"type": "score", "instructions": "Rate it", "levels": ["low", "medium", "high"]})
    check(
        "score question reads 'levels' list, order preserved",
        list(n3["options"].keys()) == ["low", "medium", "high"],
    )
    try:
        normalize_question("bad", {"type": "noul"})
        check("missing instructions rejected", False)
    except QuestionError:
        check("missing instructions rejected", True)
    try:
        normalize_question("bad", {"type": "choice", "instructions": "x", "options": {"A": "a"}})
        check("choice with <2 options rejected", False)
    except QuestionError:
        check("choice with <2 options rejected", True)
    try:
        normalize_question("bad", {"type": "frobnicate", "instructions": "x"})
        check("unknown type rejected", False)
    except QuestionError:
        check("unknown type rejected", True)

    check("calibrate_prob is a no-op at T=1", abs(calibrate_prob(0.7, 1.0) - 0.7) < 1e-9)
    check("calibrate_prob pulls toward 0.5 as T grows", calibrate_prob(0.9, 5.0) < 0.9)
    check("calibrate_prob pushes toward extremes as T shrinks", calibrate_prob(0.6, 0.2) > 0.6)
    dist, calibrated = apply_calibration_distribution({"A": 0.7, "B": 0.3}, 2.0)
    check("distribution calibration renormalizes to 1", abs(sum(dist.values()) - 1.0) < 1e-9)
    check("distribution calibration flagged true", calibrated is True)
    dist_nocal, calibrated_nocal = apply_calibration_distribution({"A": 0.7, "B": 0.3}, None)
    check("no calibration passes through unflagged", dist_nocal == {"A": 0.7, "B": 0.3} and calibrated_nocal is False)

    winner, conf = top_two_gap({"A": 0.6, "B": 0.3, "C": 0.1})
    check("top_two_gap picks the max", winner == "A")
    check("top_two_gap confidence is top-minus-second", abs(conf - 0.3) < 1e-9)

    check("_score_value reads a numeric level name directly", _score_value("2", {"1": "x", "2": "y", "3": "z"}) == 2.0)
    check(
        "_score_value falls back to 1-based rank for non-numeric levels",
        _score_value("medium", {"low": "", "medium": "", "high": ""}) == 2.0,
    )

    fb = FakeBackend(seed=1)
    questions = {
        "q_onfile": {"type": "noul", "instructions": "Evidence on file satisfies the requirement."},
        "next_step": {
            "type": "choice",
            "instructions": "Which action next?",
            "criteria": {"CONTINUE": "keep going", "RETRIEVE": "pull evidence", "REQUEST": "ask supplier"},
        },
        "urgency": {"type": "score", "instructions": "How urgent?", "levels": {"low": "", "medium": "", "high": ""}},
    }
    result = answer_questions({"note": "two sentence state"}, questions, fb, max_workers=2)
    check("answer_questions returns all question ids", set(result["answers"].keys()) == set(questions.keys()))
    check(
        "noul answer shape",
        set(result["answers"]["q_onfile"].keys())
        == {"type", "noul", "unparsed_mass", "no_family_match", "non_finite_logprob_count", "degraded"},
    )
    check("noul probability is in [0,1]", 0.0 <= result["answers"]["q_onfile"]["noul"] <= 1.0)
    ns = result["answers"]["next_step"]
    check(
        "choice answer has choice/probabilities/confidence",
        {"type", "choice", "probabilities", "confidence"} <= set(ns.keys()),
    )
    check("choice probabilities sum to 1", abs(sum(ns["probabilities"].values()) - 1.0) < 1e-6)
    sc = result["answers"]["urgency"]
    check("score answer has a numeric score", isinstance(sc["score"], float))
    check("audit has one row per question", len(result["audit"]) == len(questions))
    check("audit rows share one request_hash", len({r["request_hash"] for r in result["audit"]}) == 1)
    check(
        "audit rows carry backend/model/latency/calibrated",
        all({"backend", "model", "latency_s", "calibrated", "question_id"} <= set(r.keys()) for r in result["audit"]),
    )
    check(
        "usage summary has input_tokens/output_tokens keys",
        {"input_tokens", "output_tokens"} == set(result["usage"].keys()),
    )
    check(
        "usage is None for a backend that reports none (fake backend)",
        result["usage"]["input_tokens"] is None and result["usage"]["output_tokens"] is None,
    )

    calibration = {"per_type": {"noul": {"T": 3.0}}}
    result_cal = answer_questions({"note": "x"}, {"q": questions["q_onfile"]}, fb, calibration=calibration)
    result_nocal = answer_questions({"note": "x"}, {"q": questions["q_onfile"]}, fb)
    check(
        "calibration changes the noul probability when T != 1",
        result_cal["answers"]["q"]["noul"] != result_nocal["answers"]["q"]["noul"],
    )
    check("calibrated flag set on audit row", result_cal["audit"][0]["calibrated"] is True)

    try:
        answer_questions({"note": "x"}, {}, fb)
        check("empty questions rejected", False)
    except QuestionError:
        check("empty questions rejected", True)

    iso = isolation_check({"note": "two sentence state"}, questions, fb, max_workers=2)
    check("isolation_check covers every question", set(iso["drift"].keys()) == set(questions.keys()))
    check("isolation_check reports a numeric max_drift", isinstance(iso["max_drift"], float))
    check(
        "isolation_check flags decision_changed per question",
        all("decision_changed" in d for d in iso["drift"].values()),
    )

    h1 = request_hash({"a": 1}, {"q": {"type": "noul", "instructions": "x"}}, "m")
    h2 = request_hash({"a": 1}, {"q": {"type": "noul", "instructions": "x"}}, "m")
    h3 = request_hash({"a": 2}, {"q": {"type": "noul", "instructions": "x"}}, "m")
    check("request_hash is stable for identical inputs", h1 == h2)
    check("request_hash changes with the state", h1 != h3)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)
