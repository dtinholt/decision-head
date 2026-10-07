"""Deliberate decision mode: k sampled reasoning-then-typed-JSON answers,
aggregated by vote, as an alternative to the single deterministic-token
logprob read in :mod:`decisionhead.core` ("fast" mode).

Where fast mode asks one question with ``temperature=0``, reads the
model's belief straight off token logprobs, and gets one number back per
question, deliberate mode asks the SAME question ``k`` times at
``temperature>0``, lets the model reason briefly before answering, parses
a trailing ``{"answer": ..., "confidence": ...}`` object out of each
sample, and turns the k samples into a probability by vote share --
closer to a typed reasoning controller than to a calibrated single-token
read, and correspondingly more expensive (k generations instead of one
per decision; see the README and ``docs/deliberate.md``).

Everything else about how one question is answered is unchanged from
fast mode by design, so the two modes can share one server, one audit
log, and one hardening story: one state prefix per request, one question
per model call (never leaked into another question's prompt), the same
per-request timeout, size limits, depth guard, and bearer/CORS handling
applied by :mod:`decisionhead.server` regardless of which mode a request
resolves to, and an audit row per question in the same jsonl sink.

Invariants (each pinned by a regression test from the 2026-10-06
security review, finding 5 and 6):

* Only a JSON object that is the whole FINAL line of a reply (after
  trimming, and ignoring a closing code fence) can cast a vote, and its
  string state is scanned from the start of that line only, so a stray
  quote in the reasoning cannot shift it. JSON the model quotes earlier
  (for example from the case document) is never a fallback.
* A backend that returns fewer than ``k`` samples gets the shortfall in
  ``n_missing_samples`` and the answer is ``low_confidence``.
* A noul vote is yes/no (or a JSON boolean); anything else is malformed,
  never a "no".
* At most ``k`` samples are voted, however many the backend returns.
* When no sample casts a valid vote the answer is ``degraded`` with a
  ``null`` value, never a stand-in 0.5 or a uniform distribution.

Attribution: the idea of extracting a JSON object out of a free-form
completion and voting typed answers across samples
(:func:`parse_deliberate_reply`, :func:`aggregate_noul_votes`,
:func:`aggregate_choice_votes`) is adapted from the typed-sampling
controller arm of the JEV workflow-completion benchmark, a separate
research harness by the same author (2026-09-23). That harness is not a
dependency of this package and nothing is imported from it. The approach
is reimplemented here in modified form: one question per call rather
than all questions batched into one JSON reply, and tolerant of a
reasoning preamble and a markdown code fence around the JSON.
"""

from __future__ import annotations

import math
import re
import threading
import time
from typing import Any, Protocol

from . import deadline as _deadline
from .core import (
    JSONDict,
    QuestionError,
    _score_value,
    normalize_question,
    render_state,
    request_hash,
    top_two_gap,
)
from .safejson import loads_strict

DEFAULT_K = 8
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_CLAIM_GUARD_THRESHOLD = 0.75


class SamplingBackend(Protocol):
    """Structural type a backend must satisfy to answer deliberate-mode
    questions: same ``model`` attribute as
    :class:`decisionhead.core.TokenDistributionBackend`, plus one method
    that returns k sampled completions instead of one token
    distribution."""

    model: str

    def sample_completions(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_answers: list[str],
        *,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> tuple[list[str], JSONDict]:
        """Returns ``(completions, meta)``: ``completions`` has up to
        ``n`` raw text samples (a backend may legitimately return fewer
        than asked for; it must never fabricate one). ``meta`` carries at
        least ``requests`` (how many HTTP calls were made), and
        ``n_param_used``/``fallback_used`` flags for auditing which path
        a backend that supports a native ``n`` parameter actually took."""
        ...


# --------------------------------------------------------------------- prompts

DELIBERATE_SYSTEM_PREFIX = (
    "You answer one typed question about the state below. Reason briefly "
    "(a sentence or two is enough), then finish your reply with exactly "
    "one line containing a single strict JSON object and nothing else on "
    'that line: {"answer": <answer>, "confidence": <a number between 0 '
    'and 1>}. Do not wrap the JSON in a code fence.'
)

NOUL_DELIBERATE_TEMPLATE = 'Question: {instructions}\n"answer" must be exactly "yes" or "no".'
CHOICE_DELIBERATE_TEMPLATE = (
    "Question: {instructions}\nOptions:\n{option_lines}\n"
    '"answer" must be exactly one of the option names above, verbatim.'
)
SCORE_DELIBERATE_TEMPLATE = (
    "Question: {instructions}\nLevels (low to high):\n{option_lines}\n"
    '"answer" must be exactly one of the level names above, verbatim.'
)


def build_deliberate_question_text(qtype: str, qnorm: JSONDict) -> tuple[str, list[str]]:
    """Returns ``(question_text, allowed_answers)`` for one normalized
    question (see :func:`decisionhead.core.normalize_question`).
    ``allowed_answers`` names the answer family a sample's ``"answer"``
    is voted against -- ``["yes", "no"]`` for noul, the option/level
    names verbatim for choice/score (unlike fast mode, deliberate mode
    does not constrain the model to a single lettered token, so the full
    name is what the model is asked to echo back)."""
    if qtype == "noul":
        return NOUL_DELIBERATE_TEMPLATE.format(instructions=qnorm["instructions"]), ["yes", "no"]
    options: dict[str, str] = qnorm["options"]
    lines = "\n".join(f"- {name}: {desc}" for name, desc in options.items())
    template = CHOICE_DELIBERATE_TEMPLATE if qtype == "choice" else SCORE_DELIBERATE_TEMPLATE
    text = template.format(instructions=qnorm["instructions"], option_lines=lines)
    return text, list(options.keys())


# --------------------------------------------------------------------- parsing / repair


def _find_json_object_spans(text: str) -> list[str]:
    """Every balanced top-level ``{...}`` substring in ``text``, in
    left-to-right order, tracking string/escape state so a brace inside a
    string literal is never mistaken for structure (same bracket-tracking
    approach as ``decisionhead.schema.json_nesting_too_deep``, applied
    here to collect spans instead of just a depth count)."""
    spans: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start : i + 1])
                    start = None
    return spans


def _repair_json_text(candidate: str) -> str:
    """A couple of cheap, common fixes for an LLM's near-JSON output:
    smart quotes and single quotes to straight double quotes, and a
    trailing comma before a closing bracket. Not a general JSON5 parser --
    just the handful of mistakes models actually make when asked for a
    strict JSON object inline in a reasoning reply."""
    repaired = candidate.replace("“", '"').replace("”", '"').replace("'", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


_CODE_FENCE_LINE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def _final_object_span(text: str) -> str | None:
    """The object on the reply's FINAL non-empty line (trailing blank
    lines and closing markdown code fences ignored), but only when that
    line, trimmed, is exactly one balanced ``{...}`` object starting at
    its first character -- the one line the prompt asks the model to end
    its turn with. ``None`` otherwise.

    String state is scanned from the start of that line only. Scanning
    from the start of the whole reply let one stray ``"`` in the
    reasoning flip the scanner into a string, so an object nested inside
    a malformed final line (document JSON) could be selected
    (2026-10-06 confirmation, R3). Prose before the object on the same
    line, a second object after it, or anything after it on later lines
    makes the sample malformed."""
    lines = text.splitlines()
    while lines and (not lines[-1].strip() or _CODE_FENCE_LINE.match(lines[-1])):
        lines.pop()
    if not lines:
        return None
    final = lines[-1].strip()
    if not final.startswith("{"):
        return None
    spans = _find_json_object_spans(final)
    if len(spans) != 1 or spans[0] != final:
        return None
    return final


def parse_deliberate_reply(text: str) -> JSONDict | None:
    """Extracts ``{"answer": ..., "confidence": ...}`` from one sampled
    completion that reasons briefly before answering.

    Only an object that makes up the whole final non-empty line of the
    reply is ever considered (see :func:`_final_object_span`; trailing
    whitespace or a closing code fence aside). It is parsed as strict JSON and, only if that fails,
    again after :func:`_repair_json_text`. An earlier object -- typically
    JSON the model quoted from the case document while reasoning -- is
    never a fallback: before the 2026-10-06 security review (finding 5)
    a malformed final object made the parser fall back to the newest
    earlier span, so a document could cast the vote. Non-finite numbers
    and numeric literals longer than 64 characters make the sample
    malformed (:func:`decisionhead.safejson.loads_strict`).

    Returns ``None`` -- never raises -- for a sample with no usable final
    object, so one malformed sample degrades to "not counted in the vote"
    rather than crashing the whole question (see
    :func:`aggregate_noul_votes`/:func:`aggregate_choice_votes` for how
    an all-malformed batch of samples is handled)."""
    if not text or not text.strip():
        return None
    span = _final_object_span(text)
    if span is None:
        return None
    # The repair pass runs only on the SAME span after strict parsing
    # fails; it can never select a different object.
    for candidate in (span, _repair_json_text(span)):
        try:
            data = loads_strict(candidate)
        except (ValueError, TypeError, RecursionError):
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    return None


# --------------------------------------------------------------------- vote aggregation


def _stated_confidences(parsed: list[JSONDict]) -> list[float]:
    """Stated confidences that are finite numbers (booleans excluded).
    The strict parser already refuses ``NaN`` literals; the finiteness
    check here keeps the average finite whatever produced ``parsed``."""
    out: list[float] = []
    for s in parsed:
        value = s.get("confidence")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value):
            out.append(float(value))
    return out


def _noul_vote(answer: Any) -> str | None:
    """``"yes"``/``"no"`` for an in-vocabulary noul answer, else ``None``.
    Accepts the strings yes/no (case-insensitive, surrounding whitespace
    and one trailing ``.`` or ``!`` ignored) and the JSON booleans. Any
    other answer -- ``"maybe"``, ``"1"``, a number -- is off-vocabulary
    and counts as malformed, never as a "no" (2026-10-06 review,
    finding 5)."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if not isinstance(answer, str):
        return None
    norm = answer.strip().lower()
    if norm[-1:] in (".", "!"):
        norm = norm[:-1].rstrip()
    return norm if norm in ("yes", "no") else None


def _avg_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def aggregate_noul_votes(samples: list[JSONDict | None]) -> tuple[float | None, JSONDict]:
    """``P(yes)`` as vote share across the samples that cast a valid vote
    (see :func:`_noul_vote`). An unparsed (``None``) sample, or one whose
    answer is off-vocabulary, is excluded from the vote and counted in
    ``meta["n_malformed"]`` -- never counted as a "no". When no sample
    cast a valid vote, ``p_yes`` is ``None`` and ``meta["degraded"]`` is
    ``True``: an explicit "no reading", not a plausible-looking 0.5
    (2026-10-06 review, finding 6). ``meta`` also carries ``votes``
    (``{"yes": n, "no": n}``) and ``confidences`` -- the stated
    per-sample confidences of the valid votes."""
    valid = [(s, v) for s in samples if s is not None for v in [_noul_vote(s.get("answer"))] if v is not None]
    yes = sum(1 for _, v in valid if v == "yes")
    n_valid = len(valid)
    p_yes = (yes / n_valid) if n_valid else None
    meta: JSONDict = {
        "n_total": len(samples),
        "n_parsed": n_valid,
        "n_malformed": len(samples) - n_valid,
        "votes": {"yes": yes, "no": n_valid - yes},
        "degraded": n_valid == 0,
        "confidences": _stated_confidences([s for s, _ in valid]),
    }
    return p_yes, meta


def aggregate_choice_votes(
    samples: list[JSONDict | None], options: dict[str, str]
) -> tuple[dict[str, float], JSONDict]:
    """Vote-share distribution over ``options`` (used for both ``choice``
    and ``score`` question types, since both normalize to a named-option
    set -- see :func:`decisionhead.core.normalize_question`). A sample
    whose ``"answer"`` does not match any option name (case-insensitively)
    is excluded from the vote and counted in
    ``meta["n_unmatched_answer"]``; if NO sample votes for anything, the
    returned distribution is uniform (so it is never all-zero) and
    ``meta["degraded"]`` is ``True``. :func:`_answer_one_deliberate`
    then returns ``null`` for the choice, probabilities and confidence;
    the uniform placeholder never reaches a client (2026-10-06 review,
    finding 6)."""
    parsed = [s for s in samples if s is not None]
    names = list(options.keys())
    lower_map = {name.lower(): name for name in names}
    counts = dict.fromkeys(names, 0)
    unmatched = 0
    for s in parsed:
        canonical = lower_map.get(str(s.get("answer", "")).strip().lower())
        if canonical is not None:
            counts[canonical] += 1
        else:
            unmatched += 1
    total = sum(counts.values())
    vote_share = {name: (count / total) for name, count in counts.items()} if total else {
        name: 1.0 / len(names) for name in names
    }
    meta: JSONDict = {
        "n_total": len(samples),
        "n_parsed": len(parsed),
        "n_malformed": len(samples) - len(parsed),
        "n_unmatched_answer": unmatched,
        "votes": counts,
        # No sample voted for any option: the uniform fallback above is a
        # placeholder, not a reading (2026-10-06 review, finding 6).
        "degraded": total == 0,
        "confidences": _stated_confidences(parsed),
    }
    return vote_share, meta


# --------------------------------------------------------------------- claim guard


def apply_claim_guard(
    qdef: JSONDict,
    winner: str,
    vote_share: dict[str, float],
    *,
    enabled: bool,
    threshold: float,
) -> tuple[str, JSONDict | None]:
    """Guards a terminal "claim of done/complete" answer before it is
    returned. A question opts in by naming its terminal option(s) in
    ``qdef["terminal_answers"]`` (a list of option names, e.g.
    ``["CLAIM_COMPLETE"]``); a question that does not set this key is
    never guarded, whatever it answers.

    When the winning vote is a terminal answer, it is only returned as-is
    if BOTH hold: the winning vote share is ``>= threshold``, AND, when
    the request supplies structured completion evidence in
    ``qdef["completion_evidence"]`` (``{"open_obligations_count": int,
    "unmet_requirements": [...]}``), that evidence is empty (zero open
    obligations, no unmet requirements). Otherwise the answer is
    downgraded to the highest-vote-share option that is NOT itself marked
    terminal, and ``(downgraded_choice, guard_audit_extra)`` is returned
    so the caller can both change the answer and record why. Returns
    ``(winner, None)`` -- an ordinary no-op -- whenever the guard does not
    apply: disabled, no terminal answers declared, or the winner is not
    one of them.

    This exists because a vote share is not a calibrated probability: an
    internal development-set evaluation (k=8) found deliberate mode's raw
    vote share badly calibrated for exactly this failure mode, with
    several times the rule table's rate of wrong "done" claims, so a
    terminal claim must clear an explicit bar rather than being trusted
    at face value (``docs/deliberate.md``)."""
    terminal = qdef.get("terminal_answers")
    # Explicit type checks (2026-10-06 review, finding 5): a string here
    # used to make `winner in terminal` a substring test, and a list
    # completion_evidence raised AttributeError. normalize_question
    # rejects both with a 400 first; this is the second layer.
    if not enabled or not isinstance(terminal, list) or not terminal or winner not in terminal:
        return winner, None

    evidence = qdef.get("completion_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    open_count = evidence.get("open_obligations_count")
    if not isinstance(open_count, int) or isinstance(open_count, bool):
        open_count = 0
    unmet_requirements = evidence.get("unmet_requirements")
    if not isinstance(unmet_requirements, list):
        unmet_requirements = []
    evidence_empty = not open_count and not unmet_requirements
    winner_share = vote_share.get(winner, 0.0)

    if winner_share >= threshold and evidence_empty:
        return winner, None

    reasons = []
    if winner_share < threshold:
        reasons.append(f"vote share {winner_share:.3f} is below claim_guard_threshold {threshold}")
    if not evidence_empty:
        # Counts only, never the caller's free text: unmet_requirements
        # may quote the case document, and this reason lands in the audit
        # log (2026-10-06 review, finding 9).
        reasons.append(
            f"completion evidence is not empty (open_obligations_count={open_count}, "
            f"unmet_requirements_count={len(unmet_requirements)})"
        )

    non_terminal = {name: share for name, share in vote_share.items() if name not in terminal}
    downgraded = max(non_terminal, key=lambda name: non_terminal[name]) if non_terminal else winner
    extra: JSONDict = {
        "claim_guard_fired": True,
        "claim_guard_downgraded": downgraded != winner,
        "claim_guard_reason": "; ".join(reasons),
        "claim_guard_original_choice": winner,
        "claim_guard_original_vote_share": winner_share,
    }
    return downgraded, extra


# --------------------------------------------------------------------- answering


def _answer_one_deliberate(
    qid: str,
    qdef: JSONDict,
    prefix_messages: list[JSONDict],
    backend: SamplingBackend,
    req_hash: str,
    model: str,
    backend_name: str,
    *,
    k: int,
    temperature: float,
    max_tokens: int,
    claim_guard_enabled: bool,
    claim_guard_threshold: float,
) -> tuple[JSONDict, JSONDict, JSONDict]:
    qnorm = normalize_question(qid, qdef)
    qtype = qnorm["type"]
    question_text, allowed_answers = build_deliberate_question_text(qtype, qnorm)

    _deadline.check_current()  # never start sampling for an abandoned request
    t0 = time.time()
    texts, sample_meta = backend.sample_completions(
        prefix_messages, question_text, allowed_answers, n=k, temperature=temperature, max_tokens=max_tokens
    )
    latency_s = time.time() - t0

    # At most k samples are ever voted, whatever the backend returned;
    # extras count as malformed (2026-10-06 review, finding 5).
    n_excess = max(0, len(texts) - k)
    # Fewer samples than asked for: one crafted sample from a compromised
    # endpoint must not read as a full-confidence k-sample vote. The
    # shortfall is reported and the answer flagged low_confidence
    # (2026-10-06 confirmation, R4).
    n_missing = max(0, k - len(texts))
    samples = [parse_deliberate_reply(t) for t in texts[:k]]

    answer: JSONDict
    guard_extra: JSONDict | None = None
    if qtype == "noul":
        p_yes, agg_meta = aggregate_noul_votes(samples)
        answer = {
            "type": "noul",
            "noul": p_yes,
            "vote_share": p_yes,
        }
        result_value: Any = p_yes
    else:
        vote_share, agg_meta = aggregate_choice_votes(samples, qnorm["options"])
        if agg_meta["degraded"]:
            answer = {"type": qtype, "probabilities": None, "vote_share": None, "confidence": None}
            answer["choice" if qtype == "choice" else "score"] = None
            result_value = None
        else:
            winner, gap = top_two_gap(vote_share)
            final_choice, guard_extra = apply_claim_guard(
                qdef, winner, vote_share, enabled=claim_guard_enabled, threshold=claim_guard_threshold
            )
            answer = {
                "type": qtype,
                "probabilities": vote_share,
                "vote_share": vote_share,
                "confidence": gap,
            }
            if qtype == "choice":
                answer["choice"] = final_choice
            else:
                answer["score"] = _score_value(final_choice, qnorm["options"])
            result_value = final_choice
            if guard_extra is not None:
                answer.update(guard_extra)

    n_malformed = agg_meta["n_malformed"] + n_excess
    counts: JSONDict = {
        "n_samples": len(texts),
        "n_parsed": agg_meta["n_parsed"],
        "n_malformed": n_malformed,
        "n_unmatched_answer": agg_meta.get("n_unmatched_answer", 0),
        "n_excess_samples": n_excess,
        "n_missing_samples": n_missing,
        "degraded": bool(agg_meta["degraded"]),
    }
    answer.update(counts)
    if n_missing:
        answer["low_confidence"] = True
    answer["avg_stated_confidence"] = _avg_or_none(agg_meta["confidences"])

    audit_row: JSONDict = {
        "request_hash": req_hash,
        "question_id": qid,
        "model": model,
        "backend": backend_name,
        "mode": "deliberate",
        "k": k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **counts,
        "votes": agg_meta["votes"],
        "result": result_value,
        "sampling_requests": sample_meta.get("requests"),
        "n_param_used": sample_meta.get("n_param_used", False),
        "fallback_used": sample_meta.get("fallback_used", False),
        "latency_s": latency_s,
    }
    if guard_extra is not None:
        audit_row.update(guard_extra)

    usage: JSONDict = sample_meta.get("usage") or {}
    return answer, audit_row, usage


def answer_questions_deliberate(
    state: str | JSONDict | list[Any],
    questions: JSONDict,
    backend: SamplingBackend,
    *,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_workers: int = 4,
    backend_name: str | None = None,
    claim_guard_enabled: bool = True,
    claim_guard_threshold: float = DEFAULT_CLAIM_GUARD_THRESHOLD,
    deadline: _deadline.Deadline | None = None,
) -> JSONDict:
    """Deliberate-mode counterpart to
    :func:`decisionhead.core.answer_questions`: identical result shape
    (``{"answers", "audit", "usage", "model"}``), identical one-state-
    prefix-per-request / one-question-per-model-call isolation, and the
    identical thread-per-question harness -- but each question is
    answered by ``k`` sampled reasoning-then-JSON completions voted over
    (:func:`_answer_one_deliberate`) instead of one single-token logprob
    read. Costs ``k`` model generations per question instead of one; see
    the README and ``docs/deliberate.md``. ``deadline``: as in
    :func:`decisionhead.core.answer_questions` -- checked before every
    sampling call so a timed-out request's questions are abandoned."""
    if not isinstance(questions, dict) or not questions:
        raise QuestionError("'questions' must be a non-empty object")
    if k < 1:
        raise QuestionError("'k' (deliberate sample count) must be at least 1")

    model = getattr(backend, "model", "unknown")
    bname = backend_name or type(backend).__name__
    state_text = render_state(state)
    prefix_messages: list[JSONDict] = [
        {"role": "system", "content": DELIBERATE_SYSTEM_PREFIX},
        {"role": "user", "content": f"State:\n{state_text}"},
    ]
    req_hash = request_hash(state, questions, model) + "-deliberate"

    qids = list(questions.keys())
    results: list[tuple[str, JSONDict, JSONDict, JSONDict] | None] = [None] * len(qids)
    errors: list[BaseException | None] = [None] * len(qids)

    def worker(indices: list[int]) -> None:
        _deadline.set_current(deadline)
        for i in indices:
            qid = qids[i]
            try:
                answer, audit_row, usage = _answer_one_deliberate(
                    qid,
                    questions[qid],
                    prefix_messages,
                    backend,
                    req_hash,
                    model,
                    bname,
                    k=k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    claim_guard_enabled=claim_guard_enabled,
                    claim_guard_threshold=claim_guard_threshold,
                )
                results[i] = (qid, answer, audit_row, usage)
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
    return {"answers": answers, "audit": audit, "usage": usage_summary, "model": model}
