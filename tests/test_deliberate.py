from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from decisionhead.backends import BackendError, FakeBackend, OllamaBackend, OpenAIBackend
from decisionhead.config import ServerConfig
from decisionhead.deliberate import (
    DEFAULT_K,
    aggregate_choice_votes,
    aggregate_noul_votes,
    answer_questions_deliberate,
    apply_claim_guard,
    parse_deliberate_reply,
)
from decisionhead.logging import get_logger
from decisionhead.server import AuditLog, Metrics, ReadinessChecker, make_handler, resolve_effective_mode, run_server

QUESTIONS = {
    "q_onfile": {"type": "noul", "instructions": "Evidence on file satisfies the requirement."},
    "next_step": {
        "type": "choice",
        "instructions": "Which action next?",
        "criteria": {"CONTINUE": "keep going", "RETRIEVE": "pull evidence", "REQUEST": "ask supplier"},
    },
}


def _reply(answer: object, confidence: float | None = 0.8, preamble: str = "Some brief reasoning.") -> str:
    obj: dict[str, object] = {"answer": answer}
    if confidence is not None:
        obj["confidence"] = confidence
    return f"{preamble}\n{json.dumps(obj)}"


def _fixture_transport(fixed_response, capture=None):  # type: ignore[no-untyped-def]
    def _t(url, body, headers, timeout=60):  # type: ignore[no-untyped-def]
        if capture is not None:
            capture.setdefault("calls", []).append({"url": url, "body": body, "headers": headers})
        return fixed_response

    return _t


class ScriptedSamplingBackend:
    """A backend double with a fully scripted list of raw completion
    texts per question, used to test vote aggregation and malformed-
    sample handling end to end through
    :func:`decisionhead.deliberate.answer_questions_deliberate` without
    depending on any concrete backend's HTTP shape."""

    model = "scripted-model"

    def __init__(self, texts_by_question: dict[str, list[str]]) -> None:
        self.texts_by_question = texts_by_question
        self.calls: list[dict[str, object]] = []

    def sample_completions(self, prefix_messages, question_text, allowed_answers, *, n, temperature, max_tokens):  # type: ignore[no-untyped-def]
        # Matched by a substring of the question's own instructions text
        # (the built question_text embeds the instructions, not the
        # question id), since that is the only thing distinguishing one
        # question's prompt from another's here.
        for needle, texts in self.texts_by_question.items():
            if needle in question_text:
                self.calls.append({"question_text": question_text, "n": n})
                return texts, {"requests": len(texts), "n_param_used": False, "fallback_used": False, "usage": {}}
        raise AssertionError(f"no scripted texts for question_text={question_text!r}")


class SlowSamplingBackend:
    """A backend double that sleeps before returning, mirroring
    tests/test_server.py's SlowBackend but for the sample_completions
    interface deliberate mode uses -- exercises the request timeout path
    for deliberate mode specifically."""

    model = "slow-model"

    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay

    def token_distribution(self, prefix_messages, question_text, allowed_tokens):  # type: ignore[no-untyped-def]
        time.sleep(self.delay)
        return {t: 1.0 / len(allowed_tokens) for t in allowed_tokens} | {"_unparsed": 0.0, "_raw": {}}

    def sample_completions(self, prefix_messages, question_text, allowed_answers, *, n, temperature, max_tokens):  # type: ignore[no-untyped-def]
        time.sleep(self.delay)
        meta = {"requests": n, "n_param_used": False, "fallback_used": False, "usage": {}}
        return [_reply(allowed_answers[0])] * n, meta


def _start_custom_server(cfg: ServerConfig, backend):  # type: ignore[no-untyped-def]
    audit = AuditLog(cfg.audit_path)
    metrics = Metrics()
    readiness = ReadinessChecker(backend, cfg.ready_cache_s)
    logger = get_logger()
    handler = make_handler(backend, None, audit, cfg, metrics, readiness, logger, None)
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:  # type: ignore[type-arg]
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --------------------------------------------------------------------- parsing / repair


class TestParseDeliberateReply:
    def test_bare_json(self) -> None:
        assert parse_deliberate_reply('{"answer": "yes", "confidence": 0.9}') == {
            "answer": "yes",
            "confidence": 0.9,
        }

    def test_json_after_reasoning_preamble(self) -> None:
        text = 'The evidence looks current based on the date.\n{"answer": "yes", "confidence": 0.7}'
        assert parse_deliberate_reply(text) == {"answer": "yes", "confidence": 0.7}

    def test_json_in_markdown_fence(self) -> None:
        text = 'Reasoning here.\n```json\n{"answer": "no", "confidence": 0.6}\n```'
        parsed = parse_deliberate_reply(text)
        assert parsed is not None
        assert parsed["answer"] == "no"

    def test_trailing_comma_repaired(self) -> None:
        text = '{"answer": "yes", "confidence": 0.5,}'
        assert parse_deliberate_reply(text) == {"answer": "yes", "confidence": 0.5}

    def test_single_quotes_repaired(self) -> None:
        text = "{'answer': 'yes', 'confidence': 0.5}"
        parsed = parse_deliberate_reply(text)
        assert parsed is not None
        assert parsed["answer"] == "yes"

    def test_no_json_object_returns_none(self) -> None:
        assert parse_deliberate_reply("I think the answer is yes but I forgot to format it.") is None

    def test_json_without_answer_key_returns_none(self) -> None:
        assert parse_deliberate_reply('{"confidence": 0.9}') is None

    def test_empty_text_returns_none(self) -> None:
        assert parse_deliberate_reply("") is None
        assert parse_deliberate_reply("   ") is None

    def test_last_json_object_wins_when_several_present(self) -> None:
        text = 'First I considered {"scratch": "note"} then decided.\n{"answer": "no", "confidence": 0.4}'
        parsed = parse_deliberate_reply(text)
        assert parsed is not None
        assert parsed["answer"] == "no"


# --------------------------------------------------------------------- vote aggregation


class TestAggregateNoulVotes:
    def test_unanimous_yes(self) -> None:
        samples = [{"answer": "yes", "confidence": 0.9}] * 5
        p_yes, meta = aggregate_noul_votes(samples)
        assert p_yes == 1.0
        assert meta["n_parsed"] == 5
        assert meta["n_malformed"] == 0

    def test_mixed_votes(self) -> None:
        samples = [{"answer": "yes"}, {"answer": "yes"}, {"answer": "no"}, {"answer": "YES"}]
        p_yes, _meta = aggregate_noul_votes(samples)
        assert abs(p_yes - 0.75) < 1e-9

    def test_malformed_samples_excluded_not_counted_as_no(self) -> None:
        samples = [{"answer": "yes"}, None, {"answer": "yes"}, None]
        p_yes, meta = aggregate_noul_votes(samples)
        assert p_yes == 1.0
        assert meta["n_total"] == 4
        assert meta["n_parsed"] == 2
        assert meta["n_malformed"] == 2

    def test_all_malformed_is_degraded_not_a_plausible_half(self) -> None:
        # Was a silent 0.5 before the 2026-10-06 review (finding 6).
        p_yes, meta = aggregate_noul_votes([None, None, None])
        assert p_yes is None
        assert meta["degraded"] is True
        assert meta["n_parsed"] == 0

    def test_confidences_collected_for_averaging(self) -> None:
        samples = [{"answer": "yes", "confidence": 0.8}, {"answer": "no", "confidence": 0.4}, None]
        _p_yes, meta = aggregate_noul_votes(samples)
        assert meta["confidences"] == [0.8, 0.4]


class TestAggregateChoiceVotes:
    OPTIONS = {"CONTINUE": "keep going", "RETRIEVE": "pull evidence", "REQUEST": "ask supplier"}

    def test_majority_wins(self) -> None:
        samples = [{"answer": "CONTINUE"}] * 3 + [{"answer": "RETRIEVE"}]
        vote_share, meta = aggregate_choice_votes(samples, self.OPTIONS)
        assert vote_share["CONTINUE"] == 0.75
        assert meta["n_unmatched_answer"] == 0

    def test_case_insensitive_matching(self) -> None:
        samples = [{"answer": "continue"}, {"answer": "Continue"}]
        vote_share, _meta = aggregate_choice_votes(samples, self.OPTIONS)
        assert vote_share["CONTINUE"] == 1.0

    def test_unmatched_answer_excluded_from_vote(self) -> None:
        samples = [{"answer": "CONTINUE"}, {"answer": "not-an-option"}]
        vote_share, meta = aggregate_choice_votes(samples, self.OPTIONS)
        assert vote_share["CONTINUE"] == 1.0
        assert meta["n_unmatched_answer"] == 1

    def test_all_malformed_or_unmatched_falls_back_to_uniform(self) -> None:
        vote_share, _meta = aggregate_choice_votes([None, {"answer": "nonsense"}], self.OPTIONS)
        assert all(abs(v - 1 / 3) < 1e-9 for v in vote_share.values())


# --------------------------------------------------------------------- claim guard


class TestApplyClaimGuard:
    QDEF_NO_GUARD = {"type": "choice", "instructions": "x", "criteria": {"A": "a", "B": "b"}}
    QDEF_GUARDED = {
        "type": "choice",
        "instructions": "x",
        "criteria": {"CLAIM_COMPLETE": "done", "CONTINUE": "keep going"},
        "terminal_answers": ["CLAIM_COMPLETE"],
    }

    def test_no_terminal_answers_declared_is_a_no_op(self) -> None:
        vote_share = {"A": 0.9, "B": 0.1}
        choice, extra = apply_claim_guard(self.QDEF_NO_GUARD, "A", vote_share, enabled=True, threshold=0.75)
        assert choice == "A"
        assert extra is None

    def test_winner_not_terminal_is_a_no_op(self) -> None:
        vote_share = {"CLAIM_COMPLETE": 0.2, "CONTINUE": 0.8}
        choice, extra = apply_claim_guard(self.QDEF_GUARDED, "CONTINUE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CONTINUE"
        assert extra is None

    def test_terminal_claim_above_threshold_with_empty_evidence_passes(self) -> None:
        vote_share = {"CLAIM_COMPLETE": 0.9, "CONTINUE": 0.1}
        choice, extra = apply_claim_guard(self.QDEF_GUARDED, "CLAIM_COMPLETE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CLAIM_COMPLETE"
        assert extra is None

    def test_terminal_claim_below_threshold_is_downgraded(self) -> None:
        vote_share = {"CLAIM_COMPLETE": 0.6, "CONTINUE": 0.4}
        choice, extra = apply_claim_guard(self.QDEF_GUARDED, "CLAIM_COMPLETE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CONTINUE"
        assert extra is not None
        assert extra["claim_guard_downgraded"] is True
        assert extra["claim_guard_original_choice"] == "CLAIM_COMPLETE"
        assert "below" in extra["claim_guard_reason"]

    def test_terminal_claim_with_nonempty_evidence_is_downgraded_even_above_threshold(self) -> None:
        qdef = {**self.QDEF_GUARDED, "completion_evidence": {"unmet_requirements": ["signature missing"]}}
        vote_share = {"CLAIM_COMPLETE": 0.95, "CONTINUE": 0.05}
        choice, extra = apply_claim_guard(qdef, "CLAIM_COMPLETE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CONTINUE"
        assert extra is not None
        assert "evidence" in extra["claim_guard_reason"]

    def test_open_obligations_count_also_blocks_the_claim(self) -> None:
        qdef = {**self.QDEF_GUARDED, "completion_evidence": {"open_obligations_count": 2}}
        vote_share = {"CLAIM_COMPLETE": 1.0, "CONTINUE": 0.0}
        choice, extra = apply_claim_guard(qdef, "CLAIM_COMPLETE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CONTINUE"
        assert extra is not None

    def test_guard_disabled_never_downgrades(self) -> None:
        vote_share = {"CLAIM_COMPLETE": 0.1, "CONTINUE": 0.9}
        choice, extra = apply_claim_guard(
            self.QDEF_GUARDED, "CLAIM_COMPLETE", vote_share, enabled=False, threshold=0.75
        )
        assert choice == "CLAIM_COMPLETE"
        assert extra is None

    def test_all_options_terminal_keeps_winner_but_still_flags(self) -> None:
        qdef = {
            "type": "choice",
            "instructions": "x",
            "criteria": {"CLAIM_COMPLETE": "done", "CLAIM_COMPLETE_ALT": "also done"},
            "terminal_answers": ["CLAIM_COMPLETE", "CLAIM_COMPLETE_ALT"],
        }
        vote_share = {"CLAIM_COMPLETE": 0.5, "CLAIM_COMPLETE_ALT": 0.5}
        choice, extra = apply_claim_guard(qdef, "CLAIM_COMPLETE", vote_share, enabled=True, threshold=0.75)
        assert choice == "CLAIM_COMPLETE"  # nothing safe to downgrade to
        assert extra is not None
        assert extra["claim_guard_downgraded"] is False


# --------------------------------------------------------------------- answer_questions_deliberate


class TestAnswerQuestionsDeliberate:
    def test_rejects_empty_questions(self) -> None:
        backend = ScriptedSamplingBackend({})
        with pytest.raises(Exception):  # noqa: B017,PT011 -- QuestionError, see decisionhead.core
            answer_questions_deliberate("state", {}, backend)

    def test_rejects_k_less_than_one(self) -> None:
        backend = ScriptedSamplingBackend({})
        with pytest.raises(Exception):  # noqa: B017,PT011
            answer_questions_deliberate("state", QUESTIONS, backend, k=0)

    def test_covers_every_question_with_one_audit_row_each(self) -> None:
        backend = ScriptedSamplingBackend(
            {
                "Evidence on file": [_reply("yes")] * 6 + [_reply("no")] * 2,
                "Which action next": [_reply("CONTINUE")] * 5 + [_reply("RETRIEVE")] * 3,
            }
        )
        result = answer_questions_deliberate("some state", QUESTIONS, backend, k=8, max_workers=2)
        assert set(result["answers"].keys()) == set(QUESTIONS.keys())
        assert len(result["audit"]) == len(QUESTIONS)
        assert all(row["mode"] == "deliberate" for row in result["audit"])
        assert result["answers"]["q_onfile"]["noul"] == 0.75
        assert result["answers"]["q_onfile"]["vote_share"] == 0.75
        assert result["answers"]["next_step"]["choice"] == "CONTINUE"
        assert "vote_share" in result["answers"]["next_step"]
        # one shared request_hash across every audit row from this request
        assert len({row["request_hash"] for row in result["audit"]}) == 1

    def test_malformed_samples_do_not_crash_the_batch(self) -> None:
        backend = ScriptedSamplingBackend(
            {
                "Evidence on file": ["not json at all", "also garbage", _reply("yes")],
                "Which action next": [_reply("CONTINUE")],
            }
        )
        result = answer_questions_deliberate(
            "state", {"q_onfile": QUESTIONS["q_onfile"], "next_step": QUESTIONS["next_step"]}, backend, k=3
        )
        assert result["answers"]["q_onfile"]["noul"] == 1.0
        audit_row = next(r for r in result["audit"] if r["question_id"] == "q_onfile")
        assert audit_row["n_malformed"] == 2
        assert audit_row["n_parsed"] == 1

    def test_avg_stated_confidence_is_exposed_alongside_vote_share(self) -> None:
        backend = ScriptedSamplingBackend(
            {"Evidence on file": [_reply("yes", confidence=0.9), _reply("yes", confidence=0.7)]}
        )
        result = answer_questions_deliberate("state", {"q_onfile": QUESTIONS["q_onfile"]}, backend, k=2)
        answer = result["answers"]["q_onfile"]
        assert answer["vote_share"] == 1.0
        assert abs(answer["avg_stated_confidence"] - 0.8) < 1e-9

    def test_claim_guard_wired_into_full_answer_flow(self) -> None:
        qdef = {
            "type": "choice",
            "instructions": "Should this case be closed?",
            "criteria": {"CLAIM_COMPLETE": "close it", "CONTINUE": "keep going"},
            "terminal_answers": ["CLAIM_COMPLETE"],
            "completion_evidence": {"unmet_requirements": ["missing signature"]},
        }
        backend = ScriptedSamplingBackend({"Should this case be closed?": [_reply("CLAIM_COMPLETE")] * 10})
        result = answer_questions_deliberate(
            "state", {"q_close": qdef}, backend, k=10, claim_guard_enabled=True, claim_guard_threshold=0.75
        )
        answer = result["answers"]["q_close"]
        assert answer["choice"] == "CONTINUE"
        assert answer["claim_guard_downgraded"] is True
        # raw vote share must remain visible even though the guard fired
        assert answer["vote_share"]["CLAIM_COMPLETE"] == 1.0
        audit_row = result["audit"][0]
        assert audit_row["claim_guard_downgraded"] is True


# --------------------------------------------------------------------- backend-level sampling


class TestOllamaSampleCompletions:
    def test_always_makes_n_separate_requests(self) -> None:
        capture: dict[str, object] = {}
        resp = {"message": {"content": '{"answer": "yes", "confidence": 0.5}'}}
        backend = OllamaBackend("http://example:11434", "m", transport=_fixture_transport(resp, capture))
        texts, meta = backend.sample_completions([], "Q?", ["yes", "no"], n=4, temperature=0.7, max_tokens=64)
        assert len(texts) == 4
        assert meta["requests"] == 4
        assert meta["n_param_used"] is False
        assert len(capture["calls"]) == 4
        for call in capture["calls"]:
            assert call["url"].endswith("/api/chat")
            assert "n" not in call["body"]
            assert call["body"]["options"]["temperature"] == 0.7
            assert call["body"]["options"]["num_predict"] == 64


class TestOpenAISampleCompletions:
    def test_uses_native_n_parameter_in_one_request_when_supported(self) -> None:
        capture: dict[str, object] = {}
        resp = {
            "choices": [{"message": {"content": _reply("yes")}} for _ in range(4)],
            "usage": {"prompt_tokens": 40, "completion_tokens": 80},
        }
        backend = OpenAIBackend("http://example:8000", "m", transport=_fixture_transport(resp, capture))
        texts, meta = backend.sample_completions([], "Q?", ["yes", "no"], n=4, temperature=0.7, max_tokens=64)
        assert len(texts) == 4
        assert meta["n_param_used"] is True
        assert meta["requests"] == 1
        assert len(capture["calls"]) == 1
        assert capture["calls"][0]["body"]["n"] == 4

    def test_falls_back_to_k_requests_when_n_request_errors(self) -> None:
        calls: list[dict[str, object]] = []

        def flaky_transport(url, body, headers, timeout=60):  # type: ignore[no-untyped-def]
            calls.append(body)
            if body.get("n", 1) > 1:
                raise BackendError("HTTP 400 from server: n is not supported")
            return {"choices": [{"message": {"content": _reply("yes")}}], "usage": {}}

        backend = OpenAIBackend("http://example:8000", "m", transport=flaky_transport)
        texts, meta = backend.sample_completions([], "Q?", ["yes", "no"], n=3, temperature=0.7, max_tokens=64)
        assert len(texts) == 3
        assert meta["n_param_used"] is False
        assert meta["fallback_used"] is True
        assert meta["requests"] == 3
        # one failed n=3 attempt, then 3 separate n=1 requests
        assert len(calls) == 4
        assert all(c.get("n", 1) == 1 for c in calls[1:])

    def test_falls_back_when_n_request_returns_no_usable_choices(self) -> None:
        calls: list[dict[str, object]] = []

        def half_broken_transport(url, body, headers, timeout=60):  # type: ignore[no-untyped-def]
            calls.append(body)
            if body.get("n", 1) > 1:
                return {"choices": []}  # server silently ignored n and returned nothing usable
            return {"choices": [{"message": {"content": _reply("no")}}], "usage": {}}

        backend = OpenAIBackend("http://example:8000", "m", transport=half_broken_transport)
        texts, meta = backend.sample_completions([], "Q?", ["yes", "no"], n=2, temperature=0.7, max_tokens=64)
        assert len(texts) == 2
        assert meta["fallback_used"] is True


class TestFakeBackendSampleCompletions:
    def test_deterministic_for_same_seed_and_question(self) -> None:
        fb1 = FakeBackend(seed=5)
        fb2 = FakeBackend(seed=5)
        texts1, _m1 = fb1.sample_completions([], "question", ["yes", "no"], n=DEFAULT_K, temperature=0.7, max_tokens=64)
        texts2, _m2 = fb2.sample_completions([], "question", ["yes", "no"], n=DEFAULT_K, temperature=0.7, max_tokens=64)
        assert texts1 == texts2

    def test_samples_parse_into_allowed_answers(self) -> None:
        fb = FakeBackend(seed=1)
        texts, _meta = fb.sample_completions([], "question", ["yes", "no"], n=8, temperature=0.7, max_tokens=64)
        parsed = [parse_deliberate_reply(t) for t in texts]
        assert all(p is not None for p in parsed)
        assert all(p["answer"] in ("yes", "no") for p in parsed)  # type: ignore[index]


# --------------------------------------------------------------------- mode resolution + server dispatch


class TestResolveEffectiveMode:
    def test_default_mode_used_when_request_has_no_mode(self) -> None:
        cfg = ServerConfig(mode="fast")
        assert resolve_effective_mode(cfg, {}) == "fast"

    def test_request_override_honored_when_allowed(self) -> None:
        cfg = ServerConfig(mode="fast", mode_override_allowed=True)
        assert resolve_effective_mode(cfg, {"mode": "deliberate"}) == "deliberate"

    def test_request_override_ignored_when_not_allowed(self) -> None:
        cfg = ServerConfig(mode="fast", mode_override_allowed=False)
        assert resolve_effective_mode(cfg, {"mode": "deliberate"}) == "fast"

    def test_invalid_mode_override_raises(self) -> None:
        cfg = ServerConfig(mode="fast", mode_override_allowed=True)
        with pytest.raises(Exception):  # noqa: B017,PT011 -- QuestionError
            resolve_effective_mode(cfg, {"mode": "not-a-real-mode"})


class TestServerDeliberateIntegration:
    def test_dry_server_answers_a_deliberate_request(self) -> None:
        cfg = ServerConfig(model="dry-model", port=0, dry=True, dry_seed=2, mode="fast")
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = {
                "state": "s",
                "model": "jev-latest",
                "mode": "deliberate",
                "questions": {"q1": {"type": "noul", "instructions": "Is the evidence current?"}},
            }
            status, body = _post(base, "/v1/systemone", req)
            assert status == 200
            assert body["mode"] == "deliberate"
            assert "vote_share" in body["answers"]["q1"]
            assert 0.0 <= body["answers"]["q1"]["noul"] <= 1.0
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_mode_override_refused_when_server_disallows_it(self) -> None:
        cfg = ServerConfig(model="dry-model", port=0, dry=True, mode="fast", mode_override_allowed=False)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = {
                "state": "s",
                "model": "m",
                "mode": "deliberate",
                "questions": {"q1": {"type": "noul", "instructions": "x"}},
            }
            status, body = _post(base, "/v1/systemone", req)
            assert status == 200
            assert body["mode"] == "fast"
            assert "vote_share" not in body["answers"]["q1"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_invalid_mode_in_request_is_400(self) -> None:
        cfg = ServerConfig(model="dry-model", port=0, dry=True)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = {
                "state": "s",
                "model": "m",
                "mode": "quantum",
                "questions": {"q1": {"type": "noul", "instructions": "x"}},
            }
            status, body = _post(base, "/v1/systemone", req)
            assert status == 400
            assert body["error"]["code"] == "invalid_request"
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_deliberate_request_times_out_the_same_way_fast_mode_does(self) -> None:
        cfg = ServerConfig(model="m", port=0, request_timeout_s=0.1, max_workers=1, mode="deliberate")
        backend = SlowSamplingBackend(delay=1.0)
        httpd, thread, base = _start_custom_server(cfg, backend)
        try:
            req = {
                "state": "s",
                "model": "m",
                "questions": {"q1": {"type": "noul", "instructions": "x"}},
            }
            status, body = _post(base, "/v1/systemone", req)
            assert status == 504
            assert body["error"]["code"] == "request_timeout"
        finally:
            httpd.shutdown()
            httpd.server_close()
