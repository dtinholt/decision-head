"""Regression tests for the independent security review dated 2026-10-06
(`docs/security-review-2026-10-06.md`). Each class reproduces one
finding's evidence (the reviewer's own scenario where there is one) and
asserts the fix. Offline only: FakeBackend, fixture transports, and
servers on ephemeral loopback ports. Closure table:
`docs/security-review-2026-10-06-closures.md`.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from decisionhead import backends
from decisionhead.backends import (
    BackendError,
    ConcurrencyLimitedBackend,
    OllamaBackend,
    OpenAIBackend,
    insecure_transport_warning,
)
from decisionhead.cache import ResponseCache
from decisionhead.config import ConfigError, ServerConfig, resolve_config
from decisionhead.core import QuestionError, answer_questions, apply_abstain_band, normalize_question
from decisionhead.deadline import Deadline, DeadlineExceeded
from decisionhead.deliberate import (
    aggregate_noul_votes,
    answer_questions_deliberate,
    apply_claim_guard,
    parse_deliberate_reply,
)
from decisionhead.features import FeatureConfig, extract_features
from decisionhead.logging import get_logger
from decisionhead.safejson import dumps_strict, loads_strict
from decisionhead.server import (
    AuditLog,
    Metrics,
    ReadinessChecker,
    check_bind_allowed,
    load_calibration,
    make_handler,
    run_server,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

NOUL_Q = {"q1": {"type": "noul", "instructions": "The certificate on file is current."}}
VALID_REQUEST: dict[str, Any] = {"state": "Supplier Acme Corp.", "model": "jev-latest", "questions": NOUL_Q}


# --------------------------------------------------------------------- helpers


def _post(
    base: str,
    body: Any,
    *,
    content_type: str = "application/json",
    headers: dict[str, str] | None = None,
    raw: bytes | None = None,
    path: str = "/v1/systemone",
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    all_headers = {"Content-Type": content_type, **(headers or {})}
    req = urllib.request.Request(base + path, data=data, headers=all_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, loads_strict(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), dict(exc.headers)


def _get(base: str, path: str, headers: dict[str, str] | None = None) -> int:
    req = urllib.request.Request(base + path, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def _raw_request(port: int, request_bytes: bytes, timeout: float = 5.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request_bytes)
        chunks = []
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except OSError:
            pass
        return b"".join(chunks)


class _Server:
    """A decisionhead handler around an arbitrary backend double on an
    ephemeral loopback port (bypasses run_server's backend construction)."""

    def __init__(self, cfg: ServerConfig, backend: Any, *, cache: ResponseCache | None = None) -> None:
        self.cfg = cfg
        self.audit = AuditLog(cfg.audit_path)
        self.metrics = Metrics()
        self.cache = cache
        handler = make_handler(
            backend,
            None,
            self.audit,
            cfg,
            self.metrics,
            ReadinessChecker(backend, cfg.ready_cache_s),
            get_logger(),
            backends.env_ref(cfg.bearer_token_env),
            cache=cache,
            backend_route="test",
        )
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _start_dry(**overrides: Any) -> tuple[ThreadingHTTPServer, str]:
    cfg = ServerConfig(model="test-model", port=0, dry=True, dry_seed=1, max_workers=2, **overrides)
    httpd = run_server(cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _stop(httpd: ThreadingHTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


class RecordingSlowBackend:
    """Sleeps ``delay`` per call and records when each call STARTED and
    for which state, so a test can tell whether a backend call began
    after its request had already been answered 504."""

    model = "slow-model"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.starts: list[tuple[str, float]] = []

    def token_distribution(self, prefix_messages, question_text, allowed_tokens):  # type: ignore[no-untyped-def]
        with self.lock:
            self.starts.append((prefix_messages[-1]["content"], time.monotonic()))
        time.sleep(self.delay)
        dist = {t: (1.0 if i == 0 else 0.0) for i, t in enumerate(allowed_tokens)}
        return {**dist, "_unparsed": 0.0, "_raw": {}}

    def sample_completions(self, prefix_messages, question_text, allowed_answers, *, n, temperature, max_tokens):  # type: ignore[no-untyped-def]
        with self.lock:
            self.starts.append((prefix_messages[-1]["content"], time.monotonic()))
        time.sleep(self.delay)
        return [json.dumps({"answer": allowed_answers[0], "confidence": 0.9})] * n, {"requests": 1}


def _ollama_fixture(resp: dict[str, Any]) -> OllamaBackend:
    return OllamaBackend("http://127.0.0.1:11434", "fixture-model", transport=lambda u, b, h, t: resp)


# --------------------------------------------------------------------- finding 1


class TestFinding1OrphanedWorkAfterTimeout:
    """HIGH: a 504'd request released its slot while its worker kept
    calling the backend for every remaining question."""

    def test_timed_out_requests_leave_zero_orphan_backend_calls(self) -> None:
        # The reviewer's scenario: 0.5 s per call behind a concurrency-1
        # backend, request timeout 0.3 s, six 50-question requests.
        inner = RecordingSlowBackend(delay=0.5)
        # Built exactly as in the review (no acquire timeout): the
        # request deadline alone must stop the queued work.
        backend = ConcurrencyLimitedBackend(inner, 1)  # type: ignore[arg-type]
        cfg = ServerConfig(model="m", port=0, request_timeout_s=0.3, max_workers=4, max_concurrent_requests=16)
        baseline_threads = threading.active_count()
        server = _Server(cfg, backend)
        answered_at: dict[str, float] = {}
        try:
            questions = {f"q{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(50)}
            for r in range(6):
                state = f"state for request {r}"
                status, body, _ = _post(server.base, {"state": state, "model": "m", "questions": questions})
                answered_at[f"State:\n{state}"] = time.monotonic()
                assert status == 504, body
            # Give any orphaned work every chance to show itself.
            time.sleep(1.5)
            with inner.lock:
                starts = list(inner.starts)
            orphans = [(s, t) for s, t in starts if t > answered_at[s]]
            assert orphans == [], f"{len(orphans)} backend calls started after their request got 504"
            assert len(starts) <= 12  # at most a call or two per request, never 300
            deadline = time.monotonic() + 3
            while threading.active_count() > baseline_threads + 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert threading.active_count() <= baseline_threads + 3
        finally:
            server.close()

    def test_429_fires_while_a_timed_out_worker_still_holds_its_slot(self) -> None:
        inner = RecordingSlowBackend(delay=1.0)
        cfg = ServerConfig(model="m", port=0, request_timeout_s=0.2, max_workers=1, max_concurrent_requests=1)
        server = _Server(cfg, inner)
        try:
            status1, _, _ = _post(server.base, VALID_REQUEST)
            assert status1 == 504
            status2, body2, _ = _post(server.base, VALID_REQUEST)
            assert status2 == 429, body2  # the orphaned worker still counts
            time.sleep(1.2)  # the in-flight call ends; the worker releases the slot
            status3, _, _ = _post(server.base, VALID_REQUEST)
            assert status3 == 504
        finally:
            server.close()

    def test_backend_semaphore_acquire_times_out(self) -> None:
        inner = RecordingSlowBackend(delay=0.6)
        backend = ConcurrencyLimitedBackend(inner, 1, acquire_timeout_s=0.1)  # type: ignore[arg-type]
        holder = threading.Thread(target=backend.token_distribution, args=([{"content": "a"}], "q", ["Yes", "No"]))
        holder.start()
        time.sleep(0.05)
        t0 = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            backend.token_distribution([{"content": "b"}], "q", ["Yes", "No"])
        assert time.monotonic() - t0 < 0.4
        holder.join()
        assert len(inner.starts) == 1

    def test_queued_questions_are_abandoned_at_the_deadline(self) -> None:
        inner = RecordingSlowBackend(delay=0.1)
        questions = {f"q{i}": {"type": "noul", "instructions": f"q {i}"} for i in range(20)}
        with pytest.raises(DeadlineExceeded):
            answer_questions("s", questions, inner, max_workers=1, deadline=Deadline(0.25))
        assert len(inner.starts) <= 3

    def test_queued_deliberate_samples_are_abandoned_at_the_deadline(self) -> None:
        inner = RecordingSlowBackend(delay=0.1)
        questions = {f"q{i}": {"type": "noul", "instructions": f"q {i}"} for i in range(20)}
        with pytest.raises(DeadlineExceeded):
            answer_questions_deliberate("s", questions, inner, k=2, max_workers=1, deadline=Deadline(0.25))
        assert len(inner.starts) <= 3

    def test_design_doc_no_longer_claims_the_old_bound(self) -> None:
        design = (REPO_ROOT / "docs" / "design.md").read_text(encoding="utf-8")
        assert "roughly `questions / max_workers`" not in design


# --------------------------------------------------------------------- finding 2


class TestFinding2CrossOriginAndRebinding:
    def test_text_plain_with_json_charset_is_415(self) -> None:
        httpd, base = _start_dry()
        try:
            status, body, _ = _post(base, VALID_REQUEST, content_type="text/plain; charset=application/json")
            assert status == 415, body
        finally:
            _stop(httpd)

    def test_application_json_with_parameters_is_accepted(self) -> None:
        httpd, base = _start_dry()
        try:
            status, _, _ = _post(base, VALID_REQUEST, content_type="Application/JSON; charset=utf-8")
            assert status == 200
        finally:
            _stop(httpd)

    def test_foreign_host_header_is_refused_on_post_and_get(self) -> None:
        httpd, base = _start_dry()
        port = httpd.server_address[1]
        try:
            body = json.dumps(VALID_REQUEST).encode("utf-8")
            post = (
                b"POST /v1/systemone HTTP/1.1\r\nHost: attacker.example:%d\r\n"
                b"Content-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % (port, len(body))
            ) + body
            assert b" 421 " in _raw_request(port, post).split(b"\r\n", 1)[0]
            get = b"GET /metrics HTTP/1.1\r\nHost: attacker.example\r\nConnection: close\r\n\r\n"
            assert b" 421 " in _raw_request(port, get).split(b"\r\n", 1)[0]
            for ok_host in (b"127.0.0.1:%d" % port, b"localhost", b"[::1]:%d" % port):
                get_ok = b"GET /health HTTP/1.1\r\nHost: " + ok_host + b"\r\nConnection: close\r\n\r\n"
                assert b" 200 " in _raw_request(port, get_ok).split(b"\r\n", 1)[0]
        finally:
            _stop(httpd)

    def test_operator_allowlisted_host_is_accepted(self) -> None:
        httpd, _base = _start_dry(allowed_hosts="decisionhead, dh.internal")
        port = httpd.server_address[1]
        try:
            get = b"GET /health HTTP/1.1\r\nHost: dh.internal:8787\r\nConnection: close\r\n\r\n"
            assert b" 200 " in _raw_request(port, get).split(b"\r\n", 1)[0]
        finally:
            _stop(httpd)

    def test_cross_origin_post_is_refused_when_cors_is_off(self) -> None:
        httpd, base = _start_dry()
        try:
            status, body, _ = _post(base, VALID_REQUEST, headers={"Origin": "https://attacker.example"})
            assert status == 403
            assert body["error"]["code"] == "forbidden_origin"
        finally:
            _stop(httpd)

    def test_security_md_states_the_no_token_risk(self) -> None:
        text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        assert "Running without a bearer token" in text


# --------------------------------------------------------------------- finding 3


class _CaptureHandler(BaseHTTPRequestHandler):
    seen: list[dict[str, str]] = []

    def _record(self) -> None:
        type(self).seen.append(dict(self.headers))
        body = b'{"choices": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _record
    do_POST = _record

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


class TestFinding3CredentialForwarding:
    def test_redirect_is_refused_and_the_key_never_reaches_the_other_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _CaptureHandler.seen = []
        capture = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        threading.Thread(target=capture.serve_forever, daemon=True).start()
        target = f"http://localhost:{capture.server_address[1]}/stolen"  # a different host name

        class Redirector(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                pass

        redirector = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
        threading.Thread(target=redirector.serve_forever, daemon=True).start()
        monkeypatch.setenv("DH_TEST_API_KEY", "FAKEKEY-not-a-real-key")
        try:
            backend = OpenAIBackend(
                f"http://127.0.0.1:{redirector.server_address[1]}", "m", api_key_ref="DH_TEST_API_KEY", timeout=5
            )
            with pytest.raises(BackendError, match="redirect"):
                backend.token_distribution([], "Q?", ["Yes", "No"])
            assert _CaptureHandler.seen == []
        finally:
            for srv in (capture, redirector):
                srv.shutdown()
                srv.server_close()

    def test_followed_cross_host_redirect_drops_authorization(self) -> None:
        policy = backends._RedirectPolicy(follow=True)
        req = urllib.request.Request(
            "http://a.example/v1/x", headers={"Authorization": "Bearer FAKEKEY-x", "X-Other": "1"}
        )
        cross = policy.redirect_request(req, None, 302, "Found", {}, "http://b.example/v1/x")  # type: ignore[arg-type]
        assert cross is not None
        assert "Authorization" not in cross.headers and "authorization" not in {k.lower() for k in cross.headers}
        same = policy.redirect_request(req, None, 302, "Found", {}, "http://a.example/v1/y")  # type: ignore[arg-type]
        assert same is not None
        assert any(k.lower() == "authorization" for k in same.headers)

    def test_non_http_scheme_is_refused(self) -> None:
        with pytest.raises(BackendError):
            backends._http_post_json("file:///etc/hosts", {}, {}, 1)
        with pytest.raises(ConfigError):
            resolve_config({"base_url": "file:///etc/hosts"}, env={})

    def test_plain_http_to_a_non_loopback_host_warns(self) -> None:
        assert insecure_transport_warning("http://203.0.113.10:11434", has_credential=True)
        assert "API key" in (insecure_transport_warning("http://203.0.113.10:8000", has_credential=True) or "")
        assert insecure_transport_warning("https://203.0.113.10:8000", has_credential=True) is None
        assert insecure_transport_warning("http://127.0.0.1:11434", has_credential=True) is None

    def test_run_server_logs_the_warning_at_start_up(self, capfd: pytest.CaptureFixture[str]) -> None:
        cfg = ServerConfig(base_url="http://203.0.113.10:11434", model="m", port=0)
        httpd = run_server(cfg)  # constructs the backend; makes no network call
        httpd.server_close()
        assert "insecure_backend_transport" in capfd.readouterr().err


# --------------------------------------------------------------------- finding 4


class TestFinding4NonFiniteNumbers:
    @pytest.mark.parametrize(
        "raw",
        [
            b'{"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "x"}}, "n": NaN}',
            b'{"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "x"}}, "n": 1e999}',
            b'{"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "x"}}, "n": -Infinity}',
            b'{"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "x"}}, "n": '
            + b"1" * 100
            + b"}",
        ],
    )
    def test_request_body_with_non_finite_or_huge_number_is_400(self, raw: bytes) -> None:
        httpd, base = _start_dry()
        try:
            status, body, _ = _post(base, None, raw=raw)
            assert status == 400
            assert body["error"]["code"] == "invalid_json"
        finally:
            _stop(httpd)

    def test_numeric_looking_score_levels_never_produce_infinity(self) -> None:
        httpd, base = _start_dry()
        try:
            req = {
                "state": "s",
                "model": "m",
                "questions": {"s": {"type": "score", "instructions": "rate", "levels": ["nan", "inf", "1e999"]}},
            }
            status, body, _ = _post(base, req)  # _post parses the response strictly
            assert status == 200
            assert math.isfinite(body["answers"]["s"]["score"])
        finally:
            _stop(httpd)

    @pytest.mark.parametrize(
        "content",
        ['{"per_type": {"noul": {"T": NaN}}}', '{"noul": {"T": Infinity}}', '{"noul": {"T": -1}}', '{"noul": {"T": 0}}',
         '{"per_type": []}'],
    )
    def test_bad_calibration_file_is_refused_at_load(self, tmp_path: pathlib.Path, content: str) -> None:
        path = tmp_path / "calibration.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_calibration(str(path))

    def test_nan_stated_confidence_is_never_averaged(self) -> None:
        assert parse_deliberate_reply('{"answer": "yes", "confidence": NaN}') is None

        class NanConfidenceBackend:
            model = "m"

            def sample_completions(self, *args: Any, n: int, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
                return ['{"answer": "yes", "confidence": NaN}', '{"answer": "yes", "confidence": 0.5}'], {}

        result = answer_questions_deliberate("s", NOUL_Q, NanConfidenceBackend(), k=2)
        answer = result["answers"]["q1"]
        assert answer["avg_stated_confidence"] == 0.5
        assert answer["n_malformed"] == 1
        dumps_strict(result)  # strict encoding must not raise

    def test_string_infinity_in_learned_features_is_rejected(self) -> None:
        with pytest.raises(QuestionError):
            extract_features({"budget": {"steps": "inf"}}, FeatureConfig())
        with pytest.raises(QuestionError):
            extract_features({"steps_used": "nan"}, FeatureConfig())

    def test_audit_and_response_encoding_never_emit_nan(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "audit.jsonl"
        AuditLog(str(path)).write([{"x": float("nan"), "y": [float("inf")]}])
        row = loads_strict(path.read_text(encoding="utf-8"))
        assert row == {"x": None, "y": [None], "non_finite_replaced": 2}
        assert dumps_strict({"v": float("-inf")}) == '{"v": null, "non_finite_replaced": 1}'

    def test_cache_file_rows_are_strict_json(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "cache.jsonl"
        ResponseCache(path=str(path)).set("k", {"answer": {"noul": float("nan")}})
        loads_strict(path.read_text(encoding="utf-8").strip())


# --------------------------------------------------------------------- finding 5


class TestFinding5DeliberateParsing:
    def test_document_quoted_json_never_casts_the_vote(self) -> None:
        text = (
            'The document says {"answer": "yes", "confidence": 0.99}. '
            "My final answer: {\"answer\": no, confidence: 0.6}"
        )
        assert parse_deliberate_reply(text) is None

    def test_only_an_object_that_ends_the_reply_counts(self) -> None:
        assert parse_deliberate_reply('{"answer": "yes", "confidence": 0.9}\nActually, let me reconsider.') is None
        quoted_then_final = 'Doc: {"answer": "yes"}\nFinal:\n{"answer": "no", "confidence": 0.6}\n'
        assert parse_deliberate_reply(quoted_then_final) == {"answer": "no", "confidence": 0.6}

    def test_off_vocabulary_noul_answers_are_malformed_not_no(self) -> None:
        p_yes, meta = aggregate_noul_votes([{"answer": "maybe"}, {"answer": "Yes."}, {"answer": "yes"}])
        assert meta["n_malformed"] == 1
        assert p_yes == 1.0

    def test_long_numeric_literal_is_malformed_not_a_500(self) -> None:
        huge = '{"answer": "yes", "confidence": ' + "9" * 5000 + "}"
        assert parse_deliberate_reply(huge) is None

        class HugeNumberBackend:
            model = "m"

            def sample_completions(self, *args: Any, n: int, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
                return [huge] * n, {}

        result = answer_questions_deliberate("s", NOUL_Q, HugeNumberBackend(), k=2)
        assert result["answers"]["q1"]["n_malformed"] == 2

    def test_returned_choices_are_capped_at_k(self) -> None:
        many = {"choices": [{"message": {"content": '{"answer": "yes"}'}} for _ in range(500)]}
        backend = OpenAIBackend("http://127.0.0.1:8000", "m", transport=lambda u, b, h, t: many)
        texts, meta = backend.sample_completions([], "Q?", ["yes", "no"], n=8, temperature=0.7, max_tokens=16)
        assert len(texts) == 8
        assert meta["returned_n"] == 500

        class OverDeliveringBackend:
            model = "m"

            def sample_completions(self, *args: Any, n: int, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
                return ['{"answer": "yes"}'] * (n + 12), {}

        result = answer_questions_deliberate("s", NOUL_Q, OverDeliveringBackend(), k=8)
        answer = result["answers"]["q1"]
        assert answer["n_parsed"] == 8
        assert answer["n_excess_samples"] == 12
        assert answer["n_malformed"] == 12
        assert result["audit"][0]["votes"] == {"yes": 8, "no": 0}

    def test_claim_guard_string_terminal_answers_is_a_400_not_a_substring_match(self) -> None:
        qdef = {
            "type": "choice",
            "instructions": "x",
            "criteria": {"COMPLETE": "a", "CONTINUE": "b"},
            "terminal_answers": "CLAIM_COMPLETE",
        }
        with pytest.raises(QuestionError):
            normalize_question("q", qdef)
        # Second layer: the guard itself never substring-matches.
        choice, extra = apply_claim_guard(qdef, "COMPLETE", {"COMPLETE": 0.5, "CONTINUE": 0.5}, enabled=True,
                                          threshold=0.75)
        assert (choice, extra) == ("COMPLETE", None)

    def test_claim_guard_list_evidence_is_a_400_over_http(self) -> None:
        httpd, base = _start_dry(mode="deliberate")
        try:
            req = {
                "state": "s",
                "model": "m",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "x",
                        "criteria": {"CLAIM_COMPLETE": "a", "CONTINUE": "b"},
                        "terminal_answers": ["CLAIM_COMPLETE"],
                        "completion_evidence": ["not", "an", "object"],
                    }
                },
            }
            status, body, _ = _post(base, req)
            assert status == 400, body
        finally:
            _stop(httpd)

    def test_unmatched_answers_and_malformed_counts_are_reported(self) -> None:
        class UnmatchedBackend:
            model = "m"

            def sample_completions(self, *args: Any, n: int, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
                return ['{"answer": "SOMETHING_ELSE"}'] * n, {}

        q = {"c": {"type": "choice", "instructions": "x", "criteria": {"A": "a", "B": "b"}}}
        result = answer_questions_deliberate("s", q, UnmatchedBackend(), k=3)
        answer, row = result["answers"]["c"], result["audit"][0]
        assert answer["n_unmatched_answer"] == 3 and row["n_unmatched_answer"] == 3
        assert "n_malformed" in answer and "n_malformed" in row
        assert answer["degraded"] is True and answer["choice"] is None

    def test_claim_guard_reason_records_counts_not_document_text(self) -> None:
        qdef = {
            "type": "choice",
            "instructions": "x",
            "criteria": {"CLAIM_COMPLETE": "a", "CONTINUE": "b"},
            "terminal_answers": ["CLAIM_COMPLETE"],
            "completion_evidence": {"unmet_requirements": ["SECRET DOCUMENT TEXT"]},
        }
        _choice, extra = apply_claim_guard(qdef, "CLAIM_COMPLETE", {"CLAIM_COMPLETE": 1.0, "CONTINUE": 0.0},
                                           enabled=True, threshold=0.75)
        assert extra is not None
        assert "SECRET DOCUMENT TEXT" not in json.dumps(extra)


# --------------------------------------------------------------------- finding 6


NAN_YES_RESPONSE = {
    "message": {"content": "Yes"},
    "logprobs": [
        {"token": "Yes", "logprob": float("nan"), "top_logprobs": [{"token": "Yes", "logprob": float("nan")}]}
    ],
}


class TestFinding6DegradedReadings:
    def test_fully_degraded_fast_answer_is_flagged_not_a_plausible_half(self) -> None:
        result = answer_questions("s", NOUL_Q, _ollama_fixture(NAN_YES_RESPONSE))
        answer, row = result["answers"]["q1"], result["audit"][0]
        assert answer["noul"] is None
        assert answer["degraded"] is True
        assert answer["no_family_match"] is True and row["no_family_match"] is True
        assert answer["non_finite_logprob_count"] == 1 and row["non_finite_logprob_count"] == 1
        apply_abstain_band(result, None, "handoff")
        assert result["degraded"] is True and result["low_confidence"] is True
        assert answer["recommended_action"] == "handoff"

    def test_degraded_fields_reach_the_http_response_and_audit_row(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cfg = ServerConfig(model="m", port=0, audit_path=str(audit_path))
        server = _Server(cfg, _ollama_fixture(NAN_YES_RESPONSE))
        try:
            status, body, _ = _post(server.base, VALID_REQUEST)
        finally:
            server.close()
        assert status == 200
        assert body["degraded"] is True
        assert body["answers"]["q1"]["noul"] is None
        row = loads_strict(audit_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["no_family_match"] is True
        assert row["non_finite_logprob_count"] == 1

    def test_degraded_answers_are_not_cached(self) -> None:
        cache = ResponseCache()
        answer_questions("s", NOUL_Q, _ollama_fixture(NAN_YES_RESPONSE), cache=cache)
        assert len(cache) == 0


# --------------------------------------------------------------------- finding 7


class TestFinding7ContainerDefaults:
    def test_non_loopback_bind_requires_a_token(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, host="0.0.0.0", allow_non_local_bind=True)
        with pytest.raises(ValueError, match="bearer token"):
            check_bind_allowed(cfg, bearer_token=None)
        check_bind_allowed(cfg, bearer_token="FAKEKEY-token")
        check_bind_allowed(replace(cfg, insecure_no_auth=True), bearer_token=None)

    def test_run_server_refuses_a_tokenless_non_loopback_bind(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, host="0.0.0.0", allow_non_local_bind=True)
        with pytest.raises(ValueError, match="bearer token"):
            run_server(cfg)

    def test_image_binds_loopback_by_default(self) -> None:
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        cmd_lines = [line for line in dockerfile.splitlines() if line.startswith(("CMD", "ENTRYPOINT"))]
        assert not any("0.0.0.0" in line or "allow-non-local-bind" in line for line in cmd_lines)

    @pytest.mark.parametrize("compose", ["compose.yaml", "examples/compose.yaml"])
    def test_compose_publishes_nothing_on_all_interfaces_and_sets_a_token(self, compose: str) -> None:
        text = (REPO_ROOT / compose).read_text(encoding="utf-8")
        live = [line.split("#", 1)[0] for line in text.splitlines()]
        published = [m.group(1) for line in live for m in [re.match(r'\s*-\s*"([^"]+)"', line)] if m]
        port_maps = [p for p in published if p.count(":") >= 1 and p.split(":")[-1].isdigit()]
        assert port_maps, "expected at least one published port"
        assert all(p.startswith("127.0.0.1:") for p in port_maps), port_maps
        assert not any("11434:11434" in p and not p.startswith("127.0.0.1:") for p in port_maps)
        assert any("DECISIONHEAD_BEARER_TOKEN_ENV" in line for line in live)

    def test_dotenv_is_gitignored(self) -> None:
        assert ".env" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------- finding 8


def _audit_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return [loads_strict(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestFinding8AuditTrail:
    def test_success_rows_carry_ts_request_id_mode_status_and_answer(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        httpd, base = _start_dry(audit_path=str(audit_path))
        try:
            status, body, headers = _post(base, VALID_REQUEST)
        finally:
            _stop(httpd)
        assert status == 200
        (row,) = _audit_rows(audit_path)
        assert row["request_id"] == headers["X-Request-Id"]
        assert row["ts"].endswith("Z")
        assert row["mode"] == "fast"
        assert row["status"] == 200
        assert row["answer"] == body["answers"]["q1"]
        assert "calibration_version" in row

    def test_deliberate_rows_carry_vote_counts_and_the_result(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        httpd, base = _start_dry(audit_path=str(audit_path), mode="deliberate", deliberate_k=4)
        try:
            status, body, _ = _post(base, VALID_REQUEST)
        finally:
            _stop(httpd)
        assert status == 200
        (row,) = _audit_rows(audit_path)
        assert row["mode"] == "deliberate"
        assert sum(row["votes"].values()) == row["n_parsed"]
        assert row["result"] == body["answers"]["q1"]["noul"]
        assert row["answer"] == body["answers"]["q1"]

    def test_failed_requests_are_audited_with_their_status(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        httpd, base = _start_dry(audit_path=str(audit_path))
        try:
            status, _, headers = _post(base, {"state": "s", "model": "m", "questions": {}})
            status_415, _, _ = _post(base, VALID_REQUEST, content_type="text/plain")
        finally:
            _stop(httpd)
        assert (status, status_415) == (400, 415)
        rows = _audit_rows(audit_path)
        assert [r["status"] for r in rows] == [400, 415]
        assert rows[0]["request_id"] == headers["X-Request-Id"]
        assert rows[0]["error"] == "invalid_request"
        assert all("ts" in r for r in rows)

    def test_timed_out_request_is_audited_and_never_cached(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cache = ResponseCache()
        inner = RecordingSlowBackend(delay=0.4)
        cfg = ServerConfig(model="m", port=0, request_timeout_s=0.15, max_workers=1, audit_path=str(audit_path))
        server = _Server(cfg, inner, cache=cache)
        try:
            status, _, _ = _post(server.base, VALID_REQUEST)
            time.sleep(0.6)  # the in-flight call finishes in the background
        finally:
            server.close()
        assert status == 504
        assert len(cache) == 0
        (row,) = _audit_rows(audit_path)
        assert row["status"] == 504 and row["error"] == "request_timeout" and row["mode"] == "fast"


# --------------------------------------------------------------------- findings 10, 11, 13 (cheap closures)


class TestFinding10NoInternalTextInErrors:
    def test_502_message_is_fixed(self) -> None:
        class LeakyBackend:
            model = "m"

            def token_distribution(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise BackendError("HTTP 500 from http://user:pw@internal-host:1/api/chat: secret backend body")

        server = _Server(ServerConfig(model="m", port=0), LeakyBackend())
        try:
            status, body, _ = _post(server.base, VALID_REQUEST)
        finally:
            server.close()
        assert status == 502
        text = json.dumps(body)
        assert "internal-host" not in text and "secret backend body" not in text and "pw" not in text

    def test_redact_url_strips_userinfo(self) -> None:
        assert backends.redact_url("http://user:pw@host:8000/x") == "http://host:8000/x"


class TestFinding11NonAsciiAuthorization:
    def test_non_ascii_authorization_header_is_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DH_TEST_BEARER", "FAKEKEY-token")
        httpd, _base = _start_dry(bearer_token_env="DH_TEST_BEARER")
        port = httpd.server_address[1]
        try:
            body = json.dumps(VALID_REQUEST).encode("utf-8")
            request = (
                b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer \xff\xfe\r\n"
                b"Content-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body)
            ) + body
            response = _raw_request(port, request)
            assert b" 400 " in response.split(b"\r\n", 1)[0]
            assert b"Traceback" not in response
        finally:
            _stop(httpd)


class TestFinding13OperatorTraps:
    def test_a_token_passed_as_an_env_name_is_refused_without_echoing_it(self) -> None:
        value = "FAKEKEY-abc/123+token=="
        with pytest.raises(ConfigError) as exc_info:
            resolve_config({"bearer_token_env": value}, env={})
        assert value not in str(exc_info.value)

    def test_unset_bearer_env_refusal_does_not_echo_the_name(self, capfd: pytest.CaptureFixture[str]) -> None:
        # Since the documentation pass an unset --bearer-token-env refuses
        # to start (it used to warn and serve); the name is still never
        # echoed, in the error or the log.
        cfg = ServerConfig(model="m", port=0, dry=True, bearer_token_env="FAKEKEY_LOOKS_LIKE_A_NAME")
        with pytest.raises(ValueError) as exc_info:
            run_server(cfg)
        assert "FAKEKEY_LOOKS_LIKE_A_NAME" not in str(exc_info.value)
        assert "FAKEKEY_LOOKS_LIKE_A_NAME" not in capfd.readouterr().err

    def test_backend_timeout_s_is_wired_and_not_truncated(self) -> None:
        cfg = ServerConfig(base_url="http://127.0.0.1:11434", model="m", port=0, backend_timeout_s=2.5)
        assert backends.unwrap_backend(_backend_of(cfg)).timeout == 2.5  # was int(question_timeout_s) == 15
        cfg_short = replace(cfg, question_timeout_s=0.5)
        assert backends.unwrap_backend(_backend_of(cfg_short)).timeout == 0.5  # was int(0.5) == 0

    def test_zero_limits_are_refused(self) -> None:
        for field in ("max_concurrent_requests", "connection_read_timeout_s", "request_timeout_s"):
            with pytest.raises(ConfigError):
                resolve_config({field: 0}, env={})


def _backend_of(cfg: ServerConfig) -> Any:
    """The backend run_server would build for ``cfg`` (captured by
    patching make_handler, so nothing is served)."""
    import decisionhead.server as server_mod

    captured: dict[str, Any] = {}
    original = server_mod.make_handler

    def _capture(backend: Any, *args: Any, **kwargs: Any) -> Any:
        captured["backend"] = backend
        return original(backend, *args, **kwargs)

    server_mod.make_handler = _capture  # type: ignore[assignment]
    try:
        run_server(cfg).server_close()
    finally:
        server_mod.make_handler = original  # type: ignore[assignment]
    return captured["backend"]


class TestFinding9AuditFailClosed:
    def test_unwritable_audit_path_returns_a_json_500_not_a_dropped_connection(self, tmp_path: pathlib.Path) -> None:
        httpd, base = _start_dry(audit_path=str(tmp_path))  # a directory: every write raises IsADirectoryError
        try:
            status, body, headers = _post(base, VALID_REQUEST)
        finally:
            _stop(httpd)
        assert status == 500
        assert body["error"]["code"] == "internal_error"
        assert headers["X-Request-Id"] in body["error"]["message"]
