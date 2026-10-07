"""Regression tests for the residual items R1 to R6 in the "Confirmation"
section of `docs/security-review-2026-10-06.md`, plus the two code
findings from the documentation pass (an unset `--bearer-token-env`, and
a mode override on a learned-mode server). Offline only: in-process
doubles and servers on ephemeral loopback ports.

R5 (Host parser hygiene and deployment notes) is not closed here.
"""

from __future__ import annotations

import json
import pathlib
import socket
import threading
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import decisionhead.server as server_mod
from decisionhead import backends
from decisionhead.client import ClientError, ask, health
from decisionhead.config import ServerConfig
from decisionhead.core import apply_abstain_band
from decisionhead.deliberate import answer_questions_deliberate, parse_deliberate_reply
from decisionhead.logging import get_logger
from decisionhead.safejson import loads_strict
from decisionhead.server import (
    AuditLog,
    Metrics,
    ReadinessChecker,
    _ConcurrencySlot,
    _run_with_timeout,
    check_bind_allowed,
    make_handler,
    resolve_effective_mode,
    run_server,
)

NOUL_Q = {"q1": {"type": "noul", "instructions": "The certificate on file is current."}}
VALID_REQUEST: dict[str, Any] = {"state": "Supplier Acme Corp.", "model": "jev-latest", "questions": NOUL_Q}


# --------------------------------------------------------------------- helpers


def _post(
    base: str, body: Any, *, content_type: str = "application/json", headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + "/v1/systemone", data=data, headers={"Content-Type": content_type, **(headers or {})}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, loads_strict(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _raw_status(port: int, request_bytes: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
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
    return b"".join(chunks).split(b"\r\n", 1)[0]


class _Server:
    """A decisionhead handler around an arbitrary backend double on an
    ephemeral loopback port."""

    def __init__(self, cfg: ServerConfig, backend: Any, bearer_token: str | None = None) -> None:
        self.audit = AuditLog(cfg.audit_path)
        handler = make_handler(
            backend,
            None,
            self.audit,
            cfg,
            Metrics(),
            ReadinessChecker(backend, cfg.ready_cache_s),
            get_logger(),
            bearer_token,
            backend_route="test",
        )
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _start_dry(**overrides: Any) -> tuple[ThreadingHTTPServer, str]:
    cfg = ServerConfig(model="test-model", port=0, dry=True, dry_seed=1, max_workers=2, **overrides)
    httpd = run_server(cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _stop(httpd: ThreadingHTTPServer) -> None:
    httpd.shutdown()
    httpd.server_close()


def _audit_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [loads_strict(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _FailingThread(threading.Thread):
    """A Thread whose start() fails the way thread exhaustion does."""

    def start(self) -> None:
        raise RuntimeError("can't start new thread")


# --------------------------------------------------------------------- R1


class TestR1SlotReleasedWhenThreadStartFails:
    def test_run_with_timeout_releases_the_slot_and_does_not_mark_it_handed_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        semaphore = threading.Semaphore(1)
        assert semaphore.acquire(blocking=False)
        metrics = Metrics()
        metrics.inc_concurrent()
        slot = _ConcurrencySlot(semaphore, metrics)
        shim = types.SimpleNamespace(Thread=_FailingThread, Lock=threading.Lock, Semaphore=threading.Semaphore)
        monkeypatch.setattr(server_mod, "threading", shim)
        with pytest.raises(RuntimeError):
            _run_with_timeout(lambda: {}, 5.0, slot=slot)
        assert slot.handed_off is False
        assert semaphore.acquire(blocking=False), "the slot was leaked"

    def test_repeated_thread_start_failures_never_turn_into_permanent_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        httpd, base = _start_dry(max_concurrent_requests=1)
        try:
            shim = types.SimpleNamespace(Thread=_FailingThread, Lock=threading.Lock, Semaphore=threading.Semaphore)
            monkeypatch.setattr(server_mod, "threading", shim)
            failed = [_post(base, VALID_REQUEST)[0] for _ in range(3)]
            monkeypatch.undo()
            status, _ = _post(base, VALID_REQUEST)
        finally:
            _stop(httpd)
        assert failed == [500, 500, 500]
        assert status == 200


# --------------------------------------------------------------------- R2


class TestR2PreAuthRejectionsAreNotAudited:
    def test_401_403_421_and_bad_auth_header_write_no_audit_rows(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cfg = ServerConfig(model="m", port=0, audit_path=str(audit_path))
        server = _Server(cfg, backends.FakeBackend(seed=1), bearer_token="FAKEKEY-token")
        try:
            s401, _ = _post(server.base, VALID_REQUEST)
            s403, _ = _post(server.base, VALID_REQUEST, headers={"Origin": "http://evil.example"})
            s421 = _raw_status(
                server.port,
                b"POST /v1/systemone HTTP/1.1\r\nHost: evil.example\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            s400 = _raw_status(
                server.port,
                b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer \xff\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n",
            )
        finally:
            server.close()
        assert (s401, s403) == (401, 403)
        assert b" 421 " in s421 and b" 400 " in s400
        assert _audit_rows(audit_path) == []

    def test_authenticated_failures_are_still_audited(self, tmp_path: pathlib.Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cfg = ServerConfig(model="m", port=0, audit_path=str(audit_path))
        server = _Server(cfg, backends.FakeBackend(seed=1), bearer_token="FAKEKEY-token")
        auth = {"Authorization": "Bearer FAKEKEY-token"}
        try:
            s415, _ = _post(server.base, VALID_REQUEST, content_type="text/plain", headers=auth)
            s400, _ = _post(server.base, {"state": "s", "model": "m", "questions": {}}, headers=auth)
        finally:
            server.close()
        assert (s415, s400) == (415, 400)
        assert [r["status"] for r in _audit_rows(audit_path)] == [415, 400]


# --------------------------------------------------------------------- R3


class TestR3FinalLineAnchoring:
    def test_review_example_with_stray_quote_is_malformed(self) -> None:
        text = 'The doc " says\n{"answer": "no", "why": "x {"answer":"yes"}'
        assert parse_deliberate_reply(text) is None

    def test_review_example_without_stray_quote_is_malformed(self) -> None:
        text = 'The doc says\n{"answer": "no", "why": "x {"answer":"yes"}'
        assert parse_deliberate_reply(text) is None

    def test_object_must_start_the_final_line(self) -> None:
        assert parse_deliberate_reply('Reasoning.\nFinal: {"answer": "yes", "confidence": 0.9}') is None

    def test_two_objects_on_the_final_line_are_malformed(self) -> None:
        text = 'Reasoning.\n{"answer": "yes"} {"answer": "no", "confidence": 0.4}'
        assert parse_deliberate_reply(text) is None

    def test_stray_quote_does_not_break_a_well_formed_final_line(self) -> None:
        text = 'The doc " says {"answer": "yes"}\n{"answer": "no", "confidence": 0.6}'
        assert parse_deliberate_reply(text) == {"answer": "no", "confidence": 0.6}

    def test_last_of_several_lines_counts_and_fence_and_indentation_are_tolerated(self) -> None:
        text = 'Doc: {"answer": "yes"}\n```json\n   {"answer": "no", "confidence": 0.6}  \n```\n\n'
        assert parse_deliberate_reply(text) == {"answer": "no", "confidence": 0.6}


# --------------------------------------------------------------------- R4


class _ShortBackend:
    """Returns ``n_return`` samples whatever ``n`` was asked for."""

    model = "m"

    def __init__(self, n_return: int) -> None:
        self.n_return = n_return

    def sample_completions(self, *args: Any, n: int, **kwargs: Any) -> tuple[list[str], dict[str, Any]]:
        return ['Sure.\n{"answer": "yes", "confidence": 0.99}'] * min(n, self.n_return), {"requests": 1}


class TestR4ShortSampleSetsAreFlagged:
    def test_shortfall_is_visible_and_low_confidence(self) -> None:
        result = answer_questions_deliberate("s", NOUL_Q, _ShortBackend(1), k=8, max_workers=1)
        apply_abstain_band(result, None)
        answer, row = result["answers"]["q1"], result["audit"][0]
        assert answer["n_missing_samples"] == 7 and row["n_missing_samples"] == 7
        assert answer["low_confidence"] is True and row["low_confidence"] is True
        assert result["low_confidence"] is True
        assert answer["degraded"] is False

    def test_handoff_mode_recommends_a_handoff_for_a_shortfall(self) -> None:
        result = answer_questions_deliberate("s", NOUL_Q, _ShortBackend(3), k=4, max_workers=1)
        apply_abstain_band(result, None, mode="handoff")
        assert result["answers"]["q1"]["recommended_action"] == "handoff"

    def test_full_sample_set_is_not_flagged(self) -> None:
        result = answer_questions_deliberate("s", NOUL_Q, _ShortBackend(8), k=8, max_workers=1)
        apply_abstain_band(result, None)
        assert result["answers"]["q1"]["n_missing_samples"] == 0
        assert result["low_confidence"] is False
        assert "low_confidence" not in result["answers"]["q1"]

    def test_shortfall_reaches_the_http_response(self, tmp_path: pathlib.Path) -> None:
        cfg = ServerConfig(model="m", port=0, mode="deliberate", deliberate_k=8, audit_path=str(tmp_path / "a.jsonl"))
        server = _Server(cfg, _ShortBackend(1))
        try:
            status, body = _post(server.base, VALID_REQUEST)
        finally:
            server.close()
        assert status == 200
        assert body["low_confidence"] is True
        assert body["answers"]["q1"]["n_missing_samples"] == 7


# --------------------------------------------------------------------- R6


class _CaptureHandler(BaseHTTPRequestHandler):
    seen: list[dict[str, str]] = []

    def _record(self) -> None:
        _CaptureHandler.seen.append(dict(self.headers))
        body = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _record
    do_POST = _record

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


class TestR6ClientRefusesRedirects:
    def test_bearer_token_is_never_forwarded_on_a_redirect(self) -> None:
        _CaptureHandler.seen = []
        capture = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        threading.Thread(target=capture.serve_forever, daemon=True).start()
        target = f"http://localhost:{capture.server_address[1]}/stolen"

        class Redirector(BaseHTTPRequestHandler):
            def _redirect(self) -> None:
                self.send_response(307)  # 307 keeps the method and body
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = _redirect
            do_POST = _redirect

            def log_message(self, fmt: str, *args: Any) -> None:
                pass

        redirector = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
        threading.Thread(target=redirector.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{redirector.server_address[1]}"
        try:
            with pytest.raises(ClientError, match="redirect"):
                ask(base, "s", NOUL_Q, timeout=5, bearer_token="FAKEKEY-service-token")
            with pytest.raises(ClientError, match="redirect"):
                health(base, timeout=5)
            assert _CaptureHandler.seen == []
        finally:
            for srv in (capture, redirector):
                srv.shutdown()
                srv.server_close()


# --------------------------------------------------------------------- documentation finding 1


class TestUnsetBearerTokenEnvRefusesToStart:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "0.0.0.0"])
    def test_check_bind_allowed_refuses_on_every_bind_address(self, host: str) -> None:
        cfg = ServerConfig(
            model="m", host=host, allow_non_local_bind=True, insecure_no_auth=True, bearer_token_env="DH_UNSET"
        )
        with pytest.raises(ValueError, match="unset or empty"):
            check_bind_allowed(cfg, bearer_token=None)

    @pytest.mark.parametrize("value", [None, ""])
    def test_run_server_refuses_without_echoing_the_name(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        if value is None:
            monkeypatch.delenv("FAKEKEY_LOOKS_LIKE_A_NAME", raising=False)
        else:
            monkeypatch.setenv("FAKEKEY_LOOKS_LIKE_A_NAME", value)
        cfg = ServerConfig(model="m", port=0, dry=True, bearer_token_env="FAKEKEY_LOOKS_LIKE_A_NAME")
        with pytest.raises(ValueError) as exc_info:
            run_server(cfg)
        assert "FAKEKEY_LOOKS_LIKE_A_NAME" not in str(exc_info.value)

    def test_cli_exits_with_a_config_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("DH_UNSET_BEARER", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            server_mod.main(["--dry", "--port", "0", "--bearer-token-env", "DH_UNSET_BEARER"])
        assert exc_info.value.code == 2
        assert "unset or empty" in capsys.readouterr().err

    def test_loopback_without_bearer_token_env_still_starts(self) -> None:
        run_server(ServerConfig(model="m", port=0, dry=True)).server_close()


# --------------------------------------------------------------------- documentation finding 2


def _write_learned_model(path: pathlib.Path) -> None:
    from decisionhead.features import FeatureConfig, feature_names
    from decisionhead.learned import LearnedModel
    from decisionhead.model import MultinomialLogisticRegression, StandardScaler

    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    names = feature_names(cfg)
    LearnedModel(
        classifier=MultinomialLogisticRegression.initialize(n_classes=4, n_features=len(names), seed=0),
        scaler=StandardScaler(means=[0.0] * len(names), stds=[1.0] * len(names)),
        feature_config=cfg,
        classes=["BLOCKED", "DONE", "INFO", "REASON"],
        metadata={"name": "test-learned-model"},
    ).save(str(path))


class TestLearnedModeRefusesOtherModeOverrides:
    def test_resolve_effective_mode_raises_a_question_error(self) -> None:
        from decisionhead.core import QuestionError

        cfg = ServerConfig(mode="learned", mode_override_allowed=True)
        for requested in ("fast", "deliberate"):
            with pytest.raises(QuestionError, match="mode='learned'"):
                resolve_effective_mode(cfg, {"mode": requested})
        assert resolve_effective_mode(cfg, {"mode": "learned"}) == "learned"

    def test_learned_server_returns_400_not_500(self, tmp_path: pathlib.Path) -> None:
        model_path = tmp_path / "model.json"
        _write_learned_model(model_path)
        cfg = ServerConfig(mode="learned", head_model_path=str(model_path), port=0, mode_override_allowed=True)
        httpd = run_server(cfg)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        question = {"q": {"type": "choice", "instructions": "x", "options": ["BLOCKED", "DONE", "INFO", "REASON"]}}
        state = {"requirements": [{"item_id": "a"}], "open_items": ["a"], "history": [], "steps_used": 0}
        try:
            results = [
                _post(base, {"state": state, "model": "m", "questions": question, "mode": mode})
                for mode in ("fast", "deliberate")
            ]
        finally:
            _stop(httpd)
        for status, body in results:
            assert status == 400
            assert body["error"]["code"] == "invalid_request"
            assert "mode='learned'" in body["error"]["message"]
