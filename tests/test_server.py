from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer

import pytest

from decisionhead.config import ServerConfig
from decisionhead.core import QuestionError
from decisionhead.logging import get_logger
from decisionhead.server import (
    AuditLog,
    Metrics,
    ReadinessChecker,
    check_bind_allowed,
    make_handler,
    resolve_effective_max_concurrency_override,
    run_server,
)

VALID_REQUEST = {
    "state": "Supplier Acme Corp submitted a certificate of insurance dated last year.",
    "model": "jev-latest",
    "questions": {"q1": {"type": "noul", "instructions": "The certificate on file is current."}},
}


class SlowBackend:
    """A backend double that sleeps before answering, used to exercise
    timeout and concurrency-limiting behaviour deterministically."""

    model = "slow-model"

    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def token_distribution(self, prefix_messages, question_text, allowed_tokens):  # type: ignore[no-untyped-def]
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        dist = {t: (1.0 if i == 0 else 0.0) for i, t in enumerate(allowed_tokens)}
        return {**dist, "_unparsed": 0.0, "_raw": {}}


def _post(base: str, path: str, body: dict, content_type: str | None = "application/json") -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(base + path, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(base: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _start_custom_server(cfg: ServerConfig, backend):  # type: ignore[no-untyped-def]
    """Builds a server around an arbitrary backend double, bypassing
    run_server's normal backend construction (needed for the SlowBackend
    tests below)."""
    audit = AuditLog(cfg.audit_path)
    metrics = Metrics()
    readiness = ReadinessChecker(backend, cfg.ready_cache_s)
    logger = get_logger()
    from decisionhead.backends import env_ref

    bearer_token = env_ref(cfg.bearer_token_env)
    handler = make_handler(backend, None, audit, cfg, metrics, readiness, logger, bearer_token)
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, thread, base, metrics


class TestBasicEndpoints:
    def test_health(self, running_server: str) -> None:
        status, body = _get(running_server, "/health")
        assert status == 200
        assert json.loads(body)["status"] == "ok"

    def test_ready_is_true_for_dry_backend(self, running_server: str) -> None:
        status, body = _get(running_server, "/ready")
        assert status == 200
        assert json.loads(body)["ready"] is True

    def test_v1_models_reports_real_model_id(self, running_server: str) -> None:
        status, body = _get(running_server, "/v1/models")
        parsed = json.loads(body)
        assert parsed["models"][0]["id"] == "test-model"
        assert "jev-latest" in parsed["models"][0]["aliases"]

    def test_openapi_json_documents_systemone(self, running_server: str) -> None:
        status, body = _get(running_server, "/openapi.json")
        assert status == 200
        assert "/v1/systemone" in json.loads(body)["paths"]

    def test_metrics_reports_requests_after_a_call(self, running_server: str) -> None:
        _get(running_server, "/health")
        status, body = _get(running_server, "/metrics")
        assert status == 200
        assert "decisionhead_requests_total" in body
        assert "decisionhead_request_duration_seconds" in body

    def test_unknown_route_is_404_with_error_body(self, running_server: str) -> None:
        status, body = _get(running_server, "/nope")
        assert status == 404
        assert "error" in json.loads(body)


class TestSystemOne:
    def test_happy_path(self, running_server: str) -> None:
        status, body = _post(running_server, "/v1/systemone", VALID_REQUEST)
        assert status == 200
        assert body["model"] == "test-model"
        assert 0.0 <= body["answers"]["q1"]["noul"] <= 1.0
        assert set(body["usage"].keys()) == {"input_tokens", "output_tokens"}

    def test_model_alias_is_a_passthrough(self, running_server: str) -> None:
        req = {**VALID_REQUEST, "model": "totally-made-up-alias"}
        status, body = _post(running_server, "/v1/systemone", req)
        assert status == 200
        assert body["model"] == "test-model"

    def test_empty_questions_is_400(self, running_server: str) -> None:
        req = {"state": "x", "model": "m", "questions": {}}
        status, body = _post(running_server, "/v1/systemone", req)
        assert status == 400
        assert body["error"]["code"] == "invalid_request"

    def test_missing_state_is_400(self, running_server: str) -> None:
        req = {"model": "m", "questions": VALID_REQUEST["questions"]}
        status, body = _post(running_server, "/v1/systemone", req)
        assert status == 400

    def test_non_json_content_type_is_415(self, running_server: str) -> None:
        status, body = _post(running_server, "/v1/systemone", VALID_REQUEST, content_type="text/plain")
        assert status == 415
        assert body["error"]["code"] == "unsupported_media_type"

    def test_malformed_json_body_is_400(self, running_server: str) -> None:
        req = urllib.request.Request(
            running_server + "/v1/systemone",
            data=b"{not valid json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 400

    def test_wrong_route_is_404(self, running_server: str) -> None:
        status, _body = _post(running_server, "/v1/nope", VALID_REQUEST)
        assert status == 404

    def test_no_stack_trace_leaks_on_error(self, monkeypatch: pytest.MonkeyPatch, running_server: str) -> None:
        import decisionhead.server as server_mod

        def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("internal detail that must not leak verbatim as a traceback")

        monkeypatch.setattr(server_mod, "answer_questions", _boom)
        status, body = _post(running_server, "/v1/systemone", VALID_REQUEST)
        assert status == 500
        assert "Traceback" not in json.dumps(body)
        assert body["error"]["code"] == "internal_error"
        # The raised text and type must not reach the client at all
        # (2026-10-06 review, findings 10 and 14).
        assert "internal detail" not in json.dumps(body)
        assert "RuntimeError" not in json.dumps(body)


class TestLowConfidenceField:
    def test_response_carries_low_confidence_false_when_disabled(self, running_server: str) -> None:
        status, body = _post(running_server, "/v1/systemone", VALID_REQUEST)
        assert status == 200
        assert body["low_confidence"] is False

    def test_band_flags_a_probability_inside_it(self, dry_config: ServerConfig) -> None:
        # dry_seed=1's noul answer for VALID_REQUEST's question is
        # deterministic; a band spanning the whole [0, 1] range always
        # catches it, without depending on the exact fake probability.
        cfg = replace(dry_config, abstain_band_lo=0.0, abstain_band_hi=1.0)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 200
            assert body["low_confidence"] is True
            assert body["answers"]["q1"]["low_confidence"] is True
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_handoff_mode_recommends_a_handoff(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, abstain_band_lo=0.0, abstain_band_hi=1.0, abstain_mode="handoff")
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 200
            assert body["answers"]["q1"]["recommended_action"] == "handoff"
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestResponseCacheOverHttp:
    def test_second_identical_request_is_a_cache_hit(self, dry_config: ServerConfig, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cfg = replace(dry_config, audit_path=str(audit_path))
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status1, body1 = _post(base, "/v1/systemone", VALID_REQUEST)
            status2, body2 = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status1 == status2 == 200
            assert body1["answers"] == body2["answers"]
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert rows[0]["cache_hit"] is False
        assert rows[1]["cache_hit"] is True
        # request_hash is content-derived, so both rows share it; the
        # cache hit must point at the first request's own request_id,
        # which has its own audit row (2026-10-06 review, finding 8).
        assert rows[1]["original_trace_id"] == rows[0]["request_id"]
        assert rows[0]["request_id"] != rows[1]["request_id"]

    def test_cache_disabled_never_hits(self, dry_config: ServerConfig, tmp_path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        cfg = replace(dry_config, audit_path=str(audit_path), cache_enabled=False)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            _post(base, "/v1/systemone", VALID_REQUEST)
            _post(base, "/v1/systemone", VALID_REQUEST)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert all(row["cache_hit"] is False for row in rows)


class TestMaxConcurrencyStartup:
    def test_default_max_concurrency_is_one(self, dry_config: ServerConfig) -> None:
        assert dry_config.max_concurrency == 1

    def test_server_starts_and_serves_with_a_non_default_max_concurrency(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency=4)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_zero_max_concurrency_is_rejected_at_config_resolution(self) -> None:
        from decisionhead.config import ConfigError, resolve_config

        with pytest.raises(ConfigError):
            resolve_config({"max_concurrency": 0}, env={})


class TestResolveEffectiveMaxConcurrencyOverride:
    def test_none_when_not_requested(self, dry_config: ServerConfig) -> None:
        assert resolve_effective_max_concurrency_override(dry_config, {}) is None

    def test_none_when_override_not_allowed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency_override_allowed=False)
        assert resolve_effective_max_concurrency_override(cfg, {"max_concurrency": 8}) is None

    def test_ignored_silently_even_if_malformed_when_not_allowed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency_override_allowed=False)
        assert resolve_effective_max_concurrency_override(cfg, {"max_concurrency": "nonsense"}) is None

    def test_value_used_when_allowed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency_override_allowed=True)
        assert resolve_effective_max_concurrency_override(cfg, {"max_concurrency": 8}) == 8

    def test_rejects_non_positive_when_allowed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency_override_allowed=True)
        with pytest.raises(QuestionError):
            resolve_effective_max_concurrency_override(cfg, {"max_concurrency": 0})

    def test_rejects_non_int_when_allowed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrency_override_allowed=True)
        with pytest.raises(QuestionError):
            resolve_effective_max_concurrency_override(cfg, {"max_concurrency": "eight"})


class TestRequestSizeLimit:
    def test_oversized_body_is_413(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, max_request_bytes=100)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            big_req = {**VALID_REQUEST, "state": "x" * 10_000}
            status, body = _post(base, "/v1/systemone", big_req)
            assert status == 413
            assert body["error"]["code"] == "payload_too_large"
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestMaxQuestionsPerRequest:
    def test_too_many_questions_is_400(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, max_questions_per_request=2)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = {
                "state": "s",
                "model": "m",
                "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(5)},
            }
            status, body = _post(base, "/v1/systemone", req)
            assert status == 400
            assert "limit" in body["error"]["message"]
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestConcurrencyLimit:
    def test_second_concurrent_request_is_429(self) -> None:
        cfg = ServerConfig(model="m", port=0, max_concurrent_requests=1, request_timeout_s=5, max_workers=1)
        backend = SlowBackend(delay=0.4)
        httpd, thread, base, metrics = _start_custom_server(cfg, backend)
        try:
            results: list[int] = []
            lock = threading.Lock()

            def fire() -> None:
                status, _body = _post(base, "/v1/systemone", VALID_REQUEST)
                with lock:
                    results.append(status)

            t1 = threading.Thread(target=fire)
            t2 = threading.Thread(target=fire)
            t1.start()
            time.sleep(0.05)  # let the first request acquire the semaphore
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert 200 in results
            assert 429 in results
            assert metrics.concurrent_requests_rejected >= 1
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestRequestTimeout:
    def test_slow_backend_triggers_504(self) -> None:
        cfg = ServerConfig(model="m", port=0, request_timeout_s=0.1, max_workers=1)
        backend = SlowBackend(delay=1.0)
        httpd, thread, base, _metrics = _start_custom_server(cfg, backend)
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 504
            assert body["error"]["code"] == "request_timeout"
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestBearerAuth:
    def test_missing_token_rejected_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DH_TEST_BEARER", "FAKEKEY-correct-token")
        cfg = ServerConfig(model="m", port=0, dry=True, bearer_token_env="DH_TEST_BEARER")
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 401
            assert body["error"]["code"] == "unauthorized"
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_correct_token_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DH_TEST_BEARER", "FAKEKEY-correct-token")
        cfg = ServerConfig(model="m", port=0, dry=True, bearer_token_env="DH_TEST_BEARER")
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(
                base + "/v1/systemone",
                data=json.dumps(VALID_REQUEST).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer FAKEKEY-correct-token"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_no_bearer_env_configured_means_no_auth_required(self, running_server: str) -> None:
        status, _body = _post(running_server, "/v1/systemone", VALID_REQUEST)
        assert status == 200


class TestCORS:
    def test_off_by_default(self, running_server: str) -> None:
        req = urllib.request.Request(running_server + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert "Access-Control-Allow-Origin" not in resp.headers

    def test_enabled_adds_headers(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, cors_enabled=True)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            status, body = _post(base, "/v1/systemone", VALID_REQUEST)
            assert status == 200
            req = urllib.request.Request(base + "/v1/systemone", method="OPTIONS")
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 204
                assert resp.headers["Access-Control-Allow-Origin"] == "*"
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestBindGuard:
    def test_refuses_non_loopback_without_flag(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, host="0.0.0.0")
        with pytest.raises(ValueError, match="non-loopback"):
            check_bind_allowed(cfg)

    def test_allows_non_loopback_with_explicit_flag_and_a_token(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, host="0.0.0.0", allow_non_local_bind=True)
        check_bind_allowed(cfg, bearer_token="FAKEKEY-test-token")  # must not raise

    def test_loopback_host_never_needs_the_flag(self) -> None:
        cfg = ServerConfig(model="m", port=0, dry=True, host="127.0.0.1")
        check_bind_allowed(cfg)  # must not raise

    def test_requires_backend_config_unless_dry(self) -> None:
        cfg = ServerConfig(port=0, dry=False, base_url="", model="")
        with pytest.raises(ValueError, match="dry"):
            run_server(cfg)


class TestThreadSafety:
    def test_many_concurrent_requests_all_succeed(self, dry_config: ServerConfig) -> None:
        cfg = replace(dry_config, max_concurrent_requests=32, max_workers=4)
        httpd = run_server(cfg)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            statuses: list[int] = []
            lock = threading.Lock()

            def fire(i: int) -> None:
                req = {**VALID_REQUEST, "state": f"state number {i}"}
                status, _body = _post(base, "/v1/systemone", req)
                with lock:
                    statuses.append(status)

            threads = [threading.Thread(target=fire, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            assert statuses.count(200) == 20
        finally:
            httpd.shutdown()
            httpd.server_close()
