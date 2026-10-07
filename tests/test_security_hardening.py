"""Regression tests for the independent security review dated 2026-09-23
(`docs/security-review-2026-09-23.md`). Each test class below reproduces
the reviewer's own repro method for one finding and asserts the fix.
"""

from __future__ import annotations

import json
import pathlib
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from decisionhead.config import ServerConfig
from decisionhead.schema import json_nesting_too_deep
from decisionhead.server import run_server

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _get(base: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _post_bytes(base: str, path: str, raw: bytes, content_type: str = "application/json") -> tuple[int, dict]:
    req = urllib.request.Request(base + path, data=raw, headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _raw_http_request(host: str, port: int, request_bytes: bytes, timeout: float = 5.0) -> bytes:
    """Sends exactly ``request_bytes`` over a fresh TCP connection and
    reads whatever the server sends back (or nothing, on EOF/timeout).
    Used where urllib's own request construction would not let us send a
    deliberately malformed header (e.g. a negative Content-Length)."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(request_bytes)
        sock.settimeout(timeout)
        chunks = []
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except (TimeoutError, OSError):
            pass
        return b"".join(chunks)


def _start_server(**overrides: Any) -> tuple[ThreadingHTTPServer, threading.Thread, str, str, int]:
    cfg = ServerConfig(model="m", port=0, dry=True, max_workers=1, **overrides)
    httpd = run_server(cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    return httpd, thread, f"http://{host}:{port}", host, port


class TestFinding1SlowlorisDefense:
    """Finding 1 (HIGH): no timeout on a stalled connection let a peer
    hold a server thread open indefinitely by sending bytes slowly (or
    never finishing a request). Reproduces the reviewer's exact repro:
    open a raw socket, send a partial request, never send the
    terminating blank line."""

    def test_stalled_connection_is_dropped_within_timeout_while_another_client_is_served(self) -> None:
        httpd, _thread, base, host, port = _start_server(connection_read_timeout_s=0.3)
        try:
            sock = socket.create_connection((host, port), timeout=5)
            try:
                # Valid start line and one header; deliberately never send
                # the terminating blank line, exactly as the review did.
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n")

                # While the stalled connection is still open, a completely
                # separate client must be served immediately -- one stuck
                # thread must not affect anyone else.
                status, body = _get(base, "/health")
                assert status == 200
                assert json.loads(body)["status"] == "ok"

                # The stalled connection itself must be dropped at
                # (approximately) the configured timeout, not held open
                # for as long as the peer cares to keep it.
                sock.settimeout(5)
                t0 = time.time()
                data = sock.recv(1024)
                elapsed = time.time() - t0
            finally:
                sock.close()

            assert data == b""  # connection closed, no response ever sent
            assert elapsed < 3.0  # bounded by the 0.3s timeout, not "25+ seconds"
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_slow_body_delivery_is_also_bounded(self) -> None:
        """The second half of the reviewer's repro: valid headers with a
        Content-Length, then the body trickled in far more slowly than
        the connection read timeout, never completed."""
        httpd, _thread, base, host, port = _start_server(connection_read_timeout_s=0.3)
        try:
            sock = socket.create_connection((host, port), timeout=5)
            try:
                sock.sendall(
                    b"POST /v1/systemone HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: 100\r\n"
                    b"\r\n"
                    b"{"  # one byte of a 100-byte body, then silence
                )
                sock.settimeout(5)
                t0 = time.time()
                data = sock.recv(1024)
                elapsed = time.time() - t0
            finally:
                sock.close()

            assert data == b""
            assert elapsed < 3.0

            status, _ = _get(base, "/health")
            assert status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestFinding2MalformedInputHardening:
    """Finding 2 (MEDIUM): deeply nested JSON and a negative
    Content-Length each crashed the connection with no HTTP response and
    an unstructured traceback to stderr, instead of the documented clean
    4xx. Reproduces both of the reviewer's verified repros."""

    def test_deeply_nested_json_returns_400_not_a_crash(self) -> None:
        httpd, _thread, base, _host, _port = _start_server()
        try:
            nesting = 5000
            raw = b'{"a":' * nesting + b"1" + b"}" * nesting
            status, body = _post_bytes(base, "/v1/systemone", raw)
            assert status == 400
            assert body["error"]["code"] == "invalid_json"

            # the connection/server must still be fully usable afterward
            status2, _ = _get(base, "/health")
            assert status2 == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_recursion_error_from_the_json_decoder_is_caught_as_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt-and-suspenders: proves the except clause wrapped around
        `_parse_json_body` also returns a clean 400 `invalid_json` if a
        `RecursionError` ever escapes the JSON decoder itself, as it does
        on CPython 3.11 for a body deep enough to defeat the depth guard
        above.

        This forces that exact failure directly (monkeypatching
        `decisionhead.server._parse_json_body`) instead of trying to
        reproduce it by constructing a deeply nested body and hoping the
        interpreter's own recursion limit kicks in: whether a given
        nesting depth actually raises `RecursionError` from inside
        `json.loads` is an interpreter implementation detail, not a
        property of this codebase -- verified to raise on CPython 3.11 for
        a 5000-level-deep body, but CPython 3.12 parses the identical body
        successfully (its JSON decoder tolerates deeper nesting before
        hitting the recursion limit), which made a version of this test
        built around that body flaky across Python versions. The
        assertion below -- both the depth guard (previous test) and this
        fallback map to `invalid_json`, never `invalid_request` -- is the
        one behaviour this project commits to on every supported Python
        version: `invalid_json` is the truthful code, since the body is
        JSON the server refuses to parse."""
        import decisionhead.server as server_mod

        def _boom(raw: bytes) -> Any:
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(server_mod, "_parse_json_body", _boom)

        httpd, _thread, base, _host, _port = _start_server()
        try:
            status, body = _post_bytes(base, "/v1/systemone", b'{"state": "x", "model": "m", "questions": {}}')
            assert status == 400
            assert body["error"]["code"] == "invalid_json"

            status2, _ = _get(base, "/health")
            assert status2 == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_negative_content_length_returns_400_not_a_crash(self) -> None:
        httpd, _thread, base, host, port = _start_server()
        try:
            request = (
                b"POST /v1/systemone HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: -5\r\n"
                b"\r\n"
            )
            response = _raw_http_request(host, port, request)
            assert b" 400 " in response
            assert b'"invalid_request"' in response
            assert b"Traceback" not in response

            status, _ = _get(base, "/health")
            assert status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_absurdly_long_content_length_digit_string_returns_400(self) -> None:
        """CPython 3.11+ refuses to convert an excessively long digit
        string to int (sys.get_int_max_str_digits()); this must surface
        as the documented 400, not an uncaught ValueError."""
        httpd, _thread, base, host, port = _start_server()
        try:
            huge_digits = b"9" * 10_000
            request = (
                b"POST /v1/systemone HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + huge_digits + b"\r\n"
                b"\r\n"
            )
            response = _raw_http_request(host, port, request)
            assert b" 400 " in response
            assert b"Traceback" not in response

            status, _ = _get(base, "/health")
            assert status == 200
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestJsonNestingTooDeep:
    """Direct unit coverage of the depth-guard scan itself."""

    def test_shallow_object_is_fine(self) -> None:
        assert json_nesting_too_deep(b'{"a": {"b": 1}}', max_depth=32) is False

    def test_exactly_at_limit_is_fine(self) -> None:
        raw = b"[" * 32 + b"1" + b"]" * 32
        assert json_nesting_too_deep(raw, max_depth=32) is False

    def test_one_past_limit_is_rejected(self) -> None:
        raw = b"[" * 33 + b"1" + b"]" * 33
        assert json_nesting_too_deep(raw, max_depth=32) is True

    def test_brackets_inside_a_string_do_not_count(self) -> None:
        raw = b'{"a": "' + b"[" * 100 + b'"}'
        assert json_nesting_too_deep(raw, max_depth=32) is False

    def test_escaped_quote_does_not_end_the_string_early(self) -> None:
        # A literal `\"` inside the string must not be read as the
        # string's closing quote, or the brackets after it would be
        # miscounted as structural.
        raw = b'{"a": "esc\\"ape' + b"[" * 100 + b'"}'
        assert json_nesting_too_deep(raw, max_depth=32) is False


class TestFinding3MetricsLabelBounding:
    """Finding 3 (LOW): /metrics recorded any requested path verbatim as
    a label value, letting an unauthenticated caller grow the metrics
    state without bound by requesting many distinct nonexistent paths."""

    def test_unknown_paths_collapse_to_a_single_other_label(self) -> None:
        httpd, _thread, base, _host, _port = _start_server()
        try:
            for path in ("/etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "//etc/passwd", "/one/two/three"):
                status, _ = _get(base, path)
                assert status == 404

            status, metrics_text = _get(base, "/metrics")
            assert status == 200
            assert "/etc/passwd" not in metrics_text
            assert "one/two/three" not in metrics_text
            assert 'path="other"' in metrics_text
            # all four requests must have collapsed into ONE counter
            # series, not four -- exactly the bound this fix provides.
            other_lines = [
                line
                for line in metrics_text.splitlines()
                if line.startswith("decisionhead_requests_total") and 'path="other"' in line
            ]
            assert len(other_lines) == 1
            assert re.search(r"\} 4$", other_lines[0])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_known_routes_keep_their_own_label(self) -> None:
        httpd, _thread, base, _host, _port = _start_server()
        try:
            _get(base, "/health")
            status, metrics_text = _get(base, "/metrics")
            assert status == 200
            assert 'path="/health"' in metrics_text
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestFinding5NoRealLookingPrivateAddress:
    """Finding 5 (LOW): a real-looking private IP (a private-range address) was
    used everywhere an example backend address was needed, reading as a
    real address rather than an arbitrary example. Replaced with an
    RFC 5737 documentation address."""

    @pytest.mark.parametrize(
        "path",
        ["README.md", "docs/calibration.md", "docs/api.md", "docs/design.md", "decisionhead/server.py"],
    )
    def test_no_rfc1918_private_address_in_docs_or_code(self, path: str) -> None:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "192.168." not in text

    def test_documentation_address_is_used_instead(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "203.0.113.10" in readme


class TestFinding6ActionsPinnedBySha:
    """Finding 6 (INFORMATIONAL): GitHub Actions were pinned by a
    major-version tag rather than a full commit SHA. Every `uses:` line
    referencing a third-party action must now pin a 40-character commit
    SHA."""

    @pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
    def test_every_action_reference_is_pinned_to_a_commit_sha(self, workflow: str) -> None:
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        uses_lines = [line for line in text.splitlines() if "uses:" in line]
        assert uses_lines, f"expected at least one 'uses:' line in {workflow}"
        for line in uses_lines:
            ref = line.split("uses:", 1)[1].strip().split()[0]
            _action, _, pinned = ref.partition("@")
            assert re.fullmatch(r"[0-9a-f]{40}", pinned), f"{workflow}: {ref!r} is not pinned to a commit SHA"
