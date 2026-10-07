"""Jev-shaped HTTP server in front of a real logprob backend.

    decisionhead serve --backend ollama --base http://203.0.113.10:11434 \\
        --model qwen3.8:27b --port 8787 [--calibration calibration.json] [--audit audit.jsonl]

Endpoints:
    POST /v1/systemone   the Jev-shaped request/response (see docs/http-api.md)
    GET  /health         liveness: {"status": "ok"}
    GET  /ready          readiness: backend reachability, cached briefly
    GET  /v1/models      {"models": [{"id": <real backend model>, "aliases": [...]}]}
    GET  /metrics        Prometheus text-format counters and latency histograms
    GET  /openapi.json   the OpenAPI document for this API

Design choice -- the model field is advisory, not a route key: a request's
"model" may be "jev-latest" or any other alias. This server never rejects
it and never routes on it; it always answers with the one real backend
model id it was started with, in the response's "model" field. That is
what lets a client written against the hosted Jev API point at this
server with zero code changes beyond the base URL.

Hardening (see ``docs/design.md`` for the reasoning behind each choice):
request size limit, a per-request wall-clock timeout layered on top of
each backend call's own timeout (the "per-question" timeout), a
per-connection socket read timeout that bounds how long a stalled or
slow-drip connection can hold a server thread (slowloris defense), a
depth guard on the raw request body applied before it is ever parsed as
JSON, a cap on questions per request, a cap on concurrent requests (429
beyond it), strict JSON-only bodies with ``{"error": {"code", "message"}}``
on every 4xx/5xx, a bounded label set for ``/metrics`` so an unauthenticated
caller cannot grow the metrics state without bound, structured JSON
request logs to stderr, an optional bearer token read only from an
environment variable, CORS left off unless explicitly enabled, and a
refusal to bind a non-loopback address without an explicit opt-in flag.

Added after the 2026-10-06 independent review
(``docs/security-review-2026-10-06.md``): a timed-out request keeps its
concurrency slot until its worker thread has actually finished, and its
queued backend work is abandoned at the deadline; a Host-header
allowlist (DNS rebinding) and an exact ``application/json`` media-type
check (cross-origin "simple" requests); strict JSON in and out (no
``NaN``/``Infinity``); one audit row per request outcome, failures and
timeouts included, each with a timestamp, the request id, the mode and
the returned answer; fixed client-facing messages for 500/502; and a
bearer token required for any non-loopback bind.

Invariants a change here must keep (each has a regression test):

* every response to ``POST /v1/systemone`` carries ``X-Request-Id``, and
  that id is on every audit row the request produced;
* a request is answered only after its audit rows are written; a failed
  audit write is a ``500``, never an unaudited answer;
* a 4xx/5xx body is ``{"error": {"code", "message"}}`` with no exception
  text, stack trace, backend URL or backend body;
* only a request that ended in ``200`` writes to the response cache;
* a request's concurrency slot is released exactly once, by whichever
  of the handler or its worker thread finishes last;
* every implemented route checks the ``Host`` header before any other
  processing (an unknown POST path is a plain ``404``).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import hmac
import json
import os
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import URLError

from . import backends, schema
from .cache import ResponseCache
from .config import ALLOWED_MODES, ServerConfig, resolve_config
from .core import QuestionError, answer_questions, apply_abstain_band, validate_calibration
from .deadline import Deadline
from .deliberate import answer_questions_deliberate
from .embeddings import EmbeddingsBackend, make_embeddings_backend
from .learned import LearnedBackendAdapter, LearnedModel, answer_questions_learned
from .logging import JSONLogger, get_logger
from .logging import configure as configure_logging
from .safejson import dumps_strict, loads_strict

JSONDict = dict[str, Any]

LATENCY_BUCKETS_S: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# The small, fixed set of routes this server actually implements, used to
# bound the "path" label recorded in /metrics -- see _metrics_path_label
# and Metrics.record_request. Anything else is recorded under "other" so
# an unauthenticated caller cannot grow the in-memory metrics state (or
# the size of the /metrics response) without bound simply by requesting
# many distinct nonexistent paths.
KNOWN_GET_PATHS = frozenset({"/health", "/ready", "/v1/models", "/metrics", "/openapi.json"})
KNOWN_POST_PATHS = frozenset({"/v1/systemone"})


def _metrics_path_label(method: str, path: str) -> str:
    """Maps a request path to a bounded label for ``/metrics``: one of
    this server's actual routes, or ``"other"`` for anything else."""
    known = KNOWN_GET_PATHS if method == "GET" else KNOWN_POST_PATHS
    return path if path in known else "other"


def load_calibration(path: str | None) -> JSONDict | None:
    """Loads a ``calibration.json`` written by ``decisionhead calibrate``,
    or returns ``None`` if ``path`` is falsy (calibration is optional).
    Raises ``ValueError`` -- which stops the server at start-up -- for a
    file with ``NaN``/``Infinity`` literals or a temperature that is not
    a finite number > 0 (2026-10-06 review, findings 4 and 15)."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        loaded: Any = loads_strict(text)
        validate_calibration(loaded)
    except ValueError as exc:
        raise ValueError(f"calibration file {path!r} is not usable: {exc}") from None
    result: JSONDict = loaded
    return result


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLog:
    """Append-only jsonl audit sink, guarded by a lock (the server is
    threaded). One row per question for a successful request, one row per
    request for every other outcome (see ``docs/audit/README.md``)."""

    def __init__(self, path: str | None) -> None:
        """``path`` may be ``None``, in which case :meth:`write` is a
        no-op (audit logging is optional)."""
        self.path = path
        self._lock = threading.Lock()

    def write(self, rows: list[JSONDict]) -> None:
        """Appends each row in ``rows`` as one strict-JSON line (no bare
        ``NaN``/``Infinity``; a non-finite value becomes ``null`` and the
        row records ``non_finite_replaced``). The file is created
        owner-only (0600). Raises ``OSError`` on failure; the request
        handler turns that into a 500 rather than answering unaudited."""
        if not self.path:
            return
        lines = "".join(dumps_strict(row, default=str) + "\n" for row in rows)
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(lines)


class Metrics:
    """Prometheus-text-format counters and latency histograms, no external
    library. One instance per server process."""

    def __init__(self) -> None:
        """Starts every counter, gauge, and histogram at zero."""
        self._lock = threading.Lock()
        self._request_counts: dict[tuple[str, str, str], int] = {}
        self._bucket_hits: dict[str, list[int]] = {}
        self._latency_sum: dict[str, float] = {}
        self._latency_count: dict[str, int] = {}
        self.concurrent_requests = 0
        self.concurrent_requests_rejected = 0

    def record_request(self, method: str, path: str, status: int, duration_s: float) -> None:
        """Records one completed request's method/path/status count and
        its latency into the per-path histogram buckets."""
        with self._lock:
            key = (method, path, str(status))
            self._request_counts[key] = self._request_counts.get(key, 0) + 1
            hits = self._bucket_hits.setdefault(path, [0] * len(LATENCY_BUCKETS_S))
            for i, bucket in enumerate(LATENCY_BUCKETS_S):
                if duration_s <= bucket:
                    hits[i] += 1
            self._latency_sum[path] = self._latency_sum.get(path, 0.0) + duration_s
            self._latency_count[path] = self._latency_count.get(path, 0) + 1

    def record_rejected(self) -> None:
        """Increments the 429-rejection counter."""
        with self._lock:
            self.concurrent_requests_rejected += 1

    def inc_concurrent(self) -> None:
        """Increments the in-flight-request gauge."""
        with self._lock:
            self.concurrent_requests += 1

    def dec_concurrent(self) -> None:
        """Decrements the in-flight-request gauge."""
        with self._lock:
            self.concurrent_requests -= 1

    def render(self) -> str:
        """Renders every counter, gauge, and histogram in Prometheus text
        exposition format."""
        lines = [
            "# HELP decisionhead_requests_total Total HTTP requests handled.",
            "# TYPE decisionhead_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), count in sorted(self._request_counts.items()):
                lines.append(
                    f'decisionhead_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines.append("# HELP decisionhead_request_duration_seconds Request latency.")
            lines.append("# TYPE decisionhead_request_duration_seconds histogram")
            for path, hits in sorted(self._bucket_hits.items()):
                for i, bucket in enumerate(LATENCY_BUCKETS_S):
                    lines.append(
                        f'decisionhead_request_duration_seconds_bucket{{path="{path}",le="{bucket}"}} {hits[i]}'
                    )
                total = self._latency_count.get(path, 0)
                lines.append(
                    f'decisionhead_request_duration_seconds_bucket{{path="{path}",le="+Inf"}} {total}'
                )
                lines.append(
                    f'decisionhead_request_duration_seconds_sum{{path="{path}"}} {self._latency_sum.get(path, 0.0)}'
                )
                lines.append(f'decisionhead_request_duration_seconds_count{{path="{path}"}} {total}')
            lines.append("# HELP decisionhead_concurrent_requests_rejected_total Requests rejected with 429.")
            lines.append("# TYPE decisionhead_concurrent_requests_rejected_total counter")
            lines.append(
                f"decisionhead_concurrent_requests_rejected_total {self.concurrent_requests_rejected}"
            )
            lines.append("# HELP decisionhead_concurrent_requests In-flight requests to /v1/systemone.")
            lines.append("# TYPE decisionhead_concurrent_requests gauge")
            lines.append(f"decisionhead_concurrent_requests {self.concurrent_requests}")
        return "\n".join(lines) + "\n"


class ReadinessChecker:
    """Checks backend reachability lazily -- only when ``/ready`` is
    polled -- and caches the result for ``ttl_s`` seconds so a readiness
    probe hitting the endpoint every second does not itself generate load
    against the backend. A :class:`~decisionhead.backends.FakeBackend` is
    always ready (there is nothing to reach)."""

    def __init__(self, backend: backends.TokenDistributionBackend, ttl_s: float) -> None:
        """``ttl_s`` is how long a cached readiness result is reused
        before the next ``/ready`` poll triggers a fresh check. ``backend``
        is unwrapped first (see :func:`decisionhead.backends.unwrap_backend`)
        so a :class:`~decisionhead.backends.ConcurrencyLimitedBackend`
        still reports readiness for the real backend underneath it."""
        self._backend = backends.unwrap_backend(backend)
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._cached_at: float | None = None
        self._cached_ready = False
        self._cached_detail: str | None = None

    def _reachability_url(self) -> str | None:
        if isinstance(self._backend, backends.OllamaBackend):
            return self._backend.base_url + "/api/tags"
        if isinstance(self._backend, backends.OpenAIBackend):
            return self._backend.base_url + "/v1/models"
        return None

    def _probe(self) -> tuple[bool, str | None]:
        if isinstance(self._backend, backends.FakeBackend):
            return True, None
        url = self._reachability_url()
        if url is None:
            return True, None
        try:
            # The same no-redirect opener as every backend call (2026-10-06
            # review, finding 3). The detail is the exception type only:
            # /ready is unauthenticated and must not echo the backend URL.
            with backends.build_backend_opener().open(url, timeout=3) as resp:
                resp.read(1)
            return True, None
        except (URLError, OSError, ValueError, backends.BackendError) as exc:
            return False, f"backend unreachable ({type(exc).__name__})"

    def check(self) -> tuple[bool, str | None]:
        """Returns ``(ready, detail)``, either the cached result (if
        still within ``ttl_s``) or a fresh probe."""
        now = time.monotonic()
        with self._lock:
            if self._cached_at is not None and (now - self._cached_at) < self._ttl_s:
                return self._cached_ready, self._cached_detail
        ready, detail = self._probe()
        with self._lock:
            self._cached_at = now
            self._cached_ready = ready
            self._cached_detail = detail
        return ready, detail


def _error_body(code: str, message: str) -> bytes:
    return json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")


def _internal_error_body(request_id: str) -> bytes:
    """The fixed 500 body: no exception type or text, which used to be
    echoed verbatim (2026-10-06 review, finding 10). The detail is in
    the server log under the same ``request_id``."""
    return _error_body("internal_error", f"internal error; see the server log for request_id {request_id}")


def request_hash_for_audit(body: JSONDict) -> str:
    """A content hash of the request's state and questions for audit rows
    of requests that fail before any answer is computed (successful rows
    carry the answering mode's own ``request_hash``)."""
    payload = json.dumps(
        {"state": body.get("state"), "questions": body.get("questions")}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _media_type(content_type: str) -> str:
    """The media-type essence of a ``Content-Type`` value: everything
    before the first ``;``, stripped and lower-cased. ``text/plain;
    charset=application/json`` is ``text/plain`` -- which a browser
    sends cross-origin with no CORS preflight, so a substring test let
    any web page drive this service (2026-10-06 review, finding 2)."""
    return content_type.split(";", 1)[0].strip().lower()


def _host_name(host_header: str) -> str:
    """The hostname part of a ``Host`` header: ``[::1]:8787`` -> ``::1``,
    ``LocalHost.:8787`` -> ``localhost``."""
    value = host_header.strip()
    # Bracketed IPv6 literal, "host:port", or a bare name. A bare IPv6
    # address without brackets has several colons and is taken whole.
    # The trailing dot is dropped so "localhost." cannot bypass the
    # allowlist comparison.
    if value.startswith("["):
        end = value.find("]")
        name = value[1:end] if end > 0 else value
    elif value.count(":") == 1:
        name = value.split(":", 1)[0]
    else:
        name = value
    return name.rstrip(".").lower()


def allowed_host_names(cfg: ServerConfig) -> frozenset[str]:
    """The ``Host`` names this server answers to: the loopback names, the
    bind address, and ``cfg.allowed_hosts`` (comma-separated)."""
    # Hostnames only, never ports: a container published on another host
    # port (docker -p 9000:8787) still sends "Host: localhost:9000", which
    # must keep working. A wildcard bind adds the literal "0.0.0.0",
    # which reaches the same machine the loopback names do, so it admits
    # no attacker-chosen hostname.
    names = set(LOOPBACK_HOSTS)
    names.add(cfg.host.strip("[]").lower())
    names.update(h.strip().rstrip(".").lower() for h in cfg.allowed_hosts.split(",") if h.strip())
    return frozenset(names)


def _parse_json_body(raw: bytes) -> Any:
    """Parses a raw request body as JSON, once it has already passed the
    nesting-depth guard (`schema.json_nesting_too_deep`). May raise
    ``json.JSONDecodeError``, ``UnicodeDecodeError``, or
    ``RecursionError``; the caller catches all three and returns a clean
    400 ``invalid_json`` rather than letting a decoder failure escape as
    an unhandled exception.

    Split out as its own function, rather than inlined at the call site,
    so it can be replaced in isolation by a test: whether a given nesting
    depth actually makes ``json.loads`` raise ``RecursionError`` is an
    interpreter implementation detail, not a property of this codebase --
    verified to raise on CPython 3.11 for a body deep enough to defeat the
    depth guard above, but CPython 3.12 parses the identical body without
    error (its JSON decoder tolerates deeper nesting before hitting the
    interpreter's recursion limit). The depth guard is the real,
    version-independent defense; this except clause is a second-layer,
    best-effort catch-all for whatever a differently-tuned
    interpreter/build might still let through, and is exercised directly
    -- by forcing this exact failure -- in
    ``tests/test_security_hardening.py`` instead of depending on any one
    interpreter's actual recursion behaviour.

    Strict since the 2026-10-06 review (finding 4): ``NaN``,
    ``Infinity``, an overflowing float literal like ``1e999`` and any
    numeric literal over 64 characters raise ``ValueError``
    (:func:`decisionhead.safejson.loads_strict`), also a clean 400."""
    return loads_strict(raw.decode("utf-8")) if raw else {}


def resolve_effective_mode(cfg: ServerConfig, body: JSONDict) -> str:
    """The mode (see :data:`decisionhead.config.ALLOWED_MODES`) that
    answers this one request: the request's own top-level ``"mode"``
    field when present AND ``cfg.mode_override_allowed`` is ``True``,
    ``cfg.mode`` otherwise. An override to an unrecognized mode is a
    :class:`QuestionError` (turned into an ordinary 400 by the caller),
    same as any other malformed field -- silently ignoring a typo'd mode
    would be worse than failing loudly, per this codebase's convention
    (see e.g. ``config.load_toml_file``). On a ``mode = "learned"``
    server an override to ``"fast"`` or ``"deliberate"`` is also a
    :class:`QuestionError`: there is no model backend to answer it."""
    requested = body.get("mode")
    if requested is None or not cfg.mode_override_allowed:
        return cfg.mode
    if not isinstance(requested, str) or requested not in ALLOWED_MODES:
        raise QuestionError(f"'mode' must be one of {ALLOWED_MODES} (got {requested!r})")
    if cfg.mode == "learned" and requested != "learned":
        # A learned-mode server has no model backend behind it, so fast
        # and deliberate cannot be served; this used to surface as a 500
        # from LearnedBackendAdapter.
        raise QuestionError(
            f"this server runs mode='learned' and has no model backend; 'mode': {requested!r} "
            "cannot be served here (omit 'mode' or send 'learned')"
        )
    return requested


def resolve_effective_max_concurrency_override(cfg: ServerConfig, body: JSONDict) -> int | None:
    """The per-request backend concurrency override (fast mode only,
    see :class:`decisionhead.backends.ConcurrencyLimitedBackend`): the
    request's own top-level ``"max_concurrency"`` field when present AND
    ``cfg.max_concurrency_override_allowed`` is ``True``; ``None``
    otherwise, meaning "use the server's configured ``max_concurrency``"
    (the safe, deterministic default). Mirrors
    :func:`resolve_effective_mode`'s handling of a disallowed override:
    silently ignored (not even validated) rather than rejected, since the
    server is simply refusing to let this request change a setting it
    never had permission to touch."""
    requested = body.get("max_concurrency")
    if requested is None or not cfg.max_concurrency_override_allowed:
        return None
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
        raise QuestionError(f"'max_concurrency' must be a positive integer (got {requested!r})")
    return requested


def _backend_label(backend: backends.Backend) -> str:
    """The name used to label audit rows and logs for ``backend``: the
    wrapped backend's own class name when ``backend`` is a
    :class:`~decisionhead.backends.ConcurrencyLimitedBackend`, else
    ``type(backend).__name__`` as before."""
    wrapped = getattr(backend, "wrapped_type_name", None)
    return wrapped if isinstance(wrapped, str) else type(backend).__name__


def _calibration_version(calibration: JSONDict | None) -> str:
    """A stable, content-based version tag for a loaded calibration
    file, used as a cache-key input (see :mod:`decisionhead.cache`) so
    that recalibrating (writing a new calibration.json and restarting the
    server) never serves a stale cached probability computed under the
    old calibration: a different calibration is a different key. The
    empty string when no calibration is configured."""
    if not calibration:
        return ""
    return hashlib.sha256(json.dumps(calibration, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class _ConcurrencySlot:
    """One acquired ``max_concurrent_requests`` slot. Released exactly
    once, by whoever finishes last: the handler if the request never got
    as far as a worker thread, otherwise the worker thread itself when it
    actually ends. Before the 2026-10-06 review (finding 1) the handler
    released the slot on a 504 while the worker kept calling the backend,
    so 429 never fired however many orphans piled up."""

    def __init__(self, semaphore: threading.Semaphore, metrics: Metrics) -> None:
        """Wraps one already-acquired slot of ``semaphore``.
        ``handed_off`` is set by :func:`_run_with_timeout` once a worker
        thread owns the release."""
        self._semaphore = semaphore
        self._metrics = metrics
        self._lock = threading.Lock()
        self._released = False
        self.handed_off = False

    def release(self) -> None:
        """Returns the slot to the semaphore and decrements the
        in-flight gauge. Idempotent: the second and later calls do
        nothing, so the handler and the worker can both call it safely."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._metrics.dec_concurrent()
        self._semaphore.release()


def _run_with_timeout(
    fn: Callable[[], JSONDict],
    timeout_s: float,
    *,
    deadline: Deadline | None = None,
    slot: _ConcurrencySlot | None = None,
) -> JSONDict:
    """Runs ``fn`` in a worker thread and raises :class:`TimeoutError` if it
    has not finished within ``timeout_s``. Stdlib has no clean way to
    cancel a thread blocked in a socket call, so on a timeout the one
    backend call already in flight is allowed to finish; but ``deadline``
    is cancelled, so every backend call still queued behind it (later
    questions, later samples, waits on the backend semaphore) is
    abandoned instead of executed, and ``slot`` -- handed to the worker
    here -- stays held until the worker has actually ended, so the
    concurrency cap (and its 429) keeps counting it. See
    ``docs/design.md``."""
    box: dict[str, Any] = {}

    # The worker, not the caller, releases the slot. On a timeout the
    # caller returns 504 while the worker may still be inside one backend
    # call; holding the slot until that call ends is what lets 429 see
    # abandoned work.
    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's thread below
            box["error"] = exc
        finally:
            if slot is not None:
                slot.release()

    t = threading.Thread(target=target, daemon=True)
    try:
        t.start()
    except BaseException:
        # The worker never ran, so it will never release the slot: do it
        # here. Setting handed_off before start() used to leak the slot
        # on a thread-start failure (2026-10-06 confirmation, R1).
        if slot is not None:
            slot.release()
        raise
    if slot is not None:
        # Only now does the worker own the release. If it has already
        # finished and released, release() is idempotent.
        slot.handed_off = True
    t.join(timeout_s)
    if t.is_alive():
        if deadline is not None:
            deadline.cancel()
        raise TimeoutError(f"request exceeded the {timeout_s}s request timeout")
    if "error" in box:
        raise box["error"]
    result: JSONDict = box["result"]
    return result


def make_handler(
    backend: backends.Backend,
    calibration: JSONDict | None,
    audit: AuditLog,
    cfg: ServerConfig,
    metrics: Metrics,
    readiness: ReadinessChecker,
    logger: JSONLogger,
    bearer_token: str | None,
    learned_model: LearnedModel | None = None,
    embedder: EmbeddingsBackend | None = None,
    cache: ResponseCache | None = None,
    backend_route: str = "",
) -> type[BaseHTTPRequestHandler]:
    """Builds the ``BaseHTTPRequestHandler`` subclass this server uses,
    closing over the given backend, config, and shared state (audit log,
    metrics, readiness checker, logger, and resolved bearer token).
    ``learned_model``/``embedder`` are used only when ``cfg.mode ==
    "learned"`` (see :func:`run_server`); ``backend`` is then a
    :class:`~decisionhead.learned.LearnedBackendAdapter`, used only for
    ``/health``/``/v1/models``/readiness, never for answering a
    request. ``cache``/``backend_route`` are fast-mode-only inputs to the
    state-hash memoisation cache key (see :mod:`decisionhead.cache`);
    ``cache`` is ``None`` when ``cfg.cache_enabled`` is ``False``."""
    concurrency_semaphore = threading.Semaphore(cfg.max_concurrent_requests)
    calibration_version = _calibration_version(calibration)
    host_names = allowed_host_names(cfg)

    class Handler(BaseHTTPRequestHandler):
        server_version = "decisionhead/0.1.0"
        sys_version = ""  # no Python version in the Server header
        _request_id: str | None = None

        def log_message(self, fmt: str, *fmt_args: Any) -> None:  # quiet the default stdout access log
            logger.debug("http_access", client=self.address_string(), message=fmt % fmt_args)

        def _send_json(self, status: int, payload: bytes, extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if cfg.cors_enabled:
                self.send_header("Access-Control-Allow-Origin", "*")
            if self._request_id is not None:
                self.send_header("X-Request-Id", self._request_id)
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _check_host(self) -> bool:
            """Refuses a request whose ``Host`` is not one this server
            answers to: 400 if absent, 421 if foreign. A DNS-rebinding
            page reaches this server under the attacker's own hostname,
            so this is what stops it from reading responses (2026-10-06
            review, finding 2). Returns ``True`` if the request may
            proceed."""
            host = self.headers.get("Host")
            if not host:
                self._send_json(400, _error_body("invalid_request", "a Host header is required"))
                return False
            if _host_name(host) not in host_names:
                self._send_json(
                    421, _error_body("misdirected_request", "this server does not answer to that Host name")
                )
                return False
            return True

        def _send_text(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _check_auth(self) -> int | None:
            """``None`` if the request may proceed, else the status sent."""
            if not bearer_token:
                return None
            header = self.headers.get("Authorization", "")
            if not header.isascii():
                # hmac.compare_digest raises TypeError on non-ASCII str,
                # which used to drop the connection with a raw traceback
                # (2026-10-06 review, finding 11).
                self._send_json(400, _error_body("invalid_request", "Authorization header must be ASCII"))
                return 400
            expected = ("Bearer " + bearer_token).encode("utf-8")
            if not hmac.compare_digest(header.encode("ascii"), expected):
                self._send_json(401, _error_body("unauthorized", "missing or invalid bearer token"))
                return 401
            return None

        def do_OPTIONS(self) -> None:
            if not self._check_host():
                return
            if cfg.cors_enabled:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send_json(404, _error_body("not_found", f"no such route: {self.path}"))

        def do_GET(self) -> None:
            t0 = time.time()
            status = 200
            try:
                if not self._check_host():
                    status = 421
                    return
                if self.path == "/health":
                    health_body: JSONDict = {"status": "ok"}
                    self._send_json(200, json.dumps(health_body).encode("utf-8"))
                    return
                if self.path == "/ready":
                    ready, detail = readiness.check()
                    ready_body: JSONDict = {"status": "ok", "ready": ready, "detail": detail}
                    self._send_json(200, json.dumps(ready_body).encode("utf-8"))
                    return
                if self.path == "/v1/models":
                    models_body: JSONDict = {
                        "models": [{"id": getattr(backend, "model", None), "aliases": ["jev-latest"]}]
                    }
                    self._send_json(200, json.dumps(models_body).encode("utf-8"))
                    return
                if self.path == "/metrics":
                    self._send_text(200, metrics.render().encode("utf-8"), "text/plain; version=0.0.4")
                    return
                if self.path == "/openapi.json":
                    doc = schema.build_openapi_document(version="0.1.0", server_url=f"http://{cfg.host}:{cfg.port}")
                    self._send_json(200, json.dumps(doc).encode("utf-8"))
                    return
                status = 404
                self._send_json(404, _error_body("not_found", f"no such route: {self.path}"))
            finally:
                metrics.record_request("GET", _metrics_path_label("GET", self.path), status, time.time() - t0)

        def do_POST(self) -> None:
            request_id = uuid.uuid4().hex
            self._request_id = request_id
            t0 = time.time()
            logger.info("request_start", request_id=request_id, method="POST", path=self.path)
            ctx: JSONDict = {}
            try:
                status = self._do_post_inner(request_id, ctx)
            except OSError:
                # A socket error or read timeout on this connection (the
                # slowloris defence): handle_one_request closes the
                # connection cleanly; there is no one left to answer.
                raise
            except Exception as exc:  # noqa: BLE001 -- last resort: a JSON 500, never a dropped connection
                logger.error(
                    "unhandled_exception",
                    request_id=request_id,
                    exc_type=type(exc).__name__,
                    stack=traceback.format_exc(),
                )
                status = 500
                ctx["error"] = "internal_error"
                try:
                    self._send_json(500, _internal_error_body(request_id))
                except OSError:
                    pass
            duration_s = time.time() - t0
            # Only requests that got past authentication write a row: a
            # rejection decided before (or by) the auth check costs an
            # unauthenticated client nothing, so auditing it would let
            # anyone grow audit_path at HTTP speed (2026-10-06
            # confirmation, R2). Those rejections are still counted in
            # /metrics by status.
            if self.path == "/v1/systemone" and status != 200 and ctx.get("authenticated"):
                self._audit_outcome(request_id, status, ctx)
            metrics.record_request("POST", _metrics_path_label("POST", self.path), status, duration_s)
            logger.info(
                "request_end",
                request_id=request_id,
                method="POST",
                path=self.path,
                status=status,
                duration_ms=round(duration_s * 1000, 2),
            )

        def _audit_outcome(self, request_id: str, status: int, ctx: JSONDict) -> None:
            """One audit row for a request that passed authentication
            but did not end in 200 (2026-10-06 review, finding 8): what
            was refused or failed, when, and under which request id.
            Pre-authentication rejections (400 for a missing Host or a
            non-ASCII Authorization header, 401, 403, 421) never get a
            row (confirmation R2)."""
            row: JSONDict = {
                "ts": _utc_now_iso(),
                "request_id": request_id,
                "status": status,
                "error": ctx.get("error"),
                "mode": ctx.get("mode"),
                "request_hash": ctx.get("request_hash"),
                "n_questions": ctx.get("n_questions"),
            }
            try:
                audit.write([row])
            except (OSError, ValueError) as exc:
                logger.error("audit_write_failed", request_id=request_id, exc_type=type(exc).__name__)

        def _fail(
            self, ctx: JSONDict, status: int, code: str, message: str, headers: dict[str, str] | None = None
        ) -> int:
            ctx["error"] = code
            self._send_json(status, _error_body(code, message), extra_headers=headers)
            return status

        def _do_post_inner(self, request_id: str, ctx: JSONDict) -> int:
            if self.path != "/v1/systemone":
                self._send_json(404, _error_body("not_found", f"no such route: {self.path}"))
                return 404

            if not self._check_host():
                ctx["error"] = "misdirected_request"
                return 421 if self.headers.get("Host") else 400

            if self.headers.get("Origin") is not None and not cfg.cors_enabled:
                # Browsers send Origin on every cross-origin POST; this
                # service has no browser front end, so with CORS off any
                # such request is a page on some other site driving it.
                return self._fail(ctx, 403, "forbidden_origin", "cross-origin requests are not accepted")

            auth_status = self._check_auth()
            if auth_status is not None:
                ctx["error"] = "unauthorized" if auth_status == 401 else "invalid_request"
                return auth_status
            # From here on every outcome is audited (see do_POST).
            ctx["authenticated"] = True

            if _media_type(self.headers.get("Content-Type", "")) != "application/json":
                return self._fail(ctx, 415, "unsupported_media_type", "Content-Type must be application/json")

            length_header = self.headers.get("Content-Length")
            try:
                # int() on attacker-controlled header text is itself safe:
                # CPython (3.11+) refuses to convert a digit string longer
                # than sys.get_int_max_str_digits() and raises ValueError,
                # which the except clause below already turns into a 400.
                length = int(length_header) if length_header else 0
            except ValueError:
                return self._fail(ctx, 400, "invalid_request", "Content-Length is not a valid integer")
            if length < 0:
                return self._fail(ctx, 400, "invalid_request", "Content-Length must not be negative")
            if length > cfg.max_request_bytes:
                return self._fail(
                    ctx,
                    413,
                    "payload_too_large",
                    f"request body of {length} bytes exceeds the {cfg.max_request_bytes}-byte limit",
                )

            raw = self.rfile.read(length) if length else b""
            if schema.json_nesting_too_deep(raw, cfg.max_json_nesting_depth):
                return self._fail(
                    ctx,
                    400,
                    "invalid_json",
                    f"request body exceeds the maximum JSON nesting depth of {cfg.max_json_nesting_depth}",
                )
            try:
                body = _parse_json_body(raw)
            except (ValueError, UnicodeDecodeError, RecursionError):
                # ValueError covers json.JSONDecodeError and the strict
                # parser's non-finite/over-long number refusal.
                # RecursionError is caught defensively in addition to the
                # depth guard above (which runs first and should reject
                # any body deep enough to reach this): json.loads() has no
                # built-in depth limit and, without this, a sufficiently
                # nested body raises RecursionError straight out of
                # this handler, which previously reached socketserver's
                # own error handler: an unstructured traceback to stderr
                # and the connection dropped with no HTTP response at all.
                return self._fail(ctx, 400, "invalid_json", "request body is not valid JSON")

            if not concurrency_semaphore.acquire(blocking=False):
                metrics.record_rejected()
                return self._fail(
                    ctx,
                    429,
                    "too_many_requests",
                    "too many concurrent requests; try again shortly",
                    headers={"Retry-After": "1"},
                )

            metrics.inc_concurrent()
            slot = _ConcurrencySlot(concurrency_semaphore, metrics)
            deadline = Deadline(cfg.request_timeout_s)
            try:
                schema.validate_envelope(body, max_questions=cfg.max_questions_per_request)
                ctx["n_questions"] = len(body["questions"])
                effective_mode = resolve_effective_mode(cfg, body)
                ctx["mode"] = effective_mode
                ctx["request_hash"] = request_hash_for_audit(body)
                if effective_mode == "learned":
                    if learned_model is None:
                        raise QuestionError(
                            "this server was not started with mode='learned' (no model loaded); "
                            "restart it with --mode learned --head-model-path"
                        )
                    active_model: LearnedModel = learned_model
                    result = _run_with_timeout(
                        lambda: answer_questions_learned(
                            body["state"],
                            body["questions"],
                            active_model,
                            embedder=embedder,
                            max_workers=cfg.max_workers,
                            deadline=deadline,
                        ),
                        cfg.request_timeout_s,
                        deadline=deadline,
                        slot=slot,
                    )
                elif effective_mode == "deliberate":
                    result = _run_with_timeout(
                        lambda: answer_questions_deliberate(
                            body["state"],
                            body["questions"],
                            backend,
                            k=cfg.deliberate_k,
                            temperature=cfg.deliberate_temperature,
                            max_tokens=cfg.deliberate_max_tokens,
                            max_workers=cfg.max_workers,
                            backend_name=_backend_label(backend),
                            claim_guard_enabled=cfg.claim_guard_enabled,
                            claim_guard_threshold=cfg.claim_guard_threshold,
                            deadline=deadline,
                        ),
                        cfg.request_timeout_s,
                        deadline=deadline,
                        slot=slot,
                    )
                else:
                    max_concurrency_override = resolve_effective_max_concurrency_override(cfg, body)
                    result = _run_with_timeout(
                        lambda: answer_questions(
                            body["state"],
                            body["questions"],
                            backend,
                            calibration=calibration,
                            max_workers=cfg.max_workers,
                            backend_name=_backend_label(backend),
                            cache=cache,
                            backend_route=backend_route,
                            mode=effective_mode,
                            calibration_version=calibration_version,
                            max_concurrency_override=max_concurrency_override,
                            deadline=deadline,
                            defer_cache_writes=True,
                        ),
                        cfg.request_timeout_s,
                        deadline=deadline,
                        slot=slot,
                    )
            except QuestionError as exc:
                return self._fail(ctx, 400, "invalid_request", str(exc))
            except TimeoutError:
                return self._fail(
                    ctx, 504, "request_timeout", f"request exceeded the {cfg.request_timeout_s}s request timeout"
                )
            except backends.BackendError as exc:
                logger.error(
                    "backend_error", request_id=request_id, detail=backends.redact_url(str(exc))[:500]
                )
                return self._fail(
                    ctx, 502, "backend_error", "the model backend failed or returned an unusable response"
                )
            except Exception as exc:  # noqa: BLE001 -- last-resort 500, never a stack trace to the client
                logger.error(
                    "unhandled_exception",
                    request_id=request_id,
                    exc_type=type(exc).__name__,
                    stack=traceback.format_exc(),
                )
                ctx["error"] = "internal_error"
                self._send_json(500, _internal_error_body(request_id))
                return 500
            finally:
                if not slot.handed_off:
                    slot.release()

            # The request succeeded: only now may its answers enter the
            # cache (deep copies, so the abstain band's flags below never
            # leak into a later cache hit). original_trace_id is this
            # request's id, which has its own audit rows.
            for key, value in result.get("cache_writes", []):
                entry = copy.deepcopy(value)
                entry["original_trace_id"] = request_id
                if cache is not None:
                    cache.set(key, entry)

            apply_abstain_band(result, cfg.abstain_band, cfg.abstain_mode)
            ts = _utc_now_iso()
            answers = result["answers"]
            for row in result["audit"]:
                row["ts"] = ts
                row["request_id"] = request_id
                row["status"] = 200
                row["mode"] = effective_mode
                if effective_mode == "fast":
                    row["calibration_version"] = calibration_version
                row["answer"] = answers.get(row.get("question_id"))
            try:
                audit.write(result["audit"])
            except (OSError, ValueError) as exc:
                # Fail closed: an answer that could not be audited is not
                # returned (2026-10-06 review, finding 9).
                logger.error("audit_write_failed", request_id=request_id, exc_type=type(exc).__name__)
                ctx["error"] = "audit_write_failed"
                self._send_json(500, _internal_error_body(request_id))
                return 500
            response = {
                "model": result["model"],
                "answers": answers,
                "usage": result["usage"],
                "mode": effective_mode,
                "low_confidence": result["low_confidence"],
                "degraded": result.get("degraded", False),
            }
            self._send_json(200, dumps_strict(response).encode("utf-8"))
            return 200

    # Bounds how long the server will wait for the NEXT byte of a request
    # line, header, or body before dropping the connection --
    # socketserver.StreamRequestHandler.setup() calls
    # self.connection.settimeout(self.timeout) whenever this attribute is
    # not None, and http.server.BaseHTTPRequestHandler.handle_one_request
    # already catches the resulting TimeoutError and closes the
    # connection cleanly (no response, no traceback). Without this, a
    # peer that sends bytes slower than this timeout -- or never finishes
    # a request at all -- holds one server thread open indefinitely: the
    # slowloris pattern. This closes before authentication is checked,
    # which is exactly the gap the attack exploits (the header read
    # happens first). See docs/design.md.
    Handler.timeout = cfg.connection_read_timeout_s

    return Handler


class _DecisionHeadHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with a larger accept-queue backlog.

    ``socketserver.BaseServer.request_queue_size`` defaults to 5. Under
    realistic burst concurrency (a few dozen near-simultaneous client
    connections), the OS accept backlog can fill before any handler
    thread gets a chance to run, so the kernel resets excess connections
    (``ECONNRESET`` on the client side) even though every request would
    have succeeded once accepted. Raising the backlog here fixes requests
    being dropped at the TCP layer before they ever reach this service's
    own concurrency limiting (``cfg.max_concurrent_requests`` /
    ``concurrency_semaphore``)."""

    request_queue_size = 128


def check_bind_allowed(cfg: ServerConfig, bearer_token: str | None = None) -> None:
    """Raises :class:`ValueError` if ``cfg.bearer_token_env`` is set but
    resolves to no token (on any bind address), if ``cfg`` would bind a
    non-loopback address without the explicit opt-in flag, or -- since the 2026-10-06
    review (finding 7) -- without a bearer token, unless
    ``cfg.insecure_no_auth`` is set. ``bearer_token`` is the resolved
    token value (not the variable name). Split out from
    :func:`run_server` so this policy can be tested without actually
    opening a socket."""
    if cfg.bearer_token_env and not bearer_token:
        # On every bind address, loopback included: an operator who asked
        # for bearer auth must never get an unauthenticated server
        # because the variable is missing. The variable's NAME is
        # deliberately not echoed (2026-10-06 review, finding 13).
        raise ValueError(
            "--bearer-token-env names an environment variable that is unset or empty; refusing to start "
            "without the bearer token it asks for (set the variable, or drop --bearer-token-env; "
            "--insecure-no-auth is the only way to serve a non-loopback bind without a token)"
        )
    if cfg.host in LOOPBACK_HOSTS:
        return
    if not cfg.allow_non_local_bind:
        raise ValueError(
            f"refusing to bind non-loopback host {cfg.host!r} without --allow-non-local-bind "
            "(binding to anything other than 127.0.0.1/localhost exposes this service on the network)"
        )
    if not bearer_token and not cfg.insecure_no_auth:
        raise ValueError(
            f"refusing to bind non-loopback host {cfg.host!r} without a bearer token: set "
            "--bearer-token-env to the NAME of a non-empty environment variable holding the token "
            "(or pass --insecure-no-auth to accept an unauthenticated network-facing service)"
        )


def run_server(cfg: ServerConfig) -> ThreadingHTTPServer:
    """Builds and binds (but does not start serving) a
    ``ThreadingHTTPServer`` for ``cfg``: constructs the backend (or, in
    ``mode = "learned"``, loads the trained model file instead), loads
    any calibration file, and wires up the audit log, metrics, readiness
    checker, and bearer token. Raises :class:`ValueError` if ``cfg`` is
    not usable (see :func:`dry_or_configured` and
    :func:`check_bind_allowed`)."""
    logger = configure_logging(cfg.log_level)
    bearer_token = backends.env_ref(cfg.bearer_token_env)
    check_bind_allowed(cfg, bearer_token)
    if cfg.host not in LOOPBACK_HOSTS and not bearer_token:
        logger.warning(
            "insecure_no_auth",
            host=cfg.host,
            detail="binding a non-loopback host with no bearer token: anyone who can reach the port "
            "can use the decision service and the model behind it",
        )
    # Per-call socket timeout. A float throughout: int() used to truncate
    # 0.5 to 0, which makes a socket non-blocking (finding 12).
    call_timeout_s = min(float(cfg.backend_timeout_s), float(cfg.question_timeout_s))

    learned_model: LearnedModel | None = None
    embedder: EmbeddingsBackend | None = None
    calibration: JSONDict | None = None

    if cfg.mode == "learned":
        if not cfg.head_model_path:
            raise ValueError("mode 'learned' requires --head-model-path (or DECISIONHEAD_HEAD_MODEL_PATH)")
        learned_model = LearnedModel.load(cfg.head_model_path)
        if learned_model.feature_config.use_embedding:
            if not cfg.embedding_backend:
                raise ValueError(
                    "the loaded model was trained with embedding features on; "
                    "--embedding-backend/--embedding-base/--embedding-model are required to serve it"
                )
            embedder = make_embeddings_backend(
                cfg.embedding_backend,
                cfg.embedding_base_url,
                cfg.embedding_model,
                api_key_env=cfg.embedding_api_key_env,
                timeout=call_timeout_s,
            )
            warning = backends.insecure_transport_warning(
                cfg.embedding_base_url, has_credential=bool(backends.env_ref(cfg.embedding_api_key_env))
            )
            if warning:
                logger.warning("insecure_backend_transport", which="embeddings", detail=warning)
        backend: backends.Backend = LearnedBackendAdapter(learned_model)
    else:
        if not dry_or_configured(cfg):
            raise ValueError("--base and --model are required unless --dry is given")
        backend = backends.make_backend(
            cfg.backend,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key_env=cfg.api_key_env,
            timeout=call_timeout_s,
            dry=cfg.dry,
            dry_seed=cfg.dry_seed,
            max_concurrency=cfg.max_concurrency,
            # A backend call never waits for a semaphore slot longer than
            # the whole request may take (2026-10-06 review, finding 1).
            acquire_timeout_s=cfg.request_timeout_s,
        )
        if not cfg.dry:
            warning = backends.insecure_transport_warning(
                cfg.base_url, has_credential=bool(backends.env_ref(cfg.api_key_env))
            )
            if warning:
                logger.warning("insecure_backend_transport", which="model", detail=warning)
        calibration = load_calibration(cfg.calibration_path)
        if calibration is not None:
            logger.info("calibration_loaded", calibration_version=_calibration_version(calibration))
        logger.info(
            "backend_concurrency_configured",
            max_concurrency=cfg.max_concurrency,
            override_allowed=cfg.max_concurrency_override_allowed,
        )

    cache = (
        ResponseCache(max_entries=cfg.cache_max_entries, ttl_s=cfg.cache_ttl_s, path=cfg.cache_path)
        if cfg.cache_enabled
        else None
    )
    backend_route = "dry" if cfg.dry else f"{cfg.backend}://{cfg.base_url}"

    audit = AuditLog(cfg.audit_path)
    metrics = Metrics()
    readiness = ReadinessChecker(backend, cfg.ready_cache_s)
    handler = make_handler(
        backend,
        calibration,
        audit,
        cfg,
        metrics,
        readiness,
        logger,
        bearer_token,
        learned_model=learned_model,
        embedder=embedder,
        cache=cache,
        backend_route=backend_route,
    )
    httpd = _DecisionHeadHTTPServer((cfg.host, cfg.port), handler)
    return httpd


def dry_or_configured(cfg: ServerConfig) -> bool:
    """True if ``cfg`` is either ``--dry`` or has both a base URL and a
    model configured for a real backend. Only meaningful outside
    ``mode = "learned"``, which needs neither."""
    return cfg.dry or bool(cfg.base_url and cfg.model)


def serve_forever_with_graceful_shutdown(httpd: ThreadingHTTPServer, logger: JSONLogger) -> None:
    """Runs ``httpd.serve_forever()`` on a background thread and blocks the
    calling thread on a shutdown event set by SIGTERM/SIGINT, then shuts
    the server down cleanly. ``ThreadingHTTPServer.shutdown()`` is
    documented as safe to call from a different thread than the one
    running ``serve_forever()``, which is exactly this arrangement."""
    stop_event = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        logger.info("signal_received", signal=signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    thread = threading.Thread(target=httpd.serve_forever, name="decisionhead-serve", daemon=True)
    thread.start()
    stop_event.wait()
    logger.info("shutting_down")
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the ``decisionhead serve``
    subcommand. Every flag defaults to ``None`` (rather than a
    ``ServerConfig`` default) so :func:`config_from_args` can tell "not
    passed on the CLI" apart from "explicitly passed the default value",
    which is what makes the env/TOML fallback in
    :func:`decisionhead.config.resolve_config` work."""
    parser = argparse.ArgumentParser(prog="decisionhead serve", description="Run the decisionhead HTTP server")
    parser.add_argument("--backend", choices=["ollama", "openai"], default=None)
    parser.add_argument("--base", dest="base_url", default=None, help="backend base URL, e.g. http://203.0.113.10:11434")
    parser.add_argument("--model", default=None, help="real backend model id, e.g. qwen3.8:27b")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--allow-non-local-bind", action="store_true", default=None)
    parser.add_argument("--calibration", dest="calibration_path", default=None, help="path to a calibration.json")
    parser.add_argument("--audit", dest="audit_path", default=None, help="path to an audit.jsonl to append to")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--api-key-env", default=None, help="env var name holding an API key for --backend openai")
    parser.add_argument("--bearer-token-env", default=None, help="env var name holding this server's own bearer token")
    parser.add_argument("--max-request-bytes", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--question-timeout-s", type=float, default=None)
    parser.add_argument("--max-questions-per-request", type=int, default=None)
    parser.add_argument("--max-concurrent-requests", type=int, default=None)
    parser.add_argument(
        "--connection-read-timeout-s",
        type=float,
        default=None,
        help="per-connection socket read timeout in seconds (slowloris defense)",
    )
    parser.add_argument(
        "--max-json-nesting-depth",
        type=int,
        default=None,
        help="reject a request body nested deeper than this before parsing it",
    )
    parser.add_argument("--cors", dest="cors_enabled", action="store_true", default=None)
    parser.add_argument(
        "--allowed-hosts",
        default=None,
        help="comma-separated extra Host names to answer to (loopback names and --host are always allowed)",
    )
    parser.add_argument(
        "--insecure-no-auth",
        action="store_true",
        default=None,
        help="allow a non-loopback bind with no bearer token (not recommended)",
    )
    parser.add_argument(
        "--backend-timeout-s",
        type=float,
        default=None,
        help="socket timeout per backend HTTP call; the effective value is min(this, --question-timeout-s)",
    )
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--config", dest="config_path", default=None, help="path to a decisionhead TOML config file")
    parser.add_argument(
        "--dry", action="store_true", default=None, help="use a deterministic fake backend; no network calls"
    )
    parser.add_argument("--dry-seed", type=int, default=None)
    parser.add_argument(
        "--mode", choices=list(ALLOWED_MODES), default=None, help="decision head mode (default: fast)"
    )
    parser.add_argument(
        "--no-mode-override",
        dest="mode_override_allowed",
        action="store_false",
        default=None,
        help="refuse a per-request 'mode' override; every request uses --mode regardless",
    )
    parser.add_argument(
        "--deliberate-k", type=int, default=None, help="deliberate mode: sampled answers per question (default: 8)"
    )
    parser.add_argument(
        "--deliberate-temperature",
        type=float,
        default=None,
        help="deliberate mode: sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--deliberate-max-tokens",
        type=int,
        default=None,
        help="deliberate mode: max tokens per sampled answer (default: 256)",
    )
    parser.add_argument(
        "--no-claim-guard",
        dest="claim_guard_enabled",
        action="store_false",
        default=None,
        help="deliberate mode: disable the terminal-claim guard (see docs/deliberate.md)",
    )
    parser.add_argument(
        "--claim-guard-threshold",
        type=float,
        default=None,
        help="deliberate mode: minimum vote share to accept a terminal claim (default: 0.75)",
    )
    parser.add_argument(
        "--head-model-path",
        default=None,
        help="learned mode: path to a model.json written by `python -m decisionhead.train --out ...`",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["ollama", "openai"],
        default=None,
        help="learned mode, only if the model was trained with embedding features on",
    )
    parser.add_argument("--embedding-base", dest="embedding_base_url", default=None, help="embeddings backend base URL")
    parser.add_argument("--embedding-model", default=None, help="embeddings model id")
    parser.add_argument("--embedding-api-key-env", default=None, help="env var name holding an embeddings API key")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="serialize (or bound) backend HTTP calls behind a semaphore of this size (default: 1 -- "
        "some endpoints batch concurrent requests and return different logprobs for identical inputs)",
    )
    parser.add_argument(
        "--allow-max-concurrency-override",
        dest="max_concurrency_override_allowed",
        action="store_true",
        default=None,
        help="allow a request's own 'max_concurrency' field (fast mode only) to override --max-concurrency",
    )
    parser.add_argument(
        "--no-cache", dest="cache_enabled", action="store_false", default=None, help="disable the response cache"
    )
    parser.add_argument(
        "--cache-ttl-s", type=float, default=None, help="response cache entry lifetime in seconds (default: 86400)"
    )
    parser.add_argument(
        "--cache-max-entries", type=int, default=None, help="response cache max in-memory entries (default: 10000)"
    )
    parser.add_argument(
        "--cache-path", default=None, help="append-only JSONL file to persist the response cache across restarts"
    )
    parser.add_argument(
        "--abstain-band-lo",
        type=float,
        default=None,
        help="low-confidence band lower bound: a noul answer whose probability falls inside "
        "[--abstain-band-lo, --abstain-band-hi] is flagged low_confidence (disabled unless both are set)",
    )
    parser.add_argument("--abstain-band-hi", type=float, default=None, help="low-confidence band upper bound")
    parser.add_argument(
        "--abstain-mode",
        choices=["off", "handoff"],
        default=None,
        help="'off' (default) just flags low_confidence; 'handoff' also recommends a handoff",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    """Turns a parsed ``argparse.Namespace`` from :func:`build_arg_parser`
    into a resolved :class:`~decisionhead.config.ServerConfig`."""
    overrides = {k: v for k, v in vars(args).items() if k != "config_path"}
    return resolve_config(overrides, config_path=getattr(args, "config_path", None))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``decisionhead serve`` subcommand: resolves
    config, builds and starts the server, and blocks until a graceful
    shutdown."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a clean CLI error, not a traceback
        parser.error(str(exc))
        return 2

    try:
        httpd = run_server(cfg)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    logger = get_logger()
    port = httpd.server_address[1]
    print(
        f"decisionhead listening on http://{cfg.host}:{port} "
        f"(backend={cfg.backend if not cfg.dry else 'dry-fake'}, model={cfg.model or 'dry-fake-model'}, "
        f"mode={cfg.mode}, max_concurrency={cfg.max_concurrency}, cache_enabled={cfg.cache_enabled})"
    )
    logger.info("server_started", host=cfg.host, port=port, backend=cfg.backend, dry=cfg.dry)
    try:
        serve_forever_with_graceful_shutdown(httpd, logger)
    except KeyboardInterrupt:
        httpd.shutdown()
        httpd.server_close()
    return 0


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed.
    The equivalent, more thorough assertions -- including the hardening
    behaviour (size limits, timeouts, concurrency caps, auth, CORS) --
    live in ``tests/test_server.py``."""
    import urllib.error
    import urllib.request

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    cfg = ServerConfig(model="dry-model", port=0, dry=True, dry_seed=3, max_workers=2)
    httpd = run_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as resp:  # noqa: S310
            health = json.loads(resp.read())
        check("GET /health returns 200 with status ok", health.get("status") == "ok")

        with urllib.request.urlopen(base + "/ready", timeout=5) as resp:  # noqa: S310
            ready = json.loads(resp.read())
        check("GET /ready reports ready for a dry (fake) backend", ready.get("ready") is True)

        with urllib.request.urlopen(base + "/v1/models", timeout=5) as resp:  # noqa: S310
            models = json.loads(resp.read())
        check("GET /v1/models lists the real backend model id", models["models"][0]["id"] == "dry-model")
        check("GET /v1/models advertises the jev-latest alias", "jev-latest" in models["models"][0]["aliases"])

        with urllib.request.urlopen(base + "/openapi.json", timeout=5) as resp:  # noqa: S310
            openapi_doc = json.loads(resp.read())
        check("GET /openapi.json documents /v1/systemone", "/v1/systemone" in openapi_doc.get("paths", {}))

        req = {
            "state": "Supplier Acme Corp submitted a certificate of insurance dated last year.",
            "model": "jev-latest",
            "questions": {
                "q1": {"type": "noul", "instructions": "The certificate on file is current."},
                "q2": {
                    "type": "choice",
                    "instructions": "What should happen next?",
                    "criteria": {"CONTINUE": "keep reasoning", "REQUEST": "ask supplier for a new one"},
                },
            },
        }
        body = json.dumps(req).encode("utf-8")
        r = urllib.request.Request(
            base + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(r, timeout=5) as resp:  # noqa: S310
            sysone = json.loads(resp.read())
        check("POST /v1/systemone returns 200 with real model id, not the alias sent", sysone["model"] == "dry-model")
        check("POST /v1/systemone answers both questions", set(sysone["answers"].keys()) == {"q1", "q2"})
        check("POST /v1/systemone noul answer has a probability", 0.0 <= sysone["answers"]["q1"]["noul"] <= 1.0)
        check(
            "POST /v1/systemone choice answer has confidence as top-minus-second",
            "confidence" in sysone["answers"]["q2"],
        )
        check(
            "POST /v1/systemone reports usage keys",
            {"input_tokens", "output_tokens"} == set(sysone["usage"].keys()),
        )

        with urllib.request.urlopen(base + "/metrics", timeout=5) as resp:  # noqa: S310
            metrics_text = resp.read().decode("utf-8")
        check("GET /metrics reports the request it just served", "decisionhead_requests_total" in metrics_text)

        bad = json.dumps({"state": "x", "model": "m", "questions": {}}).encode("utf-8")
        r = urllib.request.Request(
            base + "/v1/systemone", data=bad, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(r, timeout=5)  # noqa: S310
            check("empty questions rejected with 4xx", False)
        except urllib.error.HTTPError as exc:
            error_body = json.loads(exc.read())
            check("empty questions rejected with 4xx", exc.code == 400 and "error" in error_body)

        bad2 = json.dumps({"model": "m", "questions": {"q": {"type": "noul", "instructions": "x"}}}).encode("utf-8")
        r = urllib.request.Request(
            base + "/v1/systemone", data=bad2, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(r, timeout=5)  # noqa: S310
            check("missing state rejected with 4xx", False)
        except urllib.error.HTTPError as exc:
            check("missing state rejected with 4xx", exc.code == 400)

        r = urllib.request.Request(base + "/v1/systemone", data=body, method="POST")  # no Content-Type header
        try:
            urllib.request.urlopen(r, timeout=5)  # noqa: S310
            check("non-JSON content type rejected with 415", False)
        except urllib.error.HTTPError as exc:
            check("non-JSON content type rejected with 415", exc.code == 415)

        try:
            urllib.request.urlopen(base + "/nope", timeout=5)  # noqa: S310
            check("unknown route returns 404", False)
        except urllib.error.HTTPError as exc:
            check("unknown route returns 404", exc.code == 404)

        req2 = dict(req)
        req2["model"] = "totally-made-up-alias"
        body2 = json.dumps(req2).encode("utf-8")
        r = urllib.request.Request(
            base + "/v1/systemone", data=body2, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(r, timeout=5) as resp:  # noqa: S310
            sysone2 = json.loads(resp.read())
        check(
            "any model alias in the request is accepted and answered with the real id",
            sysone2["model"] == "dry-model",
        )

    finally:
        httpd.shutdown()
        httpd.server_close()

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(main())
