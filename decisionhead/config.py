"""Configuration for the ``decisionhead serve`` command.

Three sources, in this precedence order (highest wins):

1. **Command-line flags** passed to ``decisionhead serve``.
2. **Environment variables**, all prefixed ``DECISIONHEAD_`` (see
   :data:`ENV_VAR_MAP` for the exact names).
3. **A TOML file** (stdlib ``tomllib``), pointed to by ``--config`` or the
   ``DECISIONHEAD_CONFIG`` environment variable; see
   ``decisionhead.example.toml`` for every recognized key.
4. **Built-in defaults**, in :data:`DEFAULTS` below.

A value missing from a higher-precedence source falls through to the next
one; nothing here merges partial objects beyond that per-field fallback.

Two settings are deliberately *not* configurable via the TOML file or a
CLI flag, only via an environment variable, because a secret must never
land in a config file or a process's argv (visible in `ps`):

* the API key for an OpenAI-compatible backend (``api_key_env`` names the
  *environment variable* to read it from; the CLI/TOML value is that name,
  not the key itself);
* the bearer token for this server's own API (``bearer_token_env`` works
  the same way).

Invariants enforced by :func:`resolve_config` (so a bad value stops the
server at start-up, never mid-request): ``mode`` and ``abstain_mode``
are from their allowed sets; both abstain-band bounds are set together
with ``lo <= hi``; every ``*_env`` value looks like a variable name and
is never echoed when it does not; ``base_url`` and ``embedding_base_url``
are ``http``/``https`` with a host; integer limits are ``>= 1`` and
timeouts are finite and ``> 0``. The full option reference is
``docs/configuration.md``.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

JSONDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised for a malformed config file or an out-of-range value."""


# "abstain_mode" values: "off" (default) leaves a low-confidence answer's
# recommended action untouched; "handoff" additionally turns it into a
# handoff with the reason recorded (see decisionhead.core.apply_abstain_band).
ALLOWED_ABSTAIN_MODES: tuple[str, ...] = ("off", "handoff")

# The decision head's answering strategy. "fast" (default) is the
# original single-token-logprob read (decisionhead.core.answer_questions);
# "deliberate" is the k-sampled reasoning-then-typed-JSON vote
# (decisionhead.deliberate.answer_questions_deliberate; see the README
# and docs/deliberate.md for the cost tradeoff); "learned" answers from a
# model file trained offline on labelled traces
# (decisionhead.train/decisionhead.learned; see docs/training.md) instead
# of calling any LLM backend per request. This tuple is the single place
# a new mode gets added.
ALLOWED_MODES: tuple[str, ...] = ("fast", "deliberate", "learned")


@dataclass(frozen=True)
class ServerConfig:
    """The fully-resolved configuration for ``decisionhead serve``, after
    merging CLI flags, environment variables, and a TOML file per the
    precedence documented in this module's docstring. Every field here
    has a matching entry in :data:`ENV_VAR_MAP` and, except for the two
    secret-holding fields called out above, a matching key in
    ``decisionhead.example.toml``."""

    backend: str = "ollama"
    base_url: str = ""
    model: str = ""
    host: str = "127.0.0.1"
    port: int = 8787
    allow_non_local_bind: bool = False
    calibration_path: str | None = None
    audit_path: str | None = None
    max_workers: int = 4
    api_key_env: str | None = None
    # Socket timeout for one backend HTTP call (connect, and each read).
    # The effective value is min(backend_timeout_s, question_timeout_s);
    # see decisionhead.server.run_server.
    backend_timeout_s: float = 60.0
    dry: bool = False
    dry_seed: int = 0

    # "learned" mode only: path to a model.json written by
    # `python -m decisionhead.train --out ...`, and the embeddings
    # backend settings needed only if that model's feature config has
    # embedding features turned on (off by default -- see
    # decisionhead.features.FeatureConfig).
    head_model_path: str | None = None
    embedding_backend: str | None = None
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key_env: str | None = None

    # decision head mode (see ALLOWED_MODES above)
    mode: str = "fast"
    # Whether a request's own top-level "mode" field may override this
    # server's configured default for that one request. False pins every
    # request to `mode` above regardless of what the request asks for.
    mode_override_allowed: bool = True
    # "deliberate" mode only: k sampled reasoning-then-JSON answers voted
    # over per question, instead of one single-token logprob read. Cost
    # scales with k -- see README.md and docs/deliberate.md.
    deliberate_k: int = 8
    deliberate_temperature: float = 0.7
    deliberate_max_tokens: int = 256
    # "deliberate" mode only: guards a terminal "claim of done/complete"
    # answer -- see decisionhead.deliberate.apply_claim_guard.
    claim_guard_enabled: bool = True
    claim_guard_threshold: float = 0.75

    # hardening / limits
    max_request_bytes: int = 1 * 1024 * 1024
    request_timeout_s: float = 30.0
    question_timeout_s: float = 15.0
    max_questions_per_request: int = 50
    max_concurrent_requests: int = 16
    ready_cache_s: float = 5.0
    # Per-connection socket read timeout (BaseHTTPRequestHandler.timeout):
    # bounds how long the server will wait for the NEXT byte of a request
    # line, header, or body before dropping the connection. Without this,
    # a peer that sends bytes slowly (or not at all) holds one server
    # thread open indefinitely -- the slowloris pattern -- and can do so
    # before authentication is ever checked, since the header read itself
    # happens first. See docs/design.md.
    connection_read_timeout_s: float = 30.0
    # A cheap linear-scan bracket-depth guard applied to the raw request
    # body before it is ever handed to json.loads(): a body nested deeper
    # than this is rejected with a 400 instead of being parsed, so a
    # maliciously (or accidentally) deep body can never hit Python's
    # interpreter recursion limit inside the JSON decoder. See
    # decisionhead.schema.json_nesting_too_deep and docs/design.md.
    max_json_nesting_depth: int = 32

    # auth / CORS
    bearer_token_env: str | None = None
    cors_enabled: bool = False
    # Binding a non-loopback host requires a bearer token unless this is
    # set (2026-10-06 security review, finding 7). Logged as a WARNING.
    insecure_no_auth: bool = False
    # Extra Host-header names accepted besides the loopback names and the
    # bind address (comma-separated, e.g. "decisionhead,dh.internal").
    # Any other Host is refused with 421, which is what stops a DNS-
    # rebinding page from reading responses (2026-10-06 review, finding 2).
    allowed_hosts: str = ""

    # logging
    log_level: str = "INFO"

    # backend.max_concurrency -- serializes (or bounds) every backend HTTP
    # call behind a semaphore of this size. Finding: a real DeepSeek exl3
    # endpoint returns different logprobs for otherwise-identical requests
    # when the server batches concurrent requests together; at
    # concurrency 1 (the default) they are identical. This is deliberately
    # separate from max_concurrent_requests above, which bounds how many
    # client requests this server accepts at once -- this bounds how many
    # of those requests' backend calls may be in flight against the
    # endpoint at once. See decisionhead.backends.ConcurrencyLimitedBackend.
    max_concurrency: int = 1
    # Whether a request's own "max_concurrency" field (fast mode only) may
    # raise/lower the backend concurrency for that one request. False
    # (default) means every request is served at the server's configured
    # max_concurrency regardless of what it asks for -- an explicit opt-in
    # is required to trade determinism for throughput per request.
    max_concurrency_override_allowed: bool = False

    # cache.* -- an in-process, optionally file-backed cache of full
    # per-question answers (fast mode only -- see decisionhead.cache and
    # docs/design.md). Keyed on (model, backend route, canonical state
    # JSON, question id + text, mode, calibration version), so the same
    # question about the same state is never re-sent to the backend while
    # a cache entry for it is live.
    cache_enabled: bool = True
    cache_ttl_s: float = 86400.0
    cache_max_entries: int = 10000
    # None (default): in-process only, lost on restart. Set to a file
    # path to also persist entries across restarts (append-only JSONL).
    cache_path: str | None = None

    # abstain.band / abstain.mode -- a low-confidence band applied to
    # every decisive (noul) answer's probability. Disabled by default
    # (both bounds None). When set, e.g. [0.4, 0.6], any noul answer whose
    # probability falls inside [lo, hi] gets low_confidence: true on the
    # answer, the audit row, and the response envelope; abstain_mode
    # "handoff" additionally turns that answer's recommended action into
    # a handoff with the reason recorded. Pure post-processing -- see
    # decisionhead.core.apply_abstain_band.
    abstain_band_lo: float | None = None
    abstain_band_hi: float | None = None
    abstain_mode: str = "off"

    @property
    def abstain_band(self) -> tuple[float, float] | None:
        """``(lo, hi)`` if both bounds are configured, else ``None``
        (disabled) -- the shape :func:`decisionhead.core.apply_abstain_band`
        expects."""
        if self.abstain_band_lo is None or self.abstain_band_hi is None:
            return None
        return (self.abstain_band_lo, self.abstain_band_hi)


DEFAULTS = ServerConfig()

# Maps each ServerConfig field name to the environment variable that can
# set it, and a small type-coercion function applied to the raw string.
ENV_VAR_MAP: dict[str, tuple[str, type]] = {
    "backend": ("DECISIONHEAD_BACKEND", str),
    "base_url": ("DECISIONHEAD_BASE_URL", str),
    "model": ("DECISIONHEAD_MODEL", str),
    "host": ("DECISIONHEAD_HOST", str),
    "port": ("DECISIONHEAD_PORT", int),
    "allow_non_local_bind": ("DECISIONHEAD_ALLOW_NON_LOCAL_BIND", bool),
    "calibration_path": ("DECISIONHEAD_CALIBRATION", str),
    "audit_path": ("DECISIONHEAD_AUDIT_LOG", str),
    "max_workers": ("DECISIONHEAD_MAX_WORKERS", int),
    "api_key_env": ("DECISIONHEAD_API_KEY_ENV", str),
    "backend_timeout_s": ("DECISIONHEAD_BACKEND_TIMEOUT_S", float),
    "dry": ("DECISIONHEAD_DRY", bool),
    "dry_seed": ("DECISIONHEAD_DRY_SEED", int),
    "head_model_path": ("DECISIONHEAD_HEAD_MODEL_PATH", str),
    "embedding_backend": ("DECISIONHEAD_EMBEDDING_BACKEND", str),
    "embedding_base_url": ("DECISIONHEAD_EMBEDDING_BASE_URL", str),
    "embedding_model": ("DECISIONHEAD_EMBEDDING_MODEL", str),
    "embedding_api_key_env": ("DECISIONHEAD_EMBEDDING_API_KEY_ENV", str),
    "mode": ("DECISIONHEAD_MODE", str),
    "mode_override_allowed": ("DECISIONHEAD_MODE_OVERRIDE_ALLOWED", bool),
    "deliberate_k": ("DECISIONHEAD_DELIBERATE_K", int),
    "deliberate_temperature": ("DECISIONHEAD_DELIBERATE_TEMPERATURE", float),
    "deliberate_max_tokens": ("DECISIONHEAD_DELIBERATE_MAX_TOKENS", int),
    "claim_guard_enabled": ("DECISIONHEAD_CLAIM_GUARD_ENABLED", bool),
    "claim_guard_threshold": ("DECISIONHEAD_CLAIM_GUARD_THRESHOLD", float),
    "max_request_bytes": ("DECISIONHEAD_MAX_REQUEST_BYTES", int),
    "request_timeout_s": ("DECISIONHEAD_REQUEST_TIMEOUT_S", float),
    "question_timeout_s": ("DECISIONHEAD_QUESTION_TIMEOUT_S", float),
    "max_questions_per_request": ("DECISIONHEAD_MAX_QUESTIONS_PER_REQUEST", int),
    "max_concurrent_requests": ("DECISIONHEAD_MAX_CONCURRENT_REQUESTS", int),
    "ready_cache_s": ("DECISIONHEAD_READY_CACHE_S", float),
    "connection_read_timeout_s": ("DECISIONHEAD_CONNECTION_READ_TIMEOUT_S", float),
    "max_json_nesting_depth": ("DECISIONHEAD_MAX_JSON_NESTING_DEPTH", int),
    "bearer_token_env": ("DECISIONHEAD_BEARER_TOKEN_ENV", str),
    "cors_enabled": ("DECISIONHEAD_CORS_ENABLED", bool),
    "insecure_no_auth": ("DECISIONHEAD_INSECURE_NO_AUTH", bool),
    "allowed_hosts": ("DECISIONHEAD_ALLOWED_HOSTS", str),
    "log_level": ("DECISIONHEAD_LOG_LEVEL", str),
    "max_concurrency": ("DECISIONHEAD_MAX_CONCURRENCY", int),
    "max_concurrency_override_allowed": ("DECISIONHEAD_MAX_CONCURRENCY_OVERRIDE_ALLOWED", bool),
    "cache_enabled": ("DECISIONHEAD_CACHE_ENABLED", bool),
    "cache_ttl_s": ("DECISIONHEAD_CACHE_TTL_S", float),
    "cache_max_entries": ("DECISIONHEAD_CACHE_MAX_ENTRIES", int),
    "cache_path": ("DECISIONHEAD_CACHE_PATH", str),
    "abstain_band_lo": ("DECISIONHEAD_ABSTAIN_BAND_LO", float),
    "abstain_band_hi": ("DECISIONHEAD_ABSTAIN_BAND_HI", float),
    "abstain_mode": ("DECISIONHEAD_ABSTAIN_MODE", str),
}

CONFIG_PATH_ENV_VAR = "DECISIONHEAD_CONFIG"

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def _coerce(raw: str, kind: type) -> Any:
    if kind is bool:
        low = raw.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
        raise ConfigError(f"expected a boolean-like value, got {raw!r}")
    if kind is int:
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"expected an integer, got {raw!r}") from None
    if kind is float:
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"expected a number, got {raw!r}") from None
    return raw


def load_toml_file(path: str) -> JSONDict:
    """Reads a decisionhead TOML config file. Recognized top-level
    sections are ``[server]``, ``[limits]``, ``[auth]``, ``[logging]``,
    ``[backend]``, ``[cache]``, and ``[abstain]``; every key inside them
    maps 1:1 to a :class:`ServerConfig` field name.
    An unrecognized section is ignored (forward compatibility); an
    unrecognized key inside a recognized section is an error, since a typo
    there silently doing nothing would be worse than failing loudly."""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML ({exc})") from None
    except OSError as exc:
        raise ConfigError(f"{path}: {exc}") from None

    known_fields = {f.name for f in fields(ServerConfig)}
    flat: JSONDict = {}
    for section in ("server", "limits", "auth", "logging", "backend", "cache", "abstain"):
        table = raw.get(section, {})
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [{section}] must be a table")
        for key, value in table.items():
            if key not in known_fields:
                raise ConfigError(f"{path}: unrecognized key '{key}' in [{section}]")
            flat[key] = value
    return flat


def resolve_config(
    cli_overrides: JSONDict,
    *,
    config_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ServerConfig:
    """Builds the effective :class:`ServerConfig`.

    ``cli_overrides`` should contain only the fields the user actually
    passed on the command line (an argparse convention of "default is
    ``None``/unset" makes this natural -- see
    :func:`decisionhead.server.build_arg_parser`). ``env`` is injectable
    for tests; defaults to :data:`os.environ`.
    """
    env_source = os.environ if env is None else env
    env_values = _env_values_from(env_source)

    resolved_path = config_path or env_source.get(CONFIG_PATH_ENV_VAR)
    toml_values: JSONDict = load_toml_file(resolved_path) if resolved_path else {}

    merged: JSONDict = {}
    for f in fields(ServerConfig):
        if f.name in cli_overrides and cli_overrides[f.name] is not None:
            merged[f.name] = cli_overrides[f.name]
        elif f.name in env_values:
            merged[f.name] = env_values[f.name]
        elif f.name in toml_values:
            merged[f.name] = toml_values[f.name]
        # else: leave unset, dataclass default applies

    if merged.get("mode", DEFAULTS.mode) not in ALLOWED_MODES:
        raise ConfigError(f"invalid configuration: mode must be one of {ALLOWED_MODES}, got {merged.get('mode')!r}")

    if merged.get("abstain_mode", DEFAULTS.abstain_mode) not in ALLOWED_ABSTAIN_MODES:
        raise ConfigError(
            f"invalid configuration: abstain_mode must be one of {ALLOWED_ABSTAIN_MODES}, "
            f"got {merged.get('abstain_mode')!r}"
        )

    band_lo = merged.get("abstain_band_lo", DEFAULTS.abstain_band_lo)
    band_hi = merged.get("abstain_band_hi", DEFAULTS.abstain_band_hi)
    if (band_lo is None) != (band_hi is None):
        raise ConfigError("invalid configuration: abstain_band_lo and abstain_band_hi must be set together")
    if band_lo is not None and band_hi is not None and band_lo > band_hi:
        raise ConfigError(
            f"invalid configuration: abstain_band_lo ({band_lo}) must be <= abstain_band_hi ({band_hi})"
        )

    if merged.get("max_concurrency", DEFAULTS.max_concurrency) < 1:
        raise ConfigError("invalid configuration: max_concurrency must be >= 1")

    _validate_env_names(merged)
    _validate_urls(merged)
    _validate_ranges(merged)

    try:
        return ServerConfig(**merged)
    except TypeError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from None


# Fields whose value is the NAME of an environment variable holding a
# secret, never the secret itself.
SECRET_ENV_NAME_FIELDS: tuple[str, ...] = ("api_key_env", "bearer_token_env", "embedding_api_key_env")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_names(merged: JSONDict) -> None:
    """Each ``*_env`` value must look like an environment variable name.
    An operator who passes the token itself (an easy confusion) gets a
    refusal that does NOT echo the value, so it never reaches a log line
    or terminal scrollback (2026-10-06 security review, finding 13). A
    token that happens to look like a name cannot be caught here; that is
    why the server never logs these values at all."""
    for name in SECRET_ENV_NAME_FIELDS:
        value = merged.get(name)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or not _ENV_NAME_RE.match(value):
            raise ConfigError(
                f"invalid configuration: {name} must be the NAME of an environment variable "
                "([A-Za-z_][A-Za-z0-9_]*), not a secret value; the value given is not shown"
            )


def _validate_urls(merged: JSONDict) -> None:
    """``base_url``/``embedding_base_url`` must be http(s) with a host;
    ``urlopen`` would otherwise also accept ``file:`` (finding 13)."""
    for name in ("base_url", "embedding_base_url"):
        value = merged.get(name)
        if not value:
            continue
        parts = urllib.parse.urlsplit(str(value))
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ConfigError(f"invalid configuration: {name} must be an http:// or https:// URL with a host")


_POSITIVE_INT_FIELDS = (
    "max_workers",
    "max_request_bytes",
    "max_questions_per_request",
    "max_concurrent_requests",
    "max_json_nesting_depth",
    "deliberate_k",
    "deliberate_max_tokens",
    "cache_max_entries",
)
_POSITIVE_FLOAT_FIELDS = (
    "backend_timeout_s",
    "request_timeout_s",
    "question_timeout_s",
    "connection_read_timeout_s",
    "cache_ttl_s",
)


def _validate_ranges(merged: JSONDict) -> None:
    """Numeric limits must be positive (and finite): ``0`` made every
    request 429 (``max_concurrent_requests``) or turned a socket
    non-blocking (``connection_read_timeout_s``) -- finding 13."""
    for name in _POSITIVE_INT_FIELDS:
        value = merged.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ConfigError(f"invalid configuration: {name} must be an integer >= 1")
    for name in _POSITIVE_FLOAT_FIELDS:
        value = merged.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"invalid configuration: {name} must be a finite number > 0")


def _env_values_from(env: Mapping[str, str]) -> JSONDict:
    out: JSONDict = {}
    for field_name, (var_name, kind) in ENV_VAR_MAP.items():
        raw = env.get(var_name)
        if raw is not None and raw != "":
            out[field_name] = _coerce(str(raw), kind)
    return out
