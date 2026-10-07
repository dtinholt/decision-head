"""Transport-agnostic single-token logprob backends.

Two real backends and one fake one, all exposing the same method::

    token_distribution(prefix_messages, question_text, allowed_tokens)
        -> {token: prob, ..., "_unparsed": mass, "_raw": {...}}

``prefix_messages`` is a list of ``{"role", "content"}`` chat messages
shared across every question asked about one state (the caller in
:mod:`decisionhead.core` builds this once). ``question_text`` is the single
user-turn message that names the one question being asked in isolation.
``allowed_tokens`` is the exact list of single-token answers we are willing
to read (``["Yes", "No"]`` for a noul question, ``["A", "B", "C", ...]`` for
a choice or score question).

Mechanism (verified against a local ollama host, 2026-09-23): a single
deterministic, single-token generation (temperature 0, num_predict/
max_tokens 1) with logprobs turned on reveals the model's belief over the
handful of tokens it seriously considered at that position. We read the
probability mass on each allowed token from that list and normalize over
the allowed family; whatever mass fell on anything else is "_unparsed" and
is always reported rather than silently dropped. See ``docs/design.md``
for the full mechanism writeup, including why the single-token reading is
sound and why it is not a substitute for multi-step reasoning.

Ollama: ``POST {base}/api/chat``, ``think:false``, on the NATIVE route
only. The OpenAI-compatible ``/v1`` route on ollama routes through its
thinking wrapper and returns garbage for this purpose (``"The"``, not
``"Yes"``) -- so this module never calls it. Real vLLM-style
OpenAI-compatible servers do not have that problem and are handled by
:class:`OpenAIBackend`, which disables template thinking with
``chat_template_kwargs.enable_thinking:false`` instead.

No secrets: an API key, if any, is read from an environment variable name
the caller supplies -- never a literal in code, config, or a log line.

Invariants of the outbound HTTP path (2026-10-06 security review,
findings 3 and 12):

* only ``http``/``https`` URLs with a host are ever opened;
* every redirect is refused, so an ``Authorization`` header is never
  replayed to another host and an ``https`` base is never downgraded;
* a response body is read up to :data:`MAX_RESPONSE_BYTES` and no
  further;
* a :class:`BackendError` message never carries URL userinfo or more than
  200 bytes of a backend body, and the server only ever logs it;
* a non-finite logprob contributes zero mass and is counted, never
  propagated into a probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from typing import IO, Any, Protocol

from . import deadline as _deadline

JSONDict = dict[str, Any]
Transport = Callable[[str, JSONDict, JSONDict, float], JSONDict]

DEFAULT_TIMEOUT_S: float = 60.0

#: Largest backend response body read before giving up with a
#: BackendError, so a hostile or broken endpoint cannot exhaust memory
#: with one enormous body (2026-10-06 review, finding 12).
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

ALLOWED_URL_SCHEMES = ("http", "https")
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})


class BackendError(Exception):
    """Raised for any transport or response-shape failure. Never carries a
    secret: the message includes the URL and a short body snippet only."""


class TokenDistributionBackend(Protocol):
    """Structural type every concrete backend below satisfies: a
    ``model`` attribute plus one method reading a normalized probability
    distribution over ``allowed_tokens`` at a single generated position."""

    model: str

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: Sequence[str],
    ) -> JSONDict:
        """Returns ``{token: prob, ..., "_unparsed": mass, "_raw": {...}}``
        for the single token generated after ``prefix_messages`` plus a
        final user turn holding ``question_text``."""
        ...


class Backend(TokenDistributionBackend, Protocol):
    """Structural type satisfied by every concrete backend in this
    module: :class:`TokenDistributionBackend`'s single-token logprob
    read (fast mode) plus ``sample_completions``, the k-sampled-
    completions method :mod:`decisionhead.deliberate` (deliberate mode)
    uses. :func:`make_backend` returns this combined type so one backend
    object -- constructed once per server process -- can answer either
    mode's requests, dispatched per-request by
    :func:`decisionhead.server.resolve_effective_mode`."""

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
        """Returns ``(completions, meta)`` -- see the concrete
        implementations on :class:`OllamaBackend`, :class:`OpenAIBackend`,
        and :class:`FakeBackend` for the exact contract."""
        ...


def env_ref(name: str | None) -> str | None:
    """Resolves an environment variable BY NAME at call time. Nothing ever
    hardcodes the value itself -- this is the only place a key touches the
    process, and only if the caller explicitly names one."""
    if not name:
        return None
    return os.environ.get(name)


def redact_url(url: str) -> str:
    """``url`` with any ``user:password@`` userinfo removed, for log lines
    and error messages (2026-10-06 review, finding 10)."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{port}" if port else "")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def check_url_scheme(url: str) -> None:
    """Raises ``ValueError`` unless ``url`` is ``http://`` or ``https://``
    with a host. ``urlopen`` would otherwise also accept ``file:`` and
    other schemes (2026-10-06 review, findings 3 and 13)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_URL_SCHEMES or not parts.hostname:
        raise ValueError(f"backend URL must be http:// or https:// with a host (got {redact_url(url)!r})")


def is_loopback_url(url: str) -> bool:
    """True if ``url``'s host is a loopback name or address."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in LOOPBACK_HOSTNAMES or host.startswith("127.")


def insecure_transport_warning(url: str, *, has_credential: bool) -> str | None:
    """A warning message if ``url`` is plain ``http://`` to a non-loopback
    host -- anything on the network path can read the state documents
    and, if ``has_credential``, the API key -- else ``None``. Logged at
    start-up by :func:`decisionhead.server.run_server`."""
    if not url:
        return None
    if urllib.parse.urlsplit(url).scheme.lower() != "http" or is_loopback_url(url):
        return None
    what = "case state and the backend API key" if has_credential else "case state"
    return f"backend URL is plain http to a non-loopback host; {what} cross the network unencrypted"


class _RedirectPolicy(urllib.request.HTTPRedirectHandler):
    """Redirect handling for backend calls. CPython's default handler
    copies every non-content header, ``Authorization`` included, onto the
    redirected request whatever host it points at, and turns a POST into
    a GET on 301/302/303 (2026-10-06 review, finding 3).

    ``follow=False`` (what :func:`_http_post_json` and the readiness
    probe use): every redirect is refused with a :class:`BackendError`;
    a model endpoint has no legitimate reason to redirect an API call.
    ``follow=True``: the redirect is followed, but ``Authorization`` is
    dropped whenever the target's scheme, host or port differs from the
    original's."""

    def __init__(self, follow: bool = False) -> None:
        self.follow = follow

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Called by ``urllib`` for every 3xx. Raises
        :class:`BackendError` in the default refuse mode; otherwise
        returns the stock redirected request with ``Authorization``
        removed from both header maps when the origin changes.

        ``urllib`` keeps headers in two places (``headers`` and
        ``unredirected_hdrs``); both are scrubbed, because a header added
        with ``add_unredirected_header`` would otherwise survive."""
        if not self.follow:
            raise BackendError(f"backend answered HTTP {code} with a redirect; redirects are refused")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_parts = urllib.parse.urlsplit(req.full_url)
        new_parts = urllib.parse.urlsplit(new.full_url)
        same_origin = (old_parts.scheme, old_parts.hostname, old_parts.port) == (
            new_parts.scheme,
            new_parts.hostname,
            new_parts.port,
        )
        if not same_origin:
            for name in list(new.headers):
                if name.lower() == "authorization":
                    del new.headers[name]
            for name in list(new.unredirected_hdrs):
                if name.lower() == "authorization":
                    del new.unredirected_hdrs[name]
        return new


def build_backend_opener(follow_redirects: bool = False) -> urllib.request.OpenerDirector:
    """The opener every backend HTTP call uses: the default handler chain
    with :class:`_RedirectPolicy` in place of the stock redirect
    handler."""
    return urllib.request.build_opener(_RedirectPolicy(follow=follow_redirects))


_OPENER = build_backend_opener()


def _http_post_json(url: str, body: JSONDict, headers: JSONDict, timeout: float = DEFAULT_TIMEOUT_S) -> JSONDict:
    """Real transport: POST JSON, expect JSON back. This is the default
    ``transport`` for both real backends; tests inject a fake callable with
    the identical signature instead of touching the network.

    Refuses non-http(s) URLs and every redirect, caps the response body
    at :data:`MAX_RESPONSE_BYTES`, and never puts URL userinfo into the
    error message. A :class:`BackendError` message stays in server logs;
    clients only ever see a fixed ``backend_error`` text."""
    try:
        check_url_scheme(url)
    except ValueError as exc:
        raise BackendError(str(exc)) from None
    safe_url = redact_url(url)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw_bytes = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        snippet = exc.read(200).decode("utf-8", errors="replace")
        raise BackendError(f"HTTP {exc.code} from {safe_url}: {snippet!r}") from None
    except urllib.error.URLError as exc:
        raise BackendError(f"connection failed to {safe_url}: {exc.reason}") from None
    except OSError as exc:
        raise BackendError(f"connection to {safe_url} failed: {type(exc).__name__}") from None
    if len(raw_bytes) > MAX_RESPONSE_BYTES:
        raise BackendError(f"response from {safe_url} exceeded {MAX_RESPONSE_BYTES} bytes")
    raw = raw_bytes.decode("utf-8", errors="replace")
    try:
        result: JSONDict = json.loads(raw)
        return result
    except ValueError:
        raise BackendError(f"non-JSON response from {safe_url}: {raw[:200]!r}") from None


# --------------------------------------------------------------------- parsing


def _normalize_token(token: Any) -> str:
    """Case- and whitespace-insensitive, tolerant of a trailing letter-choice
    punctuation mark ("A)", "A.", "A,"). Used identically for yes/no tokens
    and for lettered choice/score tokens -- one normalization, two uses."""
    return str(token or "").strip().lower().rstrip(").:,")


def _mass(logprob: Any) -> tuple[float, bool]:
    """``(exp(logprob), was_finite)``. A missing or non-numeric ``logprob``
    yields ``(0.0, True)`` -- an ordinary, already-handled degradation.
    A numeric ``logprob`` that produces a non-finite mass (NaN or
    infinity, including an overflow so large ``math.exp`` cannot
    represent it) yields ``(0.0, False)`` instead of the non-finite value
    itself: a NaN/inf must never enter a running sum here, since it would
    silently corrupt every probability derived from that sum and, if it
    ever reached the HTTP response, would serialize as a bare NaN or
    Infinity token -- not valid JSON per RFC 8259. The caller uses the
    second element to keep this kind of corrupted backend reading visible
    instead of indistinguishable from a normal, trustworthy zero."""
    try:
        value = math.exp(float(logprob))
    except (TypeError, ValueError):
        return 0.0, True
    except OverflowError:
        return 0.0, False
    if not math.isfinite(value):
        return 0.0, False
    return value, True


def distribution_from_top_logprobs(
    top_logprobs: Sequence[JSONDict], allowed_tokens: Sequence[str]
) -> tuple[dict[str, float], JSONDict]:
    """Core reading: given the top_logprobs list at the single generated
    position, and the exact set of tokens we're willing to accept, returns
    ``(distribution, meta)``.

    ``distribution`` is a dict over ``allowed_tokens`` (canonical spelling
    in, same spelling out) summing to 1.0 -- normalized over the mass that
    landed on an allowed token ("family mass"), matching the vendor-shape
    requirement that noul/choice/score answers carry a normalized
    probability. If NO allowed token appears anywhere in ``top_logprobs``
    at all, falls back to a uniform distribution and flags
    ``no_family_match`` so callers can see the degenerate case rather than
    trust a fabricated number.

    ``meta`` carries ``unparsed_mass``: the exp-mass in ``top_logprobs``
    that belonged to none of the allowed tokens, in the same units as
    family mass (so ``unparsed_mass / (unparsed_mass + family_mass)`` is
    directly interpretable as "share of the model's attention we could not
    read"). It also carries ``non_finite_logprob_count``: how many entries
    reported a NaN, infinite, or unrepresentably large logprob. Such an
    entry contributes zero mass here (never propagated into any sum, and
    never able to reach the JSON response as an invalid bare NaN/Infinity
    token) rather than being silently indistinguishable from a normal
    zero-probability entry.

    The uniform fallback is a placeholder for the backend layer only:
    :func:`decisionhead.core._answer_one` turns a ``no_family_match``
    reading into a ``degraded`` answer with ``null`` values, so the 0.5
    never reaches a client."""
    norm_to_canonical: dict[str, str] = {}
    for t in allowed_tokens:
        norm_to_canonical[_normalize_token(t)] = t
    family_mass = {t: 0.0 for t in allowed_tokens}
    total_mass = 0.0
    non_finite_logprob_count = 0
    for entry in top_logprobs or []:
        if not isinstance(entry, dict):
            continue
        m, finite_ok = _mass(entry.get("logprob"))
        if not finite_ok:
            non_finite_logprob_count += 1
        total_mass += m
        norm = _normalize_token(entry.get("token"))
        canonical = norm_to_canonical.get(norm)
        if canonical is not None:
            family_mass[canonical] += m
    fam_total = sum(family_mass.values())
    no_family_match = fam_total <= 0.0
    dist: dict[str, float]
    if no_family_match:
        n = len(allowed_tokens) or 1
        dist = {t: 1.0 / n for t in allowed_tokens}
    else:
        dist = {t: m / fam_total for t, m in family_mass.items()}
    meta: JSONDict = {
        "unparsed_mass": max(0.0, total_mass - fam_total),
        "total_mass": total_mass,
        "family_mass": fam_total,
        "no_family_match": no_family_match,
        "non_finite_logprob_count": non_finite_logprob_count,
    }
    return dist, meta


def _extract_top_ollama(resp: JSONDict) -> list[JSONDict]:
    """``resp['logprobs']`` is a list with one element per generated token;
    with ``num_predict:1`` that list has exactly one element, whose
    ``top_logprobs`` is what we read. Any other shape degrades to ``[]``
    rather than raising -- a malformed upstream response should read as
    fully unparsed, not crash."""
    lp = resp.get("logprobs") if isinstance(resp, dict) else None
    if not isinstance(lp, list) or not lp:
        return []
    top = lp[0].get("top_logprobs") if isinstance(lp[0], dict) else None
    return top if isinstance(top, list) else []


def _extract_top_openai(resp: JSONDict) -> list[JSONDict]:
    """OpenAI-compatible shape:
    ``choices[0].logprobs.content[0].top_logprobs``."""
    try:
        choices = resp.get("choices") or []
        content = (choices[0].get("logprobs") or {}).get("content") or []
        top = content[0].get("top_logprobs")
        return top if isinstance(top, list) else []
    except (IndexError, AttributeError, TypeError):
        return []


def _ollama_usage(resp: JSONDict) -> JSONDict:
    return {
        "input_tokens": resp.get("prompt_eval_count") if isinstance(resp, dict) else None,
        "output_tokens": resp.get("eval_count") if isinstance(resp, dict) else None,
    }


def _openai_usage(resp: JSONDict) -> JSONDict:
    usage = (resp or {}).get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


# --------------------------------------------------------------------- backends


class OllamaBackend:
    """Ollama's NATIVE ``/api/chat`` route, single deterministic token,
    logprobs on. Never touches ``/v1`` (see module docstring)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """``transport`` defaults to the real HTTP POST; tests inject a
        fixture callable with the identical signature instead."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport: Transport = transport or _http_post_json
        self.timeout = timeout

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: Sequence[str],
    ) -> JSONDict:
        """POSTs to ``{base_url}/api/chat`` with ``think:false`` and a
        single deterministic token, and reads the probability distribution
        over ``allowed_tokens`` from the returned logprobs."""
        messages = list(prefix_messages) + [{"role": "user", "content": question_text}]
        body: JSONDict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 5,
            "options": {"num_predict": 1, "temperature": 0},
        }
        t0 = time.time()
        resp = self.transport(
            self.base_url + "/api/chat", body, {"Content-Type": "application/json"}, self.timeout
        )
        latency_s = time.time() - t0
        top = _extract_top_ollama(resp)
        dist, meta = distribution_from_top_logprobs(top, allowed_tokens)
        result: JSONDict = dict(dist)
        result["_unparsed"] = meta["unparsed_mass"]
        result["_raw"] = {
            "backend": "ollama",
            "endpoint": "/api/chat",
            "top_logprobs": top,
            "no_family_match": meta["no_family_match"],
            "non_finite_logprob_count": meta["non_finite_logprob_count"],
            "latency_s": latency_s,
            "usage": _ollama_usage(resp) if isinstance(resp, dict) else {},
            "response_message": (resp.get("message") or {}).get("content") if isinstance(resp, dict) else None,
        }
        return result

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
        """Used by :mod:`decisionhead.deliberate` ("deliberate" mode):
        ``n`` real, sampled generations (``temperature>0``, up to
        ``max_tokens``), unlike :meth:`token_distribution`'s single
        deterministic token. Ollama's native ``/api/chat`` has no ``n``
        parameter (unlike an OpenAI-compatible server's ``n``), so this
        is always ``n`` separate requests here; ``allowed_answers`` is
        accepted for interface parity with other backends but unused --
        the prompt built by the caller already names the allowed answers.
        """
        messages = list(prefix_messages) + [{"role": "user", "content": question_text}]
        texts: list[str] = []
        in_tokens: list[float] = []
        out_tokens: list[float] = []
        for _ in range(n):
            _deadline.check_current()  # abandon queued samples once the request has timed out
            body: JSONDict = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            resp = self.transport(
                self.base_url + "/api/chat", body, {"Content-Type": "application/json"}, self.timeout
            )
            content = (resp.get("message") or {}).get("content") if isinstance(resp, dict) else None
            texts.append(content or "")
            usage = _ollama_usage(resp) if isinstance(resp, dict) else {}
            if isinstance(usage.get("input_tokens"), (int, float)):
                in_tokens.append(usage["input_tokens"])
            if isinstance(usage.get("output_tokens"), (int, float)):
                out_tokens.append(usage["output_tokens"])
        meta: JSONDict = {
            "requests": n,
            "n_param_used": False,
            "fallback_used": False,
            "usage": {
                "input_tokens": sum(in_tokens) if in_tokens else None,
                "output_tokens": sum(out_tokens) if out_tokens else None,
            },
        }
        return texts, meta


class OpenAIBackend:
    """vLLM-style OpenAI-compatible ``/v1/chat/completions``, single
    deterministic token, logprobs on, template thinking disabled via
    ``chat_template_kwargs``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        api_key_ref: str | None = None,
    ) -> None:
        """``api_key_ref`` is the NAME of an environment variable to read
        a bearer token from at call time, never a literal key."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport: Transport = transport or _http_post_json
        self.timeout = timeout
        self.api_key_ref = api_key_ref

    def _headers(self) -> JSONDict:
        headers: JSONDict = {"Content-Type": "application/json"}
        key = env_ref(self.api_key_ref)
        if key:
            headers["Authorization"] = "Bearer " + key
        return headers

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: Sequence[str],
    ) -> JSONDict:
        """POSTs to ``{base_url}/v1/chat/completions`` with template
        thinking disabled and a single deterministic token, and reads the
        probability distribution over ``allowed_tokens`` from the
        returned logprobs."""
        messages = list(prefix_messages) + [{"role": "user", "content": question_text}]
        body: JSONDict = {
            "model": self.model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        resp = self.transport(self.base_url + "/v1/chat/completions", body, self._headers(), self.timeout)
        latency_s = time.time() - t0
        top = _extract_top_openai(resp)
        dist, meta = distribution_from_top_logprobs(top, allowed_tokens)
        result: JSONDict = dict(dist)
        result["_unparsed"] = meta["unparsed_mass"]
        result["_raw"] = {
            "backend": "openai",
            "endpoint": "/v1/chat/completions",
            "top_logprobs": top,
            "no_family_match": meta["no_family_match"],
            "non_finite_logprob_count": meta["non_finite_logprob_count"],
            "latency_s": latency_s,
            "usage": _openai_usage(resp) if isinstance(resp, dict) else {},
        }
        return result

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
        """Used by :mod:`decisionhead.deliberate` ("deliberate" mode). An
        OpenAI-compatible server's ``n`` parameter lets ``n>1`` samples
        come back from ONE request; that is tried first. If that request
        raises :class:`BackendError`, or comes back with zero usable
        choices (a server that silently ignores ``n`` and returns one),
        this falls back to ``n`` separate ``n=1`` requests instead --
        recorded in the returned meta (``n_param_used``/``fallback_used``)
        so it is auditable which path a given backend actually took.
        ``allowed_answers`` is accepted for interface parity but unused --
        the prompt built by the caller already names the allowed answers.
        """
        messages = list(prefix_messages) + [{"role": "user", "content": question_text}]

        def _request(count: int) -> JSONDict:
            body: JSONDict = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": count,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            return self.transport(self.base_url + "/v1/chat/completions", body, self._headers(), self.timeout)

        if n > 1:
            _deadline.check_current()
            try:
                resp = _request(n)
            except BackendError:
                resp = None
            if resp is not None:
                choices = resp.get("choices") if isinstance(resp, dict) else None
                texts = (
                    [((c.get("message") or {}).get("content") or "") for c in choices if isinstance(c, dict)]
                    if isinstance(choices, list)
                    else []
                )
                if texts:
                    # A server may return more choices than asked for;
                    # only the first n are ever voted (2026-10-06 review,
                    # finding 5). returned_n records what actually came back.
                    return texts[:n], {
                        "requests": 1,
                        "n_param_used": True,
                        "fallback_used": False,
                        "requested_n": n,
                        "returned_n": len(texts),
                        "usage": _openai_usage(resp),
                    }

        texts = []
        in_tokens: list[float] = []
        out_tokens: list[float] = []
        for _ in range(n):
            _deadline.check_current()
            resp = _request(1)
            choices = resp.get("choices") if isinstance(resp, dict) else None
            text = ""
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                text = (choices[0].get("message") or {}).get("content") or ""
            texts.append(text)
            usage = _openai_usage(resp) if isinstance(resp, dict) else {}
            if isinstance(usage.get("input_tokens"), (int, float)):
                in_tokens.append(usage["input_tokens"])
            if isinstance(usage.get("output_tokens"), (int, float)):
                out_tokens.append(usage["output_tokens"])
        return texts, {
            "requests": n,
            "n_param_used": False,
            "fallback_used": n > 1,
            "usage": {
                "input_tokens": sum(in_tokens) if in_tokens else None,
                "output_tokens": sum(out_tokens) if out_tokens else None,
            },
        }


class FakeBackend:
    """Deterministic stand-in used by ``--dry`` server mode and by every
    test that must not touch the network. Given the same seed and
    question_text it always returns the same distribution (needed so
    consistency-style tests can assert zero disagreement), and different
    question texts generally produce different distributions (needed so
    calibration and isolation tests have non-degenerate structure to check
    against)."""

    def __init__(self, seed: int = 0, model: str = "fake-model-v1") -> None:
        """``seed`` selects which deterministic-but-varied distribution
        family this instance produces; ``model`` is reported as-is."""
        self.seed = seed
        self.model = model

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: Sequence[str],
    ) -> JSONDict:
        """Derives a deterministic-but-varied distribution over
        ``allowed_tokens`` from a hash of ``(seed, question_text)`` --
        no network, no model."""
        h = hashlib.sha256(f"{self.seed}|{question_text}".encode()).hexdigest()
        n = len(allowed_tokens)
        if n == 0:
            return {"_unparsed": 0.0, "_raw": {"fake": True, "hash": h}}
        idx = int(h[:8], 16) % n
        top_mass = 0.55 + 0.35 * (int(h[8:16], 16) / 0xFFFFFFFF)
        rest = (1.0 - top_mass) / (n - 1) if n > 1 else 0.0
        dist = {t: (top_mass if i == idx else rest) for i, t in enumerate(allowed_tokens)}
        return {
            **dist,
            "_unparsed": 0.0,
            "_raw": {"fake": True, "hash": h, "backend": "fake", "model": self.model, "latency_s": 0.0},
        }

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
        """Deterministic-but-varied fake completions for ``--dry``
        deliberate mode: no network, no real model. Each sample is a
        short fake "reasoning" fragment followed, on its own final line,
        by a strict JSON object naming one of ``allowed_answers``, biased the same way
        :meth:`token_distribution` is -- a hash of ``(seed,
        question_text)`` picks a modal answer that most samples agree
        with, with a modest amount of per-sample disagreement, so a vote
        over these samples has the same non-degenerate shape a vote over
        real sampled answers would."""
        names = list(allowed_answers) or ["yes"]
        modal_h = hashlib.sha256(f"{self.seed}|{question_text}".encode()).hexdigest()
        modal_idx = int(modal_h[:8], 16) % len(names)
        texts: list[str] = []
        for i in range(n):
            h = hashlib.sha256(f"{self.seed}|{question_text}|{i}".encode()).hexdigest()
            agrees_with_modal = (int(h[8:10], 16) % 10) < 7  # ~70% agreement, like a real vote
            idx = modal_idx if agrees_with_modal else int(h[:8], 16) % len(names)
            confidence = 0.5 + 0.5 * (int(h[10:18], 16) / 0xFFFFFFFF)
            answer_json = json.dumps({"answer": names[idx], "confidence": round(confidence, 3)})
            # The JSON goes on its own final line, as the prompt asks; the
            # parser only reads an object that is the whole final line.
            texts.append(f"(fake reasoning sample {i}, hash {h[:8]})\n{answer_json}")
        return texts, {
            "requests": n,
            "n_param_used": False,
            "fallback_used": False,
            "usage": {"input_tokens": None, "output_tokens": None},
        }


class ConcurrencyLimitedBackend:
    """Wraps any :class:`Backend` and serializes (or bounds) calls to
    ``token_distribution``/``sample_completions`` behind a semaphore of
    size ``max_concurrency``.

    Finding (verified against a real DeepSeek exl3 endpoint,
    2026-09): that endpoint returns different logprobs for
    otherwise-identical requests when the server batches concurrent
    requests together; at concurrency 1 (this class's default use, via
    ``ServerConfig.max_concurrency``) repeated identical requests return
    identical logprobs. This wraps the backend rather than the caller so
    every call site (:mod:`decisionhead.core`, :mod:`decisionhead.deliberate`)
    is protected uniformly with no per-call-site plumbing.

    ``max_concurrency_override`` (fast mode only, see
    :mod:`decisionhead.core`) lets one call use a different-sized
    semaphore than the default -- a distinct, lazily-created semaphore is
    kept per distinct override value, so concurrently-overridden calls
    that ask for the same value still share one limiter. This is gated
    server-side by ``ServerConfig.max_concurrency_override_allowed``
    (default ``False``); this class itself does no gating, it just
    honours whatever override value it is given.
    """

    #: Feature-detection flag :mod:`decisionhead.core` checks before
    #: passing ``max_concurrency_override`` -- other backends (including
    #: an undecorated :class:`OllamaBackend`/:class:`OpenAIBackend`/
    #: :class:`FakeBackend`) do not accept that keyword.
    accepts_concurrency_override = True

    def __init__(self, inner: Backend, max_concurrency: int, acquire_timeout_s: float | None = None) -> None:
        """``acquire_timeout_s`` bounds how long one call waits for a
        semaphore slot. When the calling thread has a current
        :class:`~decisionhead.deadline.Deadline`, the wait is also capped
        by its remaining time. A call that cannot get a slot in time
        raises :class:`~decisionhead.deadline.DeadlineExceeded` rather
        than queueing indefinitely (2026-10-06 review, finding 1)."""
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._inner = inner
        self.acquire_timeout_s = acquire_timeout_s
        # A plain attribute (not a property) so this class satisfies the
        # Backend protocol's settable `model: str`; the wrapped backend's
        # own model id never changes after construction, so copying it
        # once here is equivalent and simpler.
        self.model = inner.model
        self.max_concurrency = max_concurrency
        self._semaphore = threading.Semaphore(max_concurrency)
        self._override_lock = threading.Lock()
        self._override_semaphores: dict[int, threading.Semaphore] = {}

    @property
    def wrapped_type_name(self) -> str:
        """The wrapped backend's own class name (``"OllamaBackend"``,
        ``"OpenAIBackend"``, ...) -- callers that label audit rows or logs
        with ``type(backend).__name__`` should use this instead, so a
        wrapped backend's audit trail still names the real backend, not
        this wrapper."""
        return type(self._inner).__name__

    def _semaphore_for(self, max_concurrency_override: int | None) -> threading.Semaphore:
        if max_concurrency_override is None:
            return self._semaphore
        if max_concurrency_override < 1:
            raise ValueError(f"max_concurrency_override must be >= 1, got {max_concurrency_override}")
        with self._override_lock:
            sem = self._override_semaphores.get(max_concurrency_override)
            if sem is None:
                sem = threading.Semaphore(max_concurrency_override)
                self._override_semaphores[max_concurrency_override] = sem
            return sem

    def token_distribution(
        self,
        prefix_messages: list[JSONDict],
        question_text: str,
        allowed_tokens: Sequence[str],
        max_concurrency_override: int | None = None,
    ) -> JSONDict:
        """The wrapped backend's :meth:`token_distribution`, run while
        holding one slot of the default semaphore, or of the semaphore
        for ``max_concurrency_override`` when one is given. Raises
        :class:`~decisionhead.deadline.DeadlineExceeded` if no slot frees
        up within the time budget, without calling the backend."""
        sem = self._semaphore_for(max_concurrency_override)
        self._acquire(sem)
        try:
            return self._inner.token_distribution(prefix_messages, question_text, allowed_tokens)
        finally:
            sem.release()

    def _acquire(self, sem: threading.Semaphore) -> None:
        """Acquires ``sem`` within the time budget described in
        :meth:`__init__`, then re-checks the deadline, so a request that
        was abandoned while this call waited never reaches the backend."""
        # The deadline arrives thread-locally, set by the worker thread in
        # core/deliberate/learned, so the Backend protocol's signatures
        # stay unchanged. The wait is the smaller of the configured
        # acquire timeout and the request's remaining time.
        d = _deadline.current()
        timeout = self.acquire_timeout_s
        if d is not None:
            remaining = d.remaining()
            timeout = remaining if timeout is None else min(timeout, remaining)
        acquired = sem.acquire(timeout=timeout) if timeout is not None else sem.acquire()
        if not acquired:
            raise _deadline.DeadlineExceeded("timed out waiting for a backend concurrency slot")
        # Re-check after the wait: the request may have been cancelled
        # (504) while this thread was queued. Without this, a call queued
        # behind a slow one would still reach the backend after the
        # client had already been told the request was abandoned.
        if d is not None and d.expired():
            sem.release()
            raise _deadline.DeadlineExceeded("request deadline passed while waiting for a backend slot")

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
        """The wrapped backend's :meth:`sample_completions`, run while
        holding one slot of the default semaphore for the whole batch of
        ``n`` samples. Same time budget and
        :class:`~decisionhead.deadline.DeadlineExceeded` behaviour as
        :meth:`token_distribution`; the per-sample loops inside the real
        backends also check the deadline between samples."""
        self._acquire(self._semaphore)
        try:
            return self._inner.sample_completions(
                prefix_messages, question_text, allowed_answers, n=n, temperature=temperature, max_tokens=max_tokens
            )
        finally:
            self._semaphore.release()

    def __getattr__(self, name: str) -> Any:
        # Delegates anything not defined above (e.g. base_url, transport,
        # timeout -- attributes tests and callers occasionally reach for
        # directly) to the wrapped backend.
        return getattr(self._inner, name)

    def unwrap(self) -> Backend:
        """Returns the real backend underneath this wrapper (recursively,
        in case of nested wrapping) -- for a caller
        (:class:`decisionhead.server.ReadinessChecker`) that needs to
        ``isinstance``-check or otherwise inspect the concrete backend
        rather than this concurrency-limiting shell around it."""
        inner = self._inner
        return inner.unwrap() if hasattr(inner, "unwrap") else inner


def unwrap_backend(backend: Any) -> Any:
    """``backend.unwrap()`` if ``backend`` supports it (see
    :class:`ConcurrencyLimitedBackend`), else ``backend`` itself
    unchanged. A free function so a caller need not care whether it has a
    wrapped or a plain backend in hand."""
    return backend.unwrap() if hasattr(backend, "unwrap") else backend


def make_backend(
    name: str,
    *,
    base_url: str = "",
    model: str = "",
    api_key_env: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    dry: bool = False,
    dry_seed: int = 0,
    max_concurrency: int | None = None,
    acquire_timeout_s: float | None = None,
) -> Backend:
    """Factory shared by the server, calibrate, bench, and probe entry
    points, so backend construction lives in exactly one place.

    ``max_concurrency``, when given, wraps the constructed backend in a
    :class:`ConcurrencyLimitedBackend`. Left as the default ``None`` for
    every caller except :func:`decisionhead.server.run_server` -- bench
    and calibrate deliberately construct an unwrapped backend so they can
    measure/exercise the endpoint's own behaviour at whatever concurrency
    they choose."""
    backend: Backend
    if dry:
        backend = FakeBackend(seed=dry_seed, model=model or "dry-fake-model")
    elif name == "ollama":
        backend = OllamaBackend(base_url, model, timeout=timeout)
    elif name == "openai":
        backend = OpenAIBackend(base_url, model, timeout=timeout, api_key_ref=api_key_env)
    else:
        raise ValueError(f"unknown backend {name!r}")
    if max_concurrency is not None:
        backend = ConcurrencyLimitedBackend(backend, max_concurrency, acquire_timeout_s=acquire_timeout_s)
    return backend


# --------------------------------------------------------------------- probe


PROBE_QUESTION = "Is one plus one equal to two? Answer with exactly one word: Yes or No."


def probe_backend(backend: TokenDistributionBackend, question_text: str = PROBE_QUESTION) -> JSONDict:
    """Sends one trivial, unambiguous yes/no question straight through
    ``backend.token_distribution`` and reports the raw distribution,
    unparsed mass, and latency. This is what the ``decisionhead probe``
    subcommand runs: a fast sanity check that a newly configured backend
    actually returns a usable per-token logprob distribution, without
    needing a full question set or a running server."""
    prefix = [{"role": "system", "content": "Answer with exactly one word: Yes or No."}]
    t0 = time.time()
    dist = backend.token_distribution(prefix, question_text, ["Yes", "No"])
    latency_s = time.time() - t0
    raw = dist.get("_raw") if isinstance(dist, dict) else None
    no_family_match = bool(raw.get("no_family_match")) if isinstance(raw, dict) else False
    return {
        "model": getattr(backend, "model", None),
        "question": question_text,
        "distribution": {"Yes": dist.get("Yes"), "No": dist.get("No")},
        "unparsed_mass": dist.get("_unparsed"),
        "no_family_match": no_family_match,
        "latency_s": latency_s,
        "healthy": (not no_family_match) and dist.get("_unparsed", 1.0) < 0.5,
    }


# --------------------------------------------------------------------- selftest


def _fixture_transport(fixed_response: JSONDict, capture: JSONDict | None = None) -> Transport:
    def _t(url: str, body: JSONDict, headers: JSONDict, timeout: float = DEFAULT_TIMEOUT_S) -> JSONDict:
        if capture is not None:
            capture["url"] = url
            capture["body"] = body
            capture["headers"] = headers
        return fixed_response

    return _t


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed.
    The equivalent, more thorough assertions live in
    ``tests/test_backends.py``."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    ollama_noul_resp: JSONDict = {
        "message": {"content": "Yes"},
        "logprobs": [
            {
                "token": "Yes",
                "logprob": -0.1378,
                "top_logprobs": [
                    {"token": "Yes", "logprob": -0.1378},
                    {"token": "yes", "logprob": -2.097},
                    {"token": " yes", "logprob": -5.90},
                    {"token": "No", "logprob": -6.5},
                    {"token": "Sure", "logprob": -9.0},
                ],
            }
        ],
        "prompt_eval_count": 120,
        "eval_count": 1,
    }
    cap: JSONDict = {}
    backend = OllamaBackend("http://example:11434", "qwen3.8:27b", transport=_fixture_transport(ollama_noul_resp, cap))
    dist = backend.token_distribution([{"role": "system", "content": "s"}], "Q?", ["Yes", "No"])
    yes_mass = math.exp(-0.1378) + math.exp(-2.097) + math.exp(-5.90)
    no_mass = math.exp(-6.5)
    expected_p_yes = yes_mass / (yes_mass + no_mass)
    check("ollama noul p_yes matches hand computation", abs(dist["Yes"] - expected_p_yes) < 1e-9)
    check("ollama noul probabilities sum to 1", abs(dist["Yes"] + dist["No"] - 1.0) < 1e-9)
    check("ollama noul unparsed mass excludes yes/no family", dist["_unparsed"] > 0)
    check("ollama backend calls /api/chat, never /v1", cap["url"].endswith("/api/chat"))
    check("ollama request sets think:false", cap["body"]["think"] is False)
    check(
        "ollama request sets num_predict:1 temperature:0",
        cap["body"]["options"] == {"num_predict": 1, "temperature": 0},
    )
    check(
        "ollama usage read from prompt_eval_count/eval_count",
        dist["_raw"]["usage"] == {"input_tokens": 120, "output_tokens": 1},
    )

    ollama_choice_resp: JSONDict = {
        "message": {"content": "B)"},
        "logprobs": [
            {
                "token": "B)",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "B)", "logprob": -0.2},
                    {"token": "A", "logprob": -1.5},
                    {"token": "C.", "logprob": -3.0},
                ],
            }
        ],
    }
    backend2 = OllamaBackend("http://example:11434", "m", transport=_fixture_transport(ollama_choice_resp))
    dist2 = backend2.token_distribution([], "Q?", ["A", "B", "C"])
    check("choice letter with trailing punctuation is read", dist2["B"] > dist2["A"] > dist2["C"])
    check("choice distribution sums to 1", abs(sum(dist2[t] for t in ("A", "B", "C")) - 1.0) < 1e-9)

    no_match_resp: JSONDict = {
        "logprobs": [{"token": "X", "logprob": -0.1, "top_logprobs": [{"token": "X", "logprob": -0.1}]}]
    }
    backend3 = OllamaBackend("http://example:11434", "m", transport=_fixture_transport(no_match_resp))
    dist3 = backend3.token_distribution([], "Q?", ["Yes", "No"])
    check("no-family-match falls back to uniform", dist3["Yes"] == 0.5 and dist3["No"] == 0.5)
    check("no-family-match is flagged in _raw", dist3["_raw"]["no_family_match"] is True)

    openai_resp: JSONDict = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.05,
                            "top_logprobs": [{"token": "A", "logprob": -0.05}, {"token": "B", "logprob": -4.0}],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 1},
    }
    cap4: JSONDict = {}
    backend4 = OpenAIBackend("http://example:8000", "m", transport=_fixture_transport(openai_resp, cap4))
    dist4 = backend4.token_distribution([], "Q?", ["A", "B"])
    check("openai backend calls /v1/chat/completions", cap4["url"].endswith("/v1/chat/completions"))
    check(
        "openai request disables template thinking",
        cap4["body"]["chat_template_kwargs"] == {"enable_thinking": False},
    )
    check(
        "openai request asks for 1 max_tokens at temperature 0",
        cap4["body"]["max_tokens"] == 1 and cap4["body"]["temperature"] == 0,
    )
    check("openai dist reads choices[0].logprobs.content[0].top_logprobs", dist4["A"] > dist4["B"])
    check(
        "openai usage read from usage.prompt_tokens/completion_tokens",
        dist4["_raw"]["usage"] == {"input_tokens": 80, "output_tokens": 1},
    )

    backend5 = OllamaBackend("http://example:11434", "m", transport=_fixture_transport({"unexpected": True}))
    dist5 = backend5.token_distribution([], "Q?", ["Yes", "No"])
    check("malformed response degrades to uniform without raising", dist5["Yes"] == 0.5)

    fb = FakeBackend(seed=7)
    a1 = fb.token_distribution([], "question one", ["Yes", "No"])
    a2 = fb.token_distribution([], "question one", ["Yes", "No"])
    a3 = fb.token_distribution([], "question two", ["Yes", "No"])
    check("fake backend deterministic for same question", a1["Yes"] == a2["Yes"])
    check("fake backend distribution sums to 1", abs(a1["Yes"] + a1["No"] - 1.0) < 1e-9)
    check("fake backend varies across questions (usually)", a1["Yes"] != a3["Yes"])

    def _raising_transport(url: str, body: JSONDict, headers: JSONDict, timeout: float = DEFAULT_TIMEOUT_S) -> JSONDict:
        raise BackendError(f"HTTP 500 from {url}: boom")

    backend7 = OllamaBackend("http://example:11434", "m", transport=_raising_transport)
    try:
        backend7.token_distribution([], "Q?", ["Yes", "No"])
        check("backend error propagates as BackendError", False)
    except BackendError as exc:
        check("backend error propagates as BackendError", "HTTP 500" in str(exc))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)
