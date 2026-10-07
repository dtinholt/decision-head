"""Optional text-embedding backends for feature group (c) in
:mod:`decisionhead.features` (an embedding of the state's rendered
prefix, i.e. :func:`decisionhead.core.render_state` applied to the same
state the service receives).

This mirrors :mod:`decisionhead.backends`'s shape and transport plumbing
(``_http_post_json``, ``env_ref``, ``BackendError``, an injectable
``Transport`` for tests) but for a different API family: embeddings, not
single-token logprobs. Two real backends:

* **``ollama``** -- ``POST {base}/api/embeddings``, ``{"model": ...,
  "prompt": text}``, response ``{"embedding": [float, ...]}``. This is the
  route an ollama host with an embedding model pulled (e.g.
  ``qwen3-embedding:0.6b``) exposes.
* **``openai``** -- any OpenAI-compatible ``POST {base}/v1/embeddings``,
  ``{"model": ..., "input": text}``, response
  ``{"data": [{"embedding": [float, ...]}]}``.

Embedding features are OFF by default everywhere in this package
(:class:`decisionhead.features.FeatureConfig` defaults ``use_embedding``
to ``False``) and this module makes no network call unless a caller
explicitly constructs a real backend and calls ``embed``. It is exercised
in tests only through :class:`FakeEmbeddingsBackend`, which is
deterministic and makes no network call -- there is no reachable ollama
or OpenAI-compatible embeddings endpoint from the environment this
package is developed in, so the real backends here are implemented
against the documented API shape and are not validated against a live
server.

Invariants shared with :mod:`decisionhead.backends`: the same no-redirect,
size-capped transport; an API key read from a named environment variable
at call time only; a non-finite entry in a returned vector is a
:class:`~decisionhead.backends.BackendError`, never a feature value.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Protocol

from .backends import BackendError, Transport, _http_post_json, env_ref
from .safejson import finite_float

JSONDict = dict[str, Any]

DEFAULT_TIMEOUT_S: float = 60.0
DEFAULT_OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def _short(resp: object) -> str:
    """A bounded repr of an unusable embeddings response for an error
    message (server logs only; 2026-10-06 review, finding 10)."""
    text = repr(resp)
    return text if len(text) <= 200 else text[:200] + "...(truncated)"


def _finite_vector(embedding: list[Any], url: str) -> list[float]:
    """Every component as a finite float, or :class:`BackendError` -- a
    NaN/inf component would otherwise flow into learned-mode features
    (2026-10-06 review, finding 4)."""
    try:
        return [finite_float(v, "embedding component") for v in embedding]
    except (TypeError, ValueError):
        raise BackendError(f"embeddings response from {url} had a non-numeric or non-finite component") from None


class EmbeddingsBackend(Protocol):
    """Structural type every backend below satisfies."""

    model: str
    dim: int | None

    def embed(self, text: str) -> list[float]:
        """Returns a fixed-length embedding vector for ``text``."""
        ...


class OllamaEmbeddingsBackend:
    """ollama's native embeddings route. See module docstring."""

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim: int | None = None
        self._timeout = timeout
        self._transport = transport or (
            lambda url, body, headers, timeout: _http_post_json(url, body, headers, timeout)
        )

    def embed(self, text: str) -> list[float]:
        """POSTs ``text`` to ``{base_url}/api/embeddings`` and returns the
        vector. Raises :class:`BackendError` for a transport failure, a
        missing or empty ``embedding`` field, or a non-finite entry."""
        url = f"{self.base_url}/api/embeddings"
        resp = self._transport(url, {"model": self.model, "prompt": text}, {}, self._timeout)
        embedding = resp.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BackendError(
                f"ollama embeddings response from {url} had no usable 'embedding' field: {_short(resp)}"
            )
        vector = _finite_vector(embedding, url)
        self.dim = len(vector)
        return vector


class OpenAIEmbeddingsBackend:
    """Any OpenAI-compatible ``/v1/embeddings`` route. See module
    docstring. ``api_key_env`` names an environment variable to read a
    bearer token from, never a literal key (same convention as
    :mod:`decisionhead.backends`)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.dim: int | None = None
        self._timeout = timeout
        self._transport = transport or (
            lambda url, body, headers, timeout: _http_post_json(url, body, headers, timeout)
        )

    def embed(self, text: str) -> list[float]:
        """POSTs ``text`` to ``{base_url}/v1/embeddings`` (with a bearer
        token read from ``api_key_env`` at call time, if named) and
        returns the first vector. Raises :class:`BackendError` as
        :meth:`OllamaEmbeddingsBackend.embed` does."""
        url = f"{self.base_url}/v1/embeddings"
        headers: JSONDict = {"Content-Type": "application/json"}
        api_key = env_ref(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = self._transport(url, {"model": self.model, "input": text}, headers, self._timeout)
        data = resp.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise BackendError(
                f"OpenAI-compatible embeddings response from {url} had no usable 'data': {_short(resp)}"
            )
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BackendError(
                f"OpenAI-compatible embeddings response from {url} had no usable embedding: {_short(resp)}"
            )
        vector = _finite_vector(embedding, url)
        self.dim = len(vector)
        return vector


class FakeEmbeddingsBackend:
    """Deterministic, network-free stand-in for tests and for exercising
    :mod:`decisionhead.train`/:mod:`decisionhead.features` end to end
    without a real embeddings endpoint. The vector for a given text is a
    seeded pseudo-random function of that text's own SHA-256 digest, so
    the same text always yields the same vector (within one process and
    across processes -- it does not depend on ``random`` global state)."""

    def __init__(self, dim: int = 8, model: str = "fake-embedding-model") -> None:
        self.model = model
        self.dim: int | None = dim
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """A unit-length vector derived only from ``text``; no network."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self._dim)]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def make_embeddings_backend(
    kind: str,
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> EmbeddingsBackend:
    """Factory mirroring :func:`decisionhead.backends.make_backend`'s
    shape. ``kind`` is ``"ollama"`` or ``"openai"``; there is no ``--dry``
    embeddings backend here (use :class:`FakeEmbeddingsBackend` directly
    in tests instead, since embeddings have no natural "dry" default
    dimension to fake without a caller choosing one)."""
    if kind == "ollama":
        return OllamaEmbeddingsBackend(base_url, model=model or DEFAULT_OLLAMA_EMBEDDING_MODEL, timeout=timeout)
    if kind == "openai":
        if not model:
            raise ValueError("an OpenAI-compatible embeddings backend requires --embedding-model")
        return OpenAIEmbeddingsBackend(base_url, model=model, api_key_env=api_key_env, timeout=timeout)
    raise ValueError(f"unknown embeddings backend kind {kind!r}; expected 'ollama' or 'openai'")


def run_selftest() -> bool:
    """Standalone smoke check; see ``tests/test_embeddings.py`` for the
    thorough version."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    fake = FakeEmbeddingsBackend(dim=4)
    v1 = fake.embed("hello world")
    v2 = fake.embed("hello world")
    v3 = fake.embed("goodbye")
    check("fake embeddings are deterministic for the same text", v1 == v2)
    check("fake embeddings differ for different text", v1 != v3)
    check("fake embeddings have the requested dimension", len(v1) == 4)

    def fixture_transport(url: str, body: JSONDict, headers: JSONDict, timeout: float) -> JSONDict:
        if "/api/embeddings" in url:
            return {"embedding": [0.1, 0.2, 0.3]}
        return {"data": [{"embedding": [0.4, 0.5]}]}

    ollama_backend = OllamaEmbeddingsBackend(
        "http://127.0.0.1:11434", model="qwen3-embedding:0.6b", transport=fixture_transport
    )
    result = ollama_backend.embed("state text")
    check("ollama embeddings backend parses 'embedding'", result == [0.1, 0.2, 0.3])
    check("ollama embeddings backend records dim", ollama_backend.dim == 3)

    openai_backend = OpenAIEmbeddingsBackend("http://127.0.0.1:8000", model="e5", transport=fixture_transport)
    result2 = openai_backend.embed("state text")
    check("openai-compatible embeddings backend parses 'data[0].embedding'", result2 == [0.4, 0.5])

    made = make_embeddings_backend("ollama", "http://x", "m")
    check("make_embeddings_backend builds an ollama backend", isinstance(made, OllamaEmbeddingsBackend))

    try:
        make_embeddings_backend("bogus", "http://x", "m")
        check("unknown embeddings backend kind rejected", False)
    except ValueError:
        check("unknown embeddings backend kind rejected", True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
