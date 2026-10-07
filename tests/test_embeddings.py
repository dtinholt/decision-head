from __future__ import annotations

import pytest

from decisionhead.backends import BackendError
from decisionhead.embeddings import (
    FakeEmbeddingsBackend,
    OllamaEmbeddingsBackend,
    OpenAIEmbeddingsBackend,
    make_embeddings_backend,
)


class TestFakeEmbeddingsBackend:
    def test_deterministic_for_the_same_text(self) -> None:
        backend = FakeEmbeddingsBackend(dim=8)
        assert backend.embed("hello") == backend.embed("hello")

    def test_differs_for_different_text(self) -> None:
        backend = FakeEmbeddingsBackend(dim=8)
        assert backend.embed("hello") != backend.embed("goodbye")

    def test_has_the_requested_dimension(self) -> None:
        backend = FakeEmbeddingsBackend(dim=12)
        assert len(backend.embed("x")) == 12

    def test_is_unit_normalized(self) -> None:
        backend = FakeEmbeddingsBackend(dim=8)
        vector = backend.embed("x")
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9


class TestOllamaEmbeddingsBackend:
    def test_parses_the_embedding_field(self) -> None:
        def transport(url, body, headers, timeout):  # type: ignore[no-untyped-def]
            assert url.endswith("/api/embeddings")
            assert body == {"model": "qwen3-embedding:0.6b", "prompt": "hello"}
            return {"embedding": [0.1, 0.2, 0.3]}

        backend = OllamaEmbeddingsBackend("http://127.0.0.1:11434", transport=transport)
        assert backend.embed("hello") == [0.1, 0.2, 0.3]
        assert backend.dim == 3

    def test_missing_embedding_field_raises(self) -> None:
        backend = OllamaEmbeddingsBackend("http://x", transport=lambda *a: {})  # type: ignore[no-untyped-def]
        with pytest.raises(BackendError, match="embedding"):
            backend.embed("hello")

    def test_trailing_slash_in_base_url_is_stripped(self) -> None:
        seen = {}

        def transport(url, body, headers, timeout):  # type: ignore[no-untyped-def]
            seen["url"] = url
            return {"embedding": [1.0]}

        backend = OllamaEmbeddingsBackend("http://127.0.0.1:11434/", transport=transport)
        backend.embed("x")
        assert seen["url"] == "http://127.0.0.1:11434/api/embeddings"


class TestOpenAIEmbeddingsBackend:
    def test_parses_data_zero_embedding(self) -> None:
        def transport(url, body, headers, timeout):  # type: ignore[no-untyped-def]
            assert url.endswith("/v1/embeddings")
            assert body == {"model": "e5", "input": "hello"}
            return {"data": [{"embedding": [0.4, 0.5]}]}

        backend = OpenAIEmbeddingsBackend("http://127.0.0.1:8000", model="e5", transport=transport)
        assert backend.embed("hello") == [0.4, 0.5]

    def test_sends_bearer_token_when_api_key_env_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_EMBED_KEY", "FAKEKEY-secret-value")
        seen = {}

        def transport(url, body, headers, timeout):  # type: ignore[no-untyped-def]
            seen["headers"] = headers
            return {"data": [{"embedding": [1.0]}]}

        backend = OpenAIEmbeddingsBackend("http://x", model="e5", api_key_env="MY_EMBED_KEY", transport=transport)
        backend.embed("hello")
        assert seen["headers"]["Authorization"] == "Bearer FAKEKEY-secret-value"

    def test_missing_data_raises(self) -> None:
        backend = OpenAIEmbeddingsBackend("http://x", model="e5", transport=lambda *a: {})  # type: ignore[no-untyped-def]
        with pytest.raises(BackendError, match="data"):
            backend.embed("hello")


class TestMakeEmbeddingsBackend:
    def test_builds_ollama_backend(self) -> None:
        backend = make_embeddings_backend("ollama", "http://x", "")
        assert isinstance(backend, OllamaEmbeddingsBackend)

    def test_builds_openai_backend(self) -> None:
        backend = make_embeddings_backend("openai", "http://x", "e5")
        assert isinstance(backend, OpenAIEmbeddingsBackend)

    def test_openai_requires_a_model(self) -> None:
        with pytest.raises(ValueError, match="embedding-model"):
            make_embeddings_backend("openai", "http://x", "")

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            make_embeddings_backend("bogus", "http://x", "m")
