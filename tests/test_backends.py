from __future__ import annotations

import json
import math

import pytest

from decisionhead.backends import (
    BackendError,
    ConcurrencyLimitedBackend,
    FakeBackend,
    OllamaBackend,
    OpenAIBackend,
    distribution_from_top_logprobs,
    env_ref,
    make_backend,
    probe_backend,
    unwrap_backend,
)


def _fixture_transport(fixed_response, capture=None):
    def _t(url, body, headers, timeout=60):
        if capture is not None:
            capture["url"] = url
            capture["body"] = body
            capture["headers"] = headers
        return fixed_response

    return _t


class TestDistributionFromTopLogprobs:
    def test_normalizes_over_allowed_family(self) -> None:
        top = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.0}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert abs(dist["Yes"] + dist["No"] - 1.0) < 1e-9
        assert meta["no_family_match"] is False

    def test_case_and_whitespace_insensitive(self) -> None:
        top = [{"token": " yes", "logprob": -0.1}, {"token": "NO", "logprob": -2.0}]
        dist, _meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert dist["Yes"] > dist["No"]

    def test_trailing_punctuation_tolerated(self) -> None:
        top = [{"token": "A)", "logprob": -0.1}, {"token": "B.", "logprob": -3.0}]
        dist, _meta = distribution_from_top_logprobs(top, ["A", "B"])
        assert dist["A"] > dist["B"]

    def test_no_family_match_falls_back_to_uniform(self) -> None:
        top = [{"token": "X", "logprob": -0.1}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert dist == {"Yes": 0.5, "No": 0.5}
        assert meta["no_family_match"] is True

    def test_empty_top_logprobs_falls_back_to_uniform(self) -> None:
        dist, meta = distribution_from_top_logprobs([], ["Yes", "No"])
        assert dist == {"Yes": 0.5, "No": 0.5}
        assert meta["no_family_match"] is True

    def test_non_dict_entries_are_skipped_not_raised(self) -> None:
        top = [{"token": "Yes", "logprob": -0.1}, "garbage", None]
        dist, _meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert dist["Yes"] > 0

    def test_unparsed_mass_excludes_family_mass(self) -> None:
        top = [{"token": "Yes", "logprob": -0.1}, {"token": "Sure", "logprob": -1.0}]
        _dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert meta["unparsed_mass"] > 0


class TestNonFiniteLogprobDefense:
    """Security review 2026-09-23, finding 4: a malformed/compromised
    backend response with a NaN or infinite logprob must never produce a
    non-finite probability in the returned distribution (a bare NaN or
    Infinity token is not valid JSON per RFC 8259), and an
    unrepresentably large logprob must not crash with an uncaught
    OverflowError. Reproduces the reviewer's exact repro inputs."""

    def test_nan_logprob_does_not_propagate_into_the_distribution(self) -> None:
        top = [{"token": "Yes", "logprob": float("nan")}, {"token": "No", "logprob": -1.0}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert all(math.isfinite(v) for v in dist.values())
        assert meta["non_finite_logprob_count"] == 1
        assert "NaN" not in json.dumps(dist)  # must serialize as valid JSON

    def test_infinite_logprob_does_not_propagate(self) -> None:
        top = [{"token": "Yes", "logprob": float("inf")}, {"token": "No", "logprob": -1.0}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert all(math.isfinite(v) for v in dist.values())
        assert meta["non_finite_logprob_count"] == 1

    def test_overflowing_logprob_does_not_raise(self) -> None:
        # 800.0 -> math.exp(800.0) overflows a Python float (the real
        # boundary is ~709.78); previously an uncaught OverflowError
        # inside _mass(), now degrades to zero mass for that one entry.
        top = [{"token": "Yes", "logprob": 800.0}, {"token": "No", "logprob": -1.0}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert all(math.isfinite(v) for v in dist.values())
        assert meta["non_finite_logprob_count"] == 1
        assert dist["No"] == 1.0  # the one finite entry gets all the mass

    def test_all_non_finite_falls_back_to_uniform_and_is_flagged(self) -> None:
        top = [{"token": "Yes", "logprob": float("nan")}, {"token": "No", "logprob": float("nan")}]
        dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert dist == {"Yes": 0.5, "No": 0.5}
        assert meta["no_family_match"] is True
        assert meta["non_finite_logprob_count"] == 2

    def test_clean_input_reports_zero_non_finite_entries(self) -> None:
        top = [{"token": "Yes", "logprob": -0.1}, {"token": "No", "logprob": -2.0}]
        _dist, meta = distribution_from_top_logprobs(top, ["Yes", "No"])
        assert meta["non_finite_logprob_count"] == 0

    def test_end_to_end_through_ollama_backend_stays_finite_and_visible(self) -> None:
        resp = {
            "logprobs": [
                {
                    "token": "Yes",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "Yes", "logprob": float("nan")},
                        {"token": "No", "logprob": -1.0},
                    ],
                }
            ]
        }
        backend = OllamaBackend("http://example:11434", "m", transport=_fixture_transport(resp))
        dist = backend.token_distribution([], "Q?", ["Yes", "No"])
        assert math.isfinite(dist["Yes"])
        assert math.isfinite(dist["No"])
        assert dist["_raw"]["non_finite_logprob_count"] == 1
        json.dumps(dist)  # must not raise


class TestOllamaBackend:
    def test_noul_p_yes_matches_hand_computation(self) -> None:
        resp = {
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
                    ],
                }
            ],
            "prompt_eval_count": 120,
            "eval_count": 1,
        }
        cap: dict = {}
        backend = OllamaBackend("http://example:11434", "qwen3.8:27b", transport=_fixture_transport(resp, cap))
        dist = backend.token_distribution([{"role": "system", "content": "s"}], "Q?", ["Yes", "No"])
        yes_mass = math.exp(-0.1378) + math.exp(-2.097) + math.exp(-5.90)
        no_mass = math.exp(-6.5)
        expected = yes_mass / (yes_mass + no_mass)
        assert abs(dist["Yes"] - expected) < 1e-9
        assert cap["url"].endswith("/api/chat")
        assert cap["body"]["think"] is False
        assert cap["body"]["options"] == {"num_predict": 1, "temperature": 0}
        assert dist["_raw"]["usage"] == {"input_tokens": 120, "output_tokens": 1}

    def test_never_calls_v1_route(self) -> None:
        cap: dict = {}
        backend = OllamaBackend("http://example:11434", "m", transport=_fixture_transport({}, cap))
        backend.token_distribution([], "Q?", ["Yes", "No"])
        assert "/v1" not in cap["url"]

    def test_malformed_response_degrades_to_uniform(self) -> None:
        backend = OllamaBackend("http://example:11434", "m", transport=_fixture_transport({"unexpected": True}))
        dist = backend.token_distribution([], "Q?", ["Yes", "No"])
        assert dist["Yes"] == 0.5
        assert dist["No"] == 0.5

    def test_http_error_wrapped_as_backend_error(self) -> None:
        def _raising_transport(url, body, headers, timeout=60):
            raise BackendError(f"HTTP 500 from {url}: boom")

        backend = OllamaBackend("http://example:11434", "m", transport=_raising_transport)
        with pytest.raises(BackendError, match="HTTP 500"):
            backend.token_distribution([], "Q?", ["Yes", "No"])


class TestOpenAIBackend:
    def test_disables_template_thinking_and_reads_logprobs(self) -> None:
        resp = {
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
                    }
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 1},
        }
        cap: dict = {}
        backend = OpenAIBackend("http://example:8000", "m", transport=_fixture_transport(resp, cap))
        dist = backend.token_distribution([], "Q?", ["A", "B"])
        assert cap["url"].endswith("/v1/chat/completions")
        assert cap["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert cap["body"]["max_tokens"] == 1
        assert dist["A"] > dist["B"]
        assert dist["_raw"]["usage"] == {"input_tokens": 80, "output_tokens": 1}

    def test_api_key_read_from_named_env_var_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TEST_KEY", "FAKEKEY-sekret-value")
        cap: dict = {}
        backend = OpenAIBackend(
            "http://example:8000", "m", transport=_fixture_transport({}, cap), api_key_ref="MY_TEST_KEY"
        )
        backend.token_distribution([], "Q?", ["A", "B"])
        assert cap["headers"]["Authorization"] == "Bearer FAKEKEY-sekret-value"

    def test_no_api_key_ref_means_no_auth_header(self) -> None:
        cap: dict = {}
        backend = OpenAIBackend("http://example:8000", "m", transport=_fixture_transport({}, cap))
        backend.token_distribution([], "Q?", ["A", "B"])
        assert "Authorization" not in cap["headers"]

    def test_unset_env_var_means_no_auth_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
        cap: dict = {}
        backend = OpenAIBackend(
            "http://example:8000", "m", transport=_fixture_transport({}, cap), api_key_ref="SOME_UNSET_VAR"
        )
        backend.token_distribution([], "Q?", ["A", "B"])
        assert "Authorization" not in cap["headers"]


class TestFakeBackend:
    def test_deterministic_for_same_seed_and_question(self) -> None:
        fb = FakeBackend(seed=7)
        a1 = fb.token_distribution([], "question one", ["Yes", "No"])
        a2 = fb.token_distribution([], "question one", ["Yes", "No"])
        assert a1["Yes"] == a2["Yes"]

    def test_sums_to_one(self) -> None:
        fb = FakeBackend(seed=7)
        dist = fb.token_distribution([], "q", ["Yes", "No"])
        assert abs(dist["Yes"] + dist["No"] - 1.0) < 1e-9

    def test_varies_across_questions(self) -> None:
        fb = FakeBackend(seed=7)
        a1 = fb.token_distribution([], "question one", ["Yes", "No"])
        a3 = fb.token_distribution([], "question two", ["Yes", "No"])
        assert a1["Yes"] != a3["Yes"]

    def test_zero_allowed_tokens_does_not_crash(self) -> None:
        fb = FakeBackend(seed=1)
        dist = fb.token_distribution([], "q", [])
        assert dist["_unparsed"] == 0.0


class TestEnvRef:
    def test_returns_none_for_no_name(self) -> None:
        assert env_ref(None) is None

    def test_reads_named_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DH_TEST_VAR", "abc")
        assert env_ref("DH_TEST_VAR") == "abc"

    def test_missing_variable_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DH_TEST_VAR_MISSING", raising=False)
        assert env_ref("DH_TEST_VAR_MISSING") is None


class TestMakeBackend:
    def test_dry_returns_fake_backend(self) -> None:
        backend = make_backend("ollama", dry=True, dry_seed=3, model="m")
        assert isinstance(backend, FakeBackend)
        assert backend.seed == 3

    def test_ollama_backend(self) -> None:
        backend = make_backend("ollama", base_url="http://x", model="m")
        assert isinstance(backend, OllamaBackend)

    def test_openai_backend(self) -> None:
        backend = make_backend("openai", base_url="http://x", model="m")
        assert isinstance(backend, OpenAIBackend)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            make_backend("not-a-backend", base_url="http://x", model="m")

    def test_max_concurrency_wraps_the_backend(self) -> None:
        backend = make_backend("ollama", base_url="http://x", model="m", max_concurrency=1)
        assert isinstance(backend, ConcurrencyLimitedBackend)

    def test_no_max_concurrency_leaves_the_backend_unwrapped(self) -> None:
        backend = make_backend("ollama", base_url="http://x", model="m")
        assert isinstance(backend, OllamaBackend)


class TestConcurrencyLimitedBackend:
    def test_delegates_token_distribution(self) -> None:
        inner = FakeBackend(seed=1)
        wrapped = ConcurrencyLimitedBackend(inner, max_concurrency=1)
        result = wrapped.token_distribution([], "Q?", ["Yes", "No"])
        assert result == inner.token_distribution([], "Q?", ["Yes", "No"])

    def test_model_property_delegates(self) -> None:
        inner = FakeBackend(seed=1, model="m-1")
        wrapped = ConcurrencyLimitedBackend(inner, max_concurrency=1)
        assert wrapped.model == "m-1"

    def test_wrapped_type_name_reports_the_real_backend(self) -> None:
        inner = FakeBackend(seed=1)
        wrapped = ConcurrencyLimitedBackend(inner, max_concurrency=1)
        assert wrapped.wrapped_type_name == "FakeBackend"

    def test_unwrap_returns_the_real_backend(self) -> None:
        inner = FakeBackend(seed=1)
        wrapped = ConcurrencyLimitedBackend(inner, max_concurrency=1)
        assert wrapped.unwrap() is inner
        assert unwrap_backend(wrapped) is inner
        assert unwrap_backend(inner) is inner  # a no-op on an already-unwrapped backend

    def test_getattr_delegates_unknown_attributes(self) -> None:
        inner = OllamaBackend("http://example:11434", "m")
        wrapped = ConcurrencyLimitedBackend(inner, max_concurrency=1)
        assert wrapped.base_url == inner.base_url

    def test_rejects_non_positive_max_concurrency(self) -> None:
        with pytest.raises(ValueError):
            ConcurrencyLimitedBackend(FakeBackend(), max_concurrency=0)

    def test_serializes_concurrent_calls_at_concurrency_one(self) -> None:
        import threading
        import time

        active = {"count": 0, "max_seen": 0}
        lock = threading.Lock()

        class SlowInner:
            model = "slow"

            def token_distribution(self, prefix_messages, question_text, allowed_tokens):
                with lock:
                    active["count"] += 1
                    active["max_seen"] = max(active["max_seen"], active["count"])
                time.sleep(0.05)
                with lock:
                    active["count"] -= 1
                return {"Yes": 1.0, "No": 0.0, "_unparsed": 0.0, "_raw": {}}

        wrapped = ConcurrencyLimitedBackend(SlowInner(), max_concurrency=1)
        threads = [
            threading.Thread(target=wrapped.token_distribution, args=([], "Q?", ["Yes", "No"])) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert active["max_seen"] == 1

    def test_max_concurrency_override_uses_its_own_semaphore(self) -> None:
        import threading
        import time

        active = {"count": 0, "max_seen": 0}
        lock = threading.Lock()

        class SlowInner:
            model = "slow"

            def token_distribution(self, prefix_messages, question_text, allowed_tokens):
                with lock:
                    active["count"] += 1
                    active["max_seen"] = max(active["max_seen"], active["count"])
                time.sleep(0.05)
                with lock:
                    active["count"] -= 1
                return {"Yes": 1.0, "No": 0.0, "_unparsed": 0.0, "_raw": {}}

        wrapped = ConcurrencyLimitedBackend(SlowInner(), max_concurrency=1)
        threads = [
            threading.Thread(
                target=wrapped.token_distribution,
                args=([], "Q?", ["Yes", "No"]),
                kwargs={"max_concurrency_override": 5},
            )
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert active["max_seen"] == 5

    def test_rejects_non_positive_override(self) -> None:
        wrapped = ConcurrencyLimitedBackend(FakeBackend(), max_concurrency=1)
        with pytest.raises(ValueError):
            wrapped.token_distribution([], "Q?", ["Yes", "No"], max_concurrency_override=0)


class TestProbeBackend:
    def test_healthy_fake_backend(self) -> None:
        result = probe_backend(FakeBackend(seed=1))
        assert "distribution" in result
        assert "latency_s" in result

    def test_unhealthy_when_no_family_match(self) -> None:
        no_match_resp = {
            "logprobs": [{"token": "X", "logprob": -0.1, "top_logprobs": [{"token": "X", "logprob": -0.1}]}]
        }
        backend = OllamaBackend("http://example:11434", "m", transport=_fixture_transport(no_match_resp))
        result = probe_backend(backend)
        assert result["healthy"] is False
