from __future__ import annotations

import time
from pathlib import Path

import pytest

from decisionhead.cache import ResponseCache, cache_key


class TestCacheKey:
    def test_deterministic(self) -> None:
        kwargs = dict(
            model="m",
            backend_route="ollama://x",
            state_json='{"a": 1}',
            question_id="q1",
            question_text="Is X true?",
            mode="fast",
            calibration_version="",
        )
        assert cache_key(**kwargs) == cache_key(**kwargs)

    @pytest.mark.parametrize(
        "field",
        ["model", "backend_route", "state_json", "question_id", "question_text", "mode", "calibration_version"],
    )
    def test_changes_when_any_field_changes(self, field: str) -> None:
        base = dict(
            model="m",
            backend_route="ollama://x",
            state_json='{"a": 1}',
            question_id="q1",
            question_text="Is X true?",
            mode="fast",
            calibration_version="",
        )
        changed = dict(base)
        changed[field] = base[field] + "-changed"
        assert cache_key(**base) != cache_key(**changed)


class TestResponseCache:
    def test_miss_returns_none(self) -> None:
        cache = ResponseCache()
        assert cache.get("nope") is None

    def test_set_then_get_round_trips(self) -> None:
        cache = ResponseCache()
        cache.set("k", {"noul": 0.9})
        assert cache.get("k") == {"noul": 0.9}

    def test_ttl_expiry(self) -> None:
        cache = ResponseCache(ttl_s=0.01)
        cache.set("k", {"v": 1})
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_lru_eviction_past_max_entries(self) -> None:
        cache = ResponseCache(max_entries=2)
        cache.set("a", {"v": "a"})
        cache.set("b", {"v": "b"})
        cache.get("a")  # touch "a" so "b" becomes the LRU entry
        cache.set("c", {"v": "c"})
        assert cache.get("b") is None
        assert cache.get("a") == {"v": "a"}
        assert cache.get("c") == {"v": "c"}

    def test_len_reports_live_entry_count(self) -> None:
        cache = ResponseCache()
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        assert len(cache) == 2

    def test_rejects_non_positive_max_entries(self) -> None:
        with pytest.raises(ValueError):
            ResponseCache(max_entries=0)

    def test_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError):
            ResponseCache(ttl_s=0)

    def test_file_backed_persists_across_instances(self, tmp_path: Path) -> None:
        path = str(tmp_path / "cache.jsonl")
        cache1 = ResponseCache(path=path)
        cache1.set("k", {"v": 1})

        cache2 = ResponseCache(path=path)
        assert cache2.get("k") == {"v": 1}

    def test_file_backed_skips_expired_entries_on_load(self, tmp_path: Path) -> None:
        path = str(tmp_path / "cache.jsonl")
        cache1 = ResponseCache(path=path, ttl_s=0.01)
        cache1.set("k", {"v": 1})
        time.sleep(0.02)

        cache2 = ResponseCache(path=path)
        assert cache2.get("k") is None

    def test_missing_cache_file_starts_cold(self, tmp_path: Path) -> None:
        path = str(tmp_path / "does-not-exist.jsonl")
        cache = ResponseCache(path=path)
        assert len(cache) == 0

    def test_corrupt_line_in_cache_file_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        path.write_text("not json\n" + '{"key": "k", "expires_at": 99999999999, "value": {"v": 1}}\n')
        cache = ResponseCache(path=str(path))
        assert cache.get("k") == {"v": 1}
