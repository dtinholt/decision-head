"""State-hash memoisation for decisionhead's ``fast`` mode.

Caches at PER-QUESTION granularity -- the key includes the question id
and text, not just the state -- so the same question about the same
state, asked in two different requests (or twice in one, e.g. a
workflow loop re-checking a decision it already made), never has to call
the backend a second time while a live cache entry for it exists.

Scope note: this cache is wired up for ``mode == "fast"``
(:func:`decisionhead.core.answer_questions`) only.
``mode == "deliberate"`` samples with intentional randomness (its whole
point is to let the model reason before answering; caching it would
either pin one sampled reasoning trace forever or silently defeat the
mode), and ``mode == "learned"`` never calls a backend at all (a trained
classifier read is already cheap and local) -- see
``decisionhead.server.run_server``.

Not a distributed cache: every server process holds its own in-memory
store, optionally persisted to a JSONL file (``cache_path``) so a
restarted process does not start fully cold. The file is append-only;
entries evicted from the in-memory LRU are not removed from it, so a
long-lived ``cache_path`` grows over time -- acceptable for the
restart-warms-the-cache use case this is built for, not intended as a
long-term store. See ``docs/design.md`` and the README's caching section.

Invariants: the key covers every input that can change an answer
(model, backend route, state, question id and text, mode, calibration
version), so recalibrating or switching models never serves a stale
probability; the server writes an entry only after its request has
succeeded and never for a degraded reading; the cache file is created
``0600``, holds hashed keys and answers but no state text, and is
parsed strictly on load. Anyone who can write ``cache_path`` can plant
answers, so keep it private (``docs/configuration.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from .safejson import dumps_strict, loads_strict

JSONDict = dict[str, Any]


def cache_key(
    *,
    model: str,
    backend_route: str,
    state_json: str,
    question_id: str,
    question_text: str,
    mode: str,
    calibration_version: str,
) -> str:
    """A stable SHA-256 hex digest over exactly the fields the README's
    caching section documents: model id, backend route, canonical JSON of
    the state, the question's id and its fully-built text (so a change to
    calibration-affecting instructions or option wording is a different
    key), the answering mode, and a calibration version tag. Any change
    to any of these fields is deliberately a different key -- each one
    can change the answer."""
    payload = json.dumps(
        {
            "model": model,
            "backend_route": backend_route,
            "state": state_json,
            "question_id": question_id,
            "question_text": question_text,
            "mode": mode,
            "calibration_version": calibration_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Thread-safe in-process LRU cache with a per-entry TTL, optionally
    persisted to a JSONL file so it survives a server restart.

    ``max_entries`` bounds the IN-MEMORY store (least-recently-used
    eviction); ``ttl_s`` is the entry lifetime in seconds from the moment
    it was written. A miss (absent or expired) returns ``None``."""

    def __init__(self, max_entries: int = 10000, ttl_s: float = 86400.0, path: str | None = None) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.path = path
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[float, JSONDict]] = OrderedDict()  # key -> (expires_at, value)
        if path and os.path.exists(path):
            self._load_from_file(path)

    def _load_from_file(self, path: str) -> None:
        """Best-effort warm start: a missing, unreadable, or partially
        corrupt file just leaves the cache cold rather than failing
        server startup -- a cache is an optimization, never a
        correctness dependency."""
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        now = time.time()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = loads_strict(line)
                key = row["key"]
                expires_at = float(row["expires_at"])
                value = row["value"]
            except (KeyError, ValueError, TypeError):
                continue
            if expires_at <= now:
                continue
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def get(self, key: str) -> JSONDict | None:
        """The live entry for ``key``, or ``None`` if absent or expired
        (an expired entry is dropped). A hit marks the entry most
        recently used. The stored object itself is returned, not a copy;
        callers copy before mutating."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= time.time():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: JSONDict) -> None:
        """Stores ``value`` under ``key`` for ``ttl_s`` seconds, evicting
        the least recently used entries past ``max_entries``, and appends
        the row to ``path`` if one is configured. A failed file write is
        ignored; the in-memory entry still stands."""
        expires_at = time.time() + self.ttl_s
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        if self.path:
            try:
                # Owner-only (0600) on creation and strict JSON (no bare
                # NaN/Infinity) -- 2026-10-06 security review, findings 4
                # and 9. The mode only applies when this call creates the
                # file; an existing file keeps its permissions.
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    f.write(dumps_strict({"key": key, "expires_at": expires_at, "value": value}, default=str) + "\n")
            except (OSError, ValueError):
                pass  # best-effort persistence; the in-memory cache still works this run

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed.
    The equivalent, more thorough assertions live in
    ``tests/test_cache.py``."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    k1 = cache_key(
        model="m", backend_route="ollama", state_json='{"a": 1}', question_id="q1", question_text="Q?",
        mode="fast", calibration_version="",
    )
    k2 = cache_key(
        model="m", backend_route="ollama", state_json='{"a": 1}', question_id="q1", question_text="Q?",
        mode="fast", calibration_version="",
    )
    k3 = cache_key(
        model="m", backend_route="ollama", state_json='{"a": 2}', question_id="q1", question_text="Q?",
        mode="fast", calibration_version="",
    )
    check("cache_key is deterministic", k1 == k2)
    check("cache_key changes with the state", k1 != k3)

    cache = ResponseCache(max_entries=2, ttl_s=60)
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    check("cache returns what was set", cache.get("a") == {"v": 1})
    cache.set("c", {"v": 3})  # evicts "b" (least recently used after the get("a") above)
    check("cache evicts the LRU entry past max_entries", cache.get("b") is None)
    check("cache keeps the recently-used entry", cache.get("a") == {"v": 1})
    check("cache keeps the newest entry", cache.get("c") == {"v": 3})

    expiring = ResponseCache(max_entries=10, ttl_s=0.01)
    expiring.set("x", {"v": 1})
    time.sleep(0.02)
    check("cache expires an entry past its TTL", expiring.get("x") is None)

    check("cache reports its size", len(cache) == 2)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
