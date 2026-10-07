"""A per-request deadline that background work checks before every
backend call.

Stdlib offers no way to cancel a thread blocked in a socket call, so a
request that times out cannot stop the one backend call already in
flight. What it can stop is everything queued behind it: every question
not yet asked, every sample not yet requested, and every wait on the
backend concurrency semaphore. Before this existed, a timed-out request
released its concurrency slot and returned 504 while its worker kept
issuing backend calls for every remaining question, so a client could
keep the backend permanently busy on discarded work
(``docs/security-review-2026-10-06.md``, finding 1).

The deadline travels two ways: explicitly (``answer_questions(...,
deadline=...)`` and friends) and as a thread-local "current deadline"
set by each worker thread, which :class:`decisionhead.backends.
ConcurrencyLimitedBackend` and the multi-request sampling loops read
without any change to the backend protocol's method signatures.

Invariants:

* once :meth:`Deadline.check` has raised for a deadline, it raises on
  every later call (time only moves forward and a cancel is permanent);
* :class:`DeadlineExceeded` subclasses :class:`TimeoutError`, so the
  server maps it to the same ``504`` as its own request timeout;
* nothing here interrupts a call that has already started; the bound on
  that call is the backend socket timeout.
"""

from __future__ import annotations

import threading
import time


class DeadlineExceeded(TimeoutError):
    """Raised instead of starting a backend call once the request's
    deadline has passed or the request was cancelled."""


class Deadline:
    """A monotonic-clock deadline plus an explicit cancel flag."""

    def __init__(self, timeout_s: float) -> None:
        """``timeout_s`` seconds from now on the monotonic clock, so a
        wall-clock change cannot extend or shorten a request."""
        self.expires_at = time.monotonic() + timeout_s
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Marks the request abandoned; every later :meth:`check` raises."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        """``True`` once :meth:`cancel` has been called."""
        return self._cancelled.is_set()

    def remaining(self) -> float:
        """Seconds left, never negative; ``0.0`` once cancelled."""
        if self._cancelled.is_set():
            return 0.0
        return max(0.0, self.expires_at - time.monotonic())

    def expired(self) -> bool:
        """``True`` once the deadline has passed or was cancelled."""
        return self.remaining() <= 0.0

    def check(self) -> None:
        """Raises :class:`DeadlineExceeded` if no backend call should
        start any more."""
        if self.expired():
            raise DeadlineExceeded("request deadline passed; queued backend work abandoned")


_local = threading.local()


def set_current(deadline: Deadline | None) -> None:
    """Sets the calling thread's current deadline (``None`` clears it)."""
    _local.deadline = deadline


def current() -> Deadline | None:
    """The calling thread's current deadline, or ``None``."""
    return getattr(_local, "deadline", None)


def check_current() -> None:
    """:meth:`Deadline.check` on the calling thread's deadline, if any."""
    d = current()
    if d is not None:
        d.check()
