"""Strict JSON in and out: no NaN/Infinity tokens, no unbounded numbers.

Python's ``json`` module is lenient in both directions by default:
``json.loads`` accepts the non-standard ``NaN``/``Infinity``/``-Infinity``
literals (and silently turns an out-of-range literal like ``1e999`` into
``inf``), and ``json.dumps`` writes a non-finite float as a bare ``NaN``
or ``Infinity`` token, which RFC 8259 does not allow and strict clients
reject. Every untrusted parse (request bodies, model replies, operator
calibration files, the cache file) and every emitted document (HTTP
responses, audit rows, cache rows) goes through this module instead.
Added after the independent security review dated 2026-10-06
(``docs/security-review-2026-10-06.md``, findings 4 and 5).

Invariants: :func:`loads_strict` raises ``ValueError`` (never returns
``nan``/``inf``) for a non-finite or over-long number; :func:`dumps_strict`
never emits a bare ``NaN``/``Infinity`` token, and records
``non_finite_replaced`` on any top-level object in which it had to
substitute ``null``.
"""

from __future__ import annotations

import json
import math
from typing import Any

#: Longest numeric literal accepted by :func:`loads_strict`. No value
#: this service reads legitimately needs more than a few dozen digits;
#: the cap keeps CPython's own 4300-digit int guard (which raises a plain
#: ``ValueError``) from ever being the thing that stops a hostile literal.
DEFAULT_MAX_NUMBER_LEN = 64


class NonFiniteJSONError(ValueError):
    """A JSON document contained a non-finite number or an over-long
    numeric literal. A ``ValueError`` subclass, like
    ``json.JSONDecodeError``, so a caller catching ``ValueError`` already
    handles it."""


def _reject_constant(name: str) -> Any:
    raise NonFiniteJSONError(f"non-finite JSON literal {name!r} is not allowed")


def loads_strict(text: str | bytes, *, max_number_len: int = DEFAULT_MAX_NUMBER_LEN) -> Any:
    """``json.loads`` that raises :class:`NonFiniteJSONError` (a
    ``ValueError``) on ``NaN``/``Infinity``, on a float literal that
    overflows to infinity, and on any numeric literal longer than
    ``max_number_len`` characters."""

    def _parse_int(s: str) -> int:
        if len(s) > max_number_len:
            raise NonFiniteJSONError(f"numeric literal longer than {max_number_len} characters")
        return int(s)

    def _parse_float(s: str) -> float:
        if len(s) > max_number_len:
            raise NonFiniteJSONError(f"numeric literal longer than {max_number_len} characters")
        value = float(s)
        if not math.isfinite(value):
            raise NonFiniteJSONError("numeric literal is out of range (not finite)")
        return value

    return json.loads(text, parse_constant=_reject_constant, parse_int=_parse_int, parse_float=_parse_float)


def sanitize_non_finite(obj: Any) -> tuple[Any, int]:
    """Returns ``(copy, n_replaced)``: a deep copy of ``obj`` (dicts,
    lists and tuples) with every non-finite float replaced by ``None``,
    and how many were replaced. A last line of defence: every known
    source of a non-finite value is validated where it enters, so a
    non-zero count here means a new source slipped through, and the
    caller records it rather than emitting an invalid document."""
    count = 0

    def _walk(value: Any) -> Any:
        nonlocal count
        if isinstance(value, float) and not math.isfinite(value):
            count += 1
            return None
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_walk(v) for v in value]
        return value

    return _walk(obj), count


def dumps_strict(obj: Any, *, flag_key: str | None = "non_finite_replaced", **kwargs: Any) -> str:
    """``json.dumps(..., allow_nan=False)`` after
    :func:`sanitize_non_finite`. When anything was replaced and ``obj``
    is a dict and ``flag_key`` is set, the count is recorded under
    ``flag_key`` so the replacement is visible rather than silent.
    ``kwargs`` pass through to ``json.dumps`` (e.g. ``default=str``)."""
    clean, n = sanitize_non_finite(obj)
    if n and flag_key and isinstance(clean, dict):
        clean[flag_key] = n
    return json.dumps(clean, allow_nan=False, **kwargs)


def finite_float(value: Any, what: str) -> float:
    """``float(value)`` that raises ``ValueError`` naming ``what`` if the
    result is NaN or infinite (``float("inf")`` and ``float("nan")``
    accept the strings ``"inf"``/``"nan"``, so a string-typed field can
    smuggle one in even past a strict JSON parser). Booleans are refused
    too, since ``float(True)`` is a silent ``1.0``."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be a finite number (got {value!r})")
    return result
