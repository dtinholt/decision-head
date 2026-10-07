"""Optional numpy acceleration, used only by :mod:`decisionhead.model`.

The package guarantees zero *runtime* dependencies for serving; the
training path (:mod:`decisionhead.train`, :mod:`decisionhead.model`) may
optionally use numpy for speed via the ``decisionhead[train]`` extra, but
must work correctly with pure-Python loops when it is absent. This module
is the single place that probes for numpy so the rest of the codebase
never has a bare ``import numpy`` that would make mypy (numpy need not be
installed to type-check this package) or a zero-dependency serving
install fail.

``np`` is intentionally typed ``Any``: every caller that uses it is
already in an ``if HAVE_NUMPY:`` branch and converts results back to
plain ``list``/``float`` before they leave that branch, so no untyped
value escapes into the rest of the strictly-typed codebase.
"""

from __future__ import annotations

from typing import Any

HAVE_NUMPY: bool
np: Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover -- exercised by whichever env lacks numpy
    HAVE_NUMPY = False
    np = None
else:
    HAVE_NUMPY = True
    np = _np

__all__ = ["HAVE_NUMPY", "np"]
