"""Compatibility shim.

All logic now lives in :mod:`decisionhead.client`. This module exists only
so that code written against the original prototype -- `from dh_client
import ask` or `python3 dh_client.py --base ... --questions ...
[--selftest]` -- keeps working unchanged after the `decisionhead` package
restructuring.
"""

from __future__ import annotations

import sys

from decisionhead.client import *  # noqa: F401,F403
from decisionhead.client import main as _main
from decisionhead.client import run_selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(0 if run_selftest() else 1)
    raise SystemExit(_main(sys.argv[1:]))
