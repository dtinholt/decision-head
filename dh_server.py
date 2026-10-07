"""Compatibility shim.

All logic now lives in :mod:`decisionhead.server`. This module exists only
so that code written against the original prototype -- `python3
dh_server.py --backend ollama --base ... --model ... [--selftest]` --
keeps working unchanged after the `decisionhead` package restructuring.
The full, hardened flag set is documented by `decisionhead serve -h`.
"""

from __future__ import annotations

import sys

from decisionhead.server import *  # noqa: F401,F403
from decisionhead.server import main as _main
from decisionhead.server import run_selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(0 if run_selftest() else 1)
    raise SystemExit(_main(sys.argv[1:]))
