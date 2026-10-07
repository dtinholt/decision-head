"""Compatibility shim.

All logic now lives in :mod:`decisionhead.core`. This module exists only
so that code written against the original prototype -- `import dh_core` or
`python3 dh_core.py --selftest` -- keeps working unchanged after the
`decisionhead` package restructuring.
"""

from __future__ import annotations

from decisionhead.core import *  # noqa: F401,F403
from decisionhead.core import run_selftest

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dh_core.py -- compatibility shim for decisionhead.core")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    parser.print_help()
