"""Compatibility shim.

All logic now lives in :mod:`decisionhead.backends`. This module exists
only so that code written against the original prototype -- `import
dh_backend` or `python3 dh_backend.py --selftest` -- keeps working
unchanged after the `decisionhead` package restructuring.
"""

from __future__ import annotations

from decisionhead.backends import *  # noqa: F401,F403
from decisionhead.backends import run_selftest

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dh_backend.py -- compatibility shim for decisionhead.backends")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    parser.print_help()
