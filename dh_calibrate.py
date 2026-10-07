"""Compatibility shim.

All logic now lives in :mod:`decisionhead.calibrate`. This module exists
only so that code written against the original prototype -- `python3
dh_calibrate.py --labelled cases.jsonl --backend ollama --base ... --model
... [--selftest]` -- keeps working unchanged after the `decisionhead`
package restructuring.
"""

from __future__ import annotations

import sys

from decisionhead.calibrate import *  # noqa: F401,F403
from decisionhead.calibrate import main as _main
from decisionhead.calibrate import run_selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(0 if run_selftest() else 1)
    raise SystemExit(_main(sys.argv[1:]))
