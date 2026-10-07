"""Exercises a running decisionhead server from Python.

Start a server first, either against a real backend:
    decisionhead serve --backend ollama --base http://127.0.0.1:11434 --model qwen3.8:27b
or, with no model and no network at all, against the deterministic fake backend:
    decisionhead serve --dry

Then:
    python3 examples/python_client_example.py [base_url]
"""

from __future__ import annotations

import json
import pathlib
import sys

from decisionhead.client import ask, health

STATE = (
    "Acme Logistics BV submitted a certificate of insurance dated 2026-01-15 and a "
    "signed W-9. No endorsement schedule was attached, and the entity name on the "
    "COI (Acme Logistics Inc) does not match the W-9 (Acme Logistics BV)."
)


def main() -> int:
    """Checks ``/health``, sends the example question set about
    ``STATE``, and prints the response and the recommended next step."""
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"

    h = health(base)
    print(f"server health: {h}")

    questions_path = pathlib.Path(__file__).with_name("supplier_onboarding_questions.json")
    questions = json.loads(questions_path.read_text())

    result = ask(base, STATE, questions)
    print(json.dumps(result, indent=2))

    next_step = result["answers"]["next_step"]
    print(f"\nrecommended next step: {next_step['choice']} (confidence {next_step['confidence']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
