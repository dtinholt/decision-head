"""Verifies the six dh_*.py compatibility shims at the repository root
still work exactly as the original prototype's modules did: importable,
re-exporting the expected names, and runnable as
``python3 dh_X.py --selftest``. These are what the fleet's existing calls
depend on."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SHIMS = ["dh_core.py", "dh_backend.py", "dh_server.py", "dh_client.py", "dh_calibrate.py", "dh_bench.py"]


@pytest.fixture(autouse=True)
def _package_root_on_syspath():
    """The in-process ``import dh_X`` tests below need this package's own
    directory on ``sys.path``, found via ``__file__`` rather than assumed
    from the working directory: pytest's own invocation directory decides
    what ``sys.path[0]`` is (e.g. ``python -m pytest tools/decision-head``
    from a monorepo root puts the monorepo root there, not this package's
    directory), so relying on it would make these tests pass or fail
    depending on where pytest was launched from. The subprocess-based
    ``test_shim_selftest_passes`` below does not need this: it already
    sets ``cwd=REPO_ROOT`` on the child process, which independently makes
    the child's own ``sys.path[0]`` correct."""
    root = str(REPO_ROOT)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(root)


@pytest.mark.parametrize("shim", SHIMS)
def test_shim_selftest_passes(shim: str) -> None:
    result = subprocess.run(
        [sys.executable, shim, "--selftest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{shim} --selftest failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "FAIL" not in result.stdout, result.stdout


def test_dh_core_reexports_expected_names() -> None:
    import dh_core

    assert hasattr(dh_core, "answer_questions")
    assert hasattr(dh_core, "normalize_question")
    assert hasattr(dh_core, "QuestionError")


def test_dh_backend_reexports_expected_names() -> None:
    import dh_backend

    assert hasattr(dh_backend, "OllamaBackend")
    assert hasattr(dh_backend, "OpenAIBackend")
    assert hasattr(dh_backend, "FakeBackend")


def test_dh_client_reexports_ask() -> None:
    import dh_client

    assert hasattr(dh_client, "ask")
    assert hasattr(dh_client, "health")
