from __future__ import annotations

import json

import pytest

from decisionhead.__main__ import main


class TestTopLevelCLI:
    def test_no_args_prints_usage_and_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main([])
        assert code == 2
        assert "decisionhead" in capsys.readouterr().out

    def test_help_flag_exits_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--help"])
        assert code == 0

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--version"])
        assert code == 0
        assert "decisionhead" in capsys.readouterr().out

    def test_unknown_command_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["not-a-real-command"])
        assert code == 2
        assert "unknown command" in capsys.readouterr().err


class TestProbeCommand:
    def test_probe_dry_backend(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["probe", "--dry"])
        assert code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["healthy"] is True

    def test_probe_requires_base_and_model_unless_dry(self) -> None:
        with pytest.raises(SystemExit):
            main(["probe"])


class TestSelftestCommand:
    def test_selftest_runs_all_modules_and_passes(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["selftest"])
        out = capsys.readouterr().out
        assert "ALL PASSED" in out
        assert code == 0
