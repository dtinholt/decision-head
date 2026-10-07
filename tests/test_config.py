from __future__ import annotations

import pytest

from decisionhead.config import ConfigError, ServerConfig, load_toml_file, resolve_config


class TestDefaults:
    def test_defaults_with_no_sources(self) -> None:
        cfg = resolve_config({}, env={})
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8787
        assert cfg.backend == "ollama"
        assert cfg.max_request_bytes == 1024 * 1024


class TestPrecedence:
    def test_env_overrides_default(self) -> None:
        cfg = resolve_config({}, env={"DECISIONHEAD_PORT": "9999"})
        assert cfg.port == 9999

    def test_cli_overrides_env(self) -> None:
        cfg = resolve_config({"port": 1111}, env={"DECISIONHEAD_PORT": "9999"})
        assert cfg.port == 1111

    def test_cli_none_does_not_override_env(self) -> None:
        cfg = resolve_config({"port": None}, env={"DECISIONHEAD_PORT": "9999"})
        assert cfg.port == 9999

    def test_toml_overrides_default(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nport = 7000\n")
        cfg = resolve_config({}, config_path=str(toml_path), env={})
        assert cfg.port == 7000

    def test_env_overrides_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nport = 7000\n")
        cfg = resolve_config({}, config_path=str(toml_path), env={"DECISIONHEAD_PORT": "8000"})
        assert cfg.port == 8000

    def test_cli_overrides_toml_and_env(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nport = 7000\n")
        cfg = resolve_config({"port": 1234}, config_path=str(toml_path), env={"DECISIONHEAD_PORT": "8000"})
        assert cfg.port == 1234

    def test_config_path_from_env_var(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nmodel = 'from-toml'\n")
        cfg = resolve_config({}, env={"DECISIONHEAD_CONFIG": str(toml_path)})
        assert cfg.model == "from-toml"


class TestEnvCoercion:
    def test_bool_true_variants(self) -> None:
        for raw in ("1", "true", "TRUE", "yes", "on"):
            cfg = resolve_config({}, env={"DECISIONHEAD_DRY": raw})
            assert cfg.dry is True, raw

    def test_bool_false_variants(self) -> None:
        for raw in ("0", "false", "FALSE", "no", "off"):
            cfg = resolve_config({}, env={"DECISIONHEAD_DRY": raw})
            assert cfg.dry is False, raw

    def test_invalid_bool_raises(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({}, env={"DECISIONHEAD_DRY": "maybe"})

    def test_invalid_int_raises(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({}, env={"DECISIONHEAD_PORT": "not-a-number"})

    def test_float_field(self) -> None:
        cfg = resolve_config({}, env={"DECISIONHEAD_REQUEST_TIMEOUT_S": "12.5"})
        assert cfg.request_timeout_s == 12.5

    def test_empty_string_env_var_is_ignored(self) -> None:
        cfg = resolve_config({}, env={"DECISIONHEAD_MODEL": ""})
        assert cfg.model == ServerConfig().model


class TestTomlFile:
    def test_recognizes_all_sections(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            "[server]\nbackend = 'openai'\n\n"
            "[limits]\nmax_request_bytes = 2048\n\n"
            "[auth]\ncors_enabled = true\n\n"
            "[logging]\nlog_level = 'DEBUG'\n"
        )
        flat = load_toml_file(str(toml_path))
        assert flat == {
            "backend": "openai",
            "max_request_bytes": 2048,
            "cors_enabled": True,
            "log_level": "DEBUG",
        }

    def test_unrecognized_key_in_known_section_raises(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[server]\nnot_a_real_field = 1\n")
        with pytest.raises(ConfigError):
            load_toml_file(str(toml_path))

    def test_unrecognized_section_is_ignored(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("[future_section]\nanything = 1\n")
        flat = load_toml_file(str(toml_path))
        assert flat == {}

    def test_invalid_toml_raises_config_error(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not [ valid toml")
        with pytest.raises(ConfigError):
            load_toml_file(str(toml_path))

    def test_missing_file_raises_config_error(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            load_toml_file(str(tmp_path / "does_not_exist.toml"))

    def test_example_toml_file_loads_cleanly(self) -> None:
        import pathlib

        example = pathlib.Path(__file__).resolve().parent.parent / "decisionhead.example.toml"
        flat = load_toml_file(str(example))
        assert isinstance(flat, dict)

    def test_recognizes_new_sections(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            "[backend]\nmax_concurrency = 4\n\n"
            "[cache]\ncache_enabled = false\n\n"
            "[abstain]\nabstain_mode = 'handoff'\nabstain_band_lo = 0.4\nabstain_band_hi = 0.6\n"
        )
        flat = load_toml_file(str(toml_path))
        assert flat == {
            "max_concurrency": 4,
            "cache_enabled": False,
            "abstain_mode": "handoff",
            "abstain_band_lo": 0.4,
            "abstain_band_hi": 0.6,
        }


class TestMaxConcurrency:
    def test_default_is_one(self) -> None:
        cfg = resolve_config({}, env={})
        assert cfg.max_concurrency == 1
        assert cfg.max_concurrency_override_allowed is False

    def test_env_override(self) -> None:
        cfg = resolve_config({}, env={"DECISIONHEAD_MAX_CONCURRENCY": "4"})
        assert cfg.max_concurrency == 4

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"max_concurrency": 0}, env={})

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"max_concurrency": -1}, env={})


class TestCacheConfig:
    def test_defaults(self) -> None:
        cfg = resolve_config({}, env={})
        assert cfg.cache_enabled is True
        assert cfg.cache_ttl_s == 86400.0
        assert cfg.cache_max_entries == 10000
        assert cfg.cache_path is None

    def test_env_overrides(self, tmp_path) -> None:
        cache_path = str(tmp_path / "x.jsonl")
        cfg = resolve_config(
            {},
            env={
                "DECISIONHEAD_CACHE_ENABLED": "false",
                "DECISIONHEAD_CACHE_TTL_S": "60",
                "DECISIONHEAD_CACHE_MAX_ENTRIES": "5",
                "DECISIONHEAD_CACHE_PATH": cache_path,
            },
        )
        assert cfg.cache_enabled is False
        assert cfg.cache_ttl_s == 60.0
        assert cfg.cache_max_entries == 5
        assert cfg.cache_path == cache_path


class TestAbstainConfig:
    def test_defaults_disabled(self) -> None:
        cfg = resolve_config({}, env={})
        assert cfg.abstain_band is None
        assert cfg.abstain_mode == "off"

    def test_both_bounds_set_yields_a_band(self) -> None:
        cfg = resolve_config({"abstain_band_lo": 0.4, "abstain_band_hi": 0.6}, env={})
        assert cfg.abstain_band == (0.4, 0.6)

    def test_only_lo_set_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"abstain_band_lo": 0.4}, env={})

    def test_only_hi_set_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"abstain_band_hi": 0.6}, env={})

    def test_lo_greater_than_hi_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"abstain_band_lo": 0.7, "abstain_band_hi": 0.3}, env={})

    def test_invalid_abstain_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            resolve_config({"abstain_mode": "not-a-mode"}, env={})

    def test_handoff_mode_is_accepted(self) -> None:
        cfg = resolve_config({"abstain_mode": "handoff"}, env={})
        assert cfg.abstain_mode == "handoff"
