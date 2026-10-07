from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import replace

import pytest

from decisionhead.config import ServerConfig
from decisionhead.server import run_server


@pytest.fixture
def dry_config() -> ServerConfig:
    """A minimal, fast, network-free server configuration."""
    return ServerConfig(model="test-model", port=0, dry=True, dry_seed=1, max_workers=2)


@pytest.fixture
def running_server(dry_config: ServerConfig) -> Iterator[str]:
    """Starts a real decisionhead server against the FakeBackend on an
    ephemeral port and yields its base URL."""
    httpd = run_server(dry_config)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def make_server(config_overrides: dict) -> tuple:
    """Helper for tests that need a non-default config. Returns
    (httpd, thread, base_url); caller is responsible for shutdown."""
    cfg = replace(ServerConfig(model="test-model", port=0, dry=True, dry_seed=1, max_workers=2), **config_overrides)
    httpd = run_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, f"http://127.0.0.1:{port}"
