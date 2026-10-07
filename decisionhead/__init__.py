"""decisionhead -- a self-hosted typed-question decision service.

Reproduces the API shape of a hosted typed-question decision service
(``POST /v1/systemone``: state plus typed questions in, calibrated
probabilities out) in front of any local model that exposes per-token
logprobs, over the native ollama route or an OpenAI-compatible route
(vLLM and similar). Standard-library only at runtime.

See the README for the full picture and ``docs/design.md`` for the
mechanism and its limits.

Invariants that hold across the package: no runtime dependency outside
the standard library; no secret is accepted as a flag or config value or
written to a log, response, audit row or cache row (secrets are read from
environment variables named by ``*_env`` settings); every number this
package emits as JSON is finite.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
