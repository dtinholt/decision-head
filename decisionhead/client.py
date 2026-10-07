"""A tiny client for a running ``decisionhead serve`` instance.

    from decisionhead.client import ask
    result = ask("http://localhost:8787", state, questions)

Standard library only. Errors from the server or the transport are raised
as :class:`ClientError` carrying the server's ``{"error": ...}`` body;
the client never retries, so a ``429`` or ``504`` is the caller's to
handle. The bearer token, if any, is passed in by the caller and only
ever placed in the ``Authorization`` header.

Every call goes through the same redirect-refusing opener as the backend
path (:func:`decisionhead.backends.build_backend_opener`): CPython's
default redirect handler copies ``Authorization`` onto the redirected
request whatever host it points at, so a redirect would forward the
service bearer token (2026-10-06 confirmation, R6). A redirect is raised
as :class:`ClientError` instead.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from .backends import BackendError, build_backend_opener

JSONDict = dict[str, Any]


class ClientError(Exception):
    """Wraps any transport or server-side error into one exception type
    with the server's own error body (if any) attached, never a raw
    traceback from urllib."""


_OPENER = build_backend_opener()


def _get_json(url: str, timeout: float) -> JSONDict:
    """GET ``url`` through the redirect-refusing opener and parse the
    JSON body. A refused redirect is a :class:`ClientError`."""
    try:
        with _OPENER.open(url, timeout=timeout) as resp:
            result: JSONDict = json.loads(resp.read().decode("utf-8"))
            return result
    except BackendError as exc:
        raise ClientError(f"request to {url} refused: {exc}") from None


def ask(
    base: str,
    state: str | JSONDict,
    questions: JSONDict,
    model: str = "jev-latest",
    timeout: int = 60,
    bearer_token: str | None = None,
) -> JSONDict:
    """POSTs one Jev-shaped request to ``{base}/v1/systemone`` and returns
    the parsed response. ``model`` is an alias only -- the server always
    answers with its real backend model id regardless of what is sent
    here. ``bearer_token``, if given, is sent as an ``Authorization:
    Bearer`` header; pass the token itself (read it from your own
    environment variable before calling this), never a literal secret in
    source."""
    url = base.rstrip("/") + "/v1/systemone"
    body = json.dumps({"state": state, "model": model, "questions": questions}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = "Bearer " + bearer_token
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            result: JSONDict = json.loads(resp.read().decode("utf-8"))
            return result
    except BackendError as exc:
        # Raised by the opener's redirect policy: the request (and its
        # Authorization header) is never re-sent to the redirect target.
        raise ClientError(f"request to {url} refused: {exc}") from None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        raise ClientError(f"HTTP {exc.code} from {url}: {parsed}") from None
    except urllib.error.URLError as exc:
        raise ClientError(f"connection failed to {url}: {exc.reason}") from None


def health(base: str, timeout: int = 10) -> JSONDict:
    """GETs ``{base}/health`` and returns the parsed liveness response."""
    return _get_json(base.rstrip("/") + "/health", timeout)


def models(base: str, timeout: int = 10) -> JSONDict:
    """GETs ``{base}/v1/models`` and returns the parsed model listing."""
    return _get_json(base.rstrip("/") + "/v1/models", timeout)


# --------------------------------------------------------------------- CLI


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the ``decisionhead ask``
    subcommand."""
    parser = argparse.ArgumentParser(prog="decisionhead ask", description="Tiny CLI for a running decisionhead server")
    parser.add_argument("--base", default="http://127.0.0.1:8787")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--state-text", default=None, help="a plain-text state")
    parser.add_argument("--state-file", default=None, help="a JSON or text file holding the state")
    parser.add_argument("--questions", default=None, help="path to a JSON file with the questions map")
    parser.add_argument(
        "--bearer-token-env", default=None, help="env var name holding a bearer token, if the server requires one"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``decisionhead ask`` subcommand: parses
    ``argv``, loads the state and questions, calls :func:`ask` against a
    running server, and prints the response as JSON."""
    import os

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.questions:
        parser.error("--questions is required (path to a JSON questions map)")

    if args.state_file:
        with open(args.state_file, encoding="utf-8") as f:
            text = f.read()
        try:
            state: str | JSONDict = json.loads(text)
        except json.JSONDecodeError:
            state = text
    elif args.state_text is not None:
        state = args.state_text
    else:
        parser.error("one of --state-text or --state-file is required")
        return 2

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    bearer_token = os.environ.get(args.bearer_token_env) if args.bearer_token_env else None

    try:
        result = ask(args.base, state, questions, model=args.model, bearer_token=bearer_token)
    except ClientError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


def run_selftest() -> bool:
    """Spins up a real decisionhead server in dry mode in-process, then
    exercises ``ask()``/``health()``/``models()`` against it over real HTTP
    -- this is the one module whose selftest is necessarily an integration
    test, since its entire job is being an HTTP client. The equivalent,
    more thorough assertions live in ``tests/test_client.py``."""
    import threading

    from . import server as server_mod
    from .config import ServerConfig

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    cfg = ServerConfig(model="client-selftest-model", port=0, dry=True, dry_seed=5, max_workers=2)
    httpd = server_mod.run_server(cfg)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        h = health(base)
        check("health() reaches a running server", h.get("status") == "ok")

        m = models(base)
        check("models() lists the real backend model id", m["models"][0]["id"] == "client-selftest-model")

        questions = {
            "q1": {"type": "noul", "instructions": "Is the evidence on file current?"},
            "q2": {
                "type": "score",
                "instructions": "How urgent is this case?",
                "levels": {"low": "no deadline pressure", "medium": "due this week", "high": "overdue"},
            },
        }
        result = ask(base, {"note": "two sentence supplier state"}, questions)
        check(
            "ask() returns the real model id regardless of the alias sent",
            result["model"] == "client-selftest-model",
        )
        check("ask() returns an answer per question", set(result["answers"].keys()) == set(questions.keys()))
        check("ask() score answer has a numeric score", isinstance(result["answers"]["q2"]["score"], (int, float)))

        try:
            ask(base, "x", {})
            check("ask() surfaces server-side 4xx as ClientError", False)
        except ClientError as exc:
            check("ask() surfaces server-side 4xx as ClientError", "400" in str(exc))

        try:
            ask("http://127.0.0.1:1", "x", questions, timeout=2)
            check("ask() surfaces a connection failure as ClientError", False)
        except ClientError:
            check("ask() surfaces a connection failure as ClientError", True)

    finally:
        httpd.shutdown()
        httpd.server_close()

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(main())
