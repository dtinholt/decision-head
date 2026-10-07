"""``decisionhead`` console entry point.

    decisionhead serve      --backend ollama --base http://host:11434 --model M
    decisionhead calibrate  --labelled cases.jsonl --backend ollama --base ... --model M
    decisionhead bench      --backend ollama --base ... --model M
    decisionhead ask        --base http://localhost:8787 --state-text "..." --questions q.json
    decisionhead probe      --backend ollama --base ... --model M
    decisionhead train      --traces traces.jsonl --cv 5 --seed 0 --out model.json
    decisionhead selftest

Each subcommand's flags are documented by ``decisionhead <subcommand> -h``.
"""

from __future__ import annotations

import argparse
import sys

from . import (
    __version__,
    backends,
    bench,
    cache,
    calibrate,
    client,
    core,
    embeddings,
    features,
    learned,
    model,
    server,
    traces,
    train,
)

USAGE = __doc__ or ""


def _cmd_probe(argv: list[str]) -> int:
    import json

    parser = argparse.ArgumentParser(
        prog="decisionhead probe", description="Sanity-check a backend's per-token logprob mechanism"
    )
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base", default="", help="backend base URL")
    parser.add_argument("--model", default="", help="real backend model id")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--dry", action="store_true", help="probe the deterministic fake backend instead")
    args = parser.parse_args(argv)

    if not args.dry and not (args.base and args.model):
        parser.error("--base and --model are required unless --dry is given")

    backend = backends.make_backend(
        args.backend, base_url=args.base, model=args.model, api_key_env=args.api_key_env, dry=args.dry
    )
    result = backends.probe_backend(backend)
    print(json.dumps(result, indent=2))
    if not result["healthy"]:
        print(
            "warning: the backend did not return a clean Yes/No logprob reading; "
            "check that logprobs are enabled and, for ollama, that the NATIVE /api/chat route is reachable",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_selftest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="decisionhead selftest", description="Run every module's built-in smoke check (no pytest required)"
    )
    parser.parse_args(argv)

    modules = [
        ("core", core.run_selftest),
        ("backends", backends.run_selftest),
        ("cache", cache.run_selftest),
        ("client", client.run_selftest),
        ("calibrate", calibrate.run_selftest),
        ("bench", bench.run_selftest),
        ("server", server.run_selftest),
        ("features", features.run_selftest),
        ("embeddings", embeddings.run_selftest),
        ("model", model.run_selftest),
        ("traces", traces.run_selftest),
        ("learned", learned.run_selftest),
        ("train", train.run_selftest),
    ]
    all_ok = True
    for name, run in modules:
        print(f"=== {name} ===")
        ok = run()
        all_ok = all_ok and ok
        print()
    print("ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


COMMANDS = {
    "serve": server.main,
    "calibrate": calibrate.main,
    "bench": bench.main,
    "ask": client.main,
    "probe": _cmd_probe,
    "selftest": _cmd_selftest,
    "train": train.main,
}


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI dispatch: prints usage/version, or hands the rest of
    ``argv`` to the named subcommand's own ``main``."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("-V", "--version"):
        print(f"decisionhead {__version__}")
        return 0
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if args else 2

    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"decisionhead: unknown command {command!r}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    return COMMANDS[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
