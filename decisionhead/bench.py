"""Latency, throughput, and consistency measurements for a real backend.
Writes a JSON report.

Three measurements:

1. Latency per question at 1, 5, and 20 questions in a single
   ``answer_questions()`` call (single-threaded, ``max_workers=1``, so the
   number is per-call latency, not parallelism speedup) -- this is where a
   serving engine's prefix cache should show up: since every question in
   one call shares the same state prefix, the 2nd..Nth question's call
   should be faster than the 1st once the backend has that prefix warm.

2. Throughput at 1, 4, and 8 concurrent requests, each asking a fixed small
   question set, measuring requests/second and mean per-request latency
   under load.

3. A 20-repeat consistency check on 20 fixed questions at temperature 0:
   for each question, the same call is repeated 20 times and the share of
   repeats that produced a different top answer from the first one is
   reported. Temperature 0 is supposed to be deterministic; real serving
   stacks (batching, kernel nondeterminism) sometimes are not, and this is
   the number that would show it.

Real runs only when ``--base`` is given (this module talks to a live
backend through :mod:`decisionhead.backends`, not through a running
server). With no ``--base`` given, ``decisionhead bench`` refuses to run
against a network target and the pytest suite instead exercises the exact
same code paths against the deterministic ``FakeBackend``, so the
measurement logic itself is checked without a network.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from typing import Any

from .backends import TokenDistributionBackend, make_backend
from .core import JSONDict, answer_questions

FIXED_STATE = (
    "Acme Logistics BV (subsidiary of Acme Holding NV) submitted a certificate of "
    "insurance dated 2026-01-15 and a signed W-9. No endorsement schedule was attached."
)


def _make_noul_questions(n: int) -> JSONDict:
    return {
        f"q{i}": {"type": "noul", "instructions": f"Statement {i}: the evidence on file is sufficient and current."}
        for i in range(n)
    }


def _make_fixed_20_questions() -> JSONDict:
    qs: JSONDict = {}
    for i in range(14):
        qs[f"noul_{i}"] = {"type": "noul", "instructions": f"Fixed check {i}: this requirement is satisfied on file."}
    for i in range(3):
        qs[f"choice_{i}"] = {
            "type": "choice",
            "instructions": f"Fixed choice {i}: what should happen next?",
            "criteria": {
                "CONTINUE": "keep reasoning",
                "RETRIEVE": "pull internal evidence",
                "REQUEST": "ask the supplier",
            },
        }
    for i in range(3):
        qs[f"score_{i}"] = {
            "type": "score",
            "instructions": f"Fixed score {i}: how urgent is this item?",
            "levels": {"low": "no deadline pressure", "medium": "due this week", "high": "overdue"},
        }
    return qs


def bench_latency_by_batch_size(
    backend: TokenDistributionBackend, sizes: tuple[int, ...] = (1, 5, 20)
) -> JSONDict:
    """For each size in ``sizes``, asks that many noul questions in one
    ``answer_questions()`` call (single-threaded) and records per-question
    latency, separating the first question from the mean of the rest --
    this is where a serving engine's prefix cache should show up."""
    out: JSONDict = {}
    for n in sizes:
        questions = _make_noul_questions(n)
        t0 = time.time()
        result = answer_questions(FIXED_STATE, questions, backend, max_workers=1)
        total_s = time.time() - t0
        per_question = [row["latency_s"] for row in result["audit"]]
        out[str(n)] = {
            "n_questions": n,
            "total_s": total_s,
            "mean_per_question_s": statistics.mean(per_question) if per_question else None,
            "first_question_s": per_question[0] if per_question else None,
            "later_questions_mean_s": statistics.mean(per_question[1:]) if len(per_question) > 1 else None,
            "per_question_s": per_question,
        }
    return out


def bench_throughput(
    backend: TokenDistributionBackend, concurrencies: tuple[int, ...] = (1, 4, 8), questions: JSONDict | None = None
) -> JSONDict:
    """For each concurrency level, fires that many simultaneous
    ``answer_questions()`` calls (each asking ``questions``, default a
    small fixed noul set) and reports requests/second and per-request
    latency under that load."""
    questions = questions or _make_noul_questions(3)
    out: JSONDict = {}
    for c in concurrencies:
        latencies: list[float] = []
        lock = threading.Lock()

        def worker(lock: threading.Lock = lock, latencies: list[float] = latencies) -> None:
            t0 = time.time()
            answer_questions(FIXED_STATE, questions, backend, max_workers=1)
            dt = time.time() - t0
            with lock:
                latencies.append(dt)

        threads = [threading.Thread(target=worker) for _ in range(c)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall_s = time.time() - t0
        out[str(c)] = {
            "concurrency": c,
            "wall_s": wall_s,
            "requests_per_s": c / wall_s if wall_s > 0 else None,
            "mean_request_latency_s": statistics.mean(latencies) if latencies else None,
            "max_request_latency_s": max(latencies) if latencies else None,
        }
    return out


def bench_consistency(backend: TokenDistributionBackend, repeats: int = 20) -> JSONDict:
    """Answers a fixed set of 20 questions ``repeats`` times at
    temperature 0 and reports, per question and overall, the share of
    repeats whose top answer disagreed with the first -- ideally zero,
    since temperature 0 is supposed to be deterministic."""
    questions = _make_fixed_20_questions()
    result0 = answer_questions(FIXED_STATE, questions, backend, max_workers=4)
    first_answers = result0["answers"]

    disagreements = {qid: 0 for qid in questions}
    for _ in range(repeats - 1):
        result = answer_questions(FIXED_STATE, questions, backend, max_workers=4)
        for qid, qdef in questions.items():
            a0, a1 = first_answers[qid], result["answers"][qid]
            d0: Any
            d1: Any
            if qdef["type"] == "noul":
                d0, d1 = (a0["noul"] >= 0.5), (a1["noul"] >= 0.5)
            elif qdef["type"] == "choice":
                d0, d1 = a0["choice"], a1["choice"]
            else:
                d0, d1 = a0["score"], a1["score"]
            if d0 != d1:
                disagreements[qid] += 1

    total_comparisons = len(questions) * (repeats - 1)
    total_disagreements = sum(disagreements.values())
    return {
        "n_questions": len(questions),
        "repeats": repeats,
        "per_question_disagreement_count": disagreements,
        "share_any_disagreement": (total_disagreements / total_comparisons) if total_comparisons else 0.0,
        "questions_with_any_disagreement": sum(1 for v in disagreements.values() if v > 0),
    }


def bench_stability(backend: TokenDistributionBackend, repeats: int = 20) -> JSONDict:
    """The stability audit artefact (see ``docs/audit/README.md``):
    answers the same fixed 20 questions ``repeats`` times, each repeat a
    fresh, single-threaded (``max_workers=1``) call -- i.e. at
    concurrency 1, the deterministic-serving setting this repo's own
    ``backend.max_concurrency`` config knob defaults to (see
    :mod:`decisionhead.backends`'s ``ConcurrencyLimitedBackend``) -- so
    any variation this reports reflects the backend's own repeat-to-repeat
    behaviour, not batching-induced nondeterminism.

    Unlike :func:`bench_consistency` (which only checks each repeat's
    DECISIVE answer against the first), this also counts, per question,
    the number of DISTINCT raw values seen across every repeat (a noul
    probability rounded to 6 decimal places, or a choice/score
    probability distribution as a sorted tuple) -- catching a backend
    that returns slightly different logprobs run to run even when the
    decisive answer happens not to change, which is exactly the failure
    mode a batching endpoint exhibits."""
    questions = _make_fixed_20_questions()
    per_question_values: dict[str, list[Any]] = {qid: [] for qid in questions}
    per_question_decisive: dict[str, list[Any]] = {qid: [] for qid in questions}

    for _ in range(repeats):
        result = answer_questions(FIXED_STATE, questions, backend, max_workers=1)
        for qid, qdef in questions.items():
            answer = result["answers"][qid]
            if qdef["type"] == "noul":
                value: Any = round(answer["noul"], 6)
                decisive: Any = answer["noul"] >= 0.5
            elif qdef["type"] == "choice":
                value = tuple(sorted((k, round(v, 6)) for k, v in answer["probabilities"].items()))
                decisive = answer["choice"]
            else:
                value = tuple(sorted((k, round(v, 6)) for k, v in answer["probabilities"].items()))
                decisive = answer["score"]
            per_question_values[qid].append(value)
            per_question_decisive[qid].append(decisive)

    distinct_value_counts = {qid: len(set(vals)) for qid, vals in per_question_values.items()}
    disagreement_counts = {
        qid: sum(1 for d in decisives[1:] if d != decisives[0]) for qid, decisives in per_question_decisive.items()
    }
    total_comparisons = len(questions) * (repeats - 1) if repeats > 1 else 0
    total_disagreements = sum(disagreement_counts.values())

    return {
        "n_questions": len(questions),
        "repeats": repeats,
        "concurrency": 1,
        "distinct_value_counts": distinct_value_counts,
        "questions_with_multiple_distinct_values": sum(1 for c in distinct_value_counts.values() if c > 1),
        "per_question_disagreement_count": disagreement_counts,
        "share_any_disagreement": (total_disagreements / total_comparisons) if total_comparisons else 0.0,
        "model": getattr(backend, "model", None),
    }


def render_stability_report_markdown(report: JSONDict) -> str:
    """Renders a :func:`bench_stability` report as a short markdown
    table -- the human-readable half of the JSON+markdown pair
    ``docs/audit/README.md`` documents as this repo's stability audit
    artefact."""
    lines = [
        "# Stability report",
        "",
        f"- model: `{report.get('model')}`",
        f"- questions: {report['n_questions']}",
        f"- repeats: {report['repeats']}",
        f"- concurrency: {report['concurrency']}",
        f"- share of repeats with any disagreement: {report['share_any_disagreement']:.4f}",
        f"- questions with more than one distinct value across repeats: "
        f"{report['questions_with_multiple_distinct_values']} / {report['n_questions']}",
        "",
        "| question | distinct values across repeats | disagreements vs. first |",
        "|---|---|---|",
    ]
    for qid in report["distinct_value_counts"]:
        lines.append(
            f"| {qid} | {report['distinct_value_counts'][qid]} | {report['per_question_disagreement_count'][qid]} |"
        )
    lines.append("")
    return "\n".join(lines)


def stability_report_paths(out_path: str) -> tuple[str, str]:
    """Derives the ``(json_path, md_path)`` pair for a stability report
    from one base ``--out`` path: ``foo.json`` -> ``(foo.stability.json``,
    ``foo.stability.md)``; a base path with any other (or no) extension
    just gets ``.stability.json``/``.stability.md`` appended."""
    if out_path.endswith(".json"):
        stem = out_path[: -len(".json")]
    else:
        stem = out_path
    return f"{stem}.stability.json", f"{stem}.stability.md"


def write_stability_report(report: JSONDict, json_path: str, md_path: str) -> None:
    """Writes a :func:`bench_stability` report as both JSON (machine-
    readable) and markdown (human-readable) to the given paths."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_stability_report_markdown(report))


def run_bench(backend: TokenDistributionBackend, repeats: int = 20) -> JSONDict:
    """Runs all three measurements (latency by batch size, throughput by
    concurrency, and repeat-consistency) and assembles one report."""
    return {
        "latency_by_batch_size": bench_latency_by_batch_size(backend),
        "throughput_by_concurrency": bench_throughput(backend),
        "consistency": bench_consistency(backend, repeats=repeats),
        "model": getattr(backend, "model", None),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the ``decisionhead bench``
    subcommand."""
    parser = argparse.ArgumentParser(prog="decisionhead bench", description="Latency/throughput/consistency bench")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base", default=None, help="backend base URL; required for a real run")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", default="bench_report.json")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="use a deterministic fake backend instead of a real one; no network calls "
        "(exercises the measurement code with no hardware available)",
    )
    parser.add_argument("--dry-seed", type=int, default=0)
    parser.add_argument(
        "--stability",
        action="store_true",
        help="run ONLY the stability audit artefact (N repeats on fixed questions at concurrency 1; "
        "see docs/audit/README.md) instead of the full latency/throughput/consistency bench, and write "
        "it as JSON+markdown (paths derived from --out; see decisionhead.bench.stability_report_paths)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``decisionhead bench`` subcommand: parses
    ``argv``, runs :func:`run_bench` (or, with ``--stability``, just
    :func:`bench_stability`) against the configured backend, and writes
    the report(s) to disk."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dry and not args.base:
        parser.error("--base is required for a real bench run (or use --dry)")
    if not args.dry and not args.model:
        parser.error("--model is required for a real bench run (or use --dry)")

    backend = make_backend(
        args.backend,
        base_url=args.base or "",
        model=args.model or "",
        api_key_env=args.api_key_env,
        dry=args.dry,
        dry_seed=args.dry_seed,
    )

    if args.stability:
        report = bench_stability(backend, repeats=args.repeats)
        json_path, md_path = stability_report_paths(args.out)
        write_stability_report(report, json_path, md_path)
        print(json.dumps(report, indent=2))
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        return 0

    report = run_bench(backend, repeats=args.repeats)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed.
    The equivalent, more thorough assertions live in
    ``tests/test_bench.py``."""
    from .backends import FakeBackend

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    fb = FakeBackend(seed=11)

    lat = bench_latency_by_batch_size(fb, sizes=(1, 3, 5))
    check("latency bench covers every requested batch size", set(lat.keys()) == {"1", "3", "5"})
    check("latency bench records one per_question_s entry per question", len(lat["5"]["per_question_s"]) == 5)
    check(
        "latency bench reports first_question_s separately from the rest",
        lat["5"]["first_question_s"] is not None and lat["5"]["later_questions_mean_s"] is not None,
    )

    thr = bench_throughput(fb, concurrencies=(1, 2))
    check("throughput bench covers every requested concurrency", set(thr.keys()) == {"1", "2"})
    check("throughput bench reports a positive requests_per_s", thr["2"]["requests_per_s"] > 0)
    check(
        "throughput bench reports mean and max request latency",
        thr["2"]["mean_request_latency_s"] is not None and thr["2"]["max_request_latency_s"] is not None,
    )

    cons = bench_consistency(fb, repeats=5)
    check("consistency bench covers 20 fixed questions", cons["n_questions"] == 20)
    check(
        "consistency bench against a deterministic fake backend shows zero disagreement",
        cons["share_any_disagreement"] == 0.0,
    )
    check(
        "consistency bench reports a per-question disagreement count",
        len(cons["per_question_disagreement_count"]) == 20,
    )

    stab = bench_stability(fb, repeats=5)
    check("stability bench covers 20 fixed questions", stab["n_questions"] == 20)
    check("stability bench runs at concurrency 1", stab["concurrency"] == 1)
    check(
        "stability bench against a deterministic fake backend shows exactly one distinct value per question",
        all(c == 1 for c in stab["distinct_value_counts"].values()),
    )
    check(
        "stability bench markdown renders without raising",
        isinstance(render_stability_report_markdown(stab), str) and len(render_stability_report_markdown(stab)) > 0,
    )
    check(
        "stability_report_paths derives sibling json/md paths",
        stability_report_paths("out.json") == ("out.stability.json", "out.stability.md"),
    )

    report = run_bench(fb, repeats=5)
    check(
        "run_bench assembles all three sections plus the model id",
        {"latency_by_batch_size", "throughput_by_concurrency", "consistency", "model"} == set(report.keys()),
    )
    try:
        json.dumps(report)
        check("bench report is JSON-serializable", True)
    except (TypeError, ValueError):
        check("bench report is JSON-serializable", False)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(main())
