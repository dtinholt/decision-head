"""Fits one temperature per question type from a labelled set, writes
``calibration.json`` for ``decisionhead serve --calibration``.

Input: a JSONL file, one object per line::

    {"state": ..., "questions": {...same shape as a /v1/systemone request...},
     "answers_true": {qid: 0 | 1 | "OPTION_NAME"}}

Truth for a noul question is 0 or 1; truth for a choice or score question is
the correct option/level name.

Design choices (also in ``docs/calibration.md``):

* Calibration is fit type by type (noul, choice, score), not question by
  question -- one T per TYPE, as specified. A run needs at least 50 labelled
  items of a given type or that type is skipped (refused) with a clear
  message; if no type reaches 50, the run exits with an error rather than
  writing a useless calibration.json.

* The fitted quantity is T minimizing the negative log likelihood of the
  TRUE label under ``p' = sigmoid(logit(p)/T)`` (applied elementwise then
  renormalized for choice/score, exactly as :mod:`decisionhead.core`
  applies it at serve time). This is the standard temperature-scaling
  objective (Guo et al., 2017), generalized from noul's natural binary
  case to multi-option questions by minimizing the same elementwise
  transform's NLL on the true option's calibrated probability.

* Brier score and ECE (10 bins) are reported on two different scalars
  depending on type, because "a single probability" only exists for a noul
  question:
    - noul: the scalar is P(yes) itself, compared against the 0/1 truth.
      This is the textbook binary Brier/ECE.
    - choice/score: the scalar is the model's own top probability (its
      "confidence" in whichever option it picked), compared against whether
      that pick was correct. This is the standard multiclass generalization
      of calibration used in the literature (confidence vs. accuracy); it is
      not specified explicitly for multi-option questions in the vendor's
      published shape, so it is called out here as the interpretation used.

* T is found by a coarse-to-fine grid search over log-spaced T in roughly
  [0.05, 20], since the codebase is stdlib-only (no scipy.optimize). NLL in
  T is well-behaved (unimodal in practice for this transform) so a grid
  search is adequate and easy to verify by eye in the reliability table.

Invariant: the transform fitted here is the one :mod:`decisionhead.core`
applies at serve time, and the file it writes passes
:func:`decisionhead.core.validate_calibration` (every ``T`` finite and
``> 0``), which the server re-checks at start-up.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from .backends import TokenDistributionBackend, make_backend
from .core import (
    JSONDict,
    answer_questions,
    apply_calibration_distribution,
    calibrate_prob,
    normalize_question,
    top_two_gap,
)

MIN_ITEMS_PER_TYPE = 50

NoulPair = tuple[float, int]
MultiPair = tuple[dict[str, float], str]


def read_labelled_jsonl(path: str) -> list[JSONDict]:
    """Reads a labelled calibration file: one JSON object per line, blank
    lines skipped, any malformed line raising with its 1-based line
    number."""
    items: list[JSONDict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from None
    return items


def collect_raw_predictions(
    items: list[JSONDict], backend: TokenDistributionBackend, max_workers: int = 4
) -> dict[str, list[Any]]:
    """Runs the head (uncalibrated) over every labelled item and buckets
    the raw predictions by question type: ``{"noul": [(p_yes, truth01),
    ...], "choice": [(probs_dict, truth_name), ...], "score": [(probs_dict,
    truth_name), ...]}``."""
    buckets: dict[str, list[Any]] = {"noul": [], "choice": [], "score": []}
    for item in items:
        state = item["state"]
        questions = item["questions"]
        truths = item.get("answers_true", {})
        result = answer_questions(state, questions, backend, calibration=None, max_workers=max_workers)
        for qid, qdef in questions.items():
            if qid not in truths:
                continue
            qtype = normalize_question(qid, qdef)["type"]
            answer = result["answers"][qid]
            audit_row = next(r for r in result["audit"] if r["question_id"] == qid)
            raw = audit_row["raw_distribution"]
            if qtype == "noul":
                buckets["noul"].append((raw.get("Yes", answer["noul"]), int(truths[qid])))
            else:
                buckets[qtype].append((raw, str(truths[qid])))
    return buckets


# --------------------------------------------------------------------- fitting


def _nll_noul(pairs: list[NoulPair], temperature: float) -> float:
    total = 0.0
    for p_yes, y in pairs:
        p = calibrate_prob(p_yes, temperature)
        p_true = p if y == 1 else (1 - p)
        p_true = min(max(p_true, 1e-9), 1 - 1e-9)
        total += -math.log(p_true)
    return total / len(pairs) if pairs else float("inf")


def _nll_multi(pairs: list[MultiPair], temperature: float) -> float:
    total = 0.0
    for probs, truth in pairs:
        calibrated, _ = apply_calibration_distribution(probs, temperature)
        p_true = calibrated.get(truth, 0.0)
        p_true = min(max(p_true, 1e-9), 1 - 1e-9)
        total += -math.log(p_true)
    return total / len(pairs) if pairs else float("inf")


def fit_temperature(pairs: list[Any], qtype: str, t_min: float = 0.05, t_max: float = 20.0) -> float:
    """Coarse-to-fine grid search minimizing NLL over T. Two passes: a
    broad log-spaced sweep, then a finer linear sweep around the best
    point."""
    nll_fn = _nll_noul if qtype == "noul" else _nll_multi

    def sweep(lo: float, hi: float, n: int) -> tuple[float, float]:
        best_t, best_nll = 1.0, float("inf")
        for i in range(n):
            frac = i / (n - 1)
            t = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
            nll = nll_fn(pairs, t)
            if nll < best_nll:
                best_t, best_nll = t, nll
        return best_t, best_nll

    coarse_t, _ = sweep(t_min, t_max, 60)
    lo2 = max(t_min, coarse_t * 0.5)
    hi2 = min(t_max, coarse_t * 2.0)
    fine_t, _ = sweep(lo2, hi2, 60)
    return fine_t


# --------------------------------------------------------------------- metrics


def brier_and_ece_noul(pairs: list[NoulPair], temperature: float, bins: int = 10) -> JSONDict:
    """Brier score and ECE for noul pairs, scored on ``P(yes)`` against
    the 0/1 truth -- the textbook binary case."""
    scored = [(calibrate_prob(p, temperature), y) for p, y in pairs]
    brier = sum((p - y) ** 2 for p, y in scored) / len(scored) if scored else float("nan")
    ece, table = _ece_and_reliability(scored, bins)
    return {"brier": brier, "ece": ece, "reliability": table}


def brier_and_ece_multi(pairs: list[MultiPair], temperature: float, bins: int = 10) -> JSONDict:
    """Confidence-vs-correctness framing (see module docstring)."""
    scored: list[tuple[float, int]] = []
    for probs, truth in pairs:
        calibrated, _ = apply_calibration_distribution(probs, temperature)
        winner, _confidence = top_two_gap(calibrated)
        conf = calibrated[winner]
        correct = 1 if winner == truth else 0
        scored.append((conf, correct))
    brier = sum((c - y) ** 2 for c, y in scored) / len(scored) if scored else float("nan")
    ece, table = _ece_and_reliability(scored, bins)
    return {"brier": brier, "ece": ece, "reliability": table}


def _ece_and_reliability(scored: list[tuple[float, int]], bins: int) -> tuple[float, list[JSONDict]]:
    edges = [i / bins for i in range(bins + 1)]
    table: list[JSONDict] = []
    ece = 0.0
    n = len(scored)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [(p, y) for p, y in scored if (lo <= p < hi) or (i == bins - 1 and p == hi)]
        if not in_bin:
            table.append({"bin": [lo, hi], "count": 0, "mean_predicted": None, "observed_frequency": None, "gap": None})
            continue
        mean_p = sum(p for p, _ in in_bin) / len(in_bin)
        obs = sum(y for _, y in in_bin) / len(in_bin)
        gap = abs(mean_p - obs)
        table.append(
            {"bin": [lo, hi], "count": len(in_bin), "mean_predicted": mean_p, "observed_frequency": obs, "gap": gap}
        )
        if n:
            ece += (len(in_bin) / n) * gap
    return ece, table


# --------------------------------------------------------------------- top level


def calibrate(items: list[JSONDict], backend: TokenDistributionBackend, max_workers: int = 4) -> JSONDict:
    """Runs the full calibration pipeline: collects raw predictions from
    ``items``, fits a temperature per question type that reaches
    :data:`MIN_ITEMS_PER_TYPE`, and reports before/after metrics for each.
    Returns ``{"per_type": {...}, "skipped": [(type, n), ...],
    "min_items_per_type": int}``."""
    buckets = collect_raw_predictions(items, backend, max_workers=max_workers)
    per_type: JSONDict = {}
    skipped: list[tuple[str, int]] = []
    for qtype, pairs in buckets.items():
        if len(pairs) < MIN_ITEMS_PER_TYPE:
            if pairs:
                skipped.append((qtype, len(pairs)))
            continue
        temperature = fit_temperature(pairs, qtype)
        before = brier_and_ece_noul(pairs, 1.0) if qtype == "noul" else brier_and_ece_multi(pairs, 1.0)
        after = brier_and_ece_noul(pairs, temperature) if qtype == "noul" else brier_and_ece_multi(pairs, temperature)
        per_type[qtype] = {
            "T": temperature,
            "n": len(pairs),
            "brier_before": before["brier"],
            "brier_after": after["brier"],
            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "reliability_before": before["reliability"],
            "reliability_after": after["reliability"],
        }
    return {"per_type": per_type, "skipped": skipped, "min_items_per_type": MIN_ITEMS_PER_TYPE}


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the ``decisionhead calibrate``
    subcommand."""
    parser = argparse.ArgumentParser(prog="decisionhead calibrate", description="Fit per-type temperature scaling")
    parser.add_argument("--labelled", default=None, help="path to a labelled JSONL file")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--out", default="calibration.json")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="use a deterministic fake backend instead of a real one; no network calls "
        "(useful for exercising this command against examples/calibration_example.jsonl offline)",
    )
    parser.add_argument("--dry-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``decisionhead calibrate`` subcommand: parses
    ``argv``, runs :func:`calibrate` against the configured backend, and
    writes the result to ``--out``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.labelled:
        parser.error("--labelled is required")
    if not args.dry and (not args.base or not args.model):
        parser.error("--base and --model are required for a real calibration run (or use --dry)")

    backend = make_backend(
        args.backend,
        base_url=args.base,
        model=args.model,
        api_key_env=args.api_key_env,
        dry=args.dry,
        dry_seed=args.dry_seed,
    )

    items = read_labelled_jsonl(args.labelled)
    result = calibrate(items, backend, max_workers=args.max_workers)

    if not result["per_type"]:
        print(
            f"error: no question type reached {MIN_ITEMS_PER_TYPE} labelled items; nothing to calibrate",
            file=sys.stderr,
        )
        for qtype, n in result["skipped"]:
            print(f"  skipped: {qtype} had only {n} labelled items", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    for qtype, entry in result["per_type"].items():
        print(
            f"{qtype}: T={entry['T']:.3f} n={entry['n']} "
            f"brier {entry['brier_before']:.4f} -> {entry['brier_after']:.4f}, "
            f"ece {entry['ece_before']:.4f} -> {entry['ece_after']:.4f}"
        )
    for qtype, n in result["skipped"]:
        print(f"skipped {qtype}: only {n} labelled items (< {MIN_ITEMS_PER_TYPE})")
    print(f"wrote {args.out}")
    return 0


def run_selftest() -> bool:
    """Standalone smoke check, runnable with no test framework installed.
    The equivalent, more thorough assertions live in
    ``tests/test_calibrate.py``."""
    import random

    from .backends import FakeBackend

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    rng = random.Random(42)

    noul_pairs: list[NoulPair] = []
    for _ in range(200):
        q = rng.random()
        y = 1 if rng.random() < q else 0
        p_over = min(max(0.5 + 2 * (q - 0.5), 0.02), 0.98)
        noul_pairs.append((p_over, y))

    t_fit = fit_temperature(noul_pairs, "noul")
    check("fit_temperature finds T > 1 for an overconfident noul predictor", t_fit > 1.0)
    nll_before = _nll_noul(noul_pairs, 1.0)
    nll_after = _nll_noul(noul_pairs, t_fit)
    check("fitted T reduces NLL versus T=1", nll_after <= nll_before)

    before = brier_and_ece_noul(noul_pairs, 1.0)
    after = brier_and_ece_noul(noul_pairs, t_fit)
    check(
        "ece_after improves (or ties) ece_before for an overconfident predictor",
        after["ece"] <= before["ece"] + 1e-9,
    )
    check("reliability table has 10 bins", len(before["reliability"]) == 10)
    check(
        "reliability bins with data report mean_predicted and observed_frequency",
        all(
            (b["count"] == 0) or (b["mean_predicted"] is not None and b["observed_frequency"] is not None)
            for b in before["reliability"]
        ),
    )

    options = ["A", "B", "C"]
    multi_pairs: list[MultiPair] = []
    for _ in range(200):
        truth = rng.choice(options)
        base = {o: rng.random() * 0.2 for o in options}
        base[truth] += 0.6
        total = sum(base.values())
        probs = {o: v / total for o, v in base.items()}
        winner = max(probs, key=probs.get)  # type: ignore[arg-type]
        probs[winner] = min(0.99, probs[winner] * 1.3)
        remaining = 1.0 - probs[winner]
        others = [o for o in options if o != winner]
        rem_total = sum(probs[o] for o in others) or 1.0
        for o in others:
            probs[o] = remaining * (probs[o] / rem_total)
        multi_pairs.append((probs, truth))

    t_multi = fit_temperature(multi_pairs, "choice")
    check("fit_temperature returns a positive temperature for choice-type data", t_multi > 0)
    before_m = brier_and_ece_multi(multi_pairs, 1.0)
    check(
        "multi-option NLL improves or ties after fitting",
        _nll_multi(multi_pairs, t_multi) <= _nll_multi(multi_pairs, 1.0) + 1e-9,
    )
    check("multi-option brier is a finite number", math.isfinite(before_m["brier"]))

    tiny_items = [
        {"state": "s", "questions": {"q1": {"type": "noul", "instructions": "is it true"}}, "answers_true": {"q1": 1}}
        for _ in range(5)
    ]
    fb = FakeBackend(seed=9)
    result_tiny = calibrate(tiny_items, fb, max_workers=2)
    check("fewer than 50 labelled items for a type is skipped, not fitted", "noul" not in result_tiny["per_type"])
    check(
        "skipped types are reported by name and count",
        any(qtype == "noul" and n == 5 for qtype, n in result_tiny["skipped"]),
    )

    big_items = []
    for i in range(60):
        big_items.append(
            {
                "state": f"synthetic state {i}",
                "questions": {"q1": {"type": "noul", "instructions": f"is condition {i} true"}},
                "answers_true": {"q1": i % 2},
            }
        )
    result_big = calibrate(big_items, fb, max_workers=4)
    check("calibrate() fits noul once 50+ items are present", "noul" in result_big["per_type"])
    check("calibrate() result carries n matching the item count", result_big["per_type"]["noul"]["n"] == 60)
    check("calibrate() result carries a positive fitted T", result_big["per_type"]["noul"]["T"] > 0)

    try:
        json.dumps(result_big)
        check("calibrate() result is JSON-serializable", True)
    except (TypeError, ValueError):
        check("calibrate() result is JSON-serializable", False)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(main())
