"""``python -m decisionhead.train`` -- fits the trainable decision head on
labelled workflow traces (:mod:`decisionhead.traces`) and evaluates it
with case-clustered cross-validation.

    python -m decisionhead.train --traces dev_traces.jsonl --cv 5 --seed 0 \\
        --out model.json --report report.json

Cross-validation is CASE-CLUSTERED: folds partition traces by
``meta.case_id`` (a trace with no ``meta.case_id`` is treated as its own
singleton cluster), never by individual checkpoint, so no fold ever
trains and tests on two checkpoints from the same case -- the leak that
matters here, since checkpoints from one case share almost all of their
state. This is a DEV-SET estimate only; nothing in this module ever
reads eval/challenge-split data, and every report this module writes
carries an explicit ``"dev_only": true`` flag saying so.

Reported cross-validation metrics: ``decision_class_accuracy_cv`` (mean
accuracy of held-out, out-of-fold predictions -- the primary number),
``per_class_recall_cv`` (per label, ``None`` if a label never appears in
the held-out portion of any fold), ``ece_cv`` (expected calibration error
of the temperature-scaled out-of-fold predictions, reusing
:mod:`decisionhead.calibrate`'s own multiclass ECE code -- the same
metric ``decisionhead calibrate`` reports for the zero-shot path), and
``majority_baseline_accuracy`` (always-predict-the-most-common-label,
the floor any trained classifier should clear).

The SAVED model (``--out``) is refit on every trace this module was
given (cross-validation is for the report, not for the artifact); its
temperature is the one fit on the pooled cross-validation out-of-fold
predictions, which is the only calibration-honest set of held-out
probabilities available here (fitting temperature on the final model's
own training predictions would calibrate against data it has already
memorized).
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .calibrate import brier_and_ece_multi, fit_temperature
from .embeddings import EmbeddingsBackend, make_embeddings_backend
from .features import FeatureConfig, extract_features, feature_names
from .learned import LearnedModel
from .model import MultinomialLogisticRegression, StandardScaler
from .traces import TraceError, read_traces

JSONDict = dict[str, Any]


def _cluster_ids(traces: list[JSONDict]) -> list[str]:
    """One clustering key per trace: ``meta.case_id`` when present,
    otherwise a synthetic per-trace id (so a trace with no case
    provenance is simply its own cluster, never accidentally merged with
    another trace that also happens to lack one)."""
    ids: list[str] = []
    for i, trace in enumerate(traces):
        meta = trace.get("meta") or {}
        case_id = meta.get("case_id")
        ids.append(str(case_id) if case_id is not None else f"__no_case_id_{i}")
    return ids


def _kfold_case_clusters(cluster_ids: list[str], k: int, seed: int) -> list[list[int]]:
    """Partitions trace indices into ``k`` folds by cluster id: every
    trace sharing a cluster id lands in the same fold. Clusters are
    shuffled deterministically (seeded) before being dealt round-robin
    into folds, so fold sizes differ by at most one cluster."""
    unique_clusters = sorted(set(cluster_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_clusters)
    cluster_to_fold = {cid: i % k for i, cid in enumerate(unique_clusters)}
    fold_indices: list[list[int]] = [[] for _ in range(k)]
    for idx, cid in enumerate(cluster_ids):
        fold_indices[cluster_to_fold[cid]].append(idx)
    return fold_indices


def _argmax(values: list[float]) -> int:
    """The index of the largest value (the first one, on a tie)."""
    return max(range(len(values)), key=lambda i: values[i])


def _utc_now_iso() -> str:
    return datetime.datetime.now(tz=datetime.UTC).isoformat()


@dataclass
class TrainResult:
    """What :func:`train_and_evaluate` returns: the model refit on every
    trace, and the cross-validation report (always ``"dev_only": true``)."""

    model: LearnedModel
    report: JSONDict


def train_and_evaluate(
    traces: list[JSONDict],
    *,
    feature_config: FeatureConfig,
    cv_folds: int = 5,
    seed: int = 0,
    l2: float = 0.1,
    lr: float = 0.5,
    epochs: int = 300,
    embedder: EmbeddingsBackend | None = None,
) -> TrainResult:
    """Runs case-clustered cross-validation for the report, then fits the
    final saved model on every trace given. Raises :class:`ValueError` if
    there are fewer than two distinct labels (nothing to classify) or
    zero traces."""
    if not traces:
        raise ValueError("cannot train on zero traces")
    classes = sorted({trace["label"] for trace in traces})
    if len(classes) < 2:
        raise ValueError(f"need at least 2 distinct labels to train a classifier; found {classes}")
    class_index = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    names = feature_names(feature_config)
    n_features = len(names)
    x_raw = [extract_features(trace["state"], feature_config, embedder=embedder) for trace in traces]
    y = [class_index[trace["label"]] for trace in traces]

    cluster_ids = _cluster_ids(traces)
    n_clusters = len(set(cluster_ids))
    k = min(cv_folds, n_clusters) if n_clusters >= 2 else 1

    majority_label = Counter(trace["label"] for trace in traces).most_common(1)[0][0]
    majority_baseline_accuracy = sum(1 for trace in traces if trace["label"] == majority_label) / len(traces)

    oof_true_idx: list[int] = []
    oof_pred_idx: list[int] = []
    oof_pairs: list[tuple[dict[str, float], str]] = []

    if k >= 2:
        fold_indices = _kfold_case_clusters(cluster_ids, k, seed)
        for fold in fold_indices:
            test_idx = fold
            if not test_idx:
                continue
            test_set = set(test_idx)
            train_idx = [i for i in range(len(traces)) if i not in test_set]
            scaler = StandardScaler.fit([x_raw[i] for i in train_idx])
            x_train = scaler.transform_batch([x_raw[i] for i in train_idx])
            y_train = [y[i] for i in train_idx]
            clf = MultinomialLogisticRegression.initialize(n_classes, n_features, seed=seed)
            clf.fit(x_train, y_train, l2=l2, lr=lr, epochs=epochs)
            for i in test_idx:
                probs = clf.predict_proba(scaler.transform(x_raw[i]))
                pred_idx = _argmax(probs)
                oof_true_idx.append(y[i])
                oof_pred_idx.append(pred_idx)
                oof_pairs.append((dict(zip(classes, probs, strict=True)), classes[y[i]]))

    n_scored = len(oof_true_idx)
    decision_class_accuracy_cv = (
        sum(1 for t, p in zip(oof_true_idx, oof_pred_idx, strict=True) if t == p) / n_scored if n_scored else None
    )

    per_class_recall_cv: dict[str, float | None] = {}
    for c_idx, c_name in enumerate(classes):
        actual_positions = [i for i in range(n_scored) if oof_true_idx[i] == c_idx]
        if not actual_positions:
            per_class_recall_cv[c_name] = None
            continue
        correct = sum(1 for i in actual_positions if oof_pred_idx[i] == c_idx)
        per_class_recall_cv[c_name] = correct / len(actual_positions)

    temperature = 1.0
    ece_cv: float | None = None
    if oof_pairs:
        temperature = fit_temperature(oof_pairs, "choice")
        ece_cv = brier_and_ece_multi(oof_pairs, temperature)["ece"]

    final_scaler = StandardScaler.fit(x_raw)
    x_scaled_full = final_scaler.transform_batch(x_raw)
    final_clf = MultinomialLogisticRegression.initialize(n_classes, n_features, seed=seed)
    loss_history = final_clf.fit(x_scaled_full, y, l2=l2, lr=lr, epochs=epochs)

    model = LearnedModel(
        classifier=final_clf,
        scaler=final_scaler,
        feature_config=feature_config,
        classes=classes,
        temperature=temperature,
        metadata={
            "name": "decisionhead-learned-v1",
            "n_train": len(traces),
            "n_cases": n_clusters,
            "trained_at": _utc_now_iso(),
            "seed": seed,
            "l2": l2,
            "lr": lr,
            "epochs": epochs,
            "final_train_loss": loss_history[-1] if loss_history else None,
            # A model trained with use_prior_answers=False must never
            # request or read state["prior_answers"]/prior_p_* at serve
            # time (see decisionhead.features.extract_features and
            # decisionhead.learned.LearnedModel.__post_init__, which
            # checks this claim against feature_config for consistency).
            "uses_prior_features": feature_config.use_prior_answers,
        },
    )

    report: JSONDict = {
        "dev_only": True,
        "note": "decision_class_accuracy_cv is a DEV-SET cross-validation estimate; it is not an eval-set result.",
        "n_traces": len(traces),
        "n_cases": n_clusters,
        "cv_folds_requested": cv_folds,
        "cv_folds_used": k,
        "n_scored_cv": n_scored,
        "seed": seed,
        "classes": classes,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "decision_class_accuracy_cv": decision_class_accuracy_cv,
        "per_class_recall_cv": per_class_recall_cv,
        "ece_cv": ece_cv,
        "temperature": temperature,
        "feature_config": feature_config.to_dict(),
        "feature_names": names,
        "hyperparameters": {"l2": l2, "lr": lr, "epochs": epochs},
    }
    return TrainResult(model=model, report=report)


def build_arg_parser() -> argparse.ArgumentParser:
    """The argument parser for ``python -m decisionhead.train`` (also
    reachable as ``decisionhead train``)."""
    parser = argparse.ArgumentParser(
        prog="python -m decisionhead.train", description="Train and cross-validate the trainable decision head"
    )
    parser.add_argument("--traces", required=True, help="path to a trace JSONL file (see decisionhead.traces)")
    parser.add_argument("--out", default=None, help="path to write the trained model JSON")
    parser.add_argument("--report", default=None, help="path to write the JSON evaluation report")
    parser.add_argument("--cv", type=int, default=5, help="number of case-clustered cross-validation folds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2", type=float, default=0.1, help="L2 regularization strength")
    parser.add_argument("--lr", type=float, default=0.5, help="gradient descent learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="gradient descent epochs (with early stopping)")
    parser.add_argument(
        "--no-structured", action="store_true", help="disable feature group (a): structured state features"
    )
    parser.add_argument(
        "--no-prior-answers",
        "--no-prior-features",
        dest="no_prior_answers",
        action="store_true",
        help="disable feature group (b): prior zero-shot answers (state['prior_answers']/prior_p_*); "
        "a model trained this way never requests or uses them, and never calls a fast/zero-shot head, "
        "at serve time (recorded as metadata['uses_prior_features'] = false in the saved model file)",
    )
    parser.add_argument(
        "--use-embedding",
        action="store_true",
        help="enable feature group (c): a text embedding of the rendered state (off by default; requires network)",
    )
    parser.add_argument("--embedding-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--embedding-base", default="", help="embeddings backend base URL")
    parser.add_argument("--embedding-model", default="", help="embeddings model id")
    parser.add_argument(
        "--embedding-dim", type=int, default=None, help="expected embedding dimension (required with --use-embedding)"
    )
    parser.add_argument("--embedding-api-key-env", default=None, help="env var holding an embeddings API key")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: reads the traces, trains and cross-validates, writes
    ``--out``/``--report`` if given, and returns the process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        traces = read_traces(args.traces)
    except (TraceError, OSError) as exc:
        parser.error(str(exc))
        return 2

    embedder: EmbeddingsBackend | None = None
    if args.use_embedding:
        if not args.embedding_dim:
            parser.error("--use-embedding requires --embedding-dim")
            return 2
        embedder = make_embeddings_backend(
            args.embedding_backend, args.embedding_base, args.embedding_model, api_key_env=args.embedding_api_key_env
        )

    feature_config = FeatureConfig(
        use_structured=not args.no_structured,
        use_prior_answers=not args.no_prior_answers,
        use_embedding=args.use_embedding,
        embedding_dim=args.embedding_dim,
    )

    try:
        result = train_and_evaluate(
            traces,
            feature_config=feature_config,
            cv_folds=args.cv,
            seed=args.seed,
            l2=args.l2,
            lr=args.lr,
            epochs=args.epochs,
            embedder=embedder,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(result.report, indent=2, sort_keys=True))

    if args.out:
        result.model.save(args.out)
        print(f"model written to {args.out}", file=sys.stderr)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result.report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"report written to {args.report}", file=sys.stderr)
    return 0


def run_selftest() -> bool:
    """Standalone smoke check on a tiny synthetic trace set; see
    ``tests/test_train.py`` for the thorough version against real JEV-WC
    dev data."""
    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))

    rng = random.Random(0)
    traces: list[JSONDict] = []
    for case_num in range(20):
        n_open = rng.randint(0, 3)
        label = "DONE" if n_open == 0 else ("INFO" if n_open % 2 == 0 else "REASON")
        for step in range(2):
            traces.append(
                {
                    "state": {
                        "requirements": [{"item_id": f"r{i}"} for i in range(3)],
                        "open_items": [f"r{i}" for i in range(n_open)],
                        "history": [],
                        "budget": {"steps": 12, "expert_calls": 2, "requests": 3},
                        "steps_used": step,
                    },
                    "label": label,
                    "meta": {"case_id": f"case-{case_num}"},
                }
            )

    cfg = FeatureConfig(use_structured=True, use_prior_answers=False, use_embedding=False)
    result = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=100)
    check("report is marked dev_only", result.report["dev_only"] is True)
    check("cross-validation used more than one fold", result.report["cv_folds_used"] >= 2)
    check(
        "decision_class_accuracy_cv beats the majority baseline on a trivially learnable signal",
        (result.report["decision_class_accuracy_cv"] or 0.0) >= result.report["majority_baseline_accuracy"],
    )
    check(
        "every declared class has a recall entry",
        set(result.report["per_class_recall_cv"]) == set(result.report["classes"]),
    )

    result_again = train_and_evaluate(traces, feature_config=cfg, cv_folds=5, seed=0, epochs=100)
    check(
        "train_and_evaluate is deterministic given a seed",
        result.model.classifier.to_dict() == result_again.model.classifier.to_dict()
        and result.model.scaler.to_dict() == result_again.model.scaler.to_dict()
        and result.model.temperature == result_again.model.temperature,
    )

    try:
        train_and_evaluate([], feature_config=cfg)
        check("training on zero traces is rejected", False)
    except ValueError:
        check("training on zero traces is rejected", True)

    try:
        train_and_evaluate([{"state": {}, "label": "ONLY_ONE"}], feature_config=cfg)
        check("training on a single label is rejected", False)
    except ValueError:
        check("training on a single label is rejected", True)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"selftest summary: {passed}/{len(checks)} passed")
    return passed == len(checks)


if __name__ == "__main__":
    raise SystemExit(main())
