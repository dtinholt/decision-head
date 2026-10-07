# Audit artefacts

This page is about evidence artefacts for a deployment. For the row the
server writes per decision, see `docs/audit-log.md`.

This repo publishes six artefacts as the evidence base for trusting a
deployed decision head: three that already exist today (calibration
report, stability report, reproduction commands), and three that are
specified here as the scaffold for work not yet built (eval report,
threshold sweep, abstention report). Each entry below names the exact
command that produces (or, for the not-yet-built three, will produce) it,
and where the file lands.

Status key: **live** (the command exists and this repo's tests exercise
it) / **planned** (specified here, not yet implemented -- tracked as
follow-up work, not silently assumed to exist).

## 1. Eval report -- **planned**

Accuracy/agreement of the decision head against a HELD-OUT labelled eval
set (distinct from any set used to fit calibration or train a learned
head) -- the number a go/no-go deployment decision should actually be
made on.

* **Learned mode** already has the closest thing to this today: cross-
  validated accuracy on the training set itself (not a true held-out
  eval set, but the mechanism is in place).

  ```
  python -m decisionhead.train --traces traces.jsonl --cv 5 \
      --out model.json --report eval_report.json
  ```

  `eval_report.json`'s `decision_class_accuracy_cv`,
  `per_class_recall_cv`, and `majority_baseline_accuracy` are the numbers
  to read; see `docs/training.md`.

* **Fast/deliberate (zero-shot) mode** has no dedicated eval command yet.
  Until one exists, `decisionhead calibrate`'s before/after Brier score
  (see artefact 2) on a labelled eval file is the closest available
  substitute -- it is fit on the SAME file it reports on, so it is a
  training-set number, not a held-out one; treat it as a lower bound on
  real eval error, not a substitute for one.

## 2. Calibration report -- **live**

Per-question-type (`noul`/`choice`/`score`) fitted temperature, plus
before/after Brier score, before/after ECE (10 bins), and a full
reliability table.

```
decisionhead calibrate --labelled cases.jsonl --backend ollama \
    --base http://203.0.113.10:11434 --model qwen3.8:27b --out calibration.json
```

Refuses to fit a type with fewer than 50 labelled items. Full
walkthrough, labelled-JSONL format, and a runnable offline example
against `--dry`: `docs/calibration.md`. Tests: `tests/test_calibrate.py`.

## 3. Threshold sweep -- **planned**

How deployment metrics (coverage, accuracy on the answers NOT abstained,
handoff rate) move as `abstain_band` (`abstain_band_lo`/`abstain_band_hi`; see
`docs/configuration.md`) widens or narrows,
against a labelled eval set.

Planned command shape, once built:

```
python -m decisionhead.threshold_sweep --labelled cases.jsonl \
    --bands "0.45-0.55,0.4-0.6,0.35-0.65" --out threshold_sweep.json
```

For each band: fraction of decisive answers falling inside it
(coverage lost to abstention), and accuracy on the remaining answers,
reusing `decisionhead.core.apply_abstain_band` against each labelled
item's true answer.

## 4. Multi-seed / multi-repeat stability report -- **live**

`N` repeats of a fixed 20-question set, all at `backend.max_concurrency =
1` (see `decisionhead/backends.py`'s `ConcurrencyLimitedBackend` --
serializing backend calls is what makes this measurement mean anything;
see finding in `decisionhead.example.toml`'s `[backend]` section),
reporting per-question distinct-value counts (not just a first-vs-rest
disagreement share -- a backend can vary its raw logprobs run to run even
when the decisive answer does not change) and the disagreement share.

```
python -m decisionhead.bench --dry --repeats 20 --stability --out bench_report.json
# or against a real backend:
python -m decisionhead.bench --backend ollama --base http://203.0.113.10:11434 \
    --model qwen3.8:27b --repeats 20 --stability --out bench_report.json
```

Writes `bench_report.stability.json` (machine-readable) and
`bench_report.stability.md` (human-readable) --
`decisionhead.bench.stability_report_paths` derives both names from
`--out`. Tests: `tests/test_bench.py::TestStabilityBench`,
`TestStabilityReportIo`.

A genuine "multi-SEED" variant (repeating with different `dry_seed`
values, or different model checkpoints) is not yet wired up as a distinct
mode; `--repeats` currently means "ask the same fixed questions about the
same fixed state N times against the same running backend," which is the
multi-REPEAT half of this artefact's name.

## 5. Abstention report -- **planned**

How often the configured `abstain_band` actually fires in practice, and
what happens to the handed-off cases -- distinct from the threshold
SWEEP above (which asks "what band should I pick"); this asks "given the
band I picked, what did it do." The abstain-band fields on
each flagged audit row (`low_confidence`,
`recommended_action`/`handoff_reason` when `abstain_mode = "handoff"` --
see `decisionhead.core.apply_abstain_band`) are exactly the fields this
report will aggregate.

Planned command shape, once built:

```
python -m decisionhead.abstention_report --audit audit.jsonl --out abstention_report.json
```

Expected fields: total decisive answers, count/share flagged
`low_confidence`, count/share of those that became a handoff, and a
per-question-id breakdown (some questions abstain far more than others,
which is itself a signal worth surfacing).

## 6. Reproduction commands -- **live** (this file)

Every artefact above names its exact producing command. There is
deliberately no separate "run everything" script yet: each artefact has
a different required input (a labelled eval file for 1/2/3/5, nothing but
a reachable backend for 4) and a different cadence (4 is cheap enough to
run on every deploy; 1/2/3/5 need a labelled set and are run when that
set changes). A given engagement's own repo (not this one) is the right
place for a single "run our full audit" script that calls each of these
with its own paths filled in.
