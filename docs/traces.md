# The trace format, for training `mode: "learned"`

This is the reference for the JSONL file `decisionhead.train` consumes:
the exact schema, who is responsible for the ground truth in a pilot, how
many rows a first model needs, and the commands to train, evaluate, and
sanity-check class balance before you trust the result. For the feature
groups the classifier is trained on and the classifier itself, see
`docs/training.md`; this document is about the data, not the model.

## Schema

One JSON object per line, UTF-8, `\n`-terminated:

```json
{"state": {"...": "the same object /v1/systemone's 'state' field would carry"},
 "label": "INFO",
 "action": "RETRIEVE",
 "meta": {"case_id": "case-0042", "step": 3}}
```

* **`state`** (required) -- opaque to the trainer; `decisionhead.features`
  reads whatever shape it has (see its module docstring for the exact
  fields it looks for: `requirements`, `open_items`, `history`,
  `findings`, `budget`, `steps_used`, and an optional `prior_answers`
  block). This must be exactly what a real `/v1/systemone` request's
  `state` would carry for that same decision point -- not a summary or a
  reformatting of it.
* **`label`** (required, non-empty string) -- the ground-truth decision
  CLASS this state should have produced. This is what
  `decisionhead.train` trains against; there is no fixed class list
  built into the trainer, but the shipped `decisionhead.features`/
  `decisionhead.learned` machinery only knows how to serve a `choice`
  question back out of the four JEV-WC decision classes (`INFO`,
  `REASON`, `DONE`, `BLOCKED`) or the six-action taxonomy that collapses
  into them (`RETRIEVE`, `REQUEST`, `CONTINUE`, `CONSULT_EXPERT`,
  `HANDOFF`, `CLAIM_COMPLETE` -- see `decisionhead.features.ACTION_CLASS`).
  If your workflow's own decision classes are different names for
  materially the same four outcomes (something needs fetching, something
  needs reasoning, the case is done, the case is stuck), label with the
  four class names directly rather than inventing a fifth taxonomy the
  serving side does not know how to answer a `choice` question against.
* **`action`** (optional, string) -- the action actually taken at this
  checkpoint, kept for reference. May or may not agree with `label` (a
  controller's or a human's mistake is exactly the disagreement worth
  keeping a record of). `decisionhead.train` never reads it.
* **`meta`** (optional, object) -- free-form provenance. The ONE field
  the trainer reads is `meta.case_id`: cross-validation folds are
  clustered by it (see below), so every checkpoint from one case lands
  in the same fold. A row with no `case_id` is treated as its own
  singleton cluster. Put whatever else is useful for later debugging
  here (step index, which arm/reviewer produced the row, timestamps) --
  none of it reaches the model.

`decisionhead.traces.validate_trace`/`read_traces` enforce this shape (a
missing `state` or a missing/empty `label` is a hard error, naming the
line number) and are the same functions `decisionhead.train` calls
internally, so a file that passes `read_traces` is guaranteed loadable.

## Who labels, and the "unknown" case

`label` is a human judgement, not something this package infers. In a
pilot, the labeler is whoever already owns the decision the head is
being asked to make -- the case worker, underwriter, or ops lead who
currently signs off on this step of the process. That is deliberate: the
head is being trained to agree with the standard your organization
already holds itself to, not to some external notion of correctness.

Two labelling shapes come up in practice:

* **Shadow mode** (see `docs/pilot.md`'s week 1): the head answers, the
  human decides independently, and both are recorded at the same
  checkpoint. Here `label` is simply the human's decision, and the
  head's own answer belongs in `meta` (e.g. `meta.head_choice`,
  `meta.head_probabilities`) so agreement can be computed later without
  re-running anything.
* **Retrospective labelling**: a log was produced by a running system
  (see `examples/workflow_loop.py`, whose rows always carry the literal
  string `"UNKNOWN"` as a placeholder label) and a human reviews it after
  the fact to assign the real class.

Some rows will never get a confident label -- the case was ambiguous, it
is still open, or nobody who reviewed it was sure. Use the literal string
`"UNKNOWN"` for these (it satisfies the schema's "non-empty string"
requirement, so a file mixing labelled and not-yet-labelled rows still
loads cleanly) and then **exclude every `"UNKNOWN"` row before training**
-- `decisionhead.train` has no concept of an abstention class, so an
`"UNKNOWN"` label left in the file is trained as if it were a real,
meaningful fourth or fifth class, which will actively hurt the model
rather than being harmlessly ignored. Filter with the one-liner in
"Class balance" below (drop the `UNKNOWN` line from its output, or
literally grep it out: `grep -v '"label": *"UNKNOWN"' raw.jsonl >
labelled.jsonl`).

## How many rows

There is no fixed threshold below which training refuses to run --
`decisionhead.train` will fit a model on as few as two distinctly-labelled
rows -- but the cross-validation report becomes unreliable long before
that stops being technically true. In an internal development run on
roughly 200 labelled benchmark checkpoints, the weakest per-class recall
fell on the minority classes: case-clustered
cross-validation groups checkpoints by `meta.case_id`, so with only a few
dozen distinct cases, whichever class happens to have the fewest cases in
a given run gets a noisy, small-sample recall estimate almost by
construction, independent of whether the underlying signal is actually
weak.

**Plan for 500+ labelled rows for a pilot's first trained head**, spread
across enough distinct cases (not just checkpoints) that every decision
class has a meaningful number of CASES behind it, not just rows -- ten
cases that each produce fifty checkpoints of the same class is not the
same as fifty distinct cases. Check the actual balance (below) before
trusting any single run's per-class recall, and re-check it every time
you add more labelled data.

## Class balance check

Before training, look at how the labels are actually distributed:

```
python3 -c "
import json, collections
rows = [json.loads(l) for l in open('dev_traces.jsonl') if l.strip()]
labels = collections.Counter(r['label'] for r in rows)
cases = collections.defaultdict(set)
for r in rows:
    cases[r['label']].add((r.get('meta') or {}).get('case_id'))
total = sum(labels.values())
print(f'{\"label\":14s} {\"rows\":>6s} {\"share\":>7s} {\"cases\":>6s}')
for label, n in labels.most_common():
    print(f'{label:14s} {n:6d} {n/total:6.1%} {len(cases[label]):6d}')
print(f'{\"TOTAL\":14s} {total:6d}')
"
```

Read this before every training run, not just the first one: a class
with very few CASES (the last column, not the row count) is the one
whose `per_class_recall_cv` you should treat with real skepticism, no
matter how good the headline `decision_class_accuracy_cv` looks. If any
class the pilot cares about has fewer than roughly 20-30 cases behind it,
collect more before reading its recall as meaningful, and consider
whether `UNKNOWN` rows are silently sitting in the file (they will show
up as their own row in this table -- drop them before training, per
above).

## Train and evaluate

```
python -m decisionhead.train --traces dev_traces.jsonl --cv 5 --seed 0 \
    --out model.json --report report.json
```

Cross-validation is CASE-CLUSTERED: folds partition rows by
`meta.case_id`, never by individual checkpoint, so no fold ever trains
and tests on two checkpoints from the same case -- the leak that matters
here, since checkpoints from one case share almost all of their state.
This is a dev-set estimate only (`report.json` always carries
`"dev_only": true`); nothing in `decisionhead.train` ever reads an
eval/challenge split, and neither should you until the model is
otherwise ready to ship.

`report.json`'s fields worth reading in order:

* `n_traces`/`n_cases` -- rows vs. distinct cases; the class-balance
  check above is the per-class breakdown of the same thing.
* `majority_baseline_accuracy` -- always-predict-the-most-common-label.
  The floor `decision_class_accuracy_cv` needs to clear to be worth
  anything at all.
* `decision_class_accuracy_cv` -- mean accuracy of held-out, out-of-fold
  predictions. The primary number.
* `per_class_recall_cv` -- per label, `null` if that label never
  appeared in the held-out portion of any fold. Read this against the
  case counts from the balance check, not in isolation.
* `ece_cv` -- expected calibration error of the temperature-scaled
  out-of-fold predictions (same multiclass ECE code
  `decisionhead calibrate` uses for the zero-shot path).

Serve the trained model with:

```
decisionhead serve --mode learned --head-model-path model.json --port 8787
```

See `docs/training.md` for the feature groups, the classifier, and
serving details (embedding features, the action-taxonomy mapping for
`choice` questions, etc.) -- this document stops at the data.
