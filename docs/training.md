# Training a decision head on your own workflow traces

`mode: "fast"` and `mode: "deliberate"` both read a probability off an
LLM, zero-shot, every time. `mode: "learned"` instead answers from a
small classifier trained once, offline, on labelled examples of your own
workflow -- no LLM call at request time at all. This is the mode to use
once you have a few hundred labelled decisions and want a head that
actually learns your workflow's structure instead of reasoning about it
fresh on every request.

## The trace format

`decisionhead.train` consumes a JSONL file, one labelled decision per
line:

```json
{"state": "... same object /v1/systemone's 'state' field would carry ...",
 "label": "INFO",
 "action": "RETRIEVE",
 "meta": {"case_id": "case-0042", "step": 3}}
```

* **`state`** -- opaque to the trainer; `decisionhead.features` reads
  whatever shape it has. This should be exactly what your service would
  receive as a real `/v1/systemone` request's `state`.
* **`label`** -- the decision CLASS this state should have produced (a
  string; the classifier is multinomial over whatever distinct label
  strings appear in your trace file -- there is no fixed class list built
  into the trainer).
* **`action`** (optional) -- the actual action recorded at this
  checkpoint, kept for reference; the trainer never reads it.
* **`meta`** (optional) -- free-form provenance. `meta.case_id`, when
  present, is the ONE field the trainer reads: cross-validation folds are
  clustered by it (see below). A trace with no `case_id` is treated as
  its own singleton cluster.

### The JEV-WC converter

`decisionhead.traces.convert_jwc_trajectories` turns one JEV-WC benchmark
run's `trajectories.jsonl` into this format, deriving `label` from the
benchmark's own oracle (`jwc_score.correct_action_set`, loaded
dynamically from a path you supply -- never vendored into this package)
so the label can never silently drift from the benchmark's own PRIMARY
registered metric. It also lifts a typed controller's real per-question
Noul answers into `state["prior_answers"]`, when the run recorded them,
for feature group (b) below.

```python
from decisionhead.traces import convert_jwc_trajectories, write_traces

traces = convert_jwc_trajectories(
    "path/to/run-dev-A/trajectories.jsonl",
    "path/to/benchmark/jwc_score.py",
    arm="A",
)
write_traces("dev_traces.jsonl", traces)
```

Only ever point this at a `dev` (or your own labelled) split. `eval` and
`challenge` splits exist to be scored once, at the end, by the benchmark
harness itself -- never to inform a trained model's design or
hyperparameters.

## Feature groups

`decisionhead.features.extract_features` builds a fixed-length numeric
vector from a state, in three independently switchable groups
(`FeatureConfig`):

* **(a) structured** (`use_structured`, on by default) -- counts of
  open/closed ledger items, remaining budget, history length and
  trailing-run features, a checkpoint index, any `flags` the state
  exposes, and rule-table-shaped bookkeeping (has RETRIEVE/REQUEST
  already been tried for the focus item, which per-resource budgets are
  exhausted, and the focus item's own finding status including
  `"uncertain"`) -- the signals a hand-written rule table hard-codes and
  a zero-shot LLM head has no direct read on.
* **(b) prior_answers** (`use_prior_answers`, on by default) -- P(yes)
  values from a PRIOR zero-shot pass over the same state, read from
  `state["prior_answers"]` in the wire shape JEV-WC's own typed
  controllers bank (`"<item_id>::q_onfile"` etc, scoped to the state's
  focus item, plus a `next_step` choice answer). Optional at the state
  level too: every value has a documented neutral default and an
  explicit `prior_answers_present` flag, so a state built cold (no prior
  LLM pass at all) still produces a valid, fixed-length vector.
* **(c) embedding** (`use_embedding`, OFF by default) -- a text embedding
  of the state's rendered prefix (the same `render_state` the serving
  path uses), via an ollama-native or OpenAI-compatible embeddings
  endpoint (`decisionhead.embeddings`). Requires `--embedding-dim` and a
  reachable embeddings backend; exercised in this package's own tests
  only against a deterministic fake, since no such endpoint was reachable
  from the environment this package was built in.

Feature order matters and is saved in the model file, so serving
reproduces training's exact layout regardless of which groups were on.

## The classifier

Multinomial logistic regression with L2 regularization, trained by
full-batch gradient descent (`decisionhead.model`). Pure Python by
default; numpy accelerates the same algorithm if installed
(`pip install decisionhead[train]`) but is never required -- nothing on
the serving path imports it, so a serving-only install stays
zero-dependency. Deterministic given a seed. Features are standardized
(zero mean, unit variance) on the training data before fitting; the
fitted scaler travels in the saved model file.

## Training and cross-validation

```
python -m decisionhead.train --traces dev_traces.jsonl --cv 5 --seed 0 \
    --out model.json --report report.json
```

Cross-validation is CASE-CLUSTERED: folds partition traces by
`meta.case_id`, never by individual checkpoint, so no fold ever trains
and tests on two checkpoints from the same case. The report
(`--report`) is a JSON object carrying `decision_class_accuracy_cv` (mean
accuracy of held-out, out-of-fold predictions), `per_class_recall_cv`,
`ece_cv` (expected calibration error, reusing `decisionhead.calibrate`'s
own multiclass ECE code), and `majority_baseline_accuracy` -- always
marked `"dev_only": true`, since this package never reads eval/challenge
data.

Feature groups are toggled from the CLI: `--no-structured`,
`--no-prior-answers`, `--use-embedding --embedding-dim N
--embedding-backend ... --embedding-base ... --embedding-model ...`.

The SAVED model is refit on every trace given (cross-validation is for
the report, not the artifact); its temperature-calibration is fit on the
pooled cross-validation out-of-fold predictions -- the only
calibration-honest set of held-out probabilities available, since fitting
against the final model's own training predictions would calibrate
against data it has already memorized.

## Serving

```
decisionhead serve --mode learned --head-model-path model.json --no-mode-override --port 8787
```

`--no-mode-override` matters here: a learned-mode server has no LLM
behind it, so a request that asks for `"mode": "fast"` or
`"deliberate"` would fail with a `500`. With the flag, every request is
answered in learned mode.

Only `choice` questions are answered in learned mode (the classifier's
output is a distribution over decision classes, which has no defined
meaning for a `noul` or `score` question -- both raise a clear
`QuestionError`). A `choice` question's own `options` must be either
exactly the trained class names, or a subset of the six JEV-WC action
names (`RETRIEVE`, `REQUEST`, `CONTINUE`, `CONSULT_EXPERT`, `HANDOFF`,
`CLAIM_COMPLETE`) -- each class's probability mass is split evenly among
the requested actions in that class. Class probabilities are exposed as
the answer's ordinary `probabilities` field and are temperature-scaled by
the model's own fitted temperature, using the exact same
`sigmoid(logit(p)/T)` transform `mode: "fast"` calibration uses
(`decisionhead.core.apply_calibration_distribution`). Every learned-mode
answer is written through the same `AuditLog` as every other mode.

If the loaded model was trained with embedding features on, serving also
needs `--embedding-backend`/`--embedding-base`/`--embedding-model` (and
`--embedding-api-key-env` for an OpenAI-compatible endpoint) so the
server can embed each incoming state the same way training did.
