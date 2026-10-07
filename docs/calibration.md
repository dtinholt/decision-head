# Calibration walkthrough

`decisionhead calibrate` fits one temperature per question type (`noul`,
`choice`, `score`) so the raw probabilities coming off the model's
token-level logprobs (see `docs/design.md`) match observed correctness on
your own labelled cases.

## The labelled-JSONL format

One JSON object per line:

```json
{"state": "... same shape as a /v1/systemone request's state ...",
 "questions": {"q_onfile": {"type": "noul", "instructions": "..."}, "...": "..."},
 "answers_true": {"q_onfile": 1, "next_step": "REQUEST"}}
```

* `state` and `questions` are exactly what you would send to
  `/v1/systemone`.
* `answers_true` gives the correct answer per question id: `0` or `1` for
  a `noul` question, the correct option/level *name* (a string) for
  `choice`/`score`. A question present in `questions` but absent from
  `answers_true` is simply not used for that item (useful if you only
  have ground truth for some questions).

`examples/calibration_example.jsonl` has 60 synthetic items built this
way, with an intentionally obvious rule: each state literally contains
`STATUS: APPROVED` or `STATUS: DENIED`, and `answers_true` is `1` exactly
when it says `APPROVED`. The point of that file is to make this
walkthrough runnable and checkable end to end with no model and no
network -- not to be a realistic labelled set. Run it against the
deterministic fake backend:

```
decisionhead calibrate --labelled examples/calibration_example.jsonl \
    --dry --out calibration_demo.json
```

You should see output like:

```
noul: T=20.000 n=60 brier 0.3767 -> 0.2505, ece 0.3560 -> 0.0223
wrote calibration_demo.json
```

The fitted temperature lands at the edge of the search grid here because
the fake backend's per-question distributions are a deterministic hash of
the question text -- not actually correlated with `answers_true` -- so
there is no real signal to calibrate against. That is expected and is
exactly why this file is for exercising the *mechanism*, not for producing
a calibration you would ever load into a real server. Point `--base`/
`--model` (dropping `--dry`) at a real backend and a real labelled set to
get a calibration that means something.

## Running it for real

```
decisionhead calibrate --labelled cases.jsonl --backend ollama \
    --base http://203.0.113.10:11434 --model qwen3.8:27b --out calibration.json
```

* A question type needs at least 50 labelled items or it is skipped, with
  a message saying so, rather than fitting a temperature on noise. If no
  type reaches 50, the command exits 1 and writes nothing.
* `T` is found by a coarse-to-fine grid search (log-spaced, then a finer
  linear pass around the best point) over roughly `[0.05, 20]`, minimizing
  the negative log likelihood of the true label under
  `p' = sigmoid(logit(p) / T)` -- the standard temperature-scaling
  objective (Guo et al., 2017), applied elementwise and renormalized for
  `choice`/`score`. No `scipy.optimize`; the codebase is stdlib-only.
* `calibration.json` carries, per type: the fitted `T`, the item count
  `n`, Brier score and ECE (10 bins) before and after calibration, and a
  full reliability table (`mean_predicted` vs. `observed_frequency` per
  bin) so a fitted calibration can be inspected, not just trusted.
  * For `noul`, the scored scalar is `P(yes)` against the 0/1 truth --
    textbook binary Brier/ECE.
  * For `choice`/`score`, there is no single natural probability, so the
    scored scalar is the model's own top probability (its confidence in
    whichever option it picked) against whether that pick was correct --
    the standard multiclass generalization (confidence vs. accuracy).
    This is a design choice, not a literal reading of any spec; it is
    called out here so it can be second-guessed.
* Calibration is fit **per type**, not per question. A specific question
  that is biased in an unusual direction on its own will not be fixed by a
  type-level temperature -- check the reliability table for that before
  trusting a fitted calibration on a mixed question set.

Feed the output back into a server with `--calibration calibration.json`.
Every fast-mode `noul`/`choice`/`score` answer is then calibrated before
it is returned. Deliberate mode (vote shares) and learned mode (which
carries its own temperature in the model file) ignore this file. The
server refuses to start if any temperature in the file is not a finite
number greater than zero, logs the file's digest at start-up
(`calibration_loaded`), and records it on every fast-mode audit row as
`calibration_version` next to `calibrated` (see `docs/audit-log.md`).
