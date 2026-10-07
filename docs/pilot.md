# A two-week pilot on one real process

This is a plan for piloting `decisionhead` against one real decision
point in one real process -- not a rollout plan, and not a sales pitch.
Pick a process with a real human decision-maker who currently makes this
call, a state that can be rendered as text or JSON, and a decision that
happens often enough to accumulate a few hundred instances in two weeks.

## Week 1: shadow mode

Run the process exactly as it runs today. At the checkpoint you are
piloting, additionally call `decisionhead` (`mode: "fast"` or
`"deliberate"`, against whatever local model you already have) with the
same state the human sees, and log both answers to one JSONL file at the
same checkpoint: the head's answer (choice, probabilities, confidence)
in `meta`, and the human's actual decision as `label` (see
`docs/traces.md` for the exact row shape and who does the labelling --
in shadow mode, always the human already making this call today, never a
retrospective reviewer). The human's workflow does not change: they are
not shown the head's answer, and it does not influence their decision.
This is the point of shadow mode -- you get a real agreement measurement
against a decision the head had no chance to bias.

By the end of week 1 you should have every checkpoint from every case
that passed through the pilot's chosen decision point, plus the audit log
(`--audit audit.jsonl`) `decisionhead serve` writes automatically -- keep
both. Run the class-balance check from `docs/traces.md` partway through
the week, not just at the end: if one decision class is barely
occurring, that is useful to know before you have spent the whole week
finding out.

## Week 2: train and measure

Filter out any `"UNKNOWN"`-labelled rows (per `docs/traces.md`), then
train:

```
python -m decisionhead.train --traces week1_shadow.jsonl --cv 5 --seed 0 \
    --out model.json --report report.json
```

Read `report.json`'s `decision_class_accuracy_cv`, `per_class_recall_cv`,
and `ece_cv` first -- these are the DEV-SET numbers, useful for judging
whether training worked at all, but they are not the pilot's go/no-go
evidence, because they are cross-validated against the same shadow-mode
population the model was trained on. The go/no-go evidence is a second
comparison: run `mode: "learned"` (this trained model) and a simple
rule-based baseline (whatever hand-written logic, if any, the process
uses today, or a majority-class rule if there is none) against the SAME
week 1 cases, and compare both to the human's recorded decision.

**Go/no-go criteria** -- numbers, not adjectives, so a reader can
disagree with them before the pilot starts rather than after:

* **Decision-class agreement**: the learned head's agreement rate with
  the human's recorded label must exceed the rule baseline's agreement
  rate by **at least 0.05** (5 percentage points). Below that, the head
  is not earning its keep over what you already have.
* **Wrong-claim rate**: the rate at which the head claims a terminal
  outcome (approved, done, complete) that the human's own record later
  shows was wrong must be **at or below the human's own rate** of the
  same mistake on the same cases. A head that is more accurate on
  average but more willing to wrongly claim "done" is not a net
  improvement -- this is the specific failure mode `docs/deliberate.md`'s
  claim guard exists for, and it is worth checking even in `mode:
  "fast"`/`"learned"`, where there is no guard to catch it automatically.
* **Calibration**: ECE (`ece_cv` from the report, or recomputed against
  week 1 held out from training if you have enough data to hold any
  out) **at or below 0.1**. A head whose stated confidence does not
  track its actual accuracy is not safe to route on confidence, whatever
  its raw accuracy is.

Any one of these missed is a clear no-go for moving the decision point to
`mode: "learned"` as-is -- not a verdict on the approach, but a signal to
collect more labelled data, revisit the question wording, or keep running
in shadow mode a while longer before trying again.

## What "in-network" means here

No data leaves the network you run this on. Concretely:

* The model server (ollama or an OpenAI-compatible endpoint) runs on
  your own hardware, on your own network; `decisionhead serve --base`
  points at an address on that network, never a public one, unless you
  explicitly configure otherwise.
* The state and every question about it exist only in the HTTP request
  between `decisionhead` and that model server, and in whatever you
  choose to log (`--audit audit.jsonl`, your own shadow-mode trace file)
  on local disk. Neither ever leaves the machine or network they are
  written to.
* `model.json` (the trained head) and `calibration.json` hold only
  fitted numeric weights, a feature-name list, and small metadata
  (training date, hyperparameters, row counts) -- no case data, no raw
  state text. They can be copied between environments, or handed to
  someone reviewing the pilot, without carrying any case data with them.
* Nothing in this package makes an outbound network call except to the
  backend address and, if configured, an embeddings backend address you
  supplied yourself (`--base`/`--embedding-base`); there is no telemetry,
  no phone-home, and no third-party API call unless you deliberately
  point `--backend openai` at one.
* `--allow-non-local-bind` stays off (the default) unless you have a
  specific reason to expose the service beyond localhost, in which case
  put it behind your own network controls and a bearer token
  (`--bearer-token-env`) -- see `SECURITY.md`.
