# Deliberate mode

`decisionhead serve` has three answering strategies, selected by `mode`
(`--mode`, `DECISIONHEAD_MODE`, the TOML key `mode`, or a per-request
`"mode"` override; see below). This page covers `deliberate` and how it
differs from `fast`. `learned`, which calls no LLM at all, is in
`docs/training.md`.

* **`fast`** (default) -- the original mechanism: one model call per
  question, `temperature=0`, exactly one generated token, and `P` read
  directly off that token's logprobs. Cheap (one generation per
  question) and a genuine model belief, but a single committed token has
  no room to reason.
* **`deliberate`** -- `k` sampled model calls per question at
  `temperature>0`, each asking the model to reason briefly and then
  answer with a strict JSON object; `P` is the vote share across the `k`
  samples. More expensive (`k` generations per question instead of one),
  and gives the model room to reason before committing, at the cost of a
  probability that is a vote share, not a calibrated logprob-derived
  number -- see **Calibration is not solved by voting** below.

Everything else about how a request is handled -- one state prefix per
request, one question per model call (never leaked into another
question's prompt), request size limits, the JSON-nesting depth guard,
per-request and per-connection timeouts, the concurrency cap, bearer
auth, CORS, and the audit log -- is identical between the two modes; only
how one question's `P` gets computed differs. This is by design: the two
modes share one server, one hardening story, and one audit sink.

## Cost

`fast` mode makes one model call per question. `deliberate` mode makes
`k` (default 8) -- so a request with the same question set costs roughly
`k` times as many generations, and proportionally more wall-clock time
unless the backend supports one request per `n` samples (see below).
`bench_report.json` (`decisionhead bench`) measures `fast` mode only
today; there is no separate deliberate-mode benchmark subcommand yet --
budget for it by multiplying the `fast`-mode per-question latency by `k`
requests (serial, for a backend without native `n` support) or by the
single-request latency at `n=k` (for one that has it), plus reasoning
tokens (`deliberate_max_tokens`, default 256, versus fast mode's fixed
1-token generation).

## Configuration

All of these are `ServerConfig` fields (TOML `[server]` keys, `--flag`s,
or `DECISIONHEAD_*` environment variables, same precedence order as
everything else -- see `decisionhead/config.py` and
`decisionhead.example.toml`):

| Field | CLI flag | Default | Meaning |
|---|---|---|---|
| `mode` | `--mode` | `"fast"` | `"fast"` or `"deliberate"` |
| `mode_override_allowed` | `--no-mode-override` | `true` | whether a request's own `"mode"` field may override `mode` for that one request |
| `deliberate_k` | `--deliberate-k` | `8` | sampled answers per question |
| `deliberate_temperature` | `--deliberate-temperature` | `0.7` | sampling temperature |
| `deliberate_max_tokens` | `--deliberate-max-tokens` | `256` | max tokens per sampled answer (reasoning + JSON) |
| `claim_guard_enabled` | `--no-claim-guard` | `true` | see **Claim guard** below |
| `claim_guard_threshold` | `--claim-guard-threshold` | `0.75` | minimum vote share to accept a terminal claim |

Example: running against a local OpenAI-compatible server (vLLM) in
deliberate mode, defaults otherwise:

```
decisionhead serve --backend openai --base http://127.0.0.1:8000 \
    --model deepseek-v41-flash-exl3 --mode deliberate \
    --deliberate-k 8 --deliberate-temperature 0.7 \
    --port 8787 --audit audit.jsonl
```

or the equivalent TOML (`decisionhead.example.toml` has the full key
list):

```toml
[server]
backend = "openai"
base_url = "http://127.0.0.1:8000"   # no /v1: the backend appends /v1/chat/completions
model = "deepseek-v41-flash-exl3"
mode = "deliberate"
deliberate_k = 8
deliberate_temperature = 0.7
```

Against ollama instead: `--backend ollama --base http://127.0.0.1:11434
--model qwen3.8:27b --mode deliberate` -- ollama's native `/api/chat` has
no `n` parameter, so this always makes `k` separate requests per question
(see **One request or k requests** below); everything else is unchanged.

A one-off deliberate request against a server left at `mode = "fast"`
(with `mode_override_allowed` at its default `true`): add `"mode":
"deliberate"` to the request body. Set `mode_override_allowed = false`
(or `--no-mode-override`) to pin every request to the server's configured
mode regardless of what a request asks for.

## One request or `k` requests

An OpenAI-compatible backend's native `n` parameter lets `k` samples come
back from **one** HTTP request; `OpenAIBackend.sample_completions` tries
that first when `k>1`, and falls back to `k` separate `n=1` requests if
the `n=k` request either fails (`BackendError`) or comes back with no
usable choices (a server that silently ignores `n`). ollama's native
`/api/chat` has no `n` parameter at all, so `OllamaBackend` always makes
`k` separate requests. Either way this is recorded on the audit row
(`sampling_requests`, `n_param_used`, `fallback_used`) so it is auditable
which path a given backend actually took.

## Answer shape

A deliberate-mode `noul` answer adds `vote_share` (identical to `noul`
itself -- exposed explicitly so a calibration layer can be fit against it
later without also having to remember which field was raw) and
`avg_stated_confidence` (mean of the samples' self-reported `confidence`,
or `null` if none reported one) to the usual `{"type", "noul"}` shape. A
`choice`/`score` answer adds the same two fields, plus `n_samples` and
`n_parsed` (how many of the `k` samples actually parsed into a typed
answer -- see **Malformed samples** below). Full field reference:
`docs/http-api.md`, "Deliberate mode adds".

## Malformed samples

A sample that reasons but never produces a parseable
`{"answer": ..., "confidence": ...}` object is excluded from the vote,
not scored as a wrong or missing answer -- `n_parsed`/`n_malformed` on
each answer and audit row show how many of the `k` samples that was. A
`noul` answer other than yes/no (case-insensitive, one trailing `.` or
`!` ignored; JSON `true`/`false` accepted) is malformed too, never a
"no", and so is any sample beyond `k` that a backend returns. If no
sample casts a valid vote, the answer is `"degraded": true` with a
`null` value, not `0.5` or a uniform distribution (2026-10-06 security
review, finding 6). `decisionhead.deliberate.parse_deliberate_reply`
does the extraction: it reads only an object that is the whole final
non-empty line of the completion (after trimming, and ignoring trailing
blank lines and a closing code fence), with string state scanned from
the start of that line only, so a stray quote in the reasoning cannot
shift which object is read (the model is asked to reason first and put
the JSON alone on the last line). Prose before the object on that line,
or a second object after it, makes the sample malformed. A backend that
returns fewer than `k` samples gets the shortfall in `n_missing_samples`,
and the answer is flagged `low_confidence`. An earlier object, such as JSON quoted from the case document, is
never used as a fallback. It tolerates a markdown code fence around it.
If strict parsing of that one object fails, it retries the same object
after two cheap repairs (smart or single quotes to double quotes, a
trailing comma removed) before counting the sample as malformed. Numbers
must be finite and at most 64 characters long.

## Claim guard

A vote share is not a calibrated probability. In an internal
development-set evaluation on the synthetic supplier-onboarding
benchmark (k=8), deliberate mode's raw vote share roughly matched a
hand-written rule table on decision-class accuracy but made wrong "done"
claims several times as often, and its vote share was badly calibrated:
a terminal "this case is complete" answer at, say, 70% vote share is
nowhere near 70% likely to be correct. That evaluation is unpublished,
so its numbers are not repeated here.

The claim guard exists for exactly that failure mode. A question opts in
by naming its terminal answer(s):

```json
{
  "type": "choice",
  "instructions": "Should this case be closed?",
  "criteria": {"CLAIM_COMPLETE": "close the case", "CONTINUE": "keep working it"},
  "terminal_answers": ["CLAIM_COMPLETE"],
  "completion_evidence": {"open_obligations_count": 0, "unmet_requirements": []}
}
```

When the winning vote is a `terminal_answers` entry, it is only returned
as-is if **both** hold: the winning vote share is `>= claim_guard_threshold`
(default 0.75), and, when the question supplies `completion_evidence`,
that evidence is empty (`open_obligations_count` is 0/absent and
`unmet_requirements` is empty/absent). Otherwise the answer is downgraded
to the highest-vote-share option that is *not* itself terminal, and the
downgrade is recorded on both the answer (`claim_guard_downgraded`,
`claim_guard_reason`, `claim_guard_original_choice`,
`claim_guard_original_vote_share`) and the audit row. The raw
`vote_share`/`probabilities` are never altered by the guard -- only
`choice`/`score` (the field a caller acts on) is. A question that never
sets `terminal_answers` is never guarded, whatever it answers; the guard
can be turned off server-wide with `claim_guard_enabled = false` /
`--no-claim-guard`. See `decisionhead.deliberate.apply_claim_guard`.

## Calibration is not solved by voting

Fast mode's `decisionhead calibrate` (temperature scaling on
`logit(p)`) is fit for a single-token logprob read and is **not** applied
to deliberate mode's vote share -- a vote share does not sit on the same
scale a token-logprob-derived probability does, and the claim guard above
is a stopgap for the one failure mode (terminal claims) an eval has
actually measured, not a general calibration fix. `vote_share` and
`avg_stated_confidence` are exposed on every answer specifically so a
calibration layer (e.g. isotonic regression or its own temperature scale,
fit on labelled deliberate-mode runs) can be built on top later without
this package needing to guess the right transform today.

## Calling deliberate mode from an existing controller

A workflow engine or benchmark harness that already builds typed
questions about a case can hand the answering step to this service
without changing its own decision logic:

1. Map each of its questions to this API's
   `{"type", "instructions", "criteria"}` shape.
2. For a question whose answer can close the case, set
   `terminal_answers` to the closing option(s), and fill
   `completion_evidence` from the engine's own ledger (open obligations,
   unmet requirements). This is what engages the claim guard.
3. `POST /v1/systemone` with the engine's compact state and
   `"mode": "deliberate"` (or rely on the server's default). The server's
   `deliberate_k` and `deliberate_temperature` govern sampling.
4. Read `response["answers"][qid]` back. The deliberate-mode fields
   (`vote_share`, `avg_stated_confidence`, `claim_guard_*`, the sample
   counts) are additive; a caller that only wants the decision can ignore
   them, and an evaluation can log them to study calibration and
   wrong-claim rates.
5. Feed the answers into the engine's existing decision table unchanged.

The engine keeps the books; the service only judges.
`examples/workflow_loop.py` is a runnable version of this division of
labour against the fast mode.
