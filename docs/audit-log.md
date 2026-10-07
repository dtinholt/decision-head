# The audit log

`decisionhead serve --audit audit.jsonl` appends one JSON object per line
for every outcome of `POST /v1/systemone`:

* a **request that succeeded** (`200`) writes one row per question;
* **any other outcome decided after the authentication check**
  (invalid, too many requests, timed out, failed) writes one row for the
  request. Rejections decided before or by the authentication check
  (`400` for a missing `Host` or non-ASCII `Authorization`, `401`, `403`,
  `421`) and `404` write no row, so an unauthenticated client cannot grow
  the file; `/metrics` still counts them.

The file is created with mode `0600` and only ever appended to. Rows are
strict JSON (no `NaN` or `Infinity`). If a success row cannot be written,
the client gets `500` instead of the answer: no decision is returned
without its audit row. The server holds no rotation or retention policy;
rotate the file with your own tooling (copy-truncate is safe because each
write opens the file in append mode).

The audit log does **not** contain the state or the question text. It
contains a hash that lets you prove which state and questions a row
belongs to (see "Reconstructing a decision" below). Keep the requests
themselves wherever your workflow already keeps case records.

This page covers the per-decision log. `docs/audit/README.md` covers the
separate evidence artefacts (calibration report, stability report and
so on) that support trusting a deployment.

## A real row

From `decisionhead serve --dry --audit audit.jsonl`, for the `next_step`
question of the request in `docs/http-api.md` (line breaks added):

```json
{"request_hash": "a8ddec8ba9bbb60ffa5d0bb9dd50a1a9757d1b3a8e5d3dd976be4cd1d3a914e8",
 "question_id": "next_step",
 "model": "dry-fake-model",
 "backend": "FakeBackend",
 "calibrated": false,
 "latency_s": 1.0013580322265625e-05,
 "cache_hit": false,
 "unparsed_mass": 0.0,
 "no_family_match": false,
 "non_finite_logprob_count": 0,
 "degraded": false,
 "raw_distribution": {"APPROVE": 0.17431143392443454, "REQUEST_DOCS": 0.6513771321511309, "ESCALATE": 0.17431143392443454},
 "letters": {"APPROVE": "A", "REQUEST_DOCS": "B", "ESCALATE": "C"},
 "ts": "2026-10-06T22:22:40.610Z",
 "request_id": "81c16f5b89444043b381907858e7197c",
 "status": 200,
 "mode": "fast",
 "calibration_version": "",
 "answer": {"type": "choice", "choice": "REQUEST_DOCS",
            "probabilities": {"APPROVE": 0.17431143392443454, "REQUEST_DOCS": 0.6513771321511309, "ESCALATE": 0.17431143392443454},
            "confidence": 0.47706569822669637,
            "unparsed_mass": 0.0, "no_family_match": false, "non_finite_logprob_count": 0, "degraded": false}}
```

And a refused request (a body sent as `text/plain`):

```json
{"ts": "2026-10-06T22:22:50.351Z", "request_id": "48fb6e92f1ce45a3aee7be5df9c3f23e",
 "status": 415, "error": "unsupported_media_type", "mode": null, "request_hash": null, "n_questions": null}
```

## Fields on every success row

| Field | Type | Meaning |
|---|---|---|
| `ts` | string | When the response was produced: UTC, millisecond precision, `Z` suffix. All rows of one request share it. |
| `request_id` | string | 32 hex characters. Equal to the `X-Request-Id` response header and to the `request_id` on the server's stderr log lines for this request. |
| `status` | integer | `200`. |
| `mode` | string | `fast`, `deliberate` or `learned`: the mode that answered. |
| `question_id` | string | The question's id from the request. |
| `model` | string | The real model id (or the learned model's name). |
| `backend` | string | The backend class (`OllamaBackend`, `OpenAIBackend`, `FakeBackend`) or `learned`. |
| `request_hash` | string | SHA-256 over the state, the questions and the model id (see below). Deliberate rows append `-deliberate`. |
| `latency_s` | number | Seconds spent in the backend call (or sampling batch) for this question. |
| `answer` | object | The answer exactly as returned to the client, after calibration and the abstain band. |
| `low_confidence` | boolean | Present and `true` when the answer was flagged. |
| `recommended_action`, `handoff_reason` | string | Present when the answer was flagged and `abstain_mode` is `handoff`. |
| `non_finite_replaced` | integer | Present only if the encoder had to replace a non-finite number with `null`. Should never appear; if it does, report it. |

## Fast mode

| Field | Meaning |
|---|---|
| `raw_distribution` | The probabilities read off the model **before** calibration: `{"Yes": p, "No": 1 - p}` for `noul`, or per option name for `choice`/`score`. On a degraded row this is the backend's uniform placeholder and carries no information. |
| `letters` | `choice`/`score`: the letter each option was presented under. The model answered with one letter. |
| `calibrated` | Whether a calibration temperature was applied to this question's type. |
| `calibration_version` | SHA-256 of the loaded calibration document (empty string when none). Identifies exactly which calibration produced `answer`. |
| `unparsed_mass` | Mass on tokens outside the allowed answers. |
| `no_family_match` | No allowed token among the model's top tokens. |
| `non_finite_logprob_count` | Backend logprobs that were `NaN` or infinite. |
| `degraded` | `true` when there was no usable reading; `answer` then has `null` values. |
| `cache_hit` | `true` when the answer came from the response cache instead of a backend call. |
| `original_trace_id` | Cache hits only: the `request_id` of the request that computed the answer. That request has its own audit rows, with the original `latency_s`; a cache-hit row repeats that original latency rather than measuring a new call. |

## Deliberate mode

| Field | Meaning |
|---|---|
| `k`, `temperature`, `max_tokens` | The sampling settings used. |
| `n_samples`, `n_parsed`, `n_malformed`, `n_unmatched_answer`, `n_excess_samples` | Sample accounting, as in `docs/http-api.md`. |
| `votes` | Vote counts: `{"yes": n, "no": n}` for `noul`, per option name for `choice`/`score`. |
| `result` | The voted outcome before it became `answer`: `P(yes)` for `noul`, the chosen option name for `choice`/`score` (after the claim guard), or `null` when degraded. |
| `degraded` | No sample cast a valid vote. |
| `sampling_requests` | HTTP requests made for this question (1 when an OpenAI-compatible server honoured `n`, otherwise `k`). |
| `n_param_used`, `fallback_used` | Whether the native `n` parameter was used, and whether the server fell back to `k` separate requests after `n` failed. |
| `claim_guard_fired`, `claim_guard_downgraded`, `claim_guard_reason`, `claim_guard_original_choice`, `claim_guard_original_vote_share` | Present only when the claim guard acted. The reason records counts only, never text from the request. |

Deliberate rows carry no `calibration_version`: calibration is not
applied to vote shares.

## Learned mode

| Field | Meaning |
|---|---|
| `raw_distribution` | The model's probability per decision class (after the model's own temperature), before mapping onto the question's options. |
| `calibrated` | Whether the model file carries a temperature other than 1. |
| `unparsed_mass` | Always `0.0`. |

## Rows for requests that did not succeed

| Field | Type | Meaning |
|---|---|---|
| `ts` | string | When the outcome was recorded (UTC, `Z`). |
| `request_id` | string | Equal to the response's `X-Request-Id`. |
| `status` | integer | The HTTP status returned (`400`, `413`, `415`, `429`, `500`, `502`, `504`). |
| `error` | string | The `error.code` returned, or `audit_write_failed` when the success rows could not be written. |
| `mode` | string or `null` | The mode, if the request got far enough to resolve one. |
| `request_hash` | string or `null` | SHA-256 over the state and questions **only** (no model id), once the envelope is valid and the mode is resolved. It therefore differs from the success-row hash of the same body. |
| `n_questions` | integer or `null` | Number of questions, once known. |

A `404` (wrong path) and pre-authentication rejections (`401`, `403`,
`421`, and `400` for a missing `Host` or non-ASCII `Authorization`) are
not audited.

## Reconstructing a decision

The audit row answers "what did the service return, from which model,
under which calibration, and how sure was the reading". To rebuild one
decision:

1. **Find the rows.** Every row of one request shares a `request_id`,
   which is also the `X-Request-Id` your client received and the id on
   the server's log lines:

   ```
   grep '"request_id": "81c16f5b89444043b381907858e7197c"' audit.jsonl
   ```

2. **Prove the inputs.** Take the state and questions from your own case
   record and recompute the hash. A match proves this row answered
   exactly that state and those questions:

   ```python
   import hashlib, json

   def request_hash(state, questions, model):
       payload = json.dumps({"state": state, "questions": questions, "model": model},
                            sort_keys=True, default=str)
       return hashlib.sha256(payload.encode("utf-8")).hexdigest()

   # compare with row["request_hash"] (strip a trailing "-deliberate" first)
   ```

   Use the row's `model` field, not the alias the client sent. The
   separators must be Python's defaults (`", "` and `": "`).

3. **Prove the calibration.** Compute the same digest over your
   calibration file and compare it with `calibration_version`:

   ```python
   cal = json.load(open("calibration.json"))
   hashlib.sha256(json.dumps(cal, sort_keys=True, default=str).encode("utf-8")).hexdigest()
   ```

   The digest is over the parsed document re-serialized with sorted
   keys, not over the file's bytes, so whitespace changes do not alter
   it.

4. **Recompute the returned number** from `raw_distribution` and the
   calibration temperature `T` for the question's type, with
   `p' = sigmoid(logit(p) / T)` (probabilities clamped to
   `[1e-9, 1 - 1e-9]` before the logit). For `choice`/`score`, apply it to
   every option and divide by the sum. With `calibrated: false`,
   `answer` equals `raw_distribution` (`noul`: `answer.noul ==
   raw_distribution.Yes`).

5. **Read the confidence of the reading.** `unparsed_mass`,
   `no_family_match`, `non_finite_logprob_count` and `degraded` say how
   much of the model's output the reading could account for. A decision
   built on a row with a large `unparsed_mass` deserves a second look
   even when it is not degraded.

6. **Re-ask, if needed.** With the same state, questions, model and
   calibration, fast mode at `max_concurrency = 1` reproduces the same
   reading on backends that are deterministic at temperature 0 (the
   README's "Deterministic serving" explains why concurrency matters).
   Deliberate mode samples at temperature above 0 and is not expected to
   reproduce exactly; its `votes` are the record.

## What the audit log does not record

* Who asked. There is one shared bearer token and no caller identity
  (`SECURITY.md`). Put a gateway in front if you need that.
* The state and question text (by design; see above).
* Tamper evidence. Rows are not chained or signed; protect the file with
  file-system permissions and ship it to append-only storage if you need
  that guarantee.
