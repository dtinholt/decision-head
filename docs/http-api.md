# HTTP API reference

`decisionhead serve` exposes one decision route and five read-only
routes. Bodies are JSON (UTF-8). Every response to
`POST /v1/systemone` carries an `X-Request-Id` header; the same id is on
every audit row the request produced (`docs/audit-log.md`).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/v1/systemone` | bearer token, if configured | Answer typed questions about a state |
| `GET` | `/health` | none | Liveness |
| `GET` | `/ready` | none | Backend reachability |
| `GET` | `/v1/models` | none | The real model id and the accepted alias |
| `GET` | `/metrics` | none | Prometheus text format |
| `GET` | `/openapi.json` | none | OpenAPI 3.0 document |
| `OPTIONS` | any | none | CORS preflight (only with `--cors`) |

Every implemented route, including the GET routes, first checks the
`Host` header:
a missing `Host` is `400`, and a hostname that is not a loopback name,
the bind address or an `--allowed-hosts` entry is `421`. Ports in `Host`
are ignored.

## `POST /v1/systemone`

### Request

```json
{
  "model": "any-alias",
  "state": {"supplier": "Acme Logistics BV", "documents": ["W-9", "certificate of insurance"]},
  "questions": {
    "coi_matches": {
      "type": "noul",
      "instructions": "The entity named on the certificate of insurance matches the supplier."
    },
    "next_step": {
      "type": "choice",
      "instructions": "What should happen next?",
      "criteria": {
        "APPROVE": "onboard the supplier",
        "REQUEST_DOCS": "ask the supplier for a corrected document",
        "ESCALATE": "send to a human reviewer"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this case?",
      "levels": ["low", "medium", "high"]
    }
  }
}
```

Headers: `Content-Type` must have the media type `application/json`
exactly (parameters such as `; charset=utf-8` are accepted). With a
bearer token configured, `Authorization: Bearer <token>` is required.
With CORS off (the default), a request carrying an `Origin` header is
refused.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `state` | string, object or array | yes | What every question is asked about. Rendered once into a prompt prefix shared by every question in the request (objects are rendered as key-sorted, indented JSON). Must not be `null` or `""`. |
| `model` | non-empty string | yes | An alias. The server never routes on it and never rejects a value; the response's `model` is always the real backend model id. |
| `questions` | object | yes | Question id to question definition. At least one; at most `max_questions_per_request` (default 50). |
| `mode` | `"fast"`, `"deliberate"` or `"learned"` | no | Overrides the server's mode for this request when `mode_override_allowed` is true (the default). Ignored otherwise. An unknown value is `400`. On a server started in `learned` mode, `"fast"` or `"deliberate"` is `400` (it has no model backend). |
| `max_concurrency` | positive integer | no | Fast mode only. Overrides the backend concurrency limit for this request when the server was started with `--allow-max-concurrency-override`. Ignored otherwise. |

Question definition:

| Field | Applies to | Meaning |
|---|---|---|
| `type` | all | `"noul"` (yes/no), `"choice"` (one of a set) or `"score"` (an ordered scale). |
| `instructions` | all | Non-empty string: the question. For `noul`, phrase it as a statement or question the model can answer Yes or No to. |
| `options`, `criteria` or `levels` | `choice`, `score` | The answer set: an object (`{"NAME": "description"}`) or a list of names. 2 to 26 entries. Any of the three field names is accepted. `score` levels must be listed lowest first; nothing re-sorts them. |
| `terminal_answers` | `choice`, `score` | Deliberate mode claim guard: a list of option names that mean "done". Each must be one of the options. |
| `completion_evidence` | `choice`, `score` | Deliberate mode claim guard: `{"open_obligations_count": <int >= 0>, "unmet_requirements": [...]}`. Non-empty evidence blocks a terminal answer. |

Every question is answered in its own model call. No question's prompt
contains another question's text.

### Response (`200`)

Exact output of the request above against `decisionhead serve --dry`
(the fake backend; a real model returns the same shape with its own
numbers):

```json
{
  "model": "dry-fake-model",
  "answers": {
    "coi_matches": {"type": "noul", "noul": 0.43606700141356947,
      "unparsed_mass": 0.0, "no_family_match": false, "non_finite_logprob_count": 0, "degraded": false},
    "next_step": {"type": "choice", "choice": "REQUEST_DOCS",
      "probabilities": {"APPROVE": 0.17431143392443454, "REQUEST_DOCS": 0.6513771321511309, "ESCALATE": 0.17431143392443454},
      "confidence": 0.47706569822669637,
      "unparsed_mass": 0.0, "no_family_match": false, "non_finite_logprob_count": 0, "degraded": false},
    "urgency": {"type": "score", "score": 3.0,
      "probabilities": {"low": 0.09831820338110397, "medium": 0.09831820338110397, "high": 0.803363593237792},
      "confidence": 0.7050453898566881,
      "unparsed_mass": 0.0, "no_family_match": false, "non_finite_logprob_count": 0, "degraded": false}
  },
  "usage": {"input_tokens": null, "output_tokens": null},
  "mode": "fast",
  "low_confidence": false,
  "degraded": false
}
```

Envelope fields:

| Field | Type | Meaning |
|---|---|---|
| `model` | string | The real backend model id (or the learned model's name). |
| `answers` | object | One answer per question id. |
| `usage` | object | `input_tokens` and `output_tokens` summed over the request's backend calls, as the backend reported them. `null` when the backend reported nothing; never a made-up `0`. |
| `mode` | string | The mode that actually answered. |
| `low_confidence` | boolean | `true` if any answer is flagged `low_confidence`. Always present. |
| `degraded` | boolean | `true` if any answer is degraded. Always present. |

Answer fields, all modes:

| Field | Type | Meaning |
|---|---|---|
| `type` | string | The question type. |
| `noul` | number or `null` | `noul` only: probability of Yes, in `[0, 1]`. |
| `choice` | string or `null` | `choice` only: the option with the highest probability. |
| `score` | number or `null` | `score` only: the winning level's name if it parses as a finite number, otherwise its 1-based position in the list. |
| `probabilities` | object or `null` | `choice`/`score`: probability per option, summing to 1. |
| `confidence` | number or `null` | `choice`/`score`: top probability minus the second. |
| `degraded` | boolean | `true` when there is no usable reading. All value fields above are then `null`. |
| `low_confidence` | boolean | Present and `true` on a degraded answer, a deliberate answer with `n_missing_samples > 0`, or a `noul` answer inside the configured abstain band. |
| `recommended_action`, `handoff_reason` | string | Present only on a flagged answer when `abstain_mode` is `handoff`. `recommended_action` is `"handoff"`. |

Fast mode adds:

| Field | Meaning |
|---|---|
| `unparsed_mass` | Probability mass among the backend's top tokens that matched no allowed answer token. In the same units as the allowed tokens' mass, so a value near or above the allowed mass means the model mostly wanted to say something else. |
| `no_family_match` | `true` if no allowed token appeared among the top tokens at all. Always implies `degraded`. |
| `non_finite_logprob_count` | Backend logprobs that were `NaN` or infinite. They contribute zero mass. A non-zero value means the backend is misbehaving. |

Calibration, when configured, is applied to `noul` and `probabilities`
before they are returned; `confidence` is computed after calibration.

Deliberate mode adds:

| Field | Meaning |
|---|---|
| `vote_share` | The raw vote share (`noul`: same as `noul`; `choice`/`score`: same as `probabilities`). Kept separate for a future calibration layer. |
| `avg_stated_confidence` | Mean of the confidences the samples stated, or `null`. |
| `n_samples` | Completions the backend returned. |
| `n_parsed` | Samples that cast a valid vote (`noul`) or parsed into an answer object (`choice`/`score`). |
| `n_malformed` | Samples with no usable final JSON object (the object must be the whole final non-empty line of the reply), an off-vocabulary `noul` answer, or beyond `k`. |
| `n_unmatched_answer` | `choice`/`score`: parsed samples whose answer named no option. |
| `n_excess_samples` | Samples beyond `k` (never voted). |
| `n_missing_samples` | How many fewer than `k` samples the backend returned. When above `0`, the answer is `low_confidence` (and handed off in `abstain_mode = "handoff"`): the vote rests on fewer samples than configured. |
| `claim_guard_fired`, `claim_guard_downgraded`, `claim_guard_reason`, `claim_guard_original_choice`, `claim_guard_original_vote_share` | Present only when the claim guard acted. `choice`/`score` is then the downgraded answer; `probabilities` and `vote_share` are unchanged. |

Learned mode answers `choice` questions only, with the core fields above.
A `noul` or `score` question in learned mode is `400`.

### Degraded answers

A degraded answer means the server has no reading for that question:
in fast mode the model's reply contained none of the allowed tokens (a
wrong chat template, or thinking left on, are the usual causes); in
deliberate mode no sample cast a valid vote. The value is `null`, never a
stand-in 0.5 or a default option, the answer is flagged `low_confidence`,
and it is never cached. Treat it as "retry or hand off", not as an
answer.

### Errors

Every error body has this shape, with no stack trace, exception text,
backend URL or backend response:

```json
{"error": {"code": "invalid_request", "message": "question 'q': type must be one of noul, choice, score (got 'maybe')"}}
```

Checks run in this order; the first failure decides the response.

| Status | `error.code` | Cause |
|---|---|---|
| `404` | `not_found` | `POST` to any path other than `/v1/systemone` (checked first), or an unknown `GET` path (checked after `Host`). |
| `400` | `invalid_request` | No `Host` header. |
| `421` | `misdirected_request` | `Host` names a host this server does not answer to. |
| `403` | `forbidden_origin` | An `Origin` header is present and CORS is off. |
| `400` | `invalid_request` | Non-ASCII `Authorization` header (only when bearer auth is configured). |
| `401` | `unauthorized` | Bearer auth is configured and the token is missing or wrong. |
| `415` | `unsupported_media_type` | Media type is not exactly `application/json`. |
| `400` | `invalid_request` | `Content-Length` is not an integer, or is negative. |
| `413` | `payload_too_large` | `Content-Length` exceeds `max_request_bytes`. |
| `400` | `invalid_json` | Nested deeper than `max_json_nesting_depth`; not valid UTF-8 JSON; contains `NaN`/`Infinity`, an out-of-range number such as `1e999`, or a numeric literal over 64 characters. |
| `429` | `too_many_requests` | `max_concurrent_requests` in flight, counting timed-out requests still draining. Sent with `Retry-After: 1`. |
| `400` | `invalid_request` | Envelope or question invalid: missing field, unknown type, fewer than 2 or more than 26 options, too many questions, bad `mode` or `max_concurrency`, malformed `terminal_answers`/`completion_evidence`, an unsupported question type in learned mode, `mode: "learned"` on a server with no model loaded, `mode: "fast"` or `"deliberate"` on a learned-mode server. The message names the problem. |
| `504` | `request_timeout` | The request exceeded `request_timeout_s`. Backend calls not yet started are abandoned. |
| `502` | `backend_error` | The backend failed, refused (for example, it answered with a redirect), or returned something unreadable. Fixed message; the detail is in the server log under the request id. |
| `500` | `internal_error` | Unexpected failure, or the audit row could not be written. Fixed message naming the request id. |

Every non-`200` outcome of `POST /v1/systemone` decided after the
authentication check gets one audit row with its status and error code:
`415`, `400` (`Content-Length`, `invalid_json`, `invalid_request` for
the envelope), `413`, `429`, `504`, `502` and `500`. Rejections decided
before or by the authentication check are not audited, so an
unauthenticated client cannot grow the audit file: `404`, `400` for a
missing `Host` or a non-ASCII `Authorization` header, `421`, `403` and
`401`. They are still counted in `/metrics` by status. With bearer auth
off, every request passes the authentication check.

## `GET /health`

`{"status": "ok"}` whenever the process is serving. Never contacts the
backend.

## `GET /ready`

```json
{"status": "ok", "ready": true, "detail": null}
```

`ready` reports whether the backend answered a cheap probe:
`GET {base}/api/tags` (ollama) or `GET {base}/v1/models` (OpenAI
compatible). The fake backend and learned mode are always ready. The
result is cached for `ready_cache_s` (default 5 s). On failure `detail`
names only the exception type, never the backend URL. The status code is
`200` either way; read `ready`.

## `GET /v1/models`

```json
{"models": [{"id": "dry-fake-model", "aliases": ["jev-latest"]}]}
```

## `GET /metrics`

Prometheus text format, no client library:

* `decisionhead_requests_total{method,path,status}` (counter). `path`
  is one of the routes above or `"other"`, so unknown paths cannot grow
  the label set.
* `decisionhead_request_duration_seconds` (histogram per path).
* `decisionhead_concurrent_requests` (gauge).
* `decisionhead_concurrent_requests_rejected_total` (counter of `429`s).

## `GET /openapi.json`

An OpenAPI 3.0 document built from the same schema fragments the server
validates against (`decisionhead/schema.py`).

## `OPTIONS`

With `--cors`: `204` with `Access-Control-Allow-Origin: *`,
`Access-Control-Allow-Methods: GET, POST, OPTIONS` and
`Access-Control-Allow-Headers: Content-Type, Authorization`. Without it:
`404`.

## Authentication and CORS in brief

* Bearer auth is off by default. `--bearer-token-env NAME` names the
  environment variable holding the token; it gates `POST /v1/systemone`
  only. The comparison is constant-time. If the named variable is unset
  or empty the server refuses to start (`docs/configuration.md`).
* CORS is off by default. While off, browsers cannot drive the service
  from another site: cross-origin POSTs carry `Origin` and get `403`,
  and DNS-rebinding pages arrive under a foreign `Host` and get `421`.
* Neither is a substitute for a gateway in front of a shared service.
  See `SECURITY.md`.
