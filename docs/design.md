# Design notes

## The mechanism: reading a probability off logprobs

A typed question with a fixed, small answer vocabulary (yes/no, or one of
a handful of named options) can be answered with a single generated
token. If the serving engine returns per-token log-probabilities for that
one token, the probability mass on each candidate answer token, read
straight off the model's own distribution, is a real probability: it did
not come from sampling many replies and counting, and it did not come
from asking the model to state a confidence in words.

Verified against a local `qwen3.8:27b` on ollama, 2026-09-23: a single
call with `num_predict:1`, `temperature:0`, and `logprobs:true` on the
**native** ollama route, `POST {base}/api/chat`, returns:

```json
{"message": {"content": "Yes"},
 "logprobs": [{"token": "Yes", "logprob": -0.1378,
               "top_logprobs": [{"token": "Yes", "logprob": -0.1378},
                                 {"token": "yes", "logprob": -2.097},
                                 {"token": " yes", "logprob": -5.90},
                                 {"token": "No", "logprob": -6.5}]}]}
```

`P(yes)` is the exp-mass of every yes-variant token divided by the
exp-mass of every yes-variant plus every no-variant token, matched case-
and whitespace-insensitively (`decisionhead.backends._normalize_token`
and `distribution_from_top_logprobs`). Whatever mass fell on neither
family (some other token the model considered) is recorded as
`unparsed_mass` rather than dropped -- it is exactly the share of the
model's attention at that position this reading could not account for.
Choice and score questions work the same way over a single lettered-option
token (A, B, C, ...) instead of Yes/No.

If no allowed token appears anywhere in the returned `top_logprobs` at
all, the reading carries no information. Since the 2026-10-06 security
review (finding 6) such an answer is returned as `"degraded": true` with
a `null` value (not the uniform 0.5 the backend layer computes as a
placeholder), is flagged `low_confidence`, is never cached, and the
response envelope carries `"degraded": true`. Every fast-mode answer and
audit row also carries `unparsed_mass`, `no_family_match` and
`non_finite_logprob_count`, so partial degradation is visible too.

A backend that reports a non-finite logprob (`NaN`, `+inf`, `-inf`, or a
value so large `math.exp` cannot represent it, e.g. from a bug or a
compromised backend) is handled the same way: `decisionhead.backends._mass`
treats it as zero mass -- never propagated into any sum -- rather than
letting it corrupt every probability derived from that sum or, worse,
reach the HTTP response as a bare `NaN`/`Infinity` token (not valid JSON
per RFC 8259). Each such entry is counted in
`non_finite_logprob_count` (on the answer and the audit row) so the
anomaly stays visible instead of being silently indistinguishable from a
normal, trustworthy zero. Added after an independent security review
(2026-09-23, finding 4); see `docs/security-review-2026-09-23.md`.

The 2026-10-06 review (finding 4) found other ways a non-finite number
could still reach a response: a bare `NaN` literal in the request body,
a score level named `"inf"`, a calibration temperature of `NaN`, a
deliberate sample's `NaN` confidence, and a string `"inf"` in a
learned-mode budget field. `decisionhead.safejson` now parses every
untrusted document strictly (`NaN`/`Infinity`, overflowing literals and
numeric literals over 64 characters raise `ValueError`), each of those
sources validates its own values, and every response, audit row and
cache row is encoded with `allow_nan=False` after a last-resort pass
that turns any surviving non-finite value into `null` and records
`non_finite_replaced`.

## Why not the ollama `/v1` route

Ollama's OpenAI-compatible `/v1/chat/completions` route on a thinking
model routes through ollama's thinking wrapper and, in testing, returned
garbage for this purpose: the literal token `"The"` instead of `"Yes"` or
`"No"`. `decisionhead.backends.OllamaBackend` therefore only ever calls
the native `/api/chat` route with `think: false`. A real OpenAI-compatible
server (vLLM and similar) does not have this problem; thinking is disabled
there with `chat_template_kwargs: {"enable_thinking": false}` in the
request body (`OpenAIBackend`), and the same per-token logprobs read is
applied to `choices[0].logprobs.content[0].top_logprobs`.

## Isolation, concretely

"Evaluated in isolation" means no question's prompt ever contains another
question's text. All questions about one state share one prefix (a system
turn plus a user turn holding the state, built once in
`decisionhead.core.answer_questions`); each question is then a separate
final turn, in its own model call. This is also what lets a serving
engine's prefix cache make question 2..N of a request cheaper than
question 1, since every question in a request shares an identical prefix
(measured in `decisionhead bench`).

`decisionhead.core.isolation_check()` answers the same questions a second
way -- sharing a prefix that also lists every other question's
instructions, while still asking for one single-token answer at a time --
and reports where the answer moved (`abs_prob_diff` for noul, `l1_prob_diff`
for choice/score, plus a `decision_changed` flag per question). That is
the direct, checkable version of the isolation claim: the single-token
generation format is unchanged in both cases, so the comparison isolates
the effect of context pollution, not of a different answer format.

## Thread pool without `concurrent.futures`

`answer_questions()` partitions questions into `max_workers` buckets and
runs one thread per non-empty bucket, each writing to its own slice of a
preallocated results list, so no lock is needed for the results
themselves. The server's audit-log file writer and in-memory metrics are
the things genuinely shared across requests, and those take an explicit
lock.

## Server hardening choices

* **Request size limit** (default 1 MiB): enforced from the
  `Content-Length` header before the body is read, so an oversized
  request is rejected without materializing it.
* **Per-question timeout**: the backend's own HTTP transport timeout
  (`--question-timeout-s`, passed through to `OllamaBackend`/
  `OpenAIBackend`'s constructor `timeout`). This bounds a single model
  call.
* **Per-request timeout** (`--request-timeout-s`): layered on top, via
  `decisionhead.server._run_with_timeout`, which runs the whole
  `answer_questions()` call on a worker thread and returns a `504` to the
  client if it has not finished within the deadline. Stdlib gives no
  clean way to cancel a thread blocked in a socket call, so the one
  backend call per worker thread already in flight at the deadline runs
  to completion. Everything queued behind it is abandoned
  (`decisionhead.deadline`, added after the 2026-10-06 review, finding
  1): the request's `Deadline` is passed into `answer_questions` /
  `answer_questions_deliberate` / `answer_questions_learned`, set as the
  thread-local current deadline in every worker thread, and checked
  before every backend call and every sample request; on a timeout the
  handler also cancels it. `ConcurrencyLimitedBackend` waits for its
  semaphore with `acquire(timeout=...)`, capped by the deadline's
  remaining time and by `request_timeout_s`, and re-checks the deadline
  once it has a slot.

  The bound this gives: after a `504`, a request causes at most one more
  backend call per worker thread (`min(max_workers, questions)` calls),
  each one already started before the deadline and each bounded by the
  socket timeout `min(--backend-timeout-s, --question-timeout-s)` per
  read (not per call: a backend that drips bytes can stretch one call
  further). No backend call starts after the deadline. Before this fix
  the documented bound ("the per-question timeout times roughly
  `questions / max_workers`") ignored the process-wide backend semaphore:
  orphaned calls from many timed-out requests queued behind each other
  without limit and kept the backend busy indefinitely.
* **Max questions per request** and **max concurrent requests**: simple
  bounds checked before any backend call; the concurrency bound uses a
  non-blocking `threading.Semaphore.acquire(blocking=False)` so a request
  over the limit gets an immediate `429` rather than queuing. A request's
  slot is released by whoever finishes last: by the handler if the
  request never reached a worker thread, otherwise by the worker thread
  when it actually ends -- so a timed-out request whose in-flight call is
  still draining keeps counting against the limit, and `429` starts
  firing once abandoned work accumulates (2026-10-06 review, finding 1).
* **Per-connection read timeout** (`--connection-read-timeout-s`, default
  30s), added after an independent security review (2026-09-23; see
  `docs/security-review-2026-09-23.md`, finding 1): sets
  `Handler.timeout`, which `socketserver.StreamRequestHandler.setup()`
  turns into `self.connection.settimeout(...)` on the raw socket.
  `http.server.BaseHTTPRequestHandler.handle_one_request` already catches
  the resulting `TimeoutError` around the request-line read, the header
  read, and the handler method itself (which covers a stalled body read
  too, since `do_POST` calls `self.rfile.read(length)` from inside that
  same method), and closes the connection with no response and no
  traceback. Without this, a peer that sends a request slowly (or not at
  all) held a server thread open indefinitely -- the slowloris pattern --
  and could do so with no bearer token at all, since the header read
  happens before authentication is ever checked. This mitigation is
  partial by construction (see "Limits" below): the timeout resets on
  every byte received, so a peer sending data just faster than the
  timeout interval is not caught by it.
* **A JSON-nesting depth guard applied before parsing** (finding 2):
  `decisionhead.schema.json_nesting_too_deep` scans the raw request body
  for `{`/`[`/`}`/`]` structure (skipping string contents, including
  escaped quotes) in one linear, recursion-free pass, and the request is
  rejected with `400 invalid_json` if it nests deeper than
  `--max-json-nesting-depth` (default 32) -- before `json.loads()` is
  ever called. `json.loads()` has no depth limit of its own and, on a
  sufficiently deep body, raises `RecursionError` once it exhausts the
  interpreter's recursion limit; that exception type was not one the
  request handler caught, so it previously propagated out of the
  connection entirely (socketserver's own fallback error handler printed
  an unstructured traceback to stderr and the client got no HTTP response
  at all, not the documented 400). The `except` around `json.loads()` now
  also catches `RecursionError` directly as a second, independent layer
  in case the depth guard's limit is configured above what the
  interpreter can actually recurse through.
* **`Content-Length` is validated before it sizes a read** (finding 2):
  a negative value (`int()` happily parses `"-5"` with no sign check) is
  rejected with `400` instead of being passed to `self.rfile.read(-5)`,
  which raised an uncaught `ValueError`. An absurdly long digit string is
  already handled: CPython 3.11+'s `sys.get_int_max_str_digits()` guard
  makes `int()` raise `ValueError` (caught by the existing handler)
  rather than spend unbounded CPU converting it.
* **Bounded `/metrics` labels** (finding 3): `_metrics_path_label` maps
  any requested path outside the server's actual routes to a single
  `"other"` label before it is recorded, so an unauthenticated caller
  cannot grow the in-memory metrics state (or the `/metrics` response
  size) without bound simply by requesting many distinct nonexistent
  paths -- the standard Prometheus rule that a label's value set must be
  bounded and not attacker-controlled.
* **Bind guard**: `--host` defaults to `127.0.0.1`; any other value
  requires the explicit `--allow-non-local-bind` flag
  (`decisionhead.server.check_bind_allowed`), so exposing the service on
  the network is a decision, not a default. Since the 2026-10-06 review
  (finding 7) it also requires a bearer token, unless `--insecure-no-auth`
  is passed (logged as a WARNING).
* **Host allowlist, exact media type, Origin refusal** (2026-10-06
  review, finding 2): every request's `Host` must name a loopback
  address, the bind address, or an entry in `--allowed-hosts`, else
  `421` -- a DNS-rebinding page arrives under the attacker's hostname.
  `POST /v1/systemone` requires the media-type essence (the part before
  `;`) to equal `application/json`; the old substring test accepted
  `text/plain; charset=application/json`, which browsers send
  cross-origin without a preflight. With CORS off, a request carrying
  `Origin` is refused with `403`.
* **Outbound HTTP** (finding 3): backend calls go through an opener whose
  redirect handler refuses every redirect (CPython's default copies
  `Authorization` to whatever host a redirect names), only `http`/`https`
  URLs are accepted, response bodies are capped at 8 MiB, and a
  plain-`http` URL to a non-loopback host logs a start-up warning.
* **Bearer token**: read only from an environment variable named by
  `--bearer-token-env`, compared as bytes with `hmac.compare_digest`
  (constant-time; a non-ASCII header is a `400`), never accepted as a
  flag value and never logged -- neither is the variable's name, since an
  operator may paste the token there by mistake; a value that is not a
  valid variable name is refused at start-up without being echoed.
* **No stack traces or internal text in responses**: every error path
  returns `{"error": {"code", "message"}}` with an `X-Request-Id`
  header. A `500` carries a fixed message naming the request id, and a
  `502` a fixed "backend failed" message; the exception detail (with URL
  userinfo stripped) goes only to the structured stderr log under that
  request id (2026-10-06 review, finding 10).
* **Audit trail** (finding 8): one row per question for a successful
  request, carrying `ts`, `request_id`, `mode`, `status`, the returned
  `answer`, the degradation signals and (fast mode) `calibration_version`;
  deliberate rows add the vote counts and the result. Every other outcome
  gets one request-level row with its status and error code. Answers
  enter the response cache only after the request has succeeded
  (`answer_questions(..., defer_cache_writes=True)`), so a cache hit's
  `original_trace_id` always names a request with its own audit rows.
* **Structured logs**: one JSON object per line to stderr
  (`decisionhead.logging.JSONLogger`), with a `request_id` on every
  request-scoped line so one request's lines can be grepped out.

## Other design choices worth stating

* **Calibration on choice/score, generalized from the noul case.** See
  `docs/calibration.md` and `decisionhead/core.py`'s module docstring.
* **Score's numeric value.** If the winning level's own name parses as a
  number (`"1"`, `"2.5"`), that number is the score. Otherwise the score
  is the level's 1-based rank in the order given. Levels must be supplied
  least to greatest; nothing here re-sorts them, because names like
  "low"/"medium"/"high" carry no order the code can infer on its own.
* **Usage accounting is read from the provider, not estimated.** If a
  provider reports neither `input_tokens` nor `output_tokens`, the
  response's `usage` is `{"input_tokens": null, "output_tokens": null}`
  rather than a fabricated zero.
* **The server never validates the `model` field beyond "non-empty
  string."** It is intentionally a pass-through alias, not a routing key,
  so a client written against a hosted Jev-shaped API works against this
  server with only a base-URL change.
* **`/health` vs `/ready`.** `/health` never touches the backend (pure
  liveness); `/ready` does, but lazily and with a short cache, so a tight
  orchestrator polling loop does not itself become load on the backend.

## Limits

This inherits the base model's judgment completely; a small local model
will not out-reason a frontier hosted one, and nothing here changes that.
It is a single-token reading of a single position in the model's output --
it tells you what the model believed at the moment it had to commit to
one token, not what it would conclude given room to reason. It is not a
substitute for actual code on arithmetic or dates: asking a model "is this
date within 30 days" as a yes/no logprob question is exactly the kind of
task that should be computed in code, with the model asked only questions
that genuinely require judgment. Calibration is per question TYPE, not
per question: a specific question that is systematically biased in an
unusual direction will not be fixed by a type-level temperature, and the
reliability table in `calibration.json` is exactly the artifact to check
for that before trusting a fitted calibration.

## Residual risks (from the 2026-09-23 independent security review)

The full review is kept at `docs/security-review-2026-09-23.md`
(verdict: safe to publish after fixes; all findings closed). Its own
residual-risk list, which holds regardless of the fixes made in response
to it:

* **This service is not an authentication or authorization boundary**
  (see `SECURITY.md`). The bearer token is a shared secret with no
  per-caller identity, no scopes, and no per-caller rate limit. Anyone
  who has the token can ask anything, and there is no way to tell two
  legitimate callers apart in the audit log. Put a real gateway in front
  if more than one trusted caller needs access.
* **The connection-read-timeout fix (finding 1) is a mitigation, not a
  complete fix**, and it matters even with a bearer token configured,
  because the vulnerable code path (reading the request) runs before
  authentication is ever checked. A peer sending data just faster than
  `--connection-read-timeout-s` is not caught by it, and
  `ThreadingHTTPServer` still has no cap on the total number of
  concurrent connections. Treat "behind a gateway" as necessary but not
  sufficient, since most gateways proxy slow-client behavior straight
  through unless they impose their own read timeout; see `SECURITY.md`.
* **The service fully trusts whatever backend it is pointed at**
  (`SECURITY.md` says this explicitly). A compromised or buggy
  ollama/vLLM instance can skew every probability this service reports.
  The non-finite-logprob fix (finding 4) closes the one specific failure
  mode where that could also defeat the `no_family_match` safety net
  silently; nothing else about a compromised backend's output is
  validated. Run the backend on infrastructure you control.
* **`state` is rendered verbatim into the model's context** with no
  prompt-injection defense (`SECURITY.md` already says this). The
  single-token, fixed-vocabulary answer format substantially narrows the
  blast radius but does not eliminate it. This is a disclosed design
  tradeoff, not a bug.
* **The Docker image** used to bind `0.0.0.0` with no token by default.
  Since the 2026-10-06 review (finding 7) it binds `127.0.0.1` unless
  told otherwise, a non-loopback bind refuses to start without a bearer
  token, and the compose files publish decisionhead on host loopback
  only, with a token from `.env`, and no longer publish ollama. The base
  images are still referenced by tag, not digest (see the closure table,
  `docs/security-review-2026-10-06-closures.md`).
