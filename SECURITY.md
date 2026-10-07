# Security policy

## Reporting a vulnerability

Please do not open a public GitHub issue for a security report. Instead,
use GitHub's private vulnerability reporting
(Security tab -> "Report a vulnerability") on
https://github.com/dtinholt/decision-head, or email the maintainer listed
in that repository's profile. Include what you found, how to reproduce
it, and its impact as you understand it. Expect an initial response
within a few days.

## Threat model

Read this before deploying decisionhead anywhere it can be reached by
anyone other than you.

**decisionhead is not an authentication or authorization boundary.** The
optional bearer token (`--bearer-token-env`) is a shared secret that gates
`/v1/systemone`, useful for keeping a casual, unauthenticated client off
the service on a shared network. It is not a substitute for a real
identity system: there is no per-caller identity, no scopes, no rate
limiting per caller (only a global concurrency cap), and no audit trail
of *who* asked a question, only *that* it was asked. **Put this service
behind your own gateway, VPN, or service mesh if it needs to be reachable
by more than one trusted caller**, and let that layer handle
authentication, per-caller authorization, and rate limiting.

**Running without a bearer token** (the default on loopback) means any
process on the same machine can use the decision service and the model
behind it, and so can any web page open in a browser on that machine
that finds a way past the Host, Origin and Content-Type checks below.
Those checks stop the known browser routes (cross-origin "simple"
requests and DNS rebinding), but they are not authentication. Set
`--bearer-token-env` whenever the machine is shared, runs a browser, or
the port is reachable from anywhere else; a non-loopback bind refuses to
start without one. If the variable named by `--bearer-token-env` is
unset or empty, the server refuses to start on every bind address;
`--insecure-no-auth` is the only way to serve a non-loopback bind without
a token.

**What decisionhead does defend against**, on its own:

* Binding to a non-loopback interface requires an explicit
  `--allow-non-local-bind` flag -- the default is loopback-only -- AND a
  bearer token (a non-empty variable named by `--bearer-token-env`),
  unless `--insecure-no-auth` is also passed, which logs a WARNING. The
  container image binds `127.0.0.1` by default, and the compose files
  publish decisionhead on host loopback only, with a token, and do not
  publish the bundled ollama at all.
* Requests whose `Host` header is not a loopback name, the bind address
  or a name in `--allowed-hosts` are refused with `421` (missing `Host`:
  `400`), on every route; this is what stops a DNS-rebinding page from
  reading responses. `POST /v1/systemone` requires the media type
  `application/json` exactly (parameters such as `; charset=utf-8` are
  fine; `text/plain; charset=application/json` is `415`), and, with CORS
  off, a request carrying an `Origin` header is refused with `403`.
* A request that exceeds `--request-timeout-s` gets `504` and its
  remaining backend work is abandoned: every question or sample not yet
  sent is dropped at the deadline, a wait for a backend concurrency slot
  gives up at the deadline, and the request keeps its concurrency slot
  until its worker thread has actually finished, so `429` fires while
  abandoned work is still draining. The one backend call already in
  flight at the deadline cannot be interrupted (stdlib offers no way to
  cancel a blocked socket read) and runs to completion, bounded by
  `min(--backend-timeout-s, --question-timeout-s)` per socket operation.
* A per-connection socket read timeout (`--connection-read-timeout-s`,
  default 30s) bounds how long the server will wait for the next byte of
  a request line, header, or body before dropping the connection -- this
  runs before authentication is checked (the header read happens first),
  so it applies to every caller regardless of whether a bearer token is
  configured. See "residual risks" below for what this does and does not
  cover, and `docs/design.md` for the mechanism.
* Request size, per-request/per-question timeouts, a cap on questions per
  request, and a cap on concurrent requests bound the resource a single
  caller (trusted or not) can consume *once a request has been fully
  read*; see `docs/design.md`.
* A `Content-Length` header is validated (non-negative, within
  `--max-request-bytes`) before it is ever used to size a read, and a
  request body is scanned for excessive JSON nesting depth before it is
  ever handed to the JSON parser -- both a malformed length and a
  maliciously deep body return the documented `400`, not a dropped
  connection or a stack trace.
* `/metrics` records only the small, fixed set of routes this server
  actually implements as a label value; any other requested path is
  recorded under a single shared `"other"` label, so an unauthenticated
  caller cannot grow the metrics state (or the size of the `/metrics`
  response) without bound by requesting many distinct nonexistent paths.
* Non-finite numbers (NaN, infinity) are refused at every entry point:
  request bodies (`NaN`/`Infinity` literals, overflowing literals like
  `1e999`, and numeric literals over 64 characters are a `400`),
  calibration files (each temperature must be a finite number > 0, or the
  server refuses to start), learned-model files, learned-mode features
  (a string `"inf"` in a budget field is a `400`), embeddings vectors,
  numeric-looking score level names (they fall back to their rank), and
  deliberate-mode replies (a `NaN` confidence makes the sample
  malformed). A non-finite logprob from the backend contributes zero mass
  and is counted in `non_finite_logprob_count`. Responses, audit rows
  and cache rows are encoded with `allow_nan=False`; as a last line of
  defence any non-finite value that still slipped through becomes `null`
  and the document records `non_finite_replaced`.
* A degraded reading is never passed off as an answer. Every fast-mode
  answer and audit row carries `unparsed_mass`, `no_family_match` and
  `non_finite_logprob_count`. When the model's reply contained none of
  the allowed answer tokens (or, in deliberate mode, no sample cast a
  valid vote), the answer is `"degraded": true` with a `null` value
  instead of a plausible-looking 0.5, is flagged `low_confidence` (and
  handed off in `--abstain-mode handoff`), the response envelope carries
  `"degraded": true`, and the reading is not cached.
* Deliberate mode reads only a JSON object that is the whole final
  non-empty line of each sampled reply, scanning string state from that
  line only; JSON quoted earlier (for example from the case document) is
  never a fallback vote, and a stray quote in the reasoning cannot shift
  which object is read. A backend that returns fewer than `k` samples
  is reported in `n_missing_samples` and the answer is flagged
  `low_confidence`. A noul
  answer other than yes/no counts as malformed, never as "no"; at most
  `k` samples are voted however many the backend returns; and the
  claim-guard inputs (`terminal_answers`, `completion_evidence`) are
  type-checked, so a malformed one is a `400`.
* Every error response is a small, fixed-shape JSON object carrying the
  `X-Request-Id` header. `500` and `502` responses carry fixed messages
  only -- no exception type or text, no backend URL, no backend response
  body; the detail goes to the server log under the same request id,
  with URL userinfo stripped. A non-ASCII `Authorization` header is a
  `400`, not a dropped connection.
* Backend HTTP calls refuse every redirect (so the backend API key is
  never forwarded to another host, and an `https` base is never
  downgraded), refuse any URL scheme other than `http`/`https`, and stop
  reading a response body past 8 MiB. A plain-`http` backend URL on a
  non-loopback host logs a WARNING at start-up, since the case state and
  any API key cross the network unencrypted.
* The bundled client (`decisionhead.client`, `decisionhead ask`) uses the
  same redirect-refusing opener, so this server's bearer token is never
  forwarded on a redirect; a redirect is raised as `ClientError`.
* The audit log (`--audit`) gets one row per question for a successful
  request and one row per request for every other outcome decided after
  the authentication check (`400`, `413`, `415`, `429`, `5xx`,
  timeouts). Rejections decided before or by it (`401`, `403`, `421`,
  and `400` for a missing `Host` or non-ASCII `Authorization`) are
  counted in `/metrics` only, so an unauthenticated client cannot grow
  the audit file. Each row carries `ts` (UTC), `request_id`, `mode` and
  `status`; success rows also carry the answer returned, and deliberate
  rows the vote counts and the result. The audit and cache files are
  created owner-only (`0600`). If the audit write fails the request
  fails with `500` rather than answering unaudited. Only a request that
  completed successfully writes to the response cache.
* No secret (an API key for a backend, or this server's own bearer token)
  is ever accepted as a command-line flag or written to a log line; both
  are read from a *named* environment variable at the moment they are
  needed. A `*_env` value that does not look like a variable name (for
  example, the token itself pasted by mistake) is refused at start-up
  without being echoed, and the names themselves are never logged.

**What decisionhead does not defend against**, because it is out of
scope for a self-hosted single-tenant service, or because the mitigation
is inherently partial:

* A peer that sends bytes *faster* than `--connection-read-timeout-s`
  (i.e. never lets a single read stall long enough to hit the timeout)
  can still hold a thread open for longer than a legitimate request would
  need, since the timeout resets on every byte received, not on total
  connection age. This bounds the attack to costing the attacker
  continuous, non-trivial traffic per stalled connection -- a materially
  higher bar than the previous "send once and disappear" version of the
  same attack, which cost nothing after the first packet -- but it is not
  a complete fix. `http.server.ThreadingHTTPServer` also has no cap of
  its own on the total number of concurrent connections (as opposed to
  `--max-concurrent-requests`, which only gates `/v1/systemone`
  processing after a request is fully read); a sufficiently resourced
  peer can still open many connections up to the OS's own thread/file
  descriptor limits. Put a reverse proxy with its own connection limits
  and slow-client protection in front if this service is reachable by
  untrusted clients.
* A malicious or compromised backend (the ollama/vLLM instance it talks
  to). decisionhead trusts the backend's response shape; a backend that
  returns crafted logprobs or samples can influence the probabilities
  decisionhead reports. What is caught: non-finite logprobs, readings
  with no allowed token (reported as degraded), redirects, oversized
  bodies, and excess deliberate samples, as described above. A backend
  that drips its response slowly can still hold one worker per in-flight
  call for up to the socket timeout per read. Run the backend on
  infrastructure you control.
* Multi-tenant isolation. There is one bearer token, one audit log, one
  set of rate limits, shared by every caller. If you need per-tenant
  isolation, run one decisionhead instance per tenant or put a
  multi-tenant-aware gateway in front.
* Prompt injection via the `state` field. decisionhead renders `state`
  verbatim into the model's context; if an attacker controls `state` and
  the model's output otherwise influenced downstream action beyond the
  fixed single-token answer read here, that is the same class of risk any
  LLM-backed system has. The single-token, fixed-vocabulary answer format
  substantially narrows this in fast mode (the only thing ever extracted
  is which one of a small, caller-defined set of tokens got the most
  probability mass), but it does not eliminate the model being influenced
  by adversarial content in `state`. Deliberate mode lets the model
  reason freely before answering, which widens the steering surface;
  only the final JSON object of each reply is counted, so text the
  document plants is not counted directly, but the model can still be
  persuaded by it. Learned mode reads structured state fields, which the
  caller controls by definition.

## Independent reviews

Two independent pre-publication reviews have been run against this code.
Each finding, the change that closed it, and its regression test are
recorded alongside the review:

* `docs/security-review-2026-09-23.md` (all findings closed in 0.1.0).
* `docs/security-review-2026-10-06.md`, with the closure table in
  `docs/security-review-2026-10-06-closures.md`. The must-fix findings
  are closed; the items deferred to a point release, and why, are listed
  at the end of the closure table.

## Supported versions

Only the latest released minor version receives security fixes while this
project is pre-1.0.
