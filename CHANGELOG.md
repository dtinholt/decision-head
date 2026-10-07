# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

This section records the closures of the independent security review
dated 2026-10-06 (`docs/security-review-2026-10-06.md`; per-finding
closure table with tests in `docs/security-review-2026-10-06-closures.md`),
followed by the features added since 0.1.0.

### Security

Closes the must-fix findings of the 2026-10-06 review.

- A request that times out no longer leaves its remaining backend work
  running: queued questions and samples are abandoned at the request
  deadline (`decisionhead.deadline`), backend semaphore waits time out,
  and the request keeps its concurrency slot until its worker thread
  ends, so `429` fires while abandoned work drains (finding 1, HIGH).
- Host-header allowlist (`--allowed-hosts`, `421` otherwise), exact
  `application/json` media type, `403` for cross-origin requests while
  CORS is off (finding 2).
- Backend calls refuse redirects and non-http(s) URLs and cap response
  bodies; a plain-http, non-loopback backend URL logs a start-up warning
  (finding 3).
- No `NaN`/`Infinity` in or out: strict JSON parsing of requests, model
  replies, calibration and cache files; finite-only score values,
  calibration temperatures, learned-mode features and model files; strict
  encoding of responses, audit rows and cache rows (finding 4).
- Deliberate mode counts only each reply's final JSON object,
  off-vocabulary noul answers as malformed, and at most `k` samples;
  claim-guard inputs are type-checked (finding 5).
- Degraded readings are explicit: `"degraded": true` with a `null` value,
  `low_confidence`, never cached; every fast-mode answer and audit row
  carries `unparsed_mass`, `no_family_match`, `non_finite_logprob_count`
  (finding 6).
- The container binds loopback by default; a non-loopback bind requires a
  bearer token (or `--insecure-no-auth`); compose publishes decisionhead
  on host loopback only and no longer publishes ollama (finding 7).
- Audit rows carry `ts`, `request_id`, `mode`, `status` and the returned
  answer (deliberate rows: vote counts and result); every non-200 outcome
  is audited; only successful requests write to the cache (finding 8).
- Fixed client-facing messages for `500`/`502`; non-ASCII
  `Authorization` is a `400`; `*_env` values are validated and never
  logged; `backend_timeout_s` is wired (min with `question_timeout_s`,
  float); numeric limits are range-checked; audit and cache files are
  created `0600` (findings 9 to 13, partial where noted in the closure
  table).
- Private context removed from the tree: the real-data trace test reads
  its paths from `DECISIONHEAD_JWC_SCORE` and
  `DECISIONHEAD_JWC_DEV_A_TRAJECTORIES` and skips when they are unset;
  private repository paths in docstrings, examples and docs are replaced
  with neutral descriptions; the private address and paths quoted in the
  two review documents are redacted. Publishing from a fresh repository
  without the monorepo history is described in `docs/PUBLISHING.md`
  (finding 16).

Closes residual items R1 to R4 and R6 from the review's confirmation
section, and two code findings from the documentation pass (tests in
`tests/test_security_confirmation_2026_10_06.py`). R5 is open.

- A worker thread that fails to start no longer leaks its concurrency
  slot: the hand-off is recorded only after `Thread.start()` returns, and
  the slot is released on failure (R1).
- Rejections decided before or by the authentication check (`401`,
  `403`, `421`, and `400` for a missing `Host` or non-ASCII
  `Authorization`) no longer write audit rows, so an unauthenticated
  client cannot grow the audit file; they are still counted in
  `/metrics` (R2).
- Deliberate mode reads a sample's JSON only when the object is the
  whole final non-empty line of the reply, scanning string state from
  that line only, so a stray quote in the reasoning cannot let a
  document-embedded object cast the vote (R3).
- A deliberate answer for which the backend returned fewer than `k`
  samples carries `n_missing_samples` (also on its audit row) and is
  flagged `low_confidence` (R4).
- `decisionhead.client` uses the backend path's redirect-refusing opener,
  so the service bearer token is never forwarded on a redirect; a
  redirect is a `ClientError` (R6).
- **Breaking**: if `--bearer-token-env` names a variable that is unset
  or empty, the server refuses to start on every bind address. It used
  to serve loopback requests unauthenticated while logging that they
  would get `401`; the `bearer_token_env_unset` warning is gone.
- A learned-mode server answers a request that overrides the mode to
  `fast` or `deliberate` with `400` and a clear message instead of `500`.

### Changed

These follow from the security closures above. Several are breaking for
a deployment that relied on the old behaviour.

- **Breaking**: a non-loopback bind now needs a bearer token as well as
  `--allow-non-local-bind`, unless `--insecure-no-auth` is passed.
- **Breaking**: the container image binds `127.0.0.1` by default
  (`CMD []`); publishing the port needs `DECISIONHEAD_HOST`,
  `DECISIONHEAD_ALLOW_NON_LOCAL_BIND` and a token. The compose files
  publish on host loopback only and no longer publish ollama.
- **Breaking**: requests whose `Host` names a host the server does not
  answer to get `421`; set `--allowed-hosts` for DNS names. A POST with
  an `Origin` header gets `403` while CORS is off. `Content-Type` must be
  exactly `application/json` (parameters allowed).
- **Breaking**: a deliberate-mode reply whose final JSON object shares
  its line with prose (for example `reasoning... {"answer": "yes"}`) is
  now malformed; the prompt already asks for the object alone on the
  last line. The `--dry` fake backend now writes its samples that way.
- **Breaking**: audit rows are no longer written for `401`, `403`, `421`
  or the pre-authentication `400`s.
- **Breaking**: a reading with no allowed token is returned as
  `"degraded": true` with `null` values instead of `0.5` or a uniform
  distribution, in both fast and deliberate mode.
- `500` and `502` responses carry fixed messages; the detail is in the
  server log under the request id.
- Every `POST /v1/systemone` response carries `X-Request-Id`; the
  response envelope always carries `degraded`.
- Documentation rewritten for publication: a new README, and
  `docs/configuration.md`, `docs/http-api.md`, `docs/audit-log.md`,
  `docs/PUBLISHING.md` and a `docs/README.md` index. `docs/api.md` now
  points to `docs/http-api.md`. Module and public-function docstrings
  state each module's invariants. `CITATION.cff` added.

### Added

- **Deterministic serving**: `backend.max_concurrency` (config field
  `max_concurrency`, default `1`) serializes (or bounds) every backend
  HTTP call behind a semaphore -- `decisionhead.backends
  .ConcurrencyLimitedBackend`, applied by `decisionhead.server.run_server`
  only (bench/calibrate/probe construct an unwrapped backend on purpose,
  so they can still measure/exercise the endpoint's own behaviour at
  whatever concurrency they choose). Finding: a real DeepSeek exl3
  endpoint returns different logprobs for otherwise-identical requests
  when it batches concurrent requests together; at concurrency 1 they are
  identical. `max_concurrency_override_allowed` (default `False`) gates a
  per-request `"max_concurrency"` override (fast mode only). The
  effective concurrency is logged at startup
  (`backend_concurrency_configured`). `ReadinessChecker` and audit-row
  backend labelling now unwrap a `ConcurrencyLimitedBackend` first (see
  `decisionhead.backends.unwrap_backend`) so `/ready` and audit rows still
  reflect the real backend underneath it.

- **State-hash memoisation**: `decisionhead/cache.py` -- an in-process,
  optionally file-backed (`cache_path`, append-only JSONL) LRU cache
  (`cache_max_entries`, default `10000`) with a per-entry TTL
  (`cache_ttl_s`, default `86400`), enabled by default (`cache_enabled`).
  Caches at PER-QUESTION granularity, keyed on a SHA-256 of (model id,
  backend route, canonical JSON of state, question id + text, mode,
  calibration version -- `decisionhead.cache.cache_key`); the value is the
  full answer with probabilities. Wired into `decisionhead.core
  .answer_questions` (`mode = "fast"` only -- `deliberate` samples with
  intentional randomness and `learned` never calls a backend, so neither
  benefits); a hit skips the backend call entirely and its audit row
  carries `cache_hit: true` plus the `original_trace_id` of the request
  that first computed it. A non-hit row now always carries
  `cache_hit: false`. A calibration file's content-hash is used as its
  cache-key version tag (`decisionhead.server._calibration_version`), so
  recalibrating never serves a stale cached probability.

- **Prefix-cache-friendly prompting**: verified (this was already true of
  `decisionhead.core.answer_questions`'s and `decisionhead.deliberate
  .answer_questions_deliberate`'s existing prompt construction -- state
  first in a shared prefix built once per request, question text always
  the final appended turn) and pinned down with a regression test
  (`tests/test_core.py::TestPrefixCacheFriendlyPrompting`) asserting the
  state prefix is byte-identical across every question of one decision.
  No prompt wording changed.

- **Low-confidence band**: `abstain_band_lo`/`abstain_band_hi` (a
  `[lo, hi]` band on a decisive `noul` answer's probability, disabled by
  default) and `abstain_mode` (`"off"`, the default, or `"handoff"`) --
  pure post-processing, no model call:
  `decisionhead.core.apply_abstain_band`. A `noul` answer whose
  probability falls inside the band gets `low_confidence: true` on the
  answer, its audit row, and the response envelope
  (`response["low_confidence"]`, always present); `abstain_mode =
  "handoff"` additionally sets `recommended_action: "handoff"` and a
  `handoff_reason` string on the answer and audit row. Both band bounds
  must be set together (`ConfigError` otherwise); `abstain_mode` must be
  one of `("off", "handoff")`.

- **Audit artefacts scaffold**: `docs/audit/README.md` lists the six
  audit artefacts this repo publishes (eval report, calibration report,
  threshold sweep, multi-seed/multi-repeat stability, abstention report,
  reproduction commands), the command that produces each, and which are
  live today versus planned. `decisionhead.bench` gained
  `bench_stability` (N repeats of a fixed 20-question set at concurrency
  1, reporting per-question distinct-value counts -- not just a
  first-vs-rest disagreement share -- plus the disagreement share),
  `render_stability_report_markdown`, `stability_report_paths`, and
  `write_stability_report`; `decisionhead bench --stability` (also
  reachable via the `dh_bench.py` compatibility shim) runs only this
  measurement and writes both `<out-stem>.stability.json` and
  `<out-stem>.stability.md`.

- **Learned head without prior-answer features**: `decisionhead.train
  --no-prior-features` is now an accepted alias for the existing
  `--no-prior-answers` flag (`FeatureConfig(use_prior_answers=False)`).
  The trained model file's metadata now always records
  `uses_prior_features` (`true`/`false`, mirroring
  `feature_config.use_prior_answers`); `LearnedModel.__post_init__`
  raises if a model file's `metadata["uses_prior_features"]` disagrees
  with its own `feature_config.use_prior_answers`, so a hand-edited or
  corrupted model file cannot silently lie about its own serving
  behaviour. `decisionhead.learned.answer_questions_learned` already
  never called a backend for any feature configuration (learned mode
  reads features straight off the request's own `state`), so a model
  trained with `--no-prior-features` already could not, and still cannot,
  request or use `prior_answers`/`prior_p_*` or call a fast/zero-shot
  head at serve time.

- Pilot-readiness pieces (no change to the service's behaviour or API):
  `examples/quickstart.sh` (one-command start against `--dry` or a real
  ollama/OpenAI-compatible backend named by env vars, waits for `/ready`,
  sends one `noul` and one `choice` request, prints the answers) and a
  matching README section; `examples/compose.yaml`, a `docker compose`
  example assuming the model server runs on the host (via
  `host.docker.internal`), alongside the existing `compose.yaml` (which
  bundles its own `ollama`); the bundled `Dockerfile` now builds on
  `python:3.12-slim`. `examples/workflow_loop.py`: a small,
  dependency-free `DecisionHeadClient` plus a worked workflow-engine
  example (own ledger for open obligations/budget/tried actions,
  deterministic bookkeeping around the head's judgement -- the division
  of labour the workflow-completion benchmark's pre-registration
  describes) that logs every decision as a
  `decisionhead.traces`-shaped row, runnable end to end against the
  FakeBackend with `python examples/workflow_loop.py --dry`. `docs/traces.md`
  (the exact trace JSONL schema, who labels it and how to handle
  `"UNKNOWN"`, a class-balance check, and the train/evaluate commands)
  and `docs/pilot.md` (a two-week shadow-mode-then-train pilot plan with
  numeric go/no-go criteria and what "in-network" means concretely).

- `mode: "learned"`: a trainable decision head, answering `choice`
  questions from a classifier trained offline on your own labelled
  workflow traces instead of calling an LLM per request --
  `decisionhead/{traces,features,embeddings,model,learned,train}.py` plus
  `decisionhead/_numpy_opt.py` (an optional, training-only numpy speed
  path behind the `decisionhead[train]` extra; serving stays
  zero-dependency). `python -m decisionhead.train --traces ... --cv 5
  --seed 0 --out model.json --report report.json` fits a multinomial
  logistic regression with L2 regularization over three switchable
  feature groups (structured state/bookkeeping, prior zero-shot question
  probabilities, an optional text embedding) and reports case-clustered
  cross-validation accuracy, per-class recall, and ECE, always marked
  `"dev_only": true`. `decisionhead serve --mode learned
  --head-model-path model.json` serves from the trained model, sharing
  the same audit log, calibration transform, and `mode`/
  `mode_override_allowed` machinery as `fast`/`deliberate`. Includes a
  converter (`decisionhead.traces.convert_jwc_trajectories`) from the
  JEV-WC benchmark's own trajectories into the trainable trace format,
  using the benchmark's own scorer as the labelling oracle. See
  `docs/training.md`.

- `mode: "deliberate"`: k sampled reasoning-then-typed-JSON answers voted
  per question (default k=8), as an alternative to fast mode's
  single-token logprob read -- `decisionhead/deliberate.py`, configured
  via `mode`/`deliberate_k`/`deliberate_temperature`/
  `deliberate_max_tokens`, with a per-request `"mode"` override gated by
  `mode_override_allowed`. Exposes raw `vote_share` and
  `avg_stated_confidence` on every answer for a future calibration layer.
  Includes a terminal-claim guard (`terminal_answers`/
  `completion_evidence` per question, `claim_guard_enabled`/
  `claim_guard_threshold`) that downgrades an under-supported or
  evidence-contradicted "done" claim instead of returning it as-is. See
  `docs/deliberate.md`.

### Fixed

- `ThreadingHTTPServer`'s default accept-queue backlog
  (`socketserver.BaseServer.request_queue_size`, 5) was too small for
  realistic burst concurrency: under 20 near-simultaneous client
  connections, the OS backlog could fill before a handler thread ran,
  producing a client-side `ECONNRESET` even though every request would
  have succeeded once accepted (`tests/test_server.py::TestThreadSafety
  ::test_many_concurrent_requests_all_succeed`, flaky before this fix).
  `decisionhead.server` now binds a `_DecisionHeadHTTPServer` subclass
  with `request_queue_size = 128` instead of the bare `ThreadingHTTPServer`.

- `tests/test_traces.py::TestConvertJwcTrajectoriesAgainstRealDevData
  ::test_matches_the_known_dev_a_shape` is an opportunistic integration
  check against a real, external JEV-WC dev-data file and the
  benchmark's own `jwc_score.py` -- neither owned nor versioned by this
  package, and both located through environment variables (see
  Security above). That external `jwc_score.py` grew a
  same-directory import (`jwc_cases`) that this test's standalone module
  loader does not put on `sys.path`, so the load itself now fails
  (`ModuleNotFoundError: No module named 'jwc_cases'`) independent of any
  change here. The test now catches that specific external load/exec
  failure and `pytest.skip`s with a clear reason instead of failing the
  suite, rather than asserting a stale expected shape against a moving
  external dependency.

## [0.1.0] - 2026-09-23

### Added

- Initial release: the `decisionhead` package, restructured from a
  six-module prototype into `decisionhead/{backends,core,server,client,
  calibrate,bench,schema,config,logging}.py` with a `decisionhead`
  console script (`serve`, `calibrate`, `bench`, `ask`, `probe`,
  `selftest` subcommands).
- `POST /v1/systemone`: typed-question answering (`noul`, `choice`,
  `score`) over per-token logprobs, against ollama's native route or an
  OpenAI-compatible route (vLLM and similar).
- Server hardening: request size limit, per-question and per-request
  timeouts, a cap on questions per request, a concurrency cap with 429
  beyond it, JSON-only bodies with structured error responses, structured
  JSON logs to stderr with request ids, `/health`, `/ready`, `/metrics`
  (Prometheus text format), `/openapi.json`, graceful shutdown on
  SIGTERM, loopback-only binding by default, an optional bearer token
  read from an environment variable, and CORS off by default.
- Configuration via environment variables, a TOML file
  (`decisionhead.example.toml`), and CLI flags, in that precedence order.
- `decisionhead calibrate`: per-type temperature scaling from a labelled
  JSONL set, with Brier score, ECE, and a reliability table before/after.
- `decisionhead bench`: latency by batch size, throughput by concurrency,
  and a repeat-consistency check, runnable against a real backend or
  (`--dry`) the deterministic fake backend with no network.
- `decisionhead probe`: a fast sanity check of a backend's per-token
  logprob mechanism.
- A Dockerfile (non-root, `HEALTHCHECK`, no build dependencies), a
  `compose.yaml` pairing decisionhead with an ollama service, and CI/
  release GitHub Actions workflows.
- The original six prototype modules (`dh_core.py`, `dh_backend.py`,
  `dh_server.py`, `dh_client.py`, `dh_calibrate.py`, `dh_bench.py`) are
  kept as thin compatibility shims importing from the new package, so
  existing callers keep working unchanged.

### Security

An independent pre-publication security review
(`docs/security-review-2026-09-23.md`; verdict: safe to publish after
fixes) found and this release closes:

- **HIGH**: no timeout on a stalled connection let an unauthenticated
  peer hold a server thread open indefinitely by sending a request
  slowly (slowloris). Fixed with a per-connection socket read timeout
  (`--connection-read-timeout-s`, default 30s).
- **MEDIUM**: a deeply nested JSON body or a negative `Content-Length`
  each crashed the connection with no HTTP response and an unstructured
  traceback to stderr, instead of the documented `400`. Fixed with a
  pre-parse JSON nesting-depth guard (`--max-json-nesting-depth`, default
  32) and explicit `Content-Length` range validation.
- **LOW**: `/metrics` recorded any requested path verbatim as a label
  value, letting an unauthenticated caller grow the metrics state without
  bound. Fixed by collapsing any path outside the server's actual routes
  to a single `"other"` label.
- **LOW**: a non-finite (NaN/infinite) logprob from a malformed or
  compromised backend could produce a non-finite probability in the
  response (not valid JSON) while being reported as a normal, trusted
  reading. Fixed: such a value is excluded from every probability
  computation and counted in a new `non_finite_logprob_count` field
  instead.
- **LOW**: a real-looking private IP address (a private-range address) was used
  as the example backend address throughout the docs and code. Replaced
  with an RFC 5737 documentation address (`203.0.113.10`).
- **INFORMATIONAL**: CI/release GitHub Actions were pinned by a
  major-version tag rather than a commit SHA. All four third-party
  actions are now pinned to a full commit SHA.

Every finding above has a regression test in `tests/` (see
`tests/test_security_hardening.py` and the non-finite-logprob tests in
`tests/test_backends.py`) that reproduces the reviewer's own repro
method.

[Unreleased]: https://github.com/dtinholt/decision-head/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dtinholt/decision-head/releases/tag/v0.1.0
