# Security review: decisionhead, pre-publication

> Publication note (2026-10-06): private network addresses, home-directory
> paths, hostname patterns and private repository names quoted in this
> review have been replaced with neutral descriptions. Nothing else in the
> review was changed.

## VERDICT: SAFE TO PUBLISH AFTER FIXES

Nothing found here is a secret leak, an authentication bypass, an injection
vulnerability, or a source of embarrassment about the code's design or
honesty. The one thing that should be fixed before or immediately after
publishing is a genuine, unauthenticated resource-exhaustion bug (finding
1): any network peer can hang server threads indefinitely by sending bytes
slowly, and nothing in the code or the documented threat model catches
this. Findings 2 and 3 are smaller robustness gaps in the same family
(malformed input crashes a connection instead of returning the documented
clean error). None of the three requires more than a few lines of code to
fix, none touches the wire protocol or the API shape, and all three are
squarely within the class of issue SECURITY.md already asks reporters to
disclose privately, which this project has a working process for. Publish
once finding 1 is fixed; findings 2-3 can reasonably follow in a fast
point release.

## Scope and method

Reviewed: `README.md`, `SECURITY.md`, `docs/design.md`, `docs/api.md`,
`docs/calibration.md`, every file under `decisionhead/` and `tests/`, the
six `dh_*.py` compatibility shims, `Dockerfile`, `compose.yaml`,
`pyproject.toml`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`,
`.github/dependabot.yml`, `decisionhead.example.toml`, `LICENSE`,
`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `.gitignore`, and
the `examples/` directory.

White-box: read every network call (all four `urlopen`/`Request` call
sites, all targeting operator-configured base URLs, never request-derived
input), the threading model in `core.answer_questions` and the per-request
watcher thread in `server._run_with_timeout`, file handling (audit log,
calibration file, TOML config loader), the config precedence chain
(CLI > env > TOML > defaults), JSON parsing and its limits, probability
math (calibration, logit/sigmoid, the logprob-to-mass conversion), the
Dockerfile and compose file, the CI/release workflows, and the dependency
surface.

Black-box: ran `decisionhead serve --dry` on `127.0.0.1` on a random local
port (18787/18790/18998/18999 across runs) and attacked it directly:
oversized and near-limit bodies, malformed and non-UTF-8 bodies, deeply
nested JSON, 5000-question payloads, a 500 KB single instruction string,
wrong/missing `Content-Type`, unexpected HTTP methods, path-traversal and
null-byte URLs, a fake bearer token against both an unauthenticated and an
authenticated server, the bind guard with and without
`--allow-non-local-bind`, CORS on/off, `/metrics` and `/openapi.json`
content, concurrency floods (60 and 120 simultaneous requests), and raw-
socket tests for negative `Content-Length`, chunked transfer-encoding,
a CL/TE smuggling attempt, and slow byte-at-a-time delivery of both
headers and body (slowloris style).

Also ran the full test suite (`pytest`: 193/193 passed), `ruff check .`
(clean), `mypy decisionhead` in strict mode (clean), and `python -m
build`, then inspected the built wheel's `METADATA` directly to confirm
the zero-runtime-dependency claim rather than trusting `pyproject.toml`
alone.

No real model endpoint was ever contacted. All server runs used `--dry`.

## Findings, in severity order

### 1. HIGH -- unauthenticated thread exhaustion: no timeout on a stalled connection (slowloris)

* Location: `decisionhead/server.py`, the `Handler` class built in
  `make_handler` (never sets `timeout`), and `run_server`'s use of the
  stock `http.server.ThreadingHTTPServer` (no cap on concurrent
  connections/threads).
* Repro (verified): open a raw TCP socket to the server and send
  `GET /health HTTP/1.1\r\nHost: x\r\n`, one byte every 0.8 seconds,
  and never send the terminating blank line. The connection stayed open
  and the server kept accepting bytes for 25+ seconds in testing (limited
  only by test patience, not by any server-side limit); the server
  remained responsive to other clients throughout, but that one thread
  never returned. The identical technique against `/v1/systemone`'s
  *body* (valid headers, `Content-Length: 100`, then one byte every 0.7s)
  held a thread for 70+ seconds before completing normally, again with no
  timeout ever firing.
* Impact: `BaseHTTPRequestHandler.parse_request()` and
  `self.rfile.read(length)` block on the socket with no deadline unless
  the handler class sets a `timeout` attribute; this codebase never does.
  `ThreadingHTTPServer` spawns one OS thread per accepted connection with
  no cap. An attacker needs no valid bearer token (the auth check happens
  *after* headers are parsed, and the slowest read -- the header read
  itself -- happens before any auth check could run) and can hold open as
  many such connections as the OS allows threads/file descriptors,
  denying service to every endpoint, including `/health` and `/ready`
  that an orchestrator depends on for liveness/readiness decisions. This
  is the classic slowloris pattern and is not mentioned anywhere in
  SECURITY.md's threat model, which instead states the opposite: "Request
  size, per-request/per-question timeouts, a cap on questions per
  request, and a cap on concurrent requests bound the resource a single
  caller (trusted or not) can consume." That statement is true only once
  a request has been fully read; nothing bounds the time spent *reading*
  it. `max_concurrent_requests`'s semaphore is acquired only after the
  full body is already in memory, so it does not help here either.
* Fix: set `Handler.timeout = <n>` (a small number of seconds is enough;
  `BaseHTTPRequestHandler` already honors this attribute by calling
  `self.connection.settimeout()` and aborting a stalled read) so a
  connection that stalls mid-request is dropped instead of held forever.
  Consider also bounding total concurrent connections (not just
  concurrent `/v1/systemone` processing) with a semaphore acquired before
  `finish_request`, since `ThreadingHTTPServer` otherwise has no cap of
  its own.

### 2. MEDIUM -- two malformed-input paths crash the connection with no response, bypassing the documented "always a clean 4xx" guarantee

* Location: `decisionhead/server.py:388-393` (`_do_post_inner`).
* Repro A (verified): POST a body containing 5000 levels of JSON nesting
  (e.g. `{"a":` repeated 5000 times, then `1`, then `}` repeated 5000
  times -- about 20 KB, well under the 1 MiB limit). `json.loads()`
  raises `RecursionError: maximum recursion depth exceeded while decoding
  a JSON object`, which is not `json.JSONDecodeError` or
  `UnicodeDecodeError`, the only two exception types the surrounding
  `except` clause catches. The exception propagates out of `do_POST`
  entirely; `socketserver`'s own fallback handler prints a full Python
  traceback (with real source file paths) to stderr as plain text and
  the client's connection is dropped with **no HTTP response at all**
  (curl reported `HTTP 000` / connection reset), not the documented 400
  `invalid_json`.
* Repro B (verified): POST with header `Content-Length: -5`. The
  `int("-5")` conversion succeeds (no range check follows it), the
  `length > cfg.max_request_bytes` check is false for a negative number,
  and `self.rfile.read(-5)` raises `ValueError: read length must be
  non-negative or -1`. Same failure mode as A: unstructured traceback to
  stderr, zero-byte response to the client.
* Impact: this contradicts two explicit claims. SECURITY.md: "Every
  error response is a small, fixed-shape JSON object; no path in the
  server returns a stack trace... beyond the exception's own short text"
  -- here there is no error response at all, which is a worse outcome
  than the one the claim rules out. docs/design.md: "Structured logs: one
  JSON object per line to stderr" -- both tracebacks are printed as raw
  text by `socketserver.BaseServer.handle_error`, never touching
  `decisionhead.logging.JSONLogger`, so an operator grepping structured
  logs for `request_id` will not find these failures at all. Neither
  path leaks the request body, the bearer token, or any request content;
  the impact is availability/observability, not disclosure, and is
  bounded to the one connection that triggered it (verified: the server
  kept serving other requests immediately afterward in both cases).
* Fix: validate `length >= 0` explicitly (and consider a sane upper bound
  independent of `max_request_bytes` before ever calling `int()` on
  attacker-controlled header text) and return 400 for a negative or
  otherwise out-of-range value; wrap the `json.loads()` call so it also
  catches `RecursionError` (or reject deeply nested bodies before parsing
  -- e.g. `json.JSONDecoder` has no built-in depth limit, so either scan
  for nesting depth first or catch broadly and translate to
  `invalid_json`).

### 3. LOW -- `/metrics` records unbounded, attacker-chosen values as Prometheus label values

* Location: `decisionhead/server.py`, `Metrics.record_request`, called
  from both `do_GET` and `do_POST` with `self.path` used verbatim as the
  `path` label.
* Repro (verified): requesting `/etc/passwd`, `/%2e%2e/%2e%2e/etc/passwd`,
  `//etc/passwd`, etc. each creates a new, permanent label combination
  visible in `GET /metrics` (`decisionhead_requests_total{...,
  path="/etc/passwd",...}` and a full latency-histogram series for that
  path), even though every one of these requests is an unauthenticated
  404.
* Impact: an unauthenticated caller can grow the in-memory metrics state
  and the size of the `/metrics` response without bound simply by
  requesting many distinct nonexistent paths -- a standard Prometheus
  anti-pattern (never use unbounded/user-controlled text as a label
  value) rather than an information leak (the values reflected are
  attacker-supplied, not server secrets).
* Fix: map any path that does not match a known route to a fixed label
  (e.g. `"other"`) before recording it, or only call `record_request` for
  the small, fixed set of routes the server actually implements.

### 4. LOW -- a malformed/compromised backend response can produce non-finite (NaN) probabilities that bypass the one safety net meant to catch this

* Location: `decisionhead/backends.py`, `distribution_from_top_logprobs`
  and `_mass`.
* Repro (verified in-process): feeding
  `[{"token": "Yes", "logprob": nan}, {"token": "No", "logprob": -1.0}]`
  through `distribution_from_top_logprobs` returns
  `{"Yes": nan, "No": nan}` with `no_family_match: False` (i.e. flagged
  as a *normal, trusted* reading), and this value serializes into the
  HTTP response as a bare `NaN` token, which is not valid JSON per
  RFC 8259 and will break a strict client-side JSON parser. Separately,
  feeding `logprob: 800.0` (a value no real logprob can take, but not
  validated) raises an uncaught `OverflowError` inside `_mass()`, which
  only catches `TypeError`/`ValueError` around `math.exp()`; this is
  caught gracefully by the server's generic 500 handler (verified: clean
  `internal_error` response, not a crash), so the practical impact here
  is a request failure, not a bigger problem.
* Impact: docs/design.md states the whole point of `no_family_match` is
  to make a backend's unreadable output "visible rather than silently
  trusted" -- NaN input defeats exactly that mechanism, because mass
  technically "matched" an allowed token; the corruption is invisible in
  both the response and the audit row. This is bounded by SECURITY.md's
  explicit statement that a malicious/compromised backend is out of
  scope ("decisionhead trusts the backend's response shape... run the
  backend on infrastructure you control"), so this is not a boundary
  violation, but it is a gap in the specific safety net the docs describe
  as existing for this class of problem, and a buggy (not just malicious)
  serving engine could trigger it by accident.
* Fix: check `math.isfinite()` on computed mass/probability values and
  fold a non-finite result into `no_family_match`/`unparsed_mass` instead
  of propagating it; add `OverflowError` to the caught exceptions in
  `_mass()`.

### 5. LOW -- repository hygiene: a real-looking private IP used as the example backend address

* Location: `README.md:102,117`, `docs/calibration.md:59`,
  `decisionhead/server.py:3,538` (five occurrences total).
* One private RFC 1918 address (redacted for publication) is used consistently
  everywhere an example ollama base URL is needed. Being the same
  address every time, rather than a generic placeholder, reads as a real
  address from the author's own network rather than an arbitrary
  example. It is not independently harmful -- RFC 1918 space is not
  internet-routable and this reveals no secret -- but a public repo
  should prefer a documentation-reserved address so example commands are
  unambiguously fictional.
* Fix: replace all five occurrences with an address from RFC 5737's
  TEST-NET ranges (e.g. `192.0.2.10`).

### 6. INFORMATIONAL -- CI/release GitHub Actions pinned by tag, not by commit SHA

* `.github/workflows/ci.yml` and `release.yml` use `actions/checkout@v4`,
  `actions/setup-python@v5`, `actions/upload-artifact@v4`, and
  `softprops/action-gh-release@v2` -- all pinned to a major-version tag
  rather than a full commit SHA. All four are from GitHub itself or a
  long-established third party, `permissions:` is already correctly
  scoped (`contents: read` for CI; `contents: write` only in the
  tag-triggered release job), and no secret beyond the automatically
  provisioned `GITHUB_TOKEN` is referenced anywhere. This is a
  defense-in-depth supply-chain note, not an exploitable issue as
  configured, and is listed for completeness only.

## Internal-reference grep results

Run from the package root, excluding `__pycache__`:

| Pattern | Hits |
|---|---|
| RFC 1918 `192.168/16` prefix | 5 (README.md x2, docs/calibration.md x1, decisionhead/server.py x2 -- see finding 5) |
| `10\.0\.0\.` | 0 |
| CGNAT `100.64/10` prefix | 0 |
| author e-mail address | 0 |
| absolute home-directory paths | 0 |
| private hostname patterns (two patterns) | 0 |
| hardware product names (case-insensitive) | 2, both a generic hardware product name used as an example in README.md and compose.yaml, not a hostname |
| hardcoded API key / secret / password / token literals | 0 |

No internal hostnames, employee email addresses, absolute local file
paths, or other-machine references were found anywhere in the reviewed
tree. LICENSE carries `Copyright 2026 Dinand Tinholt` under Apache-2.0,
consistent with the stated authorship; no NOTICE file is required or
present since the package has zero runtime dependencies to attribute.

## README-claim-to-test table

| README/SECURITY.md/docs claim | Traceable to |
|---|---|
| Binding to a non-loopback host requires `--allow-non-local-bind` | `tests/test_server.py::TestBindGuard` (3 tests) + black-box verified |
| Request body size is capped (default 1 MiB), enforced before the body is read | `tests/test_server.py::TestRequestSizeLimit::test_oversized_body_is_413` + black-box verified (413 just over, 200 just under) -- **but see finding 1: the time spent reading a body of any size is unbounded** |
| A cap on questions per request | `tests/test_server.py::TestMaxQuestionsPerRequest` + black-box verified (5000 questions -> 400) |
| A cap on concurrent requests, 429 beyond it | `tests/test_server.py::TestConcurrencyLimit::test_second_concurrent_request_is_429` (uses an injected slow backend to force contention deterministically; black-box floods against `--dry` did not trigger 429 because the fake backend answers too fast to create contention -- this is expected, not a gap) |
| Per-request timeout returns 504 | `tests/test_server.py::TestConcurrencyLimit::test_slow_backend_triggers_504` |
| Every error response is `{"error": {"code","message"}}`, never a stack trace | `tests/test_server.py::test_no_stack_trace_leaks_on_error` for exceptions raised inside the covered path -- **but see finding 2: two uncaught-exception paths return no response at all, and print an unstructured traceback to stderr instead of the structured log** |
| Bearer token compared with `hmac.compare_digest`, never logged, only via `Authorization: Bearer` | Code-verified (`decisionhead/server.py::_check_auth`); no query-string/cookie fallback exists anywhere; black-box verified the token never appears in the server log or the audit file across valid/invalid/missing-token requests |
| `/health`, `/ready`, `/metrics`, `/openapi.json` never require the bearer token | `tests/test_server.py::TestBearerAuth::test_no_bearer_env_configured_means_no_auth_required` + black-box verified (all four return 200 with no `Authorization` header on an auth-configured server) |
| CORS off by default, `--cors` adds `Access-Control-Allow-Origin: *` | `tests/test_server.py::TestCORS` (2 tests) + black-box verified |
| Audit rows never contain the state text or a document | Code review (`decisionhead/core.py::_answer_one`'s `audit_extra` carries only `raw_distribution`/`letters`, never `state_text` or `instructions`) + black-box verified by grepping a live audit.jsonl for the request's state/instruction text after exercising it (zero matches) |
| Zero runtime dependencies | `pyproject.toml` (`dependencies = []`) + built the wheel and inspected `METADATA` directly: no unconditional `Requires-Dist` line, only `dev`-extra entries |
| `pytest`/`ruff`/`mypy --strict` all pass | Ran all three directly: 193/193 tests pass, ruff clean, mypy strict clean |
| Dockerfile: non-root user, no build dependencies, `HEALTHCHECK` against `/health` | Code review of `Dockerfile` (`useradd`+`USER decisionhead`, `pip install --no-deps`, `HEALTHCHECK ... /health`) |
| No path in the server returns a file path or internal exception message beyond the exception's own text | True for every path exercised by the existing test suite; **false for the two uncaught paths in finding 2**, which return nothing to the client (worse than the claim, not a violation of it in the literal sense, but the underlying guarantee -- "you always get a clean answer" -- does not hold for those two inputs) |

## Residual risks the operator must know

* This service is explicitly not an authentication or authorization
  boundary (its own SECURITY.md says so); the bearer token is a shared
  secret with no per-caller identity, no scopes, and no per-caller rate
  limit. Anyone who has the token can ask anything, and there is no way
  to tell two legitimate callers apart in the audit log. Put a real
  gateway in front if more than one trusted caller needs access, exactly
  as documented.
* Finding 1 (slowloris) means that even with a bearer token configured,
  an attacker with no token at all can still degrade the service for
  everyone, because the vulnerable code path runs before authentication
  is ever checked. Fix this before exposing the service beyond a fully
  trusted, isolated network, and treat "behind a gateway" as necessary
  but not sufficient until it is fixed, since most gateways proxy
  slow-client behavior straight through unless they impose their own
  read timeout.
* The service fully trusts whatever backend it is pointed at (SECURITY.md
  says this explicitly). A compromised or buggy ollama/vLLM instance can
  skew every probability this service reports, and finding 4 shows it
  can also, in the NaN case, do so without tripping the one flag meant to
  surface a bad reading. Run the backend on infrastructure you control,
  and do not point this at a backend you do not trust.
* `state` is rendered verbatim into the model's context with no prompt-
  injection defense; SECURITY.md already says this and the single-token,
  fixed-vocabulary answer format substantially narrows the blast radius,
  but does not eliminate it. This is a design tradeoff, not a bug, and is
  already disclosed.
* The Docker image's default `CMD` binds `0.0.0.0` with
  `--allow-non-local-bind` baked in (necessary for Docker's own port
  mapping to work at all) and ships with no bearer token configured by
  default. A user who runs the README's `docker run` example verbatim,
  without also setting `DECISIONHEAD_BEARER_TOKEN_ENV`, gets an
  unauthenticated decision endpoint reachable from anywhere that can
  reach the published port. This matches the documented threat model
  (the service is not an auth boundary by design) but is worth restating
  here because it is the single easiest way for a new user to end up
  more exposed than they intended.
* No test in the suite exercises negative/malformed `Content-Length`,
  deeply nested JSON, or slow/incomplete request delivery -- all three of
  the gaps found in this review would have been caught by adding tests
  for them, and their absence is why they shipped unnoticed through a
  project that otherwise has strong test discipline (193 passing tests,
  strict mypy, clean ruff).

---

## Second pass -- 2026-09-23 (post-fix verification)

### VERDICT: SAFE TO PUBLISH

All six findings from the first pass have been fixed and each has a
regression test that reproduces the original repro method. I re-ran the
entire black-box attack suite against a freshly started `--dry` server
(fresh process, new ports, none of the state from the first pass reused),
re-read every changed line of source, re-ran the grep sweep, and
independently confirmed the four pinned GitHub Actions SHAs resolve to
the exact tags the workflow comments claim, against the live GitHub API
rather than trusting the comment. Nothing new was found. This is now
publishable as-is.

### What was re-tested, and the result

**Finding 1 (HIGH, slowloris) -- fixed, verified.**
`decisionhead/server.py` now sets `Handler.timeout = cfg.connection_read_timeout_s`
(new `ServerConfig.connection_read_timeout_s`, default 30s, flag
`--connection-read-timeout-s`, env `DECISIONHEAD_CONNECTION_READ_TIMEOUT_S`).
Re-ran both of my original raw-socket repros against a fresh server
started with a 2s and a 1.5s timeout respectively (to keep the test fast
without changing the mechanism under test):
  * Partial headers (`GET /health HTTP/1.1\r\nHost: x\r\n`, no terminating
    blank line): connection closed at 2.00s, `data == b""`, versus 25+
    seconds of continued acceptance before the fix.
  * Partial body (valid headers, `Content-Length: 100`, one byte of body
    then silence): connection closed at 1.50s, `data == b""`, versus 70+
    seconds before the fix.
  * Re-ran the coordinator's specific ask -- a second, independent client
    hitting `/health` while the first connection was stalled -- and it was
    served in 6.5ms, confirming one stuck connection no longer affects
    anyone else (it never did, even before the fix; what changed is that
    the stuck connection itself now ends).
  * Confirmed this closes before authentication: the timeout fires on the
    header read, which happens before `_check_auth()` is ever reached, so
    the defense applies with or without a bearer token configured, exactly
    as `tests/test_security_hardening.py::TestFinding1SlowlorisDefense`
    asserts.
  * `tests/test_security_hardening.py` has both cases as permanent
    regression tests; both pass.

**Finding 2 (MEDIUM, crash-with-no-response on malformed input) -- fixed, verified.**
  * Deeply nested JSON (5000 levels, the exact repro from the first pass):
    now returns `400 invalid_json` ("request body exceeds the maximum
    JSON nesting depth of 32") instead of a dropped connection. Server
    answered `/health` normally immediately after.
  * Negative `Content-Length: -5`: now returns `400 invalid_request`
    ("Content-Length must not be negative") over a raw socket, full clean
    HTTP response, no truncation, no traceback in the bytes received.
  * New case I added beyond the original report, prompted by the fix's
    own comment about `sys.get_int_max_str_digits()`: a `Content-Length`
    header containing 10,000 digit characters. Returns a clean `400`
    ("Content-Length is not a valid integer"); confirmed CPython's own
    digit-length guard converts this to a `ValueError` the existing
    handler already caught, so no separate fix was needed there, and the
    builder added a regression test for it anyway
    (`test_absurdly_long_content_length_digit_string_returns_400`).
  * Read `decisionhead/schema.py::json_nesting_too_deep`: a linear,
    recursion-free byte scan that tracks string/escape state so brackets
    inside a string value are never miscounted as structure -- confirmed
    by re-deriving the escaped-quote case by hand
    (`{"a": "esc\"ape[[[..."` must not count as nested) and cross-checking
    it against the builder's own unit tests for the same scan, which
    match. The `except` around `json.loads()` also now catches
    `RecursionError` directly as a documented second layer.
  * In both cases the server remained fully responsive to a follow-up
    `/health` request in the same test run.

**Finding 3 (LOW, unbounded `/metrics` cardinality) -- fixed, verified.**
Requested five distinct bogus paths (`/etc/passwd`,
`/%2e%2e/%2e%2e/etc/passwd`, `//etc/passwd`, `/one/two/three`,
`/another/bogus/path`) against a fresh server and confirmed `/metrics`
shows exactly one `path="other"` counter series with count 5, and none of
the literal attacker-supplied path strings appear anywhere in the
`/metrics` output. `decisionhead/server.py::_metrics_path_label` maps
anything outside `KNOWN_GET_PATHS`/`KNOWN_POST_PATHS` to `"other"` before
it ever reaches `Metrics.record_request`.

**Finding 4 (LOW, non-finite logprobs) -- fixed, verified.**
Exercised `decisionhead.backends.distribution_from_top_logprobs` and a
full `OllamaBackend.token_distribution` call through a fixture transport
directly (no live server needed for this one, per the coordinator's
instruction to go through the fake transport) with:
  * `logprob: nan` on an allowed token -- excluded from the mass sum,
    `non_finite_logprob_count: 1`, resulting distribution
    `{"Yes": 0.0, "No": 1.0}` (finite), and the full backend result is
    `json.dumps`-safe with no bare `NaN` token.
  * `logprob: inf` -- same outcome.
  * `logprob: 800.0` (the overflow case from the first pass) -- `_mass()`
    now catches `OverflowError` explicitly and also checks
    `math.isfinite()` on the successful case, so this returns
    `(0.0, False)` instead of raising; no 500, no crash.
  * All-`-inf` logprobs (the original threat model's explicit "division
    by zero when all logprobs are -inf" scenario) -- still falls back to
    a uniform distribution via the pre-existing `no_family_match` path,
    confirmed still correct after the refactor of `_mass()`'s return type
    to a tuple.
  * Read the new `_mass()` and confirmed every one of its three exit
    paths (`TypeError`/`ValueError` on a non-numeric input, `OverflowError`,
    and a finite-check on the success path) returns `(0.0, False-or-True)`
    and never lets a non-finite float escape into a running sum.

**Finding 5 (LOW, real-looking private IP) -- fixed, verified.**
Grepped the entire tree for the private address prefix again: the only remaining hits are
in this report, `CHANGELOG.md`, and `tests/test_security_hardening.py`,
all describing the historical finding, not live documentation or code.
All five original live occurrences (README.md x2, docs/calibration.md x1,
decisionhead/server.py x2) now read `203.0.113.10` (RFC 5737 TEST-NET-3,
correctly a documentation-reserved address, not merely a different
private range).

**Finding 6 (INFORMATIONAL, unpinned Actions) -- fixed, verified independently.**
All four `uses:` lines across `ci.yml` and `release.yml` now carry a
40-character commit SHA with a `# vX.Y.Z` comment. I did not trust the
comment: I queried the live GitHub API for each repository's
`git/refs/tags/<version>` and confirmed the tag's commit object SHA
matches the pinned SHA in the workflow file exactly, for all four:

| Action | Pinned SHA | Claimed tag | Tag resolves to |
|---|---|---|---|
| `actions/checkout` | `11d5960a326750d5838078e36cf38b85af677262` | v4.4.0 | matches |
| `actions/setup-python` | `a26af69be951a213d495a4c3e4e4022e16d87065` | v5.6.0 | matches |
| `actions/upload-artifact` | `ea165f8d65b6e75b540449e92b4886f43607fa02` | v4.6.2 | matches |
| `softprops/action-gh-release` | `3bb12739c298aeb8a4eeaf626c5b8d85266b0e65` | v2.6.2 | matches |

### Full re-run of everything from the first pass

* `pytest`: 220/220 passed (up from 193; the 27 new tests are
  `tests/test_security_hardening.py` plus the new non-finite-logprob
  cases added to `tests/test_backends.py`).
* `ruff check .`: clean.
* `mypy decisionhead` (strict): clean.
* Rebuilt the wheel and re-inspected `METADATA` directly: still zero
  unconditional `Requires-Dist`, only `dev`-extra entries -- the
  zero-runtime-dependency claim still holds after the fixes (the depth
  guard and metrics bounding are pure-stdlib additions).
* Re-ran the oversized/near-limit body test (413 just over 1 MiB, 200
  just under): unchanged, still correct.
* Re-ran the bind guard (`--host 0.0.0.0` refused without
  `--allow-non-local-bind`, both from the CLI directly and matching the
  `TestBindGuard` tests): unchanged, still correct.
* Re-ran the bearer-auth checks (missing token -> 401, correct token ->
  200, token never appears in the server log or a fresh audit file):
  unchanged, still correct.
* Re-ran a heavy concurrent flood (80 simultaneous requests, 50 questions
  each) against the fresh `--dry` server: 74 succeeded, 6 got a
  connection-level `URLError` from the OS-level TCP accept backlog
  (`socketserver.TCPServer.request_queue_size` defaults to 5 and is not
  overridden), zero came back as an application-level 429 -- this is the
  same, already-understood artifact from the first pass: `--dry` answers
  fast enough that the semaphore is rarely contended, and the 429 path
  itself is correctly exercised by `tests/test_server.py`'s dedicated
  `SlowBackend` fixture, which still passes. The server was fully healthy
  immediately afterward. Not a new issue and not a regression.
* Audit-log content re-check: grepped a fresh `audit.jsonl` produced
  during this pass for state/instruction text -- zero matches, same as
  the first pass.

### README/SECURITY.md claim-to-test re-verification

Every row in the first pass's claim table was re-checked; the two rows
marked PARTIAL are now OK:

| Claim | First pass | Second pass |
|---|---|---|
| "Request size... bound the resource a single caller can consume" | PARTIAL (body-read time was unbounded) | OK -- SECURITY.md now says this explicitly holds "once a request has been fully read" and separately documents the new connection-read timeout that bounds the reading phase itself; both halves are now true and tested |
| "Every error response is `{"error": {...}}`, never a stack trace" | PARTIAL (two inputs returned nothing at all) | OK -- both inputs now return the documented shape, verified live |
| "Structured logs: one JSON object per line to stderr" | PARTIAL (the two crash paths printed raw text) | OK -- both paths now go through the normal `_send_json`/logging flow; no raw traceback observed in either repro during this pass |

All other rows from the first pass (bind guard, cap on questions, 429,
CORS, bearer token handling, audit content, zero runtime dependencies,
Dockerfile non-root/HEALTHCHECK) were re-verified this pass with the same
result as before: OK, unchanged.

Three new claims introduced by the fix itself, checked against their
tests:

| New claim | Source | Test |
|---|---|---|
| A stalled connection is dropped at `--connection-read-timeout-s`, before authentication | SECURITY.md, docs/design.md | `tests/test_security_hardening.py::TestFinding1SlowlorisDefense` (2 tests) + live re-test above |
| A body nested deeper than `--max-json-nesting-depth` is rejected with 400 before parsing | docs/api.md, docs/design.md | `tests/test_security_hardening.py::TestJsonNestingTooDeep` (5 unit tests) + `TestFinding2MalformedInputHardening` (4 tests) + live re-test above |
| A non-finite backend logprob is excluded and counted in `non_finite_logprob_count`, never reaches the response | SECURITY.md, docs/design.md | new cases in `tests/test_backends.py` + direct re-test above |

### Anything new

Nothing rising to a reportable finding. Two observations, both already
called out honestly in the builder's own updated docs rather than left
implicit:

* The connection-read timeout is, by its own documented admission, a
  partial mitigation: it resets on every byte received, so a peer that
  sends data just slightly faster than the timeout interval is not caught
  by it and can still hold a thread for a long time (bounded, just not
  tightly). `ThreadingHTTPServer` still has no cap on total concurrent
  connections independent of the `/v1/systemone`-only semaphore. This is
  a reasonable, honestly-scoped fix for the reported finding, not a full
  close of every slowloris variant -- see residual risks below.
  * "Every uses: line pinned to a commit SHA" test only regexes for a
  40-character hex string after `@`; it does not itself call the GitHub
  API. I did that verification independently in this pass (table above)
  specifically because a regex match proves the *format* is a SHA, not
  that the SHA is *authentic* for the claimed tag -- worth keeping in mind
  if this test is ever the only check relied on for a future dependency
  bump.

### Residual risks the operator must know (updated)

Everything from the first pass's residual-risks list still applies
(this service is not an auth boundary; it trusts its backend completely;
`state` is rendered into the model's context with no prompt-injection
defense; the Docker image's default CMD binds `0.0.0.0` with no bearer
token configured by default) and is unchanged by this round of fixes.
Two updates:

* The slowloris fix is a bound, not an elimination: `--connection-read-timeout-s`
  (default 30s) means a single stalled connection now costs at most ~30s
  of one thread instead of forever, but a sustained flood of many
  connections each sending data just under that rate can still tie up
  threads for up to that long, repeatedly. Operators exposing this beyond
  a fully trusted network should also rely on their gateway/load
  balancer's own connection limits, and may want a lower
  `--connection-read-timeout-s` than the 30s default if the expected
  legitimate use case (a single trusted automation caller) never needs
  anywhere near that long to finish sending a request.
* The `socketserver.TCPServer.request_queue_size` default (5) is still
  unmodified and is unrelated to any of the six findings or their fixes
  -- under a large enough burst of simultaneous new connections, some
  will see a connection-level failure at the OS level before the
  application ever sees them (observed again in this pass: 6/80 in a
  heavy synthetic flood). This is not a security vulnerability -- the
  server survives and recovers immediately, and it is exactly the kind of
  resource-exhaustion behavior SECURITY.md already scopes as a
  rate-limiting/gateway concern -- but an operator tuning this service
  for a bursty legitimate workload should know the number exists and
  that nothing in this codebase changes it.
