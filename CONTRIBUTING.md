# Contributing

Thanks for considering a contribution. This page covers setting up,
the checks a change must pass, the rules that keep the service small and
safe, and how documentation is written.

## Set up

```
git clone https://github.com/dtinholt/decision-head
cd decision-head
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

or `make dev`. Python 3.11 or later.

## Checks

```
make test        # pytest with coverage
make lint        # ruff check .
make typecheck   # mypy --strict on the decisionhead package
```

All three must pass. CI runs them on Python 3.11 and 3.12, then runs the
six compatibility shims' selftests and builds the wheel. The full suite
takes about a minute and needs no network and no model: every test runs
against `decisionhead.backends.FakeBackend`, a fixture transport, or a
loopback server the test starts itself.

One test is skipped by default. It checks the trace converter against
real benchmark data that is not part of this repository, and runs only
when both `DECISIONHEAD_JWC_SCORE` and
`DECISIONHEAD_JWC_DEV_A_TRAJECTORIES` point at existing files.

### The selftest

`decisionhead selftest` runs a smoke check of every module with nothing
but the standard library, so it works on a machine with no test
framework installed (for example, inside the container image):

```
decisionhead selftest
```

The six `dh_*.py` shims at the repository root keep code written against
the original prototype working. Each has its own `--selftest`:

```
for shim in dh_core.py dh_backend.py dh_server.py dh_client.py dh_calibrate.py dh_bench.py; do
  python "$shim" --selftest
done
```

If you move or rename something a shim imports, update the shim in the
same change.

## Rules

* **No runtime dependencies.** The `decisionhead` package imports only
  the standard library. Development tools live in the `dev` extra. The
  one optional extra, `train` (numpy), may be used only on the training
  path, behind `decisionhead/_numpy_opt.py`, with a pure-Python path that
  computes the same result; nothing on the serving path may import it.
  A pull request that adds an import outside the standard library to the
  package will not be merged.
* **No network in tests.** A new code path that talks to a backend gets
  a fixture-transport test, not a live one.
* **Types and docstrings.** Every public function and class has a full
  type signature and a docstring stating its purpose, arguments, return
  value, what it raises, and any security-relevant behaviour. `mypy
  --strict` must pass.
* **Errors stay inside.** A new error path in `decisionhead.server`
  returns `{"error": {"code", "message"}}` with a fixed message and logs
  the detail server-side under the request id. No exception text,
  backend URL or backend body reaches a client.
* **Secrets by name only.** A new secret is configured as the name of an
  environment variable (a `*_env` setting), read at the moment it is
  needed, added to `SECRET_ENV_NAME_FIELDS` in `decisionhead/config.py`,
  and never logged, echoed, or written to an audit or cache row.
* **Every emitted number is finite.** Anything that reaches a response,
  audit row or cache row goes through `decisionhead.safejson`; anything
  parsed from outside goes through `loads_strict`.

## Changing a security-relevant path

These paths carry the security properties listed in `SECURITY.md`:

* `decisionhead/server.py` (request handling, auth, Host/Origin/media
  type checks, the concurrency slot, the audit write, error bodies);
* `decisionhead/backends.py` and `decisionhead/embeddings.py`
  (outbound HTTP, redirects, size caps, logprob parsing);
* `decisionhead/deadline.py` and every place that checks it;
* `decisionhead/deliberate.py` (the reply parser and vote counting);
* `decisionhead/safejson.py`, `decisionhead/schema.py`,
  `decisionhead/config.py` (validation);
* `decisionhead/cache.py` and the audit log;
* `Dockerfile`, `compose.yaml`, `examples/compose.yaml`, the CI and
  release workflows.

A change to any of them follows this procedure:

1. **Write the regression test first** and run it against the unchanged
   code. It must fail, for the reason you are fixing. If it passes, it
   does not test the problem.
2. Name the test after what it protects, in the existing style
   (`TestFinding2CrossOriginAndRebinding::test_foreign_host_header_is_refused_on_post_and_get`).
   A fix for a reported finding goes in a class named after the finding.
3. Make the change. Run the full suite, `ruff` and `mypy`.
4. Update the documents that describe the behaviour: `SECURITY.md`,
   `docs/design.md`, `docs/http-api.md`, `docs/configuration.md` or
   `docs/audit-log.md`, whichever apply. A test may assert on a document
   (see `test_design_doc_no_longer_claims_the_old_bound`) when a stale
   sentence would be a security problem in its own right.
5. Add a `CHANGELOG.md` entry under `[Unreleased]`, in `Security` or
   `Changed`, marked **Breaking** if a deployment could notice.

Please report a vulnerability privately first (see `SECURITY.md`).

## Writing documentation

The documentation is written for an engineer who has never seen the
project. In practice:

* State what the code does now. Describe history only where it explains
  a current choice, and link to the review or changelog entry.
* Prefer the specific: name the setting, the default, the status code,
  the field. Show a command and its real output; generate example output
  by running the command, never by hand.
* Short sentences, plain words, active voice. No marketing adjectives,
  no emoji, no em dashes.
* Numbers about model quality come from a published source and are cited
  with it. Label synthetic benchmarks as synthetic. Report negative
  results next to positive ones.
* Use documentation addresses (`203.0.113.10`, `127.0.0.1`) and generic
  example names. Never a real host, private address, home-directory path
  or personal name. `docs/PUBLISHING.md` has the grep that checks this.
* Keep a fact in one place and link to it. `docs/configuration.md` owns
  settings, `docs/http-api.md` owns the wire format, `docs/audit-log.md`
  owns the audit row.

## Commits and pull requests

* Small, focused commits whose message says why.
* Reference the issue a pull request closes, if any.
* A user-visible change (a flag, a behaviour change, a fix) gets a
  `CHANGELOG.md` entry under `[Unreleased]`.

## Code of conduct

Participation is governed by `CODE_OF_CONDUCT.md` (Contributor
Covenant).
