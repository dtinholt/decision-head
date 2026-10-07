# Configuration reference

Every setting of `decisionhead serve`, with its default, its environment
variable, its CLI flag and what it does. The source of truth is
`decisionhead/config.py` (`ServerConfig` and `ENV_VAR_MAP`); this page is
written from it.

## How settings are resolved

Four sources, highest precedence first:

1. CLI flags passed to `decisionhead serve`.
2. Environment variables, all prefixed `DECISIONHEAD_`.
3. A TOML file, named by `--config PATH` or `DECISIONHEAD_CONFIG`.
4. The built-in defaults listed below.

Each setting falls through independently: a value missing from a higher
source is taken from the next one down.

**Environment values** are strings. Booleans accept `1`, `true`, `yes`,
`on` and `0`, `false`, `no`, `off` (case-insensitive); anything else is
an error. An empty variable counts as unset.

**TOML** keys live in the sections `[server]`, `[limits]`, `[auth]`,
`[logging]`, `[backend]`, `[cache]` and `[abstain]`. The section is for
readability only: any key may appear in any of them. An unknown key in a
known section is an error (a typo should fail loudly); an unknown section
is ignored. `decisionhead.example.toml` lists every key.

**CLI booleans** only move away from the default (`--cors` turns CORS
on, `--no-cache` turns the cache off). To set the other direction, use
the environment or the TOML file.

**Validation** happens once, at start-up. A value that fails it stops
the server with an error message; nothing is silently clamped. The
checks are listed with each setting below.

## Secrets

Two settings name secrets, and neither takes the secret itself:
`api_key_env` (and `embedding_api_key_env`) and `bearer_token_env`. Each
holds the **name** of an environment variable, which is read when the
value is needed. A value that does not look like a variable name
(`^[A-Za-z_][A-Za-z0-9_]*$`) is refused at start-up, and the refusal does
not print the value, because the usual cause is pasting the token itself.
The names are never logged.

```
export DH_TOKEN="$(openssl rand -hex 32)"
decisionhead serve --dry --bearer-token-env DH_TOKEN
```

## Backend

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `backend` | `"ollama"` | `DECISIONHEAD_BACKEND` | `--backend {ollama,openai}` | `ollama` calls the native `POST {base}/api/chat` route. `openai` calls `POST {base}/v1/chat/completions` on any OpenAI-compatible server (vLLM and similar). |
| `base_url` | `""` | `DECISIONHEAD_BASE_URL` | `--base` | The model server's root URL, **without** `/v1` (the backend appends `/api/chat` or `/v1/chat/completions` itself). Must be `http://` or `https://` with a host. Required unless `dry` is set or `mode` is `learned`. A plain `http://` URL to a non-loopback host logs the start-up warning `insecure_backend_transport`. |
| `model` | `""` | `DECISIONHEAD_MODEL` | `--model` | The real model id sent to the backend and returned in every response's `model` field. Required unless `dry` is set or `mode` is `learned`. |
| `api_key_env` | unset | `DECISIONHEAD_API_KEY_ENV` | `--api-key-env` | Name of the environment variable holding a bearer token for the backend. Used by `openai` only. |
| `backend_timeout_s` | `60.0` | `DECISIONHEAD_BACKEND_TIMEOUT_S` | `--backend-timeout-s` | Socket timeout for one backend HTTP call. The effective value is `min(backend_timeout_s, question_timeout_s)`. It bounds each connect and each read, not the total length of a call. Finite, `> 0`. |
| `max_concurrency` | `1` | `DECISIONHEAD_MAX_CONCURRENCY` | `--max-concurrency` | How many backend calls may be in flight at once, process-wide. `1` keeps logprobs reproducible on servers that batch concurrent requests (see the README, "Deterministic serving"). `>= 1`. |
| `max_concurrency_override_allowed` | `false` | `DECISIONHEAD_MAX_CONCURRENCY_OVERRIDE_ALLOWED` | `--allow-max-concurrency-override` | Lets a fast-mode request's own `"max_concurrency"` field choose a different limit for that request. |
| `dry` | `false` | `DECISIONHEAD_DRY` | `--dry` | Use the deterministic fake backend: no network, no model. The numbers are a hash of the question text, not a judgement. |
| `dry_seed` | `0` | `DECISIONHEAD_DRY_SEED` | `--dry-seed` | Seed for the fake backend. |

## Modes

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `mode` | `"fast"` | `DECISIONHEAD_MODE` | `--mode {fast,deliberate,learned}` | The default answering strategy. See the README, "Modes". |
| `mode_override_allowed` | `true` | `DECISIONHEAD_MODE_OVERRIDE_ALLOWED` | `--no-mode-override` | Whether a request's own `"mode"` field may change the mode for that request. Set it to `false` on a `learned` server: a `"mode": "fast"` request there has no LLM to call and fails with `500`. |
| `deliberate_k` | `8` | `DECISIONHEAD_DELIBERATE_K` | `--deliberate-k` | Samples per question in deliberate mode. Cost scales linearly with it. `>= 1`. |
| `deliberate_temperature` | `0.7` | `DECISIONHEAD_DELIBERATE_TEMPERATURE` | `--deliberate-temperature` | Sampling temperature in deliberate mode. |
| `deliberate_max_tokens` | `256` | `DECISIONHEAD_DELIBERATE_MAX_TOKENS` | `--deliberate-max-tokens` | Token limit per deliberate sample (reasoning plus the JSON answer). `>= 1`. |
| `claim_guard_enabled` | `true` | `DECISIONHEAD_CLAIM_GUARD_ENABLED` | `--no-claim-guard` | Deliberate mode: downgrade a terminal answer (named in the question's `terminal_answers`) that is under-supported. See `docs/deliberate.md`. |
| `claim_guard_threshold` | `0.75` | `DECISIONHEAD_CLAIM_GUARD_THRESHOLD` | `--claim-guard-threshold` | Minimum vote share a terminal answer needs to stand. |
| `head_model_path` | unset | `DECISIONHEAD_HEAD_MODEL_PATH` | `--head-model-path` | Learned mode: the `model.json` written by `python -m decisionhead.train --out`. Required when `mode` is `learned`. A file with an inconsistent shape or any non-finite weight stops the server. |
| `embedding_backend` | unset | `DECISIONHEAD_EMBEDDING_BACKEND` | `--embedding-backend {ollama,openai}` | Learned mode, only for a model trained with embedding features. |
| `embedding_base_url` | `""` | `DECISIONHEAD_EMBEDDING_BASE_URL` | `--embedding-base` | Embeddings server root URL. Same scheme check and transport warning as `base_url`. |
| `embedding_model` | `""` | `DECISIONHEAD_EMBEDDING_MODEL` | `--embedding-model` | Embeddings model id. |
| `embedding_api_key_env` | unset | `DECISIONHEAD_EMBEDDING_API_KEY_ENV` | `--embedding-api-key-env` | Name of the variable holding an embeddings API key (`openai` embeddings only). |

## Calibration, audit and cache

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `calibration_path` | unset | `DECISIONHEAD_CALIBRATION` | `--calibration` | A `calibration.json` from `decisionhead calibrate`. Applied in fast mode only. A file with `NaN`/`Infinity`, or a temperature that is not a finite number `> 0`, stops the server. Its SHA-256 is logged at start-up (`calibration_loaded`) and written to every fast-mode audit row as `calibration_version`. |
| `audit_path` | unset | `DECISIONHEAD_AUDIT_LOG` | `--audit` | Append-only JSONL audit log, created `0600`. Unset means no audit log. When set, a failed write turns the request into a `500`. See `docs/audit-log.md`. |
| `cache_enabled` | `true` | `DECISIONHEAD_CACHE_ENABLED` | `--no-cache` | Fast-mode response cache. A hit returns the stored answer without calling the backend. |
| `cache_ttl_s` | `86400.0` | `DECISIONHEAD_CACHE_TTL_S` | `--cache-ttl-s` | Entry lifetime in seconds. Finite, `> 0`. |
| `cache_max_entries` | `10000` | `DECISIONHEAD_CACHE_MAX_ENTRIES` | `--cache-max-entries` | In-memory LRU bound. `>= 1`. |
| `cache_path` | unset | `DECISIONHEAD_CACHE_PATH` | `--cache-path` | Optional append-only JSONL file that persists the cache across restarts. Created `0600`; holds hashed keys and answers, never state text. The file is trusted on load, so anyone who can write it can plant answers: keep it in a directory only the service user can write. It grows without bound; delete it to reset. |

## Abstention

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `abstain_band_lo` | unset | `DECISIONHEAD_ABSTAIN_BAND_LO` | `--abstain-band-lo` | Lower bound of the low-confidence band for `noul` answers. Set both bounds or neither; `lo <= hi`. |
| `abstain_band_hi` | unset | `DECISIONHEAD_ABSTAIN_BAND_HI` | `--abstain-band-hi` | Upper bound. A `noul` probability inside `[lo, hi]` (inclusive) is flagged `low_confidence`. |
| `abstain_mode` | `"off"` | `DECISIONHEAD_ABSTAIN_MODE` | `--abstain-mode {off,handoff}` | `handoff` also adds `recommended_action: "handoff"` and a `handoff_reason` to flagged answers. A degraded answer is always flagged, and handed off in `handoff` mode, whether or not a band is set. |

## Network exposure and authentication

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `host` | `"127.0.0.1"` | `DECISIONHEAD_HOST` | `--host` | Bind address. Anything other than `127.0.0.1`, `localhost` or `::1` needs `allow_non_local_bind` and a bearer token. |
| `port` | `8787` | `DECISIONHEAD_PORT` | `--port` | Bind port. `0` picks a free port. |
| `allow_non_local_bind` | `false` | `DECISIONHEAD_ALLOW_NON_LOCAL_BIND` | `--allow-non-local-bind` | Required for a non-loopback `host`. |
| `bearer_token_env` | unset | `DECISIONHEAD_BEARER_TOKEN_ENV` | `--bearer-token-env` | Name of the variable holding this server's own bearer token. When set, `POST /v1/systemone` needs `Authorization: Bearer <token>`; the GET routes stay open. If the named variable is unset or empty, the server refuses to start with a configuration error, on every bind address (loopback included); the error does not echo the variable's name. To run without a token, leave `bearer_token_env` unset (loopback) or pass `insecure_no_auth` (non-loopback). |
| `insecure_no_auth` | `false` | `DECISIONHEAD_INSECURE_NO_AUTH` | `--insecure-no-auth` | Allows a non-loopback bind with no token. Logged as the WARNING `insecure_no_auth`. Not recommended. It does not override an unset or empty `bearer_token_env`: that still refuses to start. |
| `allowed_hosts` | `""` | `DECISIONHEAD_ALLOWED_HOSTS` | `--allowed-hosts` | Comma-separated extra hostnames the server answers to. The loopback names and `host` are always allowed. Any other `Host` header gets `421`. Ports are ignored. Set this when clients reach the server by a DNS name, for example `--allowed-hosts dh.internal`. |
| `cors_enabled` | `false` | `DECISIONHEAD_CORS_ENABLED` | `--cors` | Adds `Access-Control-Allow-Origin: *` and answers `OPTIONS` with `204`. While off, a POST carrying an `Origin` header gets `403`. For local browser development only. |

The bind rules in one place:

| `host` | `allow_non_local_bind` | token set | `insecure_no_auth` | Result |
|---|---|---|---|---|
| any | any | `bearer_token_env` set, variable unset or empty | any | refuses to start |
| loopback | any | any | any | starts |
| other | `false` | any | any | refuses to start |
| other | `true` | yes | any | starts |
| other | `true` | no | `false` | refuses to start |
| other | `true` | no | `true` | starts, logs WARNING `insecure_no_auth` |

## Limits

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `max_request_bytes` | `1048576` | `DECISIONHEAD_MAX_REQUEST_BYTES` | `--max-request-bytes` | Largest accepted body, checked from `Content-Length` before reading. Over it: `413`. `>= 1`. |
| `max_json_nesting_depth` | `32` | `DECISIONHEAD_MAX_JSON_NESTING_DEPTH` | `--max-json-nesting-depth` | Deeper bodies are refused with `400` before parsing. `>= 1`. |
| `max_questions_per_request` | `50` | `DECISIONHEAD_MAX_QUESTIONS_PER_REQUEST` | `--max-questions-per-request` | More questions than this: `400`. `>= 1`. |
| `max_concurrent_requests` | `16` | `DECISIONHEAD_MAX_CONCURRENT_REQUESTS` | `--max-concurrent-requests` | `POST /v1/systemone` requests in flight at once, counting timed-out requests whose last backend call is still running. Over it: `429` with `Retry-After: 1`. `>= 1`. |
| `max_workers` | `4` | `DECISIONHEAD_MAX_WORKERS` | `--max-workers` | Worker threads per request. Questions are spread across them; each makes its own backend calls, still subject to `max_concurrency`. `>= 1`. |
| `request_timeout_s` | `30.0` | `DECISIONHEAD_REQUEST_TIMEOUT_S` | `--request-timeout-s` | Wall-clock budget for one request. Past it: `504`, and every backend call not yet started is abandoned. Also caps how long a call waits for a `max_concurrency` slot. Finite, `> 0`. |
| `question_timeout_s` | `15.0` | `DECISIONHEAD_QUESTION_TIMEOUT_S` | `--question-timeout-s` | Caps the backend socket timeout (see `backend_timeout_s`). Finite, `> 0`. |
| `connection_read_timeout_s` | `30.0` | `DECISIONHEAD_CONNECTION_READ_TIMEOUT_S` | `--connection-read-timeout-s` | Longest wait for the next byte of a request line, header or body before the connection is dropped (slow-client defence). Finite, `> 0`. |
| `ready_cache_s` | `5.0` | `DECISIONHEAD_READY_CACHE_S` | none | How long a `/ready` result is reused before the backend is probed again. |

## Logging

| Setting | Default | Environment | CLI | Effect |
|---|---|---|---|---|
| `log_level` | `"INFO"` | `DECISIONHEAD_LOG_LEVEL` | `--log-level` | Structured JSON log lines go to stderr, one object per line, with `request_id` on request-scoped lines. Secrets and secret variable names are never logged. |

## Other subcommands

`decisionhead calibrate`, `bench`, `probe` and `ask` take their own small
flag sets and do not read `ServerConfig`; run `decisionhead <command> -h`.
They share these conventions: `--backend`, `--base`, `--model` and
`--api-key-env` mean the same as above, and `--dry` uses the fake
backend. `python -m decisionhead.train -h` lists the training flags;
`docs/training.md` explains them.

| Command | Purpose | Reference |
|---|---|---|
| `decisionhead calibrate` | Fit per-type temperatures on labelled cases | `docs/calibration.md` |
| `decisionhead bench` | Latency, throughput and repeat-consistency against a backend | README, "Benchmarks" |
| `decisionhead probe` | One yes/no question to check a backend returns usable logprobs | README, "Quickstart" |
| `decisionhead ask` | Send a question file to a running server | `decisionhead ask -h` |
| `decisionhead train` | Train a learned-mode head | `docs/training.md` |
| `decisionhead selftest` | Offline smoke check of every module, no test framework needed | `CONTRIBUTING.md` |
