# Documentation

Start with the repository `README.md` for what decisionhead is and a
quickstart. The pages here go deeper.

## Running the service

| Page | What it covers |
|---|---|
| [configuration.md](configuration.md) | Every setting: default, environment variable, CLI flag, TOML key, effect and validation. The bind and token rules. |
| [http-api.md](http-api.md) | Every route, the request and response fields for each mode, degraded and low-confidence answers, and every error code in the order the server checks them. |
| [audit-log.md](audit-log.md) | Every field of an audit row, a real example, and how to reconstruct and verify a past decision. |
| [calibration.md](calibration.md) | Fitting per-type temperatures on your own labelled cases with `decisionhead calibrate`. |

## How it works

| Page | What it covers |
|---|---|
| [design.md](design.md) | The logprob reading, why the native ollama route, isolation, the request deadline, every hardening choice, and the limits of the method. |
| [deliberate.md](deliberate.md) | Deliberate mode: k sampled answers voted per question, its cost, the reply parser, and the claim guard. |
| [training.md](training.md) | Learned mode: feature groups, the classifier, case-clustered cross-validation, serving. |
| [traces.md](traces.md) | The trace file learned mode trains on: schema, labelling, how many rows, class balance. |

## Piloting and evidence

| Page | What it covers |
|---|---|
| [pilot.md](pilot.md) | A two-week shadow-mode pilot on one real decision point, with numeric go/no-go criteria. |
| [audit/README.md](audit/README.md) | The evidence artefacts behind trusting a deployment (calibration report, stability report, and planned ones), with the command for each. |

## Security and publication

| Page | What it covers |
|---|---|
| [../SECURITY.md](../SECURITY.md) | Threat model, what the service defends against and what it does not, how to report a vulnerability. |
| [security-review-2026-09-23.md](security-review-2026-09-23.md) | First independent review. All findings closed in 0.1.0. |
| [security-review-2026-10-06.md](security-review-2026-10-06.md) | Second independent review. |
| [security-review-2026-10-06-closures.md](security-review-2026-10-06-closures.md) | How each finding of the second review was closed, with its regression tests, and what is deferred. |
| [PUBLISHING.md](PUBLISHING.md) | Creating the public repository from the reviewed tree without private history, and the grep that verifies it. |

`api.md` is kept only as a pointer to `http-api.md`.
