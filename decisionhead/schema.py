"""Request/response validation and the OpenAPI document for the
``/v1/systemone`` API.

Two layers of validation exist in this package and they are deliberately
kept separate:

* :func:`validate_envelope` in this module checks the *shape of the HTTP
  request* -- is the body an object, does it have ``state``/``model``/
  ``questions``, is ``questions`` non-empty and within the configured
  per-request question limit. This is what :mod:`decisionhead.server`
  calls before doing anything else with a request.
* :func:`decisionhead.core.normalize_question` checks the shape of one
  *question definition* (its type, its instructions, its options). This
  runs per-question, inside :func:`decisionhead.core.answer_questions`,
  and is exercised for every backend call regardless of entry point
  (server, calibrate, bench).

Both raise :class:`decisionhead.core.QuestionError`, so a caller only ever
has one exception type to catch to turn "the request was malformed" into a
4xx.

:func:`build_openapi_document` renders a static OpenAPI 3.0 document from
the same JSON Schema fragments used here, so the document served at
``/openapi.json`` cannot drift from what the server actually accepts.

Invariant: nothing here parses JSON. :func:`json_nesting_too_deep` runs on
raw bytes before any parser sees them, in one linear pass with no
recursion, so a deeply nested body cannot reach the interpreter's
recursion limit.
"""

from __future__ import annotations

from typing import Any

from .core import QuestionError

JSONDict = dict[str, Any]


def validate_envelope(body: Any, *, max_questions: int) -> None:
    """Validates the top-level ``/v1/systemone`` request envelope.
    Raises :class:`QuestionError` (caught by the server and turned into a
    400) on any violation."""
    if not isinstance(body, dict):
        raise QuestionError("request body must be a JSON object")
    if "state" not in body or body["state"] in (None, ""):
        raise QuestionError("'state' is required (string or object)")
    if not isinstance(body["state"], (str, dict, list)):
        raise QuestionError("'state' must be a string or an object")
    if "model" not in body or not isinstance(body["model"], str) or not body["model"].strip():
        raise QuestionError("'model' is required and must be a non-empty string (any alias is accepted)")
    if "questions" not in body or not isinstance(body["questions"], dict) or not body["questions"]:
        raise QuestionError("'questions' is required and must be a non-empty object")
    if len(body["questions"]) > max_questions:
        raise QuestionError(
            f"request has {len(body['questions'])} questions, more than the configured limit of {max_questions}"
        )


# Kept as the previous name too, since it is the exact validation the
# original prototype's server.py exposed under this name.
validate_systemone_request = validate_envelope


_OPEN_BRACKETS = frozenset({ord("{"), ord("[")})
_CLOSE_BRACKETS = frozenset({ord("}"), ord("]")})
_QUOTE = ord('"')
_BACKSLASH = ord("\\")


def json_nesting_too_deep(raw: bytes, max_depth: int) -> bool:
    """True if ``raw`` contains a JSON object/array nested more than
    ``max_depth`` levels deep.

    This is a linear, single-pass scan over the raw bytes -- it counts
    ``{``/``[`` and ``}``/``]`` outside of string literals (tracking
    escaped quotes so a bracket character inside a string value is never
    mistaken for structure) and never calls :func:`json.loads`. The point
    is to reject a maliciously (or accidentally) deep body *before*
    parsing is ever attempted: ``json.loads`` on sufficiently deep input
    raises an uncaught ``RecursionError`` once it exhausts the
    interpreter's recursion limit, which is not one of the exception
    types the request handler otherwise catches and would previously
    crash the connection instead of returning a clean 400. This scan
    itself is recursion-free and O(len(raw)) in both time and memory, so
    it cannot itself be a resource-exhaustion vector."""
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == _BACKSLASH:
                escaped = True
            elif byte == _QUOTE:
                in_string = False
            continue
        if byte == _QUOTE:
            in_string = True
        elif byte in _OPEN_BRACKETS:
            depth += 1
            if depth > max_depth:
                return True
        elif byte in _CLOSE_BRACKETS:
            depth -= 1
    return False


# --------------------------------------------------------------------- JSON Schema fragments


NOUL_ANSWER_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["type", "noul"],
    "properties": {
        "type": {"type": "string", "enum": ["noul"]},
        "noul": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "degraded": {"type": "boolean", "description": "true: no usable reading; the value is null"},
    },
}

CHOICE_ANSWER_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["type", "choice", "probabilities", "confidence"],
    "properties": {
        "type": {"type": "string", "enum": ["choice"]},
        "choice": {"type": ["string", "null"]},
        "probabilities": {"type": ["object", "null"], "additionalProperties": {"type": "number"}},
        "confidence": {"type": ["number", "null"]},
        "degraded": {"type": "boolean"},
    },
}

SCORE_ANSWER_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["type", "score", "probabilities", "confidence"],
    "properties": {
        "type": {"type": "string", "enum": ["score"]},
        "score": {"type": ["number", "null"]},
        "probabilities": {"type": ["object", "null"], "additionalProperties": {"type": "number"}},
        "confidence": {"type": ["number", "null"]},
        "degraded": {"type": "boolean"},
    },
}

QUESTION_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["type", "instructions"],
    "properties": {
        "type": {"type": "string", "enum": ["noul", "choice", "score"]},
        "instructions": {"type": "string", "minLength": 1},
        "options": {"type": ["object", "array"]},
        "criteria": {"type": ["object", "array"]},
        "levels": {"type": ["object", "array"]},
    },
}

SYSTEMONE_REQUEST_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["state", "model", "questions"],
    "properties": {
        "state": {
            "oneOf": [{"type": "string"}, {"type": "object"}],
            "description": "The state to answer every question about; a plain string or a JSON object.",
        },
        "model": {
            "type": "string",
            "minLength": 1,
            "description": "An alias only. The server never routes on this value and always answers "
            "with the real backend model id it was started with.",
        },
        "questions": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": QUESTION_SCHEMA,
        },
        "mode": {
            "type": "string",
            "enum": ["fast", "deliberate"],
            "description": "Optional per-request override of the decision head's mode, honored only "
            "when the server's mode_override_allowed is true. See docs/deliberate.md.",
        },
    },
}

SYSTEMONE_RESPONSE_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["model", "answers", "usage"],
    "properties": {
        "model": {"type": "string"},
        "mode": {
            "type": "string",
            "enum": ["fast", "deliberate"],
            "description": "The mode that answered this request.",
        },
        "answers": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [NOUL_ANSWER_SCHEMA, CHOICE_ANSWER_SCHEMA, SCORE_ANSWER_SCHEMA]
            },
        },
        "usage": {
            "type": "object",
            "required": ["input_tokens", "output_tokens"],
            "properties": {
                "input_tokens": {"type": ["integer", "null"]},
                "output_tokens": {"type": ["integer", "null"]},
            },
        },
    },
}

ERROR_RESPONSE_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["error"],
    "properties": {
        "error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
            },
        }
    },
}

HEALTH_RESPONSE_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["status"],
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "backend": {"type": "string"},
        "model": {"type": ["string", "null"]},
        "dry": {"type": "boolean"},
    },
}

READY_RESPONSE_SCHEMA: JSONDict = {
    "type": "object",
    "required": ["status", "ready"],
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "ready": {"type": "boolean"},
        "checked_at": {"type": "string"},
        "detail": {"type": ["string", "null"]},
    },
}


# --------------------------------------------------------------------- OpenAPI document


def build_openapi_document(
    *, version: str, title: str = "decisionhead", server_url: str = "http://127.0.0.1:8787"
) -> JSONDict:
    """Renders the OpenAPI 3.0 document served at ``GET /openapi.json``.
    Built from the same schema fragments defined above, so the served
    document cannot describe an API the server does not actually
    implement."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": "Self-hosted typed-question decision service (Jev-shaped API) "
            "over any local model that exposes per-token logprobs.",
        },
        "servers": [{"url": server_url}],
        "paths": {
            "/v1/systemone": {
                "post": {
                    "summary": "Answer a set of typed questions about a state.",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": SYSTEMONE_REQUEST_SCHEMA}},
                    },
                    "responses": {
                        "200": {
                            "description": "Answers for every question in the request.",
                            "content": {"application/json": {"schema": SYSTEMONE_RESPONSE_SCHEMA}},
                        },
                        "400": {
                            "description": "Malformed request.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "401": {
                            "description": "Missing or invalid bearer token (only when auth is configured).",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "403": {
                            "description": "Cross-origin request (an Origin header) while CORS is off.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "421": {
                            "description": "The Host header is not a name this server answers to.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "413": {
                            "description": "Request body exceeded the configured size limit.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "415": {
                            "description": "Request was not JSON.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "429": {
                            "description": "Too many concurrent requests.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "502": {
                            "description": "The backend model server failed or returned an unreadable response.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                        "504": {
                            "description": "The request exceeded the configured request timeout.",
                            "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
                        },
                    },
                }
            },
            "/health": {
                "get": {
                    "summary": "Liveness probe. Always 200 once the process is up.",
                    "responses": {
                        "200": {
                            "description": "The process is alive.",
                            "content": {"application/json": {"schema": HEALTH_RESPONSE_SCHEMA}},
                        }
                    },
                }
            },
            "/ready": {
                "get": {
                    "summary": "Readiness probe. Checks backend reachability, cached briefly.",
                    "responses": {
                        "200": {
                            "description": "Readiness state (see the 'ready' field; the HTTP status is "
                            "always 200 so a caller does not need special-case handling).",
                            "content": {"application/json": {"schema": READY_RESPONSE_SCHEMA}},
                        }
                    },
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus text-format metrics.",
                    "responses": {
                        "200": {
                            "description": "Counters and latency histograms in Prometheus exposition format.",
                            "content": {"text/plain": {"schema": {"type": "string"}}},
                        }
                    },
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "This document.",
                    "responses": {"200": {"description": "The OpenAPI document."}},
                }
            },
        },
        "components": {
            "schemas": {
                "SystemOneRequest": SYSTEMONE_REQUEST_SCHEMA,
                "SystemOneResponse": SYSTEMONE_RESPONSE_SCHEMA,
                "Error": ERROR_RESPONSE_SCHEMA,
                "Health": HEALTH_RESPONSE_SCHEMA,
                "Ready": READY_RESPONSE_SCHEMA,
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Optional. Enabled only when the server is started with a bearer "
                    "token environment variable configured.",
                }
            },
        },
    }
