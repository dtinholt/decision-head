#!/usr/bin/env bash
# One-command start: brings up a decisionhead server, waits for it to be
# ready, sends one noul question and one choice question, and prints the
# answers.
#
# Points at a real backend when both DECISIONHEAD_BASE_URL and
# DECISIONHEAD_MODEL are set in the environment:
#
#   export DECISIONHEAD_BACKEND=ollama          # or: openai
#   export DECISIONHEAD_BASE_URL=http://127.0.0.1:11434
#   export DECISIONHEAD_MODEL=qwen3.8:27b
#   ./examples/quickstart.sh
#
# With neither set, runs against the deterministic fake backend instead
# (no model, no network, same request/response shapes) -- this is also
# what makes the script runnable in CI with no inference hardware:
#
#   ./examples/quickstart.sh
#
# Optional env vars: DECISIONHEAD_PORT (default 8787), DECISIONHEAD_HOST
# (default 127.0.0.1), DECISIONHEAD_API_KEY_ENV (name of an env var
# holding a bearer token for --backend openai, if the backend requires
# one -- never the key itself).

set -euo pipefail

PORT="${DECISIONHEAD_PORT:-8787}"
HOST="${DECISIONHEAD_HOST:-127.0.0.1}"
BASE="http://${HOST}:${PORT}"
BACKEND="${DECISIONHEAD_BACKEND:-ollama}"

# Prefer the installed console script; fall back to `python3 -m` so this
# also works from a checkout with an editable install but no PATH entry,
# as long as it is run from the repository root.
DECISIONHEAD_CMD=(decisionhead)
if ! command -v decisionhead >/dev/null 2>&1; then
    DECISIONHEAD_CMD=(python3 -m decisionhead)
fi

SERVER_ARGS=(serve --host "${HOST}" --port "${PORT}")
if [[ -n "${DECISIONHEAD_BASE_URL:-}" && -n "${DECISIONHEAD_MODEL:-}" ]]; then
    echo "Starting decisionhead against a real backend: ${BACKEND} @ ${DECISIONHEAD_BASE_URL} (${DECISIONHEAD_MODEL})"
    SERVER_ARGS+=(--backend "${BACKEND}" --base "${DECISIONHEAD_BASE_URL}" --model "${DECISIONHEAD_MODEL}")
    if [[ -n "${DECISIONHEAD_API_KEY_ENV:-}" ]]; then
        SERVER_ARGS+=(--api-key-env "${DECISIONHEAD_API_KEY_ENV}")
    fi
else
    echo "DECISIONHEAD_BASE_URL/DECISIONHEAD_MODEL not both set -- starting against the deterministic fake backend (--dry)."
    SERVER_ARGS+=(--dry)
fi

echo "+ ${DECISIONHEAD_CMD[*]} ${SERVER_ARGS[*]}"
"${DECISIONHEAD_CMD[@]}" "${SERVER_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for ${BASE}/ready ..."
READY=""
for _ in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "decisionhead exited before becoming ready" >&2
        exit 1
    fi
    if RESPONSE="$(curl -sS --max-time 2 "${BASE}/ready" 2>/dev/null)"; then
        if printf '%s' "${RESPONSE}" | grep -q '"ready": *true'; then
            READY="1"
            break
        fi
    fi
    sleep 0.5
done
if [[ -z "${READY}" ]]; then
    echo "decisionhead never reported ready at ${BASE}/ready" >&2
    exit 1
fi
echo "Ready: ${RESPONSE}"

echo
echo "POST ${BASE}/v1/systemone (one noul question, one choice question)"
read -r -d '' BODY <<'JSON' || true
{
  "state": "Acme Logistics BV submitted a certificate of insurance dated 2026-01-15 and a signed W-9. No endorsement schedule was attached, and the entity name on the COI (Acme Logistics Inc) does not match the W-9 (Acme Logistics BV).",
  "model": "jev-latest",
  "questions": {
    "coi_current": {
      "type": "noul",
      "instructions": "The certificate of insurance on file is dated within the last 12 months and lists coverage limits meeting the standard supplier requirement."
    },
    "next_step": {
      "type": "choice",
      "instructions": "Given everything in the state, what should happen to this supplier onboarding case next?",
      "criteria": {
        "APPROVE": "all required documents are present and current; onboard the supplier",
        "REQUEST_DOCS": "one or more required documents are missing or expired; ask the supplier to submit them",
        "ESCALATE": "something in the state looks contradictory or unusual enough to need a human reviewer"
      }
    }
  }
}
JSON

curl -sS -X POST "${BASE}/v1/systemone" \
    -H "Content-Type: application/json" \
    -d "${BODY}" | python3 -m json.tool

echo
echo "decisionhead is running at ${BASE} (pid ${SERVER_PID}); stopping it now."
