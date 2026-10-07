#!/usr/bin/env bash
# Exercises a running decisionhead server with curl.
#
# Start a server first, either against a real backend:
#   decisionhead serve --backend ollama --base http://127.0.0.1:11434 --model qwen3.8:27b
# or, with no model and no network at all, against the deterministic fake backend:
#   decisionhead serve --dry
#
# Then, from this directory:
#   ./curl_example.sh [base_url]

set -euo pipefail

BASE="${1:-http://127.0.0.1:8787}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTIONS_FILE="${SCRIPT_DIR}/supplier_onboarding_questions.json"

QUESTIONS_JSON="$(cat "${QUESTIONS_FILE}")"

read -r -d '' BODY <<JSON || true
{
  "state": "Acme Logistics BV submitted a certificate of insurance dated 2026-01-15 and a signed W-9. No endorsement schedule was attached, and the entity name on the COI (Acme Logistics Inc) does not match the W-9 (Acme Logistics BV).",
  "model": "jev-latest",
  "questions": ${QUESTIONS_JSON}
}
JSON

echo "POST ${BASE}/v1/systemone"
curl -sS -X POST "${BASE}/v1/systemone" \
  -H "Content-Type: application/json" \
  -d "${BODY}" | python3 -m json.tool

echo
echo "GET ${BASE}/health"
curl -sS "${BASE}/health" | python3 -m json.tool

echo
echo "GET ${BASE}/ready"
curl -sS "${BASE}/ready" | python3 -m json.tool
