#!/usr/bin/env bash
# Run vLLM A/B experiments across three Modal container profiles.
#
# Each profile maps to a distinct vLLM configuration deployed on Modal:
#   standard  — baseline flags only                            (technique: baseline)
#   optimized — chunked prefill + prefix caching               (technique: optimized)
#   hardcore  — optimized + fp8 KV cache + larger batch budget (technique: hardcore)
#
# For each profile the script:
#   1. Optionally deploys the Modal container (--deploy).
#   2. Waits until the backend is reachable through the gateway.
#   3. Runs crew.py N times with the matching X-Technique label.
#   4. Prints a per-technique summary and an overall comparison table.
#
# Usage:
#   bash scripts/run_experiments.sh [OPTIONS]
#
# Options:
#   --deploy          Deploy Modal containers before running (adds ~5-10 min per profile)
#   --profiles LIST   Comma-separated subset of profiles to run (default: all three)
#                     Example: --profiles standard,optimized
#   --topic TEXT      Research topic for the crew (default: see _common.sh)
#   --runs N          Crew runs per profile (default: 3)
#   --wait S          Seconds to poll for backend readiness (default: 480)
#   --help            Show this message and exit
#
# Prerequisites:
#   - Gateway running:   uv run python main.py
#   - Nginx LB running:  nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"
#   - Modal CLI logged in and 'huggingface-secret' created
#   - config.yaml contains the modal-gemma4-* backends with correct URLs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env so OPENAI_API_KEY / API_KEY are available to curl health-poll
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DEPLOY=false
PROFILES_ARG="standard,optimized,hardcore"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy)          DEPLOY=true; shift ;;
    --profiles)        PROFILES_ARG="$2"; shift 2 ;;
    --topic)           TOPIC="$2"; shift 2 ;;
    --runs)            N_RUNS="$2"; shift 2 ;;
    --wait)            BACKEND_WAIT_S="$2"; shift 2 ;;
    --help)
      sed -n '/^# Usage:/,/^[^#]/{ /^#/p }' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1  (run with --help for usage)" >&2
      exit 1
      ;;
  esac
done

IFS=',' read -ra PROFILES <<< "$PROFILES_ARG"

# ---------------------------------------------------------------------------
# Profile → (backend name, technique label, modal file)
# ---------------------------------------------------------------------------
profile_backend()   { case "$1" in standard) echo "modal-gemma4-standard" ;; optimized) echo "modal-gemma4-optimized" ;; hardcore) echo "modal-gemma4-hardcore" ;; esac }
profile_technique() { case "$1" in standard) echo "baseline" ;; optimized) echo "optimized" ;; hardcore) echo "hardcore" ;; esac }
profile_modal()     { case "$1" in standard) echo "modal/vllm_gemma4.py" ;; optimized) echo "modal/vllm_gemma4_optimized.py" ;; hardcore) echo "modal/vllm_gemma4_hardcore.py" ;; esac }

# ---------------------------------------------------------------------------
# Helper: wait for a specific backend to be reachable through the gateway
# ---------------------------------------------------------------------------
wait_for_backend() {
  local backend="$1"
  local deadline=$(( $(date +%s) + BACKEND_WAIT_S ))
  local attempt=0

  echo ""
  echo "⏳ Waiting for backend '${backend}' (up to ${BACKEND_WAIT_S}s, polling every ${POLL_INTERVAL}s)..."
  echo "   Cold start + weight download on A10G typically takes 3-8 minutes."

  while true; do
    attempt=$(( attempt + 1 ))
    now=$(date +%s)
    remaining=$(( deadline - now ))

    if [[ $remaining -le 0 ]]; then
      echo ""
      echo "❌ Timed out after ${BACKEND_WAIT_S}s waiting for '${backend}'." >&2
      echo "   Check Modal dashboard or increase --wait." >&2
      exit 1
    fi

    # Send a minimal request to the specific backend; 200 + non-empty content = ready
    http_code=$(curl -s -o /tmp/_exp_resp.json -w "%{http_code}" \
      --max-time 30 \
      -X POST "${GATEWAY_BASE}/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${OPENAI_API_KEY:-${API_KEY:-dummy}}" \
      -d "{\"model\":\"${backend}\",\"messages\":[{\"role\":\"user\",\"content\":\"ready?\"}],\"max_tokens\":1}" \
      2>/dev/null || echo "000")

    if [[ "$http_code" == "200" ]]; then
      echo "✅ Backend '${backend}' is ready (attempt ${attempt}, ${remaining}s remaining)."
      return 0
    fi

    printf "   [%3ds left] attempt %d — HTTP %s, retrying in %ds...\n" \
      "$remaining" "$attempt" "$http_code" "$POLL_INTERVAL"
    sleep "$POLL_INTERVAL"
  done
}

# ---------------------------------------------------------------------------
# Deploy (optional)
# ---------------------------------------------------------------------------
if $DEPLOY; then
  echo "═══════════════════════════════════════════════════════"
  echo " Deploying Modal containers"
  echo "═══════════════════════════════════════════════════════"
  for profile in "${PROFILES[@]}"; do
    echo ""
    echo "── Deploying profile: $profile ──"
    bash "$SCRIPT_DIR/deploy_modal_vllm.sh" "$profile"
  done
  echo ""
  echo "All containers deployed. Waiting for them to warm up..."
fi

# ---------------------------------------------------------------------------
# CSV output — written to data/experiments.csv (one row per run)
# ---------------------------------------------------------------------------
CSV_OUT="$REPO_ROOT/data/experiments.csv"
mkdir -p "$(dirname "$CSV_OUT")"
echo "technique,profile,run,wall_clock_s,success,error" > "$CSV_OUT"
echo "Writing per-run results to: $CSV_OUT"

# ---------------------------------------------------------------------------
# Results accumulator (one pre-formatted line per profile)
# ---------------------------------------------------------------------------
RESULT_LINES=""

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Running experiments"
echo " Topic:    $TOPIC"
echo " Runs/profile: $N_RUNS"
echo " Profiles: ${PROFILES[*]}"
echo "═══════════════════════════════════════════════════════"

for profile in "${PROFILES[@]}"; do
  backend=$(profile_backend "$profile")
  technique=$(profile_technique "$profile")

  echo ""
  echo "───────────────────────────────────────────────────────"
  echo " Profile:   $profile"
  echo " Backend:   $backend"
  echo " Technique: $technique"
  echo "───────────────────────────────────────────────────────"

  # Wait for this backend to be reachable
  wait_for_backend "$backend"

  # Run crew N times
  ok=0
  fail=0
  total_ms=0

  for run in $(seq 1 "$N_RUNS"); do
    echo ""
    echo "  ── Run $run / $N_RUNS ──"
    start_ms=$(date +%s)

    crew_exit=0
    if MODEL_NAME="$backend" \
       CREW_VLLM_WAIT_S=0 \
       uv run --group crew --group otel python "$REPO_ROOT/crew.py" \
         --topic "$TOPIC" \
         --technique "$technique"; then
      ok=$(( ok + 1 ))
    else
      crew_exit=$?
      fail=$(( fail + 1 ))
      echo "  ⚠️  Run $run failed (exit $crew_exit)."
    fi

    end_ms=$(date +%s)
    elapsed=$(( end_ms - start_ms ))
    total_ms=$(( total_ms + elapsed ))
    echo "  ⏱  Run $run wall-clock: ${elapsed}s"

    # Append one CSV row (ttft_s/tokens are left empty; read from Prometheus if needed)
    if [[ $crew_exit -eq 0 ]]; then
      echo "${technique},${profile},${run},${elapsed},true," >> "$CSV_OUT"
    else
      echo "${technique},${profile},${run},${elapsed},false,exit_code_${crew_exit}" >> "$CSV_OUT"
    fi
  done

  if [[ $N_RUNS -gt 0 ]]; then
    avg_ms=$(( total_ms / N_RUNS ))
  else
    avg_ms=0
  fi

  echo ""
  echo "  Profile '$profile': $ok/$N_RUNS succeeded, avg wall-clock ${avg_ms}s"
  RESULT_LINES="${RESULT_LINES}$(printf "  %-12s %-15s %d/%d       %ds\n" "$profile" "$technique" "$ok" "$N_RUNS" "$avg_ms")"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Experiment Summary"
echo "═══════════════════════════════════════════════════════"
printf "  %-12s %-15s %-8s %s\n" "Profile" "Technique" "Success" "Avg (s)"
printf "  %-12s %-15s %-8s %s\n" "-------" "---------" "-------" "-------"
printf "%s" "$RESULT_LINES"
echo ""
echo "Results CSV:           $CSV_OUT"
echo "Metrics per technique: curl http://localhost:9101/metrics | grep gateway_requests_total"
echo "Grafana dashboard:     http://localhost:3000"
