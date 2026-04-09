#!/usr/bin/env bash
# Start the full local stack: two gateway instances + Nginx load balancer.
#
# Processes run in the background; logs go to /tmp/gw1.log, /tmp/gw2.log,
# and /tmp/nginx.log. Press Ctrl-C to stop everything cleanly.
#
# Usage:
#   ./scripts/start_stack.sh
#
# Environment (all optional — defaults shown):
#   API_KEY              auth key forwarded to both gateway instances (default: empty)
#   PORT1                port for gateway instance 1 (default: 8080)
#   PORT2                port for gateway instance 2 (default: 8081)
#   METRICS_PORT1        Prometheus scrape port for instance 1 (default: 9101)
#   METRICS_PORT2        Prometheus scrape port for instance 2 (default: 9102)
#   VLLM_SERVER_PROFILE  profile label on all metrics (default: baseline)
#   SKIP_NGINX           set to 1 to skip the Nginx LB (default: 0)
#
# Requires:
#   uv    (Python env manager — https://docs.astral.sh/uv/)
#   nginx (with ngx_http_stub_status_module — standard on macOS/Linux)

set -uo pipefail   # NOTE: no -e so nginx failures don't kill the whole script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env from repo root if it exists (picks up API_KEY, VLLM_SERVER_PROFILE, etc.)
if [[ -f "$REPO_ROOT/.env" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$REPO_ROOT/.env" && set +a
fi

PORT1="${PORT1:-8080}"
PORT2="${PORT2:-8081}"
METRICS_PORT1="${METRICS_PORT1:-9101}"
METRICS_PORT2="${METRICS_PORT2:-9102}"
VLLM_SERVER_PROFILE="${VLLM_SERVER_PROFILE:-baseline}"
SKIP_NGINX="${SKIP_NGINX:-0}"
API_KEY="${API_KEY:-}"

LOG_GW1="/tmp/gw1.log"
LOG_GW2="/tmp/gw2.log"
LOG_NGINX="/tmp/nginx.log"

NGINX_CONF="$REPO_ROOT/nginx-gateway-lb.conf"

GW1_PID=""
GW2_PID=""
SHUTTING_DOWN=0   # set to 1 in cleanup so the monitor loop exits cleanly

# ---------------------------------------------------------------------------
# Cleanup — stop all child processes on Ctrl-C or SIGTERM
# ---------------------------------------------------------------------------
cleanup() {
  SHUTTING_DOWN=1
  echo ""
  echo "Stopping stack..."
  [[ -n "$GW1_PID" ]] && kill "$GW1_PID" 2>/dev/null || true
  [[ -n "$GW2_PID" ]] && kill "$GW2_PID" 2>/dev/null || true
  if [[ "$SKIP_NGINX" != "1" ]] && command -v nginx &>/dev/null; then
    nginx -p /tmp -s stop 2>/dev/null || true
  fi
  echo "Done."
  exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Stop any existing nginx on /tmp to avoid conflict
# ---------------------------------------------------------------------------
if [[ "$SKIP_NGINX" != "1" ]] && command -v nginx &>/dev/null; then
  nginx -p /tmp -s stop 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Start gateway instance 1
# ---------------------------------------------------------------------------
echo "Starting gateway 1  (port $PORT1, metrics :$METRICS_PORT1) → $LOG_GW1"
PORT="$PORT1" \
  GATEWAY_METRICS_PORT="$METRICS_PORT1" \
  VLLM_SERVER_PROFILE="$VLLM_SERVER_PROFILE" \
  API_KEY="$API_KEY" \
  uv run python "$REPO_ROOT/main.py" >"$LOG_GW1" 2>&1 &
GW1_PID=$!

# ---------------------------------------------------------------------------
# Start gateway instance 2
# ---------------------------------------------------------------------------
echo "Starting gateway 2  (port $PORT2, metrics :$METRICS_PORT2) → $LOG_GW2"
PORT="$PORT2" \
  GATEWAY_METRICS_PORT="$METRICS_PORT2" \
  VLLM_SERVER_PROFILE="$VLLM_SERVER_PROFILE" \
  API_KEY="$API_KEY" \
  uv run python "$REPO_ROOT/main.py" >"$LOG_GW2" 2>&1 &
GW2_PID=$!

# ---------------------------------------------------------------------------
# Poll until both gateways are healthy (or give up after 30s)
# ---------------------------------------------------------------------------
echo "Waiting for gateways to start..."
deadline=$(( $(date +%s) + 30 ))
while true; do
  gw1_ok=false; gw2_ok=false
  curl -sf "http://localhost:$PORT1/healthz" -o /dev/null 2>/dev/null && gw1_ok=true
  curl -sf "http://localhost:$PORT2/healthz" -o /dev/null 2>/dev/null && gw2_ok=true
  $gw1_ok && $gw2_ok && break
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "Gateways did not start in 30s. Check $LOG_GW1 and $LOG_GW2." >&2
    exit 1
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# Start Nginx load balancer
# ---------------------------------------------------------------------------
if [[ "$SKIP_NGINX" != "1" ]]; then
  if ! command -v nginx &>/dev/null; then
    echo "nginx not found — skipping LB. Install nginx or set SKIP_NGINX=1."
  else
    echo "Starting Nginx LB (port 8780) → $LOG_NGINX"
    nginx -p /tmp -c "$NGINX_CONF" 2>"$LOG_NGINX" || {
      echo "  nginx failed to start — check $LOG_NGINX"
    }
  fi
fi

# ---------------------------------------------------------------------------
# Ready
# ---------------------------------------------------------------------------
echo ""
echo "Stack is up:"
echo "  Gateway 1  : http://localhost:$PORT1       metrics: http://localhost:$METRICS_PORT1/metrics"
echo "  Gateway 2  : http://localhost:$PORT2       metrics: http://localhost:$METRICS_PORT2/metrics"
if [[ "$SKIP_NGINX" != "1" ]]; then
  echo "  Nginx LB   : http://localhost:8780"
fi
echo ""
echo "Logs: tail -f $LOG_GW1 $LOG_GW2"
echo "Press Ctrl-C to stop all processes."
echo ""

# ---------------------------------------------------------------------------
# Keep script alive; restart gateways if they crash unexpectedly
# ---------------------------------------------------------------------------
while [[ $SHUTTING_DOWN -eq 0 ]]; do
  if ! kill -0 "$GW1_PID" 2>/dev/null; then
    echo "Gateway 1 exited unexpectedly — restarting..."
    PORT="$PORT1" \
      GATEWAY_METRICS_PORT="$METRICS_PORT1" \
      VLLM_SERVER_PROFILE="$VLLM_SERVER_PROFILE" \
      API_KEY="$API_KEY" \
      uv run python "$REPO_ROOT/main.py" >>"$LOG_GW1" 2>&1 &
    GW1_PID=$!
  fi
  if ! kill -0 "$GW2_PID" 2>/dev/null; then
    echo "Gateway 2 exited unexpectedly — restarting..."
    PORT="$PORT2" \
      GATEWAY_METRICS_PORT="$METRICS_PORT2" \
      VLLM_SERVER_PROFILE="$VLLM_SERVER_PROFILE" \
      API_KEY="$API_KEY" \
      uv run python "$REPO_ROOT/main.py" >>"$LOG_GW2" 2>&1 &
    GW2_PID=$!
  fi
  sleep 5
done
