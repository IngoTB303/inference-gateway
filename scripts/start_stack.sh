#!/usr/bin/env bash
# Start the full local stack: two gateway instances + Nginx load balancer.
#
# Processes run in the background; logs go to /tmp/gw1.log, /tmp/gw2.log,
# and /tmp/nginx.log. A cleanup trap stops everything on Ctrl-C or EXIT.
#
# Usage:
#   bash scripts/start_stack.sh
#
# Environment (all optional — defaults shown):
#   API_KEY                 auth key forwarded to both gateway instances (default: empty)
#   PORT1                   port for gateway instance 1 (default: 8080)
#   PORT2                   port for gateway instance 2 (default: 8081)
#   METRICS_PORT1           Prometheus scrape port for instance 1 (default: 9101)
#   METRICS_PORT2           Prometheus scrape port for instance 2 (default: 9102)
#   VLLM_SERVER_PROFILE     profile label on all metrics (default: baseline)
#   SKIP_NGINX              set to 1 to skip the Nginx LB (default: 0)
#
# Requires:
#   uv    (Python env manager — https://docs.astral.sh/uv/)
#   nginx (with ngx_http_stub_status_module — standard on macOS/Linux)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# ---------------------------------------------------------------------------
# Cleanup — kill all child processes on exit
# ---------------------------------------------------------------------------
cleanup() {
  echo ""
  echo "Stopping stack..."
  kill "$GW1_PID" "$GW2_PID" 2>/dev/null || true
  if [[ "$SKIP_NGINX" != "1" ]]; then
    nginx -p /tmp -s stop 2>/dev/null || true
  fi
  echo "Done."
}
trap cleanup EXIT INT TERM

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

# Give the gateways a moment to bind their ports
sleep 2

# ---------------------------------------------------------------------------
# Start Nginx load balancer
# ---------------------------------------------------------------------------
if [[ "$SKIP_NGINX" != "1" ]]; then
  if ! command -v nginx &>/dev/null; then
    echo "nginx not found — skipping LB. Install nginx or set SKIP_NGINX=1." >&2
  else
    echo "Starting Nginx LB (port 8780) → $LOG_NGINX"
    nginx -p /tmp -c "$NGINX_CONF" >"$LOG_NGINX" 2>&1
  fi
fi

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
echo ""
echo "Stack is up:"
echo "  Gateway 1  : http://localhost:$PORT1     metrics: http://localhost:$METRICS_PORT1/metrics"
echo "  Gateway 2  : http://localhost:$PORT2     metrics: http://localhost:$METRICS_PORT2/metrics"
if [[ "$SKIP_NGINX" != "1" ]]; then
  echo "  Nginx LB   : http://localhost:8780"
fi
echo ""
echo "Logs: tail -f $LOG_GW1  $LOG_GW2"
echo "Press Ctrl-C to stop all processes."
echo ""

# Wait for any child to exit (keeps script alive)
wait "$GW1_PID" "$GW2_PID"
