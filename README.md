# Inference Gateway

An OpenAI-compatible HTTP gateway that routes chat completion requests to multiple configurable inference backends (llama.cpp, vLLM, or any OpenAI-compatible server) or falls back to an echo response when no real backend is reachable.

**Team members:** Dev Jadhav, Ingo Villow

---

## How to Run

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd inference-gateway

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env to set PORT and optionally API_KEY

# 4. Start the gateway
uv run python main.py
```

The server listens on `http://localhost:8080` by default.

---

## Configuration

### Environment variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port the HTTP server listens on. Set to `8081` for a second instance. |
| `API_KEY` | *(unset)* | If set, all POST requests must include `Authorization: Bearer <key>` or `Authorization: Api-Key <key>`. Leave empty to disable auth. |
| `GATEWAY_METRICS_PORT` | `9101` | Port for the Prometheus metrics scrape endpoint. Set to `9102` for a second instance. |
| `GPU_HOURLY_COST_USD` | `1.10` | Hourly GPU cost used to estimate `gateway_gpu_cost_usd_total` (A10 default) |
| `VLLM_SERVER_PROFILE` | `default` | Server profile label attached to all Prometheus metrics (e.g. `chunked_prefill`, `baseline`) |

### Backend configuration (`config.yaml`)

Backends are configured in `config.yaml`. The gateway routes requests by matching the `model` field in the request body to a backend `name`. Unrecognised or absent `model` values fall back to `default_backend`.

```yaml
backends:
  - name: local                  # echo mode — no real backend needed
    type: echo

  - name: modal-gemma4-standard  # vLLM Gemma4 standard on Modal
    type: http
    url: https://ingo-villnow--vllm-gemma4-standard-serve.modal.run/
    timeout: 120
    model: gemma-4-e2b-it        # vLLM requires this field

  - name: modal-gemma4-optimized # vLLM Gemma4 optimized on Modal
    type: http
    url: https://ingo-villnow--vllm-gemma4-optimized-serve.modal.run/
    timeout: 120
    model: gemma-4-e2b-it

default_backend: local
```

**Adding a new backend:** Add one entry to `config.yaml` — no code changes needed.

**Echo-only mode (no config.yaml):** The gateway falls back to the `BACKEND_URL` environment variable; if that is also unset, it runs in echo mode.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Index — lists all available endpoints |
| `POST` | `/v1/chat/completions` | Main inference endpoint |
| `GET` | `/v1/models` | Proxy to the default HTTP backend's `/v1/models` |
| `GET` | `/v1/backends` | List configured backends and default |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/health` | Extended health check with upstream probe |
| `GET` | `/metrics` | Legacy JSON counters (request count, latency, tokens) |
| `GET` | `:9101/metrics` | Prometheus-format metrics (dedicated scrape port) |

### Request body (`POST /v1/chat/completions`)

| Field | Type | Required | Notes |
|---|---|---|---|
| `messages` | array | yes | Each element must have `role` (system/user/assistant) and `content` |
| `model` | string | no | Backend name from `config.yaml`; defaults to `default_backend` |
| `stream` | boolean | no | SSE streaming (default `false`) |
| `max_tokens` | integer | no | 1–100 000 |
| `temperature` | number | no | 0–2 |
| `stop` | string or array | no | Stop sequences |

### Response shape

```json
{
  "id": "<request-id>",
  "object": "chat.completion",
  "model": "<backend-model-name>",
  "backend": "<gateway-backend-name>",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "latency_ms": 42.15}
}
```

The `backend` field tells the client which gateway backend handled the request. `latency_ms` is the gateway-measured wall-clock time for the request in milliseconds.

---

## curl Examples

### Route to local echo backend

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hello"}]}'
```

Response includes `"backend": "local"` and content `"Echo: hello"`.

### Route to Modal vLLM (standard)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "modal-gemma4-standard", "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 50}'
```

### Route to Modal vLLM (optimized)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "modal-gemma4-optimized", "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 50}'
```

### Omit model — falls back to default backend

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

### With auth (when `API_KEY` is set)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-key' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hello"}]}'
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

### List configured backends

```bash
curl http://localhost:8080/v1/backends
# → {"backends": [...], "default": "local"}
```

### Health check

```bash
curl http://localhost:8080/healthz
# → {"status": "ok"}
```

### Metrics

```bash
# Legacy JSON counters
curl http://localhost:8080/metrics
# → {"request_count": N, "error_count": N, "avg_latency_ms": N, ...}

# Prometheus scrape endpoint (separate port)
curl http://localhost:9101/metrics
# → # HELP gateway_requests_total ...
#   gateway_requests_total{status_code="200",model="local",technique="baseline",...} 5.0
```

Pass `X-Technique` on requests to segment metrics by inference technique (e.g. `chunked_prefill`, `prefix_caching`). Missing header defaults to `baseline`.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Technique: chunked_prefill' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hello"}]}'
```

---

## Running Tests

### Unit tests

```bash
uv run pytest
uv run pytest tests/test_gateway.py::test_routing_by_model_echo   # single test
```

The suite uses a FastAPI `TestClient` and [respx](https://lundberg.github.io/respx/) to mock backend `httpx` calls. No running backend is required.

**60 tests** covering: GET endpoints, echo shape, request-ID, validation errors (400), auth (401), SSE streaming, backend proxy, response normalization, multi-backend routing, metrics, `latency_ms` in usage, Prometheus counter/histogram/gauge behaviour, `X-Technique` label propagation.

### Live backend tests

Tests that hit real backends are marked `@pytest.mark.live` and **skipped by default**. Run them when backends are available:

```bash
# All live tests
uv run pytest -m live -v

# Single live test
uv run pytest -m live tests/test_gateway.py::test_live_remote_modal_llama
```

Live tests skip gracefully (rather than fail) when a backend returns 502/504.

### Bruno collection

Open `tests/bruno/` in [Bruno](https://www.usebruno.com/) and select the **local** environment (`base_url = http://localhost:8080`, `metrics_url = http://localhost:9101`).

---

## Nginx Load Balancer

**Prerequisites:** [nginx](https://nginx.org/) (any standard build with `ngx_http_stub_status_module`)

`nginx-gateway-lb.conf` (project root) distributes traffic round-robin across two gateway instances and exposes an `/nginx_status` endpoint for Prometheus.

### Full stack with two gateway instances

```
Client → :8780 (Nginx LB) → :8080 / :8081 (Gateway instances) → Modal vLLM
```

```bash
# Terminal 1 — gateway instance 1
PORT=8080 GATEWAY_METRICS_PORT=9101 uv run python main.py

# Terminal 2 — gateway instance 2
PORT=8081 GATEWAY_METRICS_PORT=9102 uv run python main.py

# Terminal 3 — Nginx load balancer (rootless, logs to /tmp)
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"

# Stop Nginx when done
nginx -p /tmp -s stop
```

All client traffic goes to `:8780`; Nginx round-robins requests between the two gateway instances.

### Load balancing test scenario

Send 10 requests through the load balancer and verify they are split evenly:

```bash
# Send 10 requests through the LB
for i in $(seq 10); do
  curl -s -X POST http://127.0.0.1:8780/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}' \
    | jq -r .backend
done
# Expected output: alternating "local" entries served by instance 1 and instance 2

# Verify the split — each instance should show ~5 requests
curl -s http://localhost:9101/metrics | grep 'gateway_requests_total{' | head -5
curl -s http://localhost:9102/metrics | grep 'gateway_requests_total{' | head -5

# Nginx connection stats (active connections, total requests handled)
curl -s http://localhost:8780/nginx_status
```

### Single-gateway mode

To use only one gateway (skip instance 2), comment out the second `server` line in `nginx-gateway-lb.conf`:

```nginx
upstream inference_gateways {
    server 127.0.0.1:8080;
    # server 127.0.0.1:8081;
}
```

---

## Monitoring Stack (Prometheus + Grafana)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) with Compose

The monitoring stack scrapes **both** gateway instances, plus Nginx metrics via `nginx-prometheus-exporter`.

```bash
# 1. Start both gateway instances and Nginx (see above)

# 2. Launch Prometheus + Grafana + Nginx exporter
cd monitoring
docker compose up -d
```

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Nginx exporter | http://localhost:9113/metrics | — |

Prometheus scrapes:
- `gateway` — instance 1 metrics on `:9101`
- `gateway2` — instance 2 metrics on `:9102`
- `nginx` — Nginx exporter on `:9113` (active connections, requests/s, upstream health)
- `vllm_standard` — Modal vLLM standard deployment over HTTPS
- `vllm_optimized` — Modal vLLM optimized deployment over HTTPS

Check all targets are **UP**: http://localhost:9090/targets

Set `VLLM_SERVER_PROFILE=default` or `VLLM_SERVER_PROFILE=optimized` in `.env` to label gateway metrics with the active deployment.

Grafana loads with a pre-provisioned Prometheus datasource and four dashboards: **gateway-proxy**, **overview**, **technique-cost**, **tinyllama-ops**.

---

## Error Responses

| Situation | Status | Body |
|---|---|---|
| Invalid JSON body | 400 | `{"error": "invalid_json"}` |
| Missing/malformed messages | 400 | `{"error": "invalid_messages"}` |
| Invalid stream/max_tokens/temperature/stop | 400 | `{"error": "invalid_stream"}` etc. |
| Missing or wrong API key | 401 | `{"error": "unauthorized"}` |
| Backend did not respond in time | 504 | `{"error": "gateway_timeout"}` |
| Backend returned 5xx | 502 | `{"error": "backend_error"}` |
| Backend unreachable | 502 | `{"error": "backend_unavailable"}` |
