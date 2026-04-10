# Inference Gateway

An OpenAI-compatible HTTP gateway that routes chat completion requests to multiple configurable inference backends (llama.cpp, vLLM, or any OpenAI-compatible server) or falls back to an echo response when no real backend is reachable.

**Team members:** Ingo Villnow

> **Reviewer?** See [SUBMISSION.md](SUBMISSION.md) for the complete step-by-step setup guide, and [submission.ipynb](submission.ipynb) / [submission.pdf](submission.pdf) for experiment analysis, SLIs/SLOs, and hardware justification.

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
| `GPU_HOURLY_COST_USD` | `1.10` | Hourly GPU cost used to estimate `gateway_gpu_cost_usd_total` (A10G default). |
| `VLLM_SERVER_PROFILE` | `default` | Server profile label attached to all Prometheus metrics (e.g. `baseline`, `optimized`, `hardcore`). |
| `BACKEND_URL` | *(unset)* | Legacy single-backend URL. Overridden by `config.yaml` when present; if both are unset the gateway runs in echo mode. |
| `OTEL_TRACES_EXPORTER` | `none` | Set to `otlp` to enable distributed tracing. Any other value disables tracing entirely (zero overhead). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://127.0.0.1:4317` | gRPC endpoint for the OTLP collector (e.g. Jaeger). Only used when `OTEL_TRACES_EXPORTER=otlp`. |
| `OTEL_SERVICE_NAME` | `inference-gateway` | Service name reported to the OTLP collector / Jaeger UI. |

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
| `GET` | `/health` | Health check with upstream probe |
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
curl http://localhost:8080/health
# → {"status": "ok", "upstream": null}
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

**144 tests** covering: GET endpoints, echo shape, request-ID, validation errors (400), auth (401), SSE streaming, backend proxy, response normalization, multi-backend routing, metrics, `latency_ms` in usage, Prometheus counter/histogram/gauge behaviour, `X-Technique` label propagation, Nginx config validation, crew health checks, and experiment script behaviour.

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

## CrewAI Agentic Client

`crew.py` runs a two-agent **Researcher → Writer** pipeline through the gateway. The Researcher collects 3–5 bullet points on a topic; the Writer turns them into a short paragraph (≤120 words).

### Install crew dependencies

```bash
uv sync --group crew
```

### Run the crew

```bash
# Default topic, baseline technique
uv run --group crew python crew.py

# Custom topic
uv run --group crew python crew.py --topic "speculative decoding in vLLM"

# A/B technique label (shows up in Prometheus metrics)
uv run --group crew python crew.py --topic "prefix caching for RAG" --technique chunked_prefill
```

### Crew environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_OPENAI_BASE` | `http://127.0.0.1:8780/v1` | Full gateway URL including `/v1`. Alias: `OPENAI_API_BASE`. |
| `GATEWAY_USE_LOAD_BALANCER` | `true` | Set to `false` to bypass Nginx and hit `:8080` directly. |
| `GATEWAY_HOST` | `127.0.0.1` | Gateway host override (used when `GATEWAY_USE_LOAD_BALANCER=false`). |
| `GATEWAY_PORT` | `8080` | Gateway port override (used when `GATEWAY_USE_LOAD_BALANCER=false`). |
| `GATEWAY_LB_HOST` | `127.0.0.1` | Nginx LB host override. |
| `GATEWAY_LB_PORT` | `8780` | Nginx LB port override. |
| `MODEL_NAME` | `modal-gemma4-optimized` | Model name passed to the gateway (must match a backend name in `config.yaml`). |
| `OPENAI_API_KEY` / `API_KEY` | `dummy` | API key forwarded to the gateway. |
| `CREW_VLLM_WAIT_S` | `0` | Seconds to poll `/v1/models` before starting the crew (0 = skip). |
| `CREW_VLLM_POLL_S` | `8` | Polling interval in seconds while waiting for vLLM to become ready. |
| `CREW_LLM_STREAM` | `true` | Enable streaming for LLM calls. |

### Failure modes

| Situation | Behaviour |
|---|---|
| Gateway unreachable | Exits with clear error message before crew starts |
| Gateway reports misconfigured upstream (`/health`) | Exits with `status=2` and instructions |
| vLLM not ready within `CREW_VLLM_WAIT_S` | Exits with `status=3` and timeout message |
| Empty / bad model response | CrewAI retries internally; logged to stderr |

---

## Chat UI

`chat_ui.py` is a Gradio web app with two tabs:

- **💬 Chat** — streaming chat directly against `/v1/chat/completions`; select the `X-Technique` label per request
- **🤖 CrewAI** — editable topic and technique; runs the Researcher→Writer crew and shows the output

### Start the Chat UI

```bash
# Gateway must already be running (uv run python main.py)
uv run --group crew python chat_ui.py
```

Open **http://localhost:7860** in your browser.

The UI inherits `GATEWAY_OPENAI_BASE`, `MODEL_NAME`, and `API_KEY` from `.env` / the shell environment.

| Variable | Default | Purpose |
|---|---|---|
| `CHAT_UI_PORT` | `7860` | Port Gradio listens on |

---

## Modal vLLM Backends

Three container profiles are available, each a distinct vLLM configuration on an A10G GPU:

| Profile | Modal file | Key flags | Technique label |
|---|---|---|---|
| `standard` | `modal/vllm_gemma4.py` | baseline | `baseline` |
| `optimized` | `modal/vllm_gemma4_optimized.py` | chunked prefill + prefix caching | `optimized` |
| `hardcore` | `modal/vllm_gemma4_hardcore.py` | optimized + fp8 KV cache, 4096-token batches | `hardcore` |

### Deploy a container

```bash
# Deploy one profile
# (modal is in the 'deploy' group; the script uses 'uv run --group deploy modal deploy' internally)
bash scripts/deploy_modal_vllm.sh standard
bash scripts/deploy_modal_vllm.sh optimized
bash scripts/deploy_modal_vllm.sh hardcore
```

After deploy, paste the printed URL into `config.yaml` under the matching backend name.

### Run A/B experiments

Before running experiments, ensure the full stack is up:

```bash
# Gateway (terminal 1)
uv run python main.py

# Nginx LB (terminal 2)
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"

# Monitoring (terminal 3 — optional)
cd monitoring && docker compose up -d
```

Then run the experiments:

```bash
# Run 3 crew passes per profile (backends must already be deployed)
bash scripts/run_experiments.sh

# Deploy Modal containers first, then wait for them to warm up and run
bash scripts/run_experiments.sh --deploy

# Customise profiles, number of runs, or topic
bash scripts/run_experiments.sh --profiles standard,hardcore --runs 5 --topic "prefix caching for RAG"
```

The script:
1. (Optionally) deploys each Modal container via `deploy_modal_vllm.sh`.
2. Polls the gateway until the backend responds (cold start + weight download ≈ 3–8 min).
3. Runs `uv run --group crew python crew.py --technique <label>` N times per profile.
4. Writes per-run results to `data/experiments.csv` (technique, wall-clock, success/error).
5. Prints a comparison table of success rate and wall-clock time.

Technique labels flow through as the `X-Technique` header and appear as Prometheus metric labels for per-technique comparison in Grafana.

### Analyse results in the submission notebook

After running experiments, open `submission.ipynb` to visualise latency, success rate, and GPU cost:

```bash
# Install notebook dependencies once
uv sync --group notebook

# Open in VS Code (select the project venv as kernel)
code submission.ipynb
```

`data/experiments.csv` is committed as sample data so the notebook runs offline. `run_experiments.sh` overwrites it with real results.

---

## Nginx Load Balancer

**Prerequisites:** [nginx](https://nginx.org/) (any standard build with `ngx_http_stub_status_module`)

`nginx-gateway-lb.conf` (project root) distributes traffic round-robin across two gateway instances and exposes an `/nginx_status` endpoint for Prometheus.

### Full stack with two gateway instances

```
Client → :8780 (Nginx LB) → :8080 / :8081 (Gateway instances) → Modal vLLM
```

**Quick start — one script:**

```bash
# Start both gateways + Nginx in the background (Ctrl-C to stop all)
API_KEY=test-key bash scripts/start_stack.sh
```

Logs stream to `/tmp/gw1.log`, `/tmp/gw2.log`. Override ports or skip Nginx:

```bash
# Skip Nginx (single gateway mode)
SKIP_NGINX=1 bash scripts/start_stack.sh

# Custom ports
PORT1=8080 PORT2=8090 API_KEY=test-key bash scripts/start_stack.sh
```

**Manual start (separate terminals):**

```bash
# Terminal 1 — gateway instance 1
PORT=8080 GATEWAY_METRICS_PORT=9101 API_KEY=test-key uv run python main.py

# Terminal 2 — gateway instance 2
PORT=8081 GATEWAY_METRICS_PORT=9102 API_KEY=test-key uv run python main.py

# Terminal 3 — Nginx load balancer (rootless, logs to /tmp)
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"

# Stop Nginx when done
nginx -p /tmp -s stop
```

All client traffic goes to `:8780`; Nginx round-robins requests between the two gateway instances.

> **Auth note:** Nginx is auth-transparent — it passes the `Authorization` header straight through to the gateway unchanged. If `API_KEY` is set on the gateway instances, every request through the LB must include the header. If `API_KEY` is unset (omit it from the command above), no header is required.

### Load balancing test scenario

Send 10 requests through the load balancer and verify they are split evenly:

```bash
# Without auth (API_KEY unset on gateway instances)
for i in $(seq 10); do
  curl -s -X POST http://127.0.0.1:8780/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}' \
    | jq -r .backend
done

# With auth (API_KEY=test-key set on gateway instances)
for i in $(seq 10); do
  curl -s -X POST http://127.0.0.1:8780/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer test-key' \
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

## Distributed Tracing (OpenTelemetry)

Tracing is a **stretch-goal feature** that is off by default. When enabled, each gateway request is wrapped in a span, and `crew.py` propagates W3C trace context so you can see the full `crew → gateway → vLLM` trace in one view.

**Prerequisites:** install the optional OTel packages:

```bash
uv sync --group otel
```

**1. Start Jaeger (all-in-one) via the monitoring stack:**

```bash
cd monitoring && docker compose up -d jaeger
# Jaeger UI: http://localhost:16686
# OTLP gRPC receiver: localhost:4317
```

**2. Enable tracing in `.env`:**

```bash
OTEL_TRACES_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
OTEL_SERVICE_NAME=inference-gateway   # name shown in Jaeger UI (optional, this is the default)
```

**3. Restart the gateway, then run a request:**

```bash
uv run python main.py
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
```

Open http://localhost:16686, select service `inference-gateway`, and you should see a `chat.completions` span with attributes `technique`, `server_profile`, `backend`, and `http.status_code`.

**4. Run the crew with tracing:**

```bash
OTEL_TRACES_EXPORTER=otlp uv run --group crew --group otel python crew.py --technique chunked_prefill
```

This creates a `crew.run` root span (with `llm.technique=chunked_prefill`) that is the parent of all gateway spans generated during that run.

**Default (`OTEL_TRACES_EXPORTER=none`):** no OTel packages are imported, no connections are attempted — zero performance cost.

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

---

## Reflection

**Biggest surprise:** Gemma 4's heterogeneous attention head dimensions (256 for local layers, 512 for global layers) forced vLLM to fall back from FlashAttention to the Triton attention backend. This was undocumented for vLLM 0.19.0 and only surfaced at serve-time via a log line: `Using Triton attention backend`. The practical impact was lower throughput (~20–40 tok/s on A10G) than comparably sized models with uniform head dimensions. The workaround was `--async-scheduling` with tuned batch budgets (`--max-num-batched-tokens`) to pipeline kernel calls and prevent the Triton backend from becoming the bottleneck.

**What broke first (and how I diagnosed it):** The first experiment run timed out. I saw `gateway_errors_total{status_code="504"}` spike to 1 in Grafana immediately. Checking `gateway_request_duration_seconds`, latency was flatlined at exactly 120 seconds — the backend timeout in `config.yaml`. This ruled out a vLLM OOM (which would have returned 5xx faster) and pointed to Modal cold-start: image pull, CUDA initialisation, and weight download exceeding the timeout. The fix was raising `scaledown_window` to 15 minutes and increasing the backend timeout to 300 seconds for initial deploys.

**Next steps for production:** Three changes would have the most impact: (1) **FP8 KV cache on H100** — Hopper (compute capability ≥ 9.0) unlocks `--kv-cache-dtype fp8`, halving KV memory and allowing `max_model_len=32768` or far higher concurrency on the same VRAM budget. (2) **Modal autoscaling** — replacing `min_containers=max_containers=1` with a proper autoscaler that scales to zero when idle would make the cost SLO self-enforcing. (3) **Structured crew output** — adding a Pydantic output schema to the CrewAI Writer task would enforce the 120-word contract at the framework level and eliminate silent application-layer SLO violations.
