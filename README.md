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
| `PORT` | `8080` | Port the HTTP server listens on |
| `API_KEY` | *(unset)* | If set, all POST requests must include `Authorization: Bearer <key>` or `Authorization: Api-Key <key>`. Leave empty to disable auth. |

### Backend configuration (`config.yaml`)

Backends are configured in `config.yaml`. The gateway routes requests by matching the `model` field in the request body to a backend `name`. Unrecognised or absent `model` values fall back to `default_backend`.

```yaml
backends:
  - name: local          # echo mode — no real backend needed
    type: echo

  - name: local-llama    # local llama.cpp server
    type: http
    url: http://localhost:8081
    timeout: 60

  - name: remote-modal-llama   # llama.cpp hosted on Modal
    type: http
    url: https://ingo-villnow--relay-llama-server-web.modal.run/
    timeout: 60

  - name: remote-modal-vllm   # vLLM hosted on Modal
    type: http
    url: https://ingo-villnow--relay-vllm-server-web.modal.run/
    timeout: 120

default_backend: local
```

**Adding a new backend:** Add one entry to `config.yaml` — no code changes needed.

**Echo-only mode (no config.yaml):** The gateway falls back to the `BACKEND_URL` environment variable; if that is also unset, it runs in echo mode.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main inference endpoint |
| `GET` | `/v1/backends` | List configured backends and default |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Request/latency/token counters |

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
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

The `backend` field tells the client which gateway backend handled the request.

---

## curl Examples

### Route to local echo backend

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "hello"}]}'
```

Response includes `"backend": "local"` and content `"Echo: hello"`.

### Route to remote llama.cpp on Modal

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "remote-modal-llama", "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 50}'
```

### Route to remote vLLM on Modal

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "remote-modal-vllm", "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 50}'
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
curl http://localhost:8080/metrics
# → {"request_count": N, "error_count": N, "total_latency_ms": N, ...}
```

---

## Running Tests

### Unit tests

```bash
uv run pytest
uv run pytest tests/test_gateway.py::test_routing_by_model_echo   # single test
```

The suite starts real `HTTPServer` instances on free ports and uses [respx](https://lundberg.github.io/respx/) to mock backend `httpx` calls. No running backend is required.

**47 tests** covering: GET endpoints, echo shape, request-ID, validation errors (400), auth (401), SSE streaming, backend proxy, response normalization, multi-backend routing, metrics.

### Bruno collection

Open `tests/bruno/` in [Bruno](https://www.usebruno.com/) and select the **local** environment (`base_url = http://localhost:8080`).

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
