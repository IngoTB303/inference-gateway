# Inference Gateway

An OpenAI-compatible HTTP proxy that forwards chat completion requests to a backend inference server (e.g. llama.cpp) or returns echo responses when no backend is configured.

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
# Edit .env to set PORT and optionally BACKEND_URL

# 4. Start the gateway
uv run python main.py
```

The server listens on `http://localhost:8080` by default.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port the HTTP server listens on |
| `BACKEND_URL` | *(unset)* | Upstream inference server URL. If unset, echo mode is used. |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main inference endpoint |
| `GET` | `/healthz` | Health check |
| `GET` | `/v1/models` | Model list |
| `GET` | `/metrics` | Request/latency/token counters |

---

## Running Tests

### Unit tests

```bash
# Run all tests
uv run pytest tests/test_gateway.py -v

# Run a single test
uv run pytest tests/test_gateway.py::test_echo_returns_correct_shape
```

The test suite starts a real `HTTPServer` on a free port (no mocking of the HTTP layer) and uses [respx](https://lundberg.github.io/respx/) to mock backend `httpx` calls. No running backend is required.

**Coverage:** GET endpoints, echo response shape, request-ID echoing/generation, validation errors (400), SSE streaming, backend proxy (success), and metrics counters.

### Bruno collection

Open `tests/bruno/` as a collection in [Bruno](https://www.usebruno.com/) and select the **local** environment (`base_url = http://localhost:8080`).

The collection has two folders:

| Folder | Requests |
|---|---|
| `GET endpoints` | `/healthz`, `/v1/models`, unknown route |
| `POST endpoints` | echo shape/content/usage, request-ID variants, validation errors (400), SSE streaming, backend success |

---

## curl Examples

### Non-streaming request (echo mode, no BACKEND_URL set)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

Expected response:
```json
{
  "id": "<uuid>",
  "object": "chat.completion",
  "model": "echo",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Echo: hello"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
}
```

### Request-ID echoed back

The client-supplied `X-Request-ID` is returned in both the response header and the `id` field:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: my-trace-id-42' \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
# → "id": "my-trace-id-42" in the response body
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'
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

### With a backend (e.g. llama.cpp)

```bash
BACKEND_URL=http://localhost:8081 uv run python main.py
```

The gateway forwards the full request body to `$BACKEND_URL/v1/chat/completions` and proxies the response back. On timeout (>60 s) it returns `504`; on backend 5xx it returns `502`.

---

## Out of Scope / Future Work

The following items are intentionally excluded from the MVP and are candidates for later enhancement:

- **Worker queue and thread pool** — concurrent request handling via a bounded queue; currently each request is handled synchronously.
- **Scheduler and device selection** — routing requests to GPU/CPU backends based on availability or load.
- **Multiple backends / load balancing** — routing across a pool of inference servers with health probing.
- **KV cache awareness** — prefix caching hints or session affinity to reuse cached KV state on the backend.
- **Auth and rate limiting** — API key / JWT validation, per-key rate limiting with `429 Retry-After`.
- **Request validation (advanced)** — strict schema enforcement (model field, max_tokens range, stop sequences).
- **OpenTelemetry push** — emit spans and metrics to an OTLP collector instead of the in-memory `/metrics` endpoint.
- **Detailed streaming usage** — aggregate `usage` from SSE final chunks for billing/analytics.
