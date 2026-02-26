# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python AI inference gateway — an HTTP proxy that accepts OpenAI-style chat completion requests and forwards them to a backend inference server (e.g., llama.cpp) or returns an echo response when no backend is configured.

**Status**: In progress. See `implementation_plan.md` for the full specification.

## Commands

The project uses **uv** for dependency and environment management. The linter is **ruff**.

```bash
# Run the gateway
uv run python main.py

# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest
uv run pytest tests/test_gateway.py::test_name   # single test

# Copy and edit env
cp .env.example .env
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | Port the HTTP server listens on |
| `BACKEND_URL` | _(unset)_ | Upstream inference server URL; if unset, use echo mode |
| `API_KEY` / `AUTHORIZATION` | _(unset)_ | Auth token for gateway (advanced feature) |

## Architecture

### Request lifecycle

```
Client POST /v1/chat/completions
  → Auth/policy check (advanced)
  → Request validation (messages[], model, stream, max_tokens)
  → Extract last user message as prompt; read/generate X-Request-ID
  → If BACKEND_URL set: forward full request to backend
    → Handle timeout (504), 5xx (502), connection error (502)
  → Else: echo mode — return "Echo: <prompt>"
  → Build OpenAI-shaped response (id, choices, usage)
  → Log request-id, path, status, latency
  → Emit metrics (OpenTelemetry)
```

### Response shape (non-streaming)

```json
{
  "id": "<request-id>",
  "choices": [{ "message": { "role": "assistant", "content": "..." }, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N }
}
```

Streaming (`stream: true`) uses Server-Sent Events: `data: {...}\n\n` chunks ending with `data: [DONE]`. Chunks must be forwarded as they arrive — do not buffer the full backend response.

### Key design points

- **Settings**: A `Settings` class reads from `.env`. No hard-coded config.
- **Echo mode**: When `BACKEND_URL` is not set, the gateway returns a standalone echo response so it runs without any backend.
- **Request-ID**: Read from `X-Request-ID` or `Request-Id` header; generate a UUID v4 if absent. Echo the ID in the response header and the top-level `id` field.
- **Metrics**: Expose request count, latency histogram, token usage — either via `GET /metrics` or pushed to an OpenTelemetry collector.
- **Out of scope (MVP)**: No queue, no scheduler, no load balancing, no multi-backend support.

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Main inference endpoint |
| `GET` | `/healthz` | Health check |
| `GET` | `/v1/models` | Model list |
| `GET` | `/metrics` | Prometheus-style or OTLP metrics (advanced) |

## Advanced Features (post-MVP)

Described in `implementation_plan.md` sections 1–6:
1. Request validation with structured 400 errors
2. API key / JWT auth + rate limiting (429 with `Retry-After`)
3. Error normalization: 504 on timeout, 502 on backend error
4. True streaming — proxy SSE chunks without buffering
5. Post-response logging and usage aggregation
6. Full request contract: `model`, `messages[]`, `stream`, `max_tokens`, `temperature`, `stop`
