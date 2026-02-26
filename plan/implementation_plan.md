# TAKE-HOME exercise: Simple AI inference gateway

## Goal

Create a simple inference gateway which handles multiple concurrent user requests, checks the requests regarding requirements and schedule valid requests to various backend inference servers like llama.cpp. If no backend inference server is available, use a mock server which simply echoes the request data within the needed response structure.

## Implementation tasks

- Use a worker which handles the creation of request and response
- implement metrics (e. g. OpenTelemetry), see chapter metrics
- update the README.md file, add the following project members: 1) Dev Jadhav and 2) Ingo Villow
- to setup the application, use a settings class and an .env file
- dependencies: json, http handler, queue, threading
- use basic python without the need of pip if possible, but use external libraries, if this make the app more stable and faster with less code
- for the MVP: No queue, no scheduler, no device probing—just: request in → call backend (or echo) → response/stream out

## API

- the gateway should accept POST requests on /v1/chat/completions with an OpenAI-style JSON body (messages, optional stream)
- Forwards the request to a backend (another HTTP server that speaks the same API), or returns a simple echo if no backend is configured
- Returns a non-streaming response: one JSON object with choices[0].message.content, usage, and an id (request-id) -> if no id is inside the request, please create one
- Optional) Supports streaming: when stream: true, return Server-Sent Events (SSE) until data: [DONE].

## MVP must-haves

- HTTP server that listens on a configurable port (e.g. env PORT or 8080)
- Single route: POST /v1/chat/completions
- Request body: JSON with messages (list of {role, content}). Extract the last user content as the prompt
- Backend: Configurable backend URL (e.g. env BACKEND_URL). If set, POST the same shape to the backend and return its response (or a normalized form). If not set, return a simple echo response (e.g. "Echo: <prompt>")
- Request-id: Read X-Request-ID (or Request-Id) from the request; if missing, generate a UUID. Include it in the response (e.g. top-level id and/or response header X-Request-ID)
- Response shape (non-streaming): JSON with at least:
	- id (request-id)
	- choices: [{ "message": { "role": "assistant", "content": "<reply>" }, "finish_reason": "stop" }]
	- usage: { "prompt_tokens", "completion_tokens", "total_tokens" } 
- Optional (stretch)
	- Streaming: If stream: true in the request body and the backend supports streaming, proxy SSE from the backend to the client; otherwise return one SSE chunk with the full reply then data: [DONE].
- implement also GET endpoints: 
	- /healthz or GET /v1/models for health 
	- model list

## Out of scope (do not implement for now

Do not implement the following items, but add them into the README.md for a later enhancement.
- Queue or worker threads
- Scheduler or device selection
- Multiple backends/load balancing
- KV cache, sharding, or detailed metrics (more than mentioned above)

## Deliverables
- Code: A runnable minimal gateway, written in Python. Include a short README: how to run, env vars, how to point at a backend or run without one.
- Test: At least one way to verify it (e.g. curl commands or a small script) that shows:
	- Non-streaming POST /v1/chat/completions returns JSON with id and choices[0].message.content.
	- Request-id from the client is echoed in the response.
- Submission: one repository. List group members in the README.

## Success criteria
- Running the gateway and sending POST /v1/chat/completions with valid messages (role=user) returns 200 and a JSON body with the expected shape and request-id.
- With BACKEND_URL set to a running backend (e.g. llama.cpp or the notebook’s mock), the gateway returns the backend’s reply (or a normalized version).
- Without a backend, the gateway returns an echo (or a clear placeholder) so it runs standalone

## Reference

- For the take-home, you only need the gateway path: one route, one backend (or echo), request-id, and optional streaming.
- Request/response shapes: OpenAI Chat Completions API (you only need a minimal subset)

# Advanced TAKE-HOME exercise
The inference gateway should be extended with some features (later implementation).
After the basic take-home, extend the gateway with the following. Each item is part of the gateway lifecycle.

## 1. Request validation (separate from parsing)
- Do: Validate the request body after decoding JSON and before calling the backend. Checks: 
	- model present (or allow default), 
	- messages is a list and each element has role and content, 
	- stream is a boolean 
	- if present, max_tokens is an integer in a sane range if present. 
- If any check fails, return 400 with a structured body, e.g. {"error": "invalid_messages"} or {"error": "invalid_stream"}.
- Why: Schema enforcement belongs at the gateway; the backend should never see malformed requests.

## 2. Auth and policy hooks

- Do: Before parsing the body (or right after), add an API key or JWT check: 
	- read Authorization (e.g. Bearer <token> or Api-Key <key>), 
	- validate against a config or secret store. 
	- If invalid or missing when required → 401 and {"error": "unauthorized"}; do not call the backend. 
	- Optionally add a rate limit stub: per key or per IP, return 429 and {"error": "rate_limit_exceeded"} with a Retry-After header when over limit.
- Why: Backend never sees unauthorized or over-limit traffic; policy lives at the gateway.

## 3. Error normalization and resilience

- Do: When calling the backend, handle failures and map them to gateway responses instead of hiding them:
	- Backend timeout (e.g. no response within 60s) → 504 and {"error": "gateway_timeout"}.
	- Backend returns 5xx or invalid response → 502 and {"error": "backend_error"} (or similar).
	- Connection error (refused, DNS, etc.) → 502 and {"error": "backend_unavailable"}. 
- Do not return 200 with an echo when the backend actually failed. 
- Optionally: retry once or try the next backend if you have multiple.
- Why: Clients get honest status codes and a stable error format; the gateway normalizes backend failures.

## 4. Streaming without buffering the full response

- Do: When stream: true, open the connection to the backend and forward chunks to the client as they arrive. Read the backend response in a loop (e.g. by line for SSE), write each data: ... line to the client immediately, then pass through data: [DONE]. Do not read the entire backend response with resp.read() and then replay it.
- Why: True streaming keeps latency low and avoids buffering the full response at the gateway.

## 5. Logging and usage after response (and stream) completes
- Do: After the response is sent (or after the stream completes with data: [DONE]):
	- Capture latency (e.g. time from request start to response/stream end) and log it (e.g. one structured log line per request with request-id, path, status, latency).
	- Record usage: If the backend returns usage (prompt_tokens, completion_tokens), log or store it (e.g. for billing or analytics); for streaming, aggregate from the final chunk or from the stream.
	- Log request-id in every log line for that request so you can trace the full lifecycle.
	- Emit metrics (e.g. request count, latency histogram, tokens) to an in-memory store and expose them (e.g. GET /metrics), or push to a metrics system (e. g. OpenTelemetry). Do this for streaming requests too when the stream completes.
- Why: Logging and metrics occur after the response/stream completes; this closes the lifecycle and enables observability.

## 6. Full request contract
- Do: Accept and document the full request shape: model, messages[], stream, max_tokens (and optionally temperature, stop). Validate as in (1). Forward the fields you support to the backend (or normalize them) instead of ignoring them.
- Why: Clients integrate against a stable, complete contract; the gateway may normalize before sending to the backend.

## Summary: Deliverables for Advanced
- Extend the gateway with as many of the above as you can.
- Document what you implemented and how (README or short doc).
- Tests or curl examples for at least: validation (400), auth (401), rate limit (429), backend timeout (504), and one log/metrics check.
- Submission can be the same repo with an “advanced” feature branch 