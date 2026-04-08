"""Inference gateway — OpenAI-compatible HTTP proxy with echo fallback."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8080")))
    backend_url: str | None = field(
        default_factory=lambda: os.environ.get("BACKEND_URL") or None
    )
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("API_KEY") or None
    )
    metrics_port: int = field(
        default_factory=lambda: int(os.environ.get("GATEWAY_METRICS_PORT", "9101"))
    )
    gpu_hourly_cost_usd: float = field(
        default_factory=lambda: float(os.environ.get("GPU_HOURLY_COST_USD", "1.10"))
    )
    vllm_server_profile: str = field(
        default_factory=lambda: os.environ.get("VLLM_SERVER_PROFILE", "default")
    )


settings = Settings()


# ---------------------------------------------------------------------------
# Multi-backend config (loaded once at startup; None if no config.yaml)
# ---------------------------------------------------------------------------

from gateway.config import GatewayConfig, load_config  # noqa: E402
from gateway.backends.http_backend import HttpBackend  # noqa: E402
from gateway.prom_metrics import (  # noqa: E402
    ACTIVE_REQUESTS,
    ERRORS_TOTAL,
    GPU_COST_USD_TOTAL,
    INTER_CHUNK_SECONDS,
    REQUEST_DURATION_SECONDS,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    TTFT_SECONDS,
)

# Load config.yaml if present; otherwise gateway_config stays None and the
# handler falls back to the legacy settings.backend_url path.
gateway_config: GatewayConfig | None = (
    load_config() if Path("config.yaml").exists() else None
)


# ---------------------------------------------------------------------------
# In-memory metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0


metrics = Metrics()


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


def _normalize_response(
    data: dict[str, Any], request_id: str, latency_ms: float = 0.0
) -> dict[str, Any]:
    """Return a clean, spec-compliant response dict regardless of backend shape.

    Only the documented fields are included; any extra keys sent by the backend
    (e.g. created, system_fingerprint, timings, logprobs) are silently dropped
    so that all backends produce an identical response contract.
    """
    clean_choices = []
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        clean_choices.append(
            {
                "index": choice.get("index", 0),
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        )

    usage_raw = data.get("usage", {})
    prompt_tokens = usage_raw.get("prompt_tokens", 0)
    completion_tokens = usage_raw.get("completion_tokens", 0)

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": data.get("model", "unknown"),
        "choices": clean_choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage_raw.get(
                "total_tokens", prompt_tokens + completion_tokens
            ),
            "latency_ms": round(latency_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _validate_body(body: dict[str, Any]) -> str | None:
    """Validate request body fields. Returns an error key or None if valid."""
    # model: string if present
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        return "invalid_model"

    # messages: already checked non-empty above; validate each element
    messages = body.get("messages", [])
    for msg in messages:
        if not isinstance(msg, dict):
            return "invalid_messages"
        if "role" not in msg or "content" not in msg:
            return "invalid_messages"
        if msg["role"] not in ("system", "user", "assistant"):
            return "invalid_messages"

    # stream: boolean if present
    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        return "invalid_stream"

    # max_tokens: positive integer in sane range if present
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            return "invalid_max_tokens"
        if max_tokens < 1 or max_tokens > 100_000:
            return "invalid_max_tokens"

    # temperature: float/int in [0, 2] if present
    temperature = body.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            return "invalid_temperature"
        if temperature < 0 or temperature > 2:
            return "invalid_temperature"

    # stop: string or list of strings if present
    stop = body.get("stop")
    if stop is not None:
        if isinstance(stop, str):
            pass
        elif isinstance(stop, list):
            if not all(isinstance(s, str) for s in stop):
                return "invalid_stop"
        else:
            return "invalid_stop"

    return None


# ---------------------------------------------------------------------------
# Helper: build an OpenAI-shaped response dict
# ---------------------------------------------------------------------------


def build_response(
    request_id: str,
    content: str,
    model: str = "echo",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Gateway HTTP handler
# ---------------------------------------------------------------------------


class GatewayHandler(BaseHTTPRequestHandler):
    """Handle all incoming HTTP requests for the inference gateway."""

    # Suppress default access logging — we do structured logging instead.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/metrics":
            count = metrics.request_count
            avg_latency_ms = metrics.total_latency_ms / count if count > 0 else 0.0
            self._send_json(
                200,
                {
                    "request_count": count,
                    "error_count": metrics.error_count,
                    "avg_latency_ms": round(avg_latency_ms, 2),
                    "prompt_tokens_total": metrics.prompt_tokens_total,
                    "completion_tokens_total": metrics.completion_tokens_total,
                },
            )
        elif self.path == "/v1/backends":
            if gateway_config is not None:
                self._send_json(
                    200,
                    {
                        "backends": [
                            {"name": b.name, "type": type(b).__name__}
                            for b in gateway_config.all_backends
                        ],
                        "default": gateway_config.default_backend.name,
                    },
                )
            else:
                self._send_json(404, {"error": "not_found"})
        else:
            self._send_json(404, {"error": "not_found"})

    # ------------------------------------------------------------------
    # POST routes
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        if not self._check_auth():
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json(404, {"error": "not_found"})

    # ------------------------------------------------------------------
    # /v1/chat/completions implementation
    # ------------------------------------------------------------------

    def _handle_chat_completions(self) -> None:
        ACTIVE_REQUESTS.inc()
        try:
            self._do_handle_chat_completions()
        finally:
            ACTIVE_REQUESTS.dec()

    def _do_handle_chat_completions(self) -> None:
        start = time.monotonic()
        request_id = self._get_request_id()
        technique = self.headers.get("X-Technique", "baseline")
        server_profile = settings.vllm_server_profile
        log = logging.LoggerAdapter(logger, {"request_id": request_id})

        # Parse body
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Bad request body: %s", exc)
            self._record_metrics(
                400,
                (time.monotonic() - start) * 1000,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            self._send_json(400, {"error": "invalid_json"})
            return

        # Basic structure check
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._record_metrics(
                400,
                (time.monotonic() - start) * 1000,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            self._send_json(400, {"error": "invalid_messages"})
            return

        # Full validation
        error = _validate_body(body)
        if error:
            self._record_metrics(
                400,
                (time.monotonic() - start) * 1000,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            self._send_json(400, {"error": error})
            return

        stream = body.get("stream", False)

        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        model = body.get("model")

        # Build a clean forward body with only known fields
        forward_body: dict[str, Any] = {
            "messages": messages,
            "model": model or "default",
            "stream": stream,
        }
        for key in ("max_tokens", "temperature", "stop"):
            if key in body:
                forward_body[key] = body[key]

        # Dispatch — config-based routing takes precedence over legacy settings
        if gateway_config is not None:
            self._handle_with_config(
                request_id,
                forward_body,
                model,
                prompt,
                stream,
                start,
                log,
                technique=technique,
                server_profile=server_profile,
            )
        elif settings.backend_url:
            self._handle_backend(
                request_id,
                forward_body,
                stream,
                start,
                log,
                technique=technique,
                server_profile=server_profile,
            )
        else:
            self._handle_echo(
                request_id,
                prompt,
                stream,
                start,
                log,
                technique=technique,
                server_profile=server_profile,
            )

    def _handle_echo(
        self,
        request_id: str,
        prompt: str,
        stream: bool,
        start: float,
        log: logging.LoggerAdapter,
        *,
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        content = f"Echo: {prompt}"
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())

        latency_ms = (time.monotonic() - start) * 1000

        if stream:
            self._send_sse_echo(request_id, content)
        else:
            resp = build_response(
                request_id,
                content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
            self._send_json(200, resp, request_id=request_id)

        self._record_metrics(
            200,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            technique=technique,
            server_profile=server_profile,
        )
        log.info(
            "POST /v1/chat/completions status=200 latency_ms=%.1f mode=echo", latency_ms
        )

    def _handle_with_config(
        self,
        request_id: str,
        forward_body: dict[str, Any],
        model: str | None,
        prompt: str,
        stream: bool,
        start: float,
        log: logging.LoggerAdapter,
        *,
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        """Route the request through the multi-backend config."""
        assert gateway_config is not None
        backend = gateway_config.get_backend_for_model(model)

        # The "model" field in the request is the gateway routing key, not a
        # backend model name.  Strip it before forwarding so that backends
        # (e.g. vLLM) do not reject an unknown name.
        # If the backend config specifies a model override, inject it instead.
        backend_body = {k: v for k, v in forward_body.items() if k != "model"}
        if isinstance(backend, HttpBackend) and backend.model is not None:
            backend_body["model"] = backend.model

        if stream:
            # Streaming: delegate to echo or proxy path based on backend type
            if isinstance(backend, HttpBackend):
                try:
                    self._proxy_stream(
                        request_id,
                        backend.completions_url,
                        backend_body,
                        start,
                        log,
                        technique=technique,
                        server_profile=server_profile,
                    )
                except httpx.TimeoutException:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._record_metrics(
                        504,
                        latency_ms,
                        0,
                        0,
                        technique=technique,
                        server_profile=server_profile,
                    )
                    log.error("Backend timeout after %.1f ms", latency_ms)
                    self._send_json(
                        504, {"error": "gateway_timeout"}, request_id=request_id
                    )
                except httpx.RequestError as exc:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._record_metrics(
                        502,
                        latency_ms,
                        0,
                        0,
                        technique=technique,
                        server_profile=server_profile,
                    )
                    log.error("Backend connection error: %s", exc)
                    self._send_json(
                        502, {"error": "backend_unavailable"}, request_id=request_id
                    )
            else:
                self._handle_echo(
                    request_id,
                    prompt,
                    stream,
                    start,
                    log,
                    technique=technique,
                    server_profile=server_profile,
                )
            return

        # Non-streaming: call backend.generate()
        try:
            data = backend.generate(backend_body, request_id)
        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                504,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend %s timeout after %.1f ms", backend.name, latency_ms)
            self._send_json(504, {"error": "gateway_timeout"}, request_id=request_id)
            return
        except httpx.RequestError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend %s connection error: %s", backend.name, exc)
            self._send_json(
                502, {"error": "backend_unavailable"}, request_id=request_id
            )
            return
        except RuntimeError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend %s error: %s", backend.name, exc)
            self._send_json(502, {"error": "backend_error"}, request_id=request_id)
            return

        latency_ms = (time.monotonic() - start) * 1000
        data = _normalize_response(data, request_id, latency_ms=latency_ms)
        data["backend"] = backend.name

        usage = data["usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self._record_metrics(
            200,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            technique=technique,
            server_profile=server_profile,
        )
        log.info(
            "POST /v1/chat/completions status=200 latency_ms=%.1f mode=config backend=%s",
            latency_ms,
            backend.name,
        )
        self._send_json(200, data, request_id=request_id)

    def _handle_backend(
        self,
        request_id: str,
        body: dict[str, Any],
        stream: bool,
        start: float,
        log: logging.LoggerAdapter,
        *,
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        assert settings.backend_url is not None
        url = settings.backend_url.rstrip("/") + "/v1/chat/completions"

        try:
            if stream:
                self._proxy_stream(
                    request_id,
                    url,
                    body,
                    start,
                    log,
                    technique=technique,
                    server_profile=server_profile,
                )
            else:
                self._proxy_non_stream(
                    request_id,
                    url,
                    body,
                    start,
                    log,
                    technique=technique,
                    server_profile=server_profile,
                )
        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                504,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend timeout after %.1f ms", latency_ms)
            self._send_json(504, {"error": "gateway_timeout"}, request_id=request_id)
        except httpx.RequestError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend connection error: %s", exc)
            self._send_json(
                502, {"error": "backend_unavailable"}, request_id=request_id
            )

    def _proxy_non_stream(
        self,
        request_id: str,
        url: str,
        body: dict[str, Any],
        start: float,
        log: logging.LoggerAdapter,
        *,
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=body, headers={"X-Technique": technique})

        if resp.status_code >= 500:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend returned %d", resp.status_code)
            self._send_json(502, {"error": "backend_error"}, request_id=request_id)
            return

        try:
            data = resp.json()
        except Exception:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            self._send_json(
                502, {"error": "backend_invalid_response"}, request_id=request_id
            )
            return

        # Normalize response shape and ensure it carries our request-id
        latency_ms = (time.monotonic() - start) * 1000
        data = _normalize_response(data, request_id, latency_ms=latency_ms)
        usage = data["usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self._record_metrics(
            resp.status_code,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            technique=technique,
            server_profile=server_profile,
        )
        log.info(
            "POST /v1/chat/completions status=%d latency_ms=%.1f mode=backend",
            resp.status_code,
            latency_ms,
        )
        self._send_json(resp.status_code, data, request_id=request_id)

    def _proxy_stream(
        self,
        request_id: str,
        url: str,
        body: dict[str, Any],
        start: float,
        log: logging.LoggerAdapter,
        *,
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        """Forward SSE chunks from backend to client without buffering."""
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST", url, json=body, headers={"X-Technique": technique}
            ) as resp:
                if resp.status_code >= 500:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._record_metrics(
                        502,
                        latency_ms,
                        0,
                        0,
                        technique=technique,
                        server_profile=server_profile,
                    )
                    self._send_json(
                        502, {"error": "backend_error"}, request_id=request_id
                    )
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Request-ID", request_id)
                self.end_headers()

                prompt_tokens = 0
                completion_tokens = 0
                first_chunk = True
                last_chunk_time = time.monotonic()
                for line in resp.iter_lines():
                    self.wfile.write((line + "\n").encode())
                    if line.startswith("data: ") and line != "data: [DONE]":
                        now = time.monotonic()
                        if first_chunk:
                            TTFT_SECONDS.observe(now - start)
                            first_chunk = False
                        else:
                            INTER_CHUNK_SECONDS.observe(now - last_chunk_time)
                        last_chunk_time = now
                        try:
                            chunk = json.loads(line[6:])
                            if usage := chunk.get("usage"):
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                self.wfile.write(b"\n")
                self.wfile.flush()

        latency_ms = (time.monotonic() - start) * 1000
        self._record_metrics(
            200,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            technique=technique,
            server_profile=server_profile,
        )
        log.info(
            "POST /v1/chat/completions status=200 latency_ms=%.1f mode=backend-stream",
            latency_ms,
        )

    def _send_sse_echo(self, request_id: str, content: str) -> None:
        """Return a single SSE chunk followed by [DONE] for echo mode."""
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self) -> bool:
        """Return True if auth passes. If auth fails, send 401 and return False."""
        if settings.api_key is None:
            return True

        auth_header = self.headers.get("Authorization", "")
        token: str | None = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        elif auth_header.startswith("Api-Key "):
            token = auth_header[8:]

        if token != settings.api_key:
            self._send_json(401, {"error": "unauthorized"})
            return False
        return True

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _get_request_id(self) -> str:
        return (
            self.headers.get("X-Request-ID")
            or self.headers.get("Request-Id")
            or str(uuid.uuid4())
        )

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _send_json(
        self,
        status: int,
        data: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _record_metrics(
        self,
        status: int,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "unknown",
        technique: str = "baseline",
        server_profile: str = "default",
    ) -> None:
        metrics.request_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.prompt_tokens_total += prompt_tokens
        metrics.completion_tokens_total += completion_tokens
        if status >= 400:
            metrics.error_count += 1

        latency_s = latency_ms / 1000.0
        REQUESTS_TOTAL.labels(
            status_code=str(status),
            model=model,
            technique=technique,
            server_profile=server_profile,
        ).inc()
        REQUEST_DURATION_SECONDS.labels(
            technique=technique,
            server_profile=server_profile,
        ).observe(latency_s)
        TOKENS_TOTAL.labels(type="prompt").inc(prompt_tokens)
        TOKENS_TOTAL.labels(type="completion").inc(completion_tokens)
        if status >= 400:
            ERRORS_TOTAL.labels(
                status_code=str(status),
                technique=technique,
                server_profile=server_profile,
            ).inc()
        cost = latency_s * settings.gpu_hourly_cost_usd / 3600.0
        GPU_COST_USD_TOTAL.labels(
            technique=technique,
            server_profile=server_profile,
        ).inc(cost)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import prometheus_client

    prometheus_client.start_http_server(settings.metrics_port)
    logger.info(
        "Prometheus metrics available on port %d",
        settings.metrics_port,
        extra={"request_id": "-"},
    )

    server = HTTPServer(("0.0.0.0", settings.port), GatewayHandler)
    mode = f"backend={settings.backend_url}" if settings.backend_url else "echo mode"
    logger.info(
        "Inference gateway listening on port %d (%s)",
        settings.port,
        mode,
        extra={"request_id": "-"},
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.", extra={"request_id": "-"})


if __name__ == "__main__":
    main()
