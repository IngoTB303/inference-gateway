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


settings = Settings()


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
# Helper: build an OpenAI-shaped response dict
# ---------------------------------------------------------------------------


def build_response(
    request_id: str,
    content: str,
    model: str = "echo",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
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
        elif self.path == "/v1/models":
            if settings.backend_url:
                self._proxy_models()
            else:
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "echo",
                                "object": "model",
                                "owned_by": "inference-gateway",
                            }
                        ],
                    },
                )
        elif self.path == "/metrics":
            self._send_json(
                200,
                {
                    "request_count": metrics.request_count,
                    "error_count": metrics.error_count,
                    "total_latency_ms": round(metrics.total_latency_ms, 2),
                    "prompt_tokens_total": metrics.prompt_tokens_total,
                    "completion_tokens_total": metrics.completion_tokens_total,
                },
            )
        else:
            self._send_json(404, {"error": "not_found"})

    # ------------------------------------------------------------------
    # POST routes
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json(404, {"error": "not_found"})

    # ------------------------------------------------------------------
    # /v1/chat/completions implementation
    # ------------------------------------------------------------------

    def _handle_chat_completions(self) -> None:
        start = time.monotonic()
        request_id = self._get_request_id()
        log = logging.LoggerAdapter(logger, {"request_id": request_id})

        # Parse body
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Bad request body: %s", exc)
            self._record_metrics(400, (time.monotonic() - start) * 1000, 0, 0)
            self._send_json(400, {"error": "invalid_json"})
            return

        # Basic validation
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._record_metrics(400, (time.monotonic() - start) * 1000, 0, 0)
            self._send_json(400, {"error": "invalid_messages"})
            return

        stream = body.get("stream", False)

        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        # Dispatch
        if settings.backend_url:
            self._handle_backend(request_id, body, stream, start, log)
        else:
            self._handle_echo(request_id, prompt, stream, start, log)

    def _handle_echo(
        self,
        request_id: str,
        prompt: str,
        stream: bool,
        start: float,
        log: logging.LoggerAdapter,
    ) -> None:
        content = f"Echo: {prompt}"
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())

        if stream:
            self._send_sse_echo(request_id, content)
        else:
            resp = build_response(
                request_id,
                content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self._send_json(200, resp, request_id=request_id)

        latency_ms = (time.monotonic() - start) * 1000
        self._record_metrics(200, latency_ms, prompt_tokens, completion_tokens)
        log.info(
            "POST /v1/chat/completions status=200 latency_ms=%.1f mode=echo", latency_ms
        )

    def _handle_backend(
        self,
        request_id: str,
        body: dict[str, Any],
        stream: bool,
        start: float,
        log: logging.LoggerAdapter,
    ) -> None:
        assert settings.backend_url is not None
        url = settings.backend_url.rstrip("/") + "/v1/chat/completions"

        try:
            if stream:
                self._proxy_stream(request_id, url, body, start, log)
            else:
                self._proxy_non_stream(request_id, url, body, start, log)
        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(504, latency_ms, 0, 0)
            log.error("Backend timeout after %.1f ms", latency_ms)
            self._send_json(504, {"error": "gateway_timeout"}, request_id=request_id)
        except httpx.RequestError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(502, latency_ms, 0, 0)
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
    ) -> None:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=body)

        if resp.status_code >= 500:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(502, latency_ms, 0, 0)
            log.error("Backend returned %d", resp.status_code)
            self._send_json(502, {"error": "backend_error"}, request_id=request_id)
            return

        try:
            data = resp.json()
        except Exception:
            latency_ms = (time.monotonic() - start) * 1000
            self._record_metrics(502, latency_ms, 0, 0)
            self._send_json(
                502, {"error": "backend_invalid_response"}, request_id=request_id
            )
            return

        # Ensure the response carries our request-id
        data["id"] = request_id
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        latency_ms = (time.monotonic() - start) * 1000
        self._record_metrics(
            resp.status_code, latency_ms, prompt_tokens, completion_tokens
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
    ) -> None:
        """Forward SSE chunks from backend to client without buffering."""
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", url, json=body) as resp:
                if resp.status_code >= 500:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._record_metrics(502, latency_ms, 0, 0)
                    self._send_json(
                        502, {"error": "backend_error"}, request_id=request_id
                    )
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Request-ID", request_id)
                self.end_headers()

                for line in resp.iter_lines():
                    self.wfile.write((line + "\n").encode())
                self.wfile.write(b"\n")
                self.wfile.flush()

        latency_ms = (time.monotonic() - start) * 1000
        self._record_metrics(200, latency_ms, 0, 0)
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

    def _proxy_models(self) -> None:
        assert settings.backend_url is not None
        url = settings.backend_url.rstrip("/") + "/v1/models"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
            self._send_json(resp.status_code, resp.json())
        except httpx.TimeoutException:
            self._send_json(504, {"error": "gateway_timeout"})
        except httpx.RequestError as exc:
            logger.error("Backend /v1/models error: %s", exc)
            self._send_json(502, {"error": "backend_unavailable"})

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
    ) -> None:
        metrics.request_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.prompt_tokens_total += prompt_tokens
        metrics.completion_tokens_total += completion_tokens
        if status >= 400:
            metrics.error_count += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
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
