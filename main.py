"""Inference gateway — OpenAI-compatible HTTP proxy with echo fallback."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import httpx
import prometheus_client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import os
from pathlib import Path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenTelemetry (optional — zero overhead when OTEL_TRACES_EXPORTER != "otlp")
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.propagate import extract as _otel_extract, inject as _otel_inject

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _otlp_traces_enabled() -> bool:
    return os.environ.get("OTEL_TRACES_EXPORTER", "none").lower() == "otlp"


def _setup_otel() -> None:
    """Configure TracerProvider with OTLPSpanExporter when OTEL_TRACES_EXPORTER=otlp."""
    if not _otlp_traces_enabled() or not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    except ImportError:
        logger.warning("opentelemetry-sdk / exporter not installed; tracing disabled")
        return
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    resource = Resource(
        {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "inference-gateway")}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    _otel_trace.set_tracer_provider(provider)
    logger.info("OTel tracing enabled → %s", endpoint)


@contextlib.contextmanager
def _span(name: str, carrier: dict | None = None, **attrs):
    """Start a tracing span from the incoming W3C context. Yields None when OTel is off."""
    if not _OTEL_AVAILABLE:
        yield None
        return
    ctx = _otel_extract(carrier or {})
    tracer = _otel_trace.get_tracer("inference-gateway")
    with tracer.start_as_current_span(name, context=ctx) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s


def _span_event(name: str, **attrs) -> None:
    """Record a named event on the current active span. No-op when OTel is off."""
    if not _OTEL_AVAILABLE:
        return
    _otel_trace.get_current_span().add_event(
        name, attributes={k: str(v) for k, v in attrs.items()}
    )


def _outgoing_headers(technique: str) -> dict[str, str]:
    """Build upstream request headers, injecting W3C trace context when OTel is active."""
    headers: dict[str, str] = {"X-Technique": technique}
    if _OTEL_AVAILABLE:
        _otel_inject(headers)
    return headers


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
    metrics_log_dir: str | None = field(
        default_factory=lambda: (
            os.environ.get("GATEWAY_METRICS_LOG_DIR", "logs/gateway") or "logs/gateway"
        )
    )


settings = Settings()


# ---------------------------------------------------------------------------
# Multi-backend config
# ---------------------------------------------------------------------------

from gateway.config import GatewayConfig, load_config  # noqa: E402
from gateway.backends.http_backend import HttpBackend  # noqa: E402
from gateway.prom_metrics import (  # noqa: E402
    ACTIVE_REQUESTS,
    BACKEND_REQUEST_DURATION_SECONDS,
    BACKEND_SELECTION_TOTAL,
    ERRORS_TOTAL,
    GPU_COST_USD_TOTAL,
    INTER_CHUNK_SECONDS,
    REQUEST_DURATION_SECONDS,
    REQUEST_SIZE_BYTES,
    REQUESTS_TOTAL,
    RESPONSE_SIZE_BYTES,
    STREAMING_CHUNKS_TOTAL,
    TOKENS_PER_SECOND,
    TOKENS_TOTAL,
    TTFT_SECONDS,
)

gateway_config: GatewayConfig | None = (
    load_config() if Path("config.yaml").exists() else None
)


# ---------------------------------------------------------------------------
# In-memory metrics (legacy JSON endpoint)
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
# JSONL per-request metrics log (mirrors fullstack-inferencing schema)
# ---------------------------------------------------------------------------


def _smol_style_row(
    *,
    ttft_s: float,
    e2e_s: float,
    tpot_avg_s: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, float | int]:
    return {
        "prompt_tokens_total": prompt_tokens,
        "generation_tokens_total": completion_tokens,
        "time_to_first_token_avg_ms": ttft_s * 1000.0,
        "tpot_avg_ms": tpot_avg_s * 1000.0,
        "e2e_request_latency_avg_ms": e2e_s * 1000.0,
        "prompt_len_avg": float(prompt_tokens),
        "prefill_latency_avg_ms": ttft_s * 1000.0,
        "decode_latency_avg_ms": max(0.0, e2e_s - ttft_s) * 1000.0,
    }


def _build_metrics_log_row(
    *,
    technique: str,
    server_profile: str,
    upstream_path: str,
    streaming: bool,
    status_code: int,
    trace_id: str,
    e2e_s: float,
    ttft_s: float,
    tpot_avg_s: float,
    inter_chunk_delays_s: list[float],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    inter_ms = [d * 1000.0 for d in inter_chunk_delays_s]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "technique": technique,
        "server_profile": server_profile,
        "upstream_path": upstream_path,
        "streaming": streaming,
        "status_code": status_code,
        "trace_id": trace_id,
        "e2e_latency_s": e2e_s,
        "time_to_first_token_s": ttft_s,
        "tpot_avg_s": tpot_avg_s,
        "inter_chunk_count": len(inter_chunk_delays_s),
        "inter_chunk_delay_avg_ms": (
            sum(inter_ms) / len(inter_ms) if inter_ms else 0.0
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "smol_style": _smol_style_row(
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            tpot_avg_s=tpot_avg_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    }


async def _append_gateway_metrics_log(row: dict[str, Any]) -> None:
    """Append one JSON row to logs/gateway/gateway_metrics_YYYY-MM-DD.jsonl."""
    mdir = settings.metrics_log_dir
    if not mdir or mdir.lower() in ("-", "none", "false", "0"):
        return
    lock: asyncio.Lock | None = getattr(app.state, "metrics_log_lock", None)
    if lock is None:
        return
    async with lock:
        path = Path(mdir)
        path.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fpath = path / f"gateway_metrics_{day}.jsonl"

        def _write() -> None:
            with open(fpath, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _normalize_response(
    data: dict[str, Any], request_id: str, latency_ms: float = 0.0
) -> dict[str, Any]:
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
# Metrics recording
# ---------------------------------------------------------------------------


def record_metrics(
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
# Request validation
# ---------------------------------------------------------------------------


def _validate_body(body: dict[str, Any]) -> str | None:
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        return "invalid_model"

    messages = body.get("messages", [])
    for msg in messages:
        if not isinstance(msg, dict):
            return "invalid_messages"
        if "role" not in msg or "content" not in msg:
            return "invalid_messages"
        if msg["role"] not in ("system", "user", "assistant"):
            return "invalid_messages"

    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        return "invalid_stream"

    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            return "invalid_max_tokens"
        if max_tokens < 1 or max_tokens > 100_000:
            return "invalid_max_tokens"

    temperature = body.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            return "invalid_temperature"
        if temperature < 0 or temperature > 2:
            return "invalid_temperature"

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
# Auth dependency
# ---------------------------------------------------------------------------


def check_auth(request: Request) -> None:
    if settings.api_key is None:
        return
    auth_header = request.headers.get("Authorization", "")
    token: str | None = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif auth_header.startswith("Api-Key "):
        token = auth_header[8:]
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.metrics_log_lock = asyncio.Lock()
    mdir = settings.metrics_log_dir
    if not mdir or mdir.lower() in ("-", "none", "false", "0"):
        logger.info("JSONL request metrics logging disabled.")
    else:
        logger.info("JSONL metrics log dir: %s", Path(mdir).resolve())
    yield


app = FastAPI(title="Inference Gateway", lifespan=_lifespan)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


_ENDPOINTS = [
    {
        "method": "GET",
        "path": "/",
        "description": "This index — all available endpoints",
    },
    {
        "method": "GET",
        "path": "/health",
        "description": "Health check with upstream status",
    },
    {
        "method": "GET",
        "path": "/metrics",
        "description": "Legacy JSON counters (request count, latency, tokens)",
    },
    {
        "method": "GET",
        "path": "/v1/backends",
        "description": "List configured backends and default",
    },
    {
        "method": "GET",
        "path": "/v1/models",
        "description": "Proxy GET /v1/models to the default HTTP backend",
    },
    {
        "method": "POST",
        "path": "/v1/chat/completions",
        "description": "OpenAI-compatible chat completion (streaming supported)",
    },
    {
        "method": "GET",
        "path": ":9101/metrics",
        "description": "Prometheus scrape endpoint (dedicated port)",
    },
]


@app.get("/")
async def index() -> dict[str, Any]:
    return {"service": "inference-gateway", "endpoints": _ENDPOINTS}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check — returns ok; probes the default HTTP backend's /health when configured."""
    from gateway.backends.http_backend import HttpBackend

    if gateway_config is None or not isinstance(
        gateway_config.default_backend, HttpBackend
    ):
        return {"status": "ok", "upstream": None}

    backend = gateway_config.default_backend
    probe_url = backend.base_url + "/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(probe_url)
        if resp.status_code < 500:
            return {"status": "ok", "default_upstream": backend.base_url}
        return {
            "status": "degraded",
            "warning": f"upstream returned {resp.status_code}",
            "default_upstream": backend.base_url,
        }
    except httpx.RequestError as exc:
        return {
            "status": "misconfigured",
            "warning": str(exc),
            "default_upstream": backend.base_url,
        }


@app.get("/v1/models")
async def list_models() -> Response:
    """Proxy GET /v1/models to the default HTTP backend."""
    from gateway.backends.http_backend import HttpBackend

    if gateway_config is None or not isinstance(
        gateway_config.default_backend, HttpBackend
    ):
        raise HTTPException(status_code=404, detail={"error": "not_found"})

    backend = gateway_config.default_backend
    url = backend.base_url + "/v1/models"
    try:
        async with httpx.AsyncClient(timeout=backend.timeout) as client:
            resp = await client.get(url)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "upstream_unavailable", "detail": "upstream timed out"},
            status_code=504,
        )
    except httpx.RequestError as exc:
        return JSONResponse(
            {"error": "upstream_unavailable", "detail": str(exc)},
            status_code=502,
        )


@app.get("/metrics")
async def metrics_json() -> dict[str, Any]:
    count = metrics.request_count
    avg_latency_ms = metrics.total_latency_ms / count if count > 0 else 0.0
    return {
        "request_count": count,
        "error_count": metrics.error_count,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "prompt_tokens_total": metrics.prompt_tokens_total,
        "completion_tokens_total": metrics.completion_tokens_total,
    }


@app.get("/v1/backends")
async def list_backends() -> dict[str, Any]:
    if gateway_config is None:
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    return {
        "backends": [
            {"name": b.name, "type": type(b).__name__}
            for b in gateway_config.all_backends
        ],
        "default": gateway_config.default_backend.name,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, _auth: None = Depends(check_auth)
) -> Response:
    start = time.monotonic()
    request_id = (
        request.headers.get("X-Request-ID")
        or request.headers.get("Request-Id")
        or str(uuid.uuid4())
    )
    technique = request.headers.get("X-Technique", "baseline")
    server_profile = settings.vllm_server_profile
    log = logging.LoggerAdapter(logger, {"request_id": request_id})

    with _span(
        "chat.completions",
        dict(request.headers),
        technique=technique,
        server_profile=server_profile,
    ) as span:
        ACTIVE_REQUESTS.inc()
        try:
            response = await _handle_completions(
                request, request_id, technique, server_profile, start, log
            )
            if span is not None:
                span.set_attribute("http.status_code", response.status_code)
            return response
        finally:
            ACTIVE_REQUESTS.dec()


async def _handle_completions(
    request: Request,
    request_id: str,
    technique: str,
    server_profile: str,
    start: float,
    log: logging.LoggerAdapter,
) -> Response:
    # Parse body
    try:
        body = await request.json()
    except Exception as exc:
        log.warning("Bad request body: %s", exc)
        record_metrics(
            400,
            (time.monotonic() - start) * 1000,
            0,
            0,
            technique=technique,
            server_profile=server_profile,
        )
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    # Basic structure check
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        record_metrics(
            400,
            (time.monotonic() - start) * 1000,
            0,
            0,
            technique=technique,
            server_profile=server_profile,
        )
        return JSONResponse({"error": "invalid_messages"}, status_code=400)

    # Full validation
    error = _validate_body(body)
    if error:
        record_metrics(
            400,
            (time.monotonic() - start) * 1000,
            0,
            0,
            technique=technique,
            server_profile=server_profile,
        )
        return JSONResponse({"error": error}, status_code=400)

    stream = body.get("stream", False)
    model = body.get("model")

    _span_event("validation.done", model=str(model), stream=str(stream))

    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
            break

    forward_body: dict[str, Any] = {
        "messages": messages,
        "model": model or "default",
        "stream": stream,
    }
    for key in ("max_tokens", "temperature", "stop"):
        if key in body:
            forward_body[key] = body[key]

    if gateway_config is not None:
        return await _handle_with_config(
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
    if settings.backend_url:
        return await _handle_backend(
            request_id,
            forward_body,
            stream,
            start,
            log,
            technique=technique,
            server_profile=server_profile,
        )
    return await _handle_echo(
        request_id,
        prompt,
        stream,
        start,
        log,
        technique=technique,
        server_profile=server_profile,
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


async def _handle_echo(
    request_id: str,
    prompt: str,
    stream: bool,
    start: float,
    log: logging.LoggerAdapter,
    *,
    technique: str = "baseline",
    server_profile: str = "default",
) -> Response:
    content = f"Echo: {prompt}"
    prompt_tokens = len(prompt.split())
    completion_tokens = len(content.split())
    latency_ms = (time.monotonic() - start) * 1000

    record_metrics(
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

    e2e_s = latency_ms / 1000.0
    await _append_gateway_metrics_log(
        _build_metrics_log_row(
            technique=technique,
            server_profile=server_profile,
            upstream_path="/v1/chat/completions",
            streaming=stream,
            status_code=200,
            trace_id=request_id,
            e2e_s=e2e_s,
            ttft_s=e2e_s,
            tpot_avg_s=(e2e_s / completion_tokens) if completion_tokens > 0 else 0.0,
            inter_chunk_delays_s=[],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )

    if stream:
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

        async def _sse() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"},
        )

    resp = build_response(
        request_id,
        content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )
    return JSONResponse(resp, headers={"X-Request-ID": request_id})


async def _handle_with_config(
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
) -> Response:
    assert gateway_config is not None
    backend = gateway_config.get_backend_for_model(model)
    if _OTEL_AVAILABLE:
        _otel_trace.get_current_span().set_attribute("backend", backend.name)

    # Record which backend was selected and why
    reason = (
        "model_match"
        if (
            model
            and model in {b.name for b in gateway_config.all_backends}
            and model != gateway_config.default_backend.name
        )
        else "default_fallback"
    )
    BACKEND_SELECTION_TOTAL.labels(backend=backend.name, reason=reason).inc()
    _span_event("backend.selected", backend=backend.name, reason=reason)

    backend_body = {k: v for k, v in forward_body.items() if k != "model"}
    if isinstance(backend, HttpBackend) and backend.model is not None:
        backend_body["model"] = backend.model

    if stream:
        if isinstance(backend, HttpBackend):
            return await _proxy_stream(
                request_id,
                backend.completions_url,
                backend_body,
                start,
                log,
                technique=technique,
                server_profile=server_profile,
                backend_name=backend.name,
            )
        return await _handle_echo(
            request_id,
            prompt,
            stream,
            start,
            log,
            technique=technique,
            server_profile=server_profile,
        )

    backend_start = time.monotonic()
    _span_event("backend.request.start", backend=backend.name)
    try:
        data = await backend.generate(backend_body, request_id)
    except httpx.TimeoutException:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            504, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend %s timeout after %.1f ms", backend.name, latency_ms)
        return JSONResponse(
            {"error": "gateway_timeout"},
            status_code=504,
            headers={"X-Request-ID": request_id},
        )
    except httpx.RequestError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            502, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend %s connection error: %s", backend.name, exc)
        return JSONResponse(
            {"error": "backend_unavailable"},
            status_code=502,
            headers={"X-Request-ID": request_id},
        )
    except RuntimeError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            502, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend %s error: %s", backend.name, exc)
        return JSONResponse(
            {"error": "backend_error"},
            status_code=502,
            headers={"X-Request-ID": request_id},
        )

    backend_duration_s = time.monotonic() - backend_start
    BACKEND_REQUEST_DURATION_SECONDS.labels(
        backend=backend.name, technique=technique
    ).observe(backend_duration_s)
    REQUEST_SIZE_BYTES.labels(backend=backend.name, technique=technique).observe(
        len(json.dumps(backend_body).encode())
    )

    latency_ms = (time.monotonic() - start) * 1000
    data = _normalize_response(data, request_id, latency_ms=latency_ms)
    data["backend"] = backend.name

    RESPONSE_SIZE_BYTES.labels(backend=backend.name, technique=technique).observe(
        len(json.dumps(data).encode())
    )

    usage = data["usage"]
    completion_tokens = usage.get("completion_tokens", 0)
    record_metrics(
        200,
        latency_ms,
        usage.get("prompt_tokens", 0),
        completion_tokens,
        technique=technique,
        server_profile=server_profile,
    )
    if backend_duration_s > 0 and completion_tokens > 0:
        TOKENS_PER_SECOND.labels(
            technique=technique, server_profile=server_profile
        ).observe(completion_tokens / backend_duration_s)
    _span_event(
        "backend.response.complete",
        backend=backend.name,
        completion_tokens=str(completion_tokens),
        backend_duration_ms=f"{backend_duration_s * 1000:.1f}",
    )
    log.info(
        "POST /v1/chat/completions status=200 latency_ms=%.1f backend_ms=%.1f mode=config backend=%s",
        latency_ms,
        backend_duration_s * 1000,
        backend.name,
    )
    e2e_s = latency_ms / 1000.0
    tpot_avg_s = (
        (backend_duration_s / completion_tokens) if completion_tokens > 0 else 0.0
    )
    await _append_gateway_metrics_log(
        _build_metrics_log_row(
            technique=technique,
            server_profile=server_profile,
            upstream_path="/v1/chat/completions",
            streaming=False,
            status_code=200,
            trace_id=request_id,
            e2e_s=e2e_s,
            ttft_s=e2e_s,
            tpot_avg_s=tpot_avg_s,
            inter_chunk_delays_s=[],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
        )
    )
    return JSONResponse(data, headers={"X-Request-ID": request_id})


async def _handle_backend(
    request_id: str,
    body: dict[str, Any],
    stream: bool,
    start: float,
    log: logging.LoggerAdapter,
    *,
    technique: str = "baseline",
    server_profile: str = "default",
) -> Response:
    assert settings.backend_url is not None
    url = settings.backend_url.rstrip("/") + "/v1/chat/completions"
    if stream:
        return await _proxy_stream(
            request_id,
            url,
            body,
            start,
            log,
            technique=technique,
            server_profile=server_profile,
        )
    return await _proxy_non_stream(
        request_id,
        url,
        body,
        start,
        log,
        technique=technique,
        server_profile=server_profile,
    )


async def _proxy_non_stream(
    request_id: str,
    url: str,
    body: dict[str, Any],
    start: float,
    log: logging.LoggerAdapter,
    *,
    technique: str = "baseline",
    server_profile: str = "default",
    backend_name: str = "legacy",
) -> Response:
    backend_start = time.monotonic()
    _span_event("backend.request.start", backend=backend_name)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url, json=body, headers=_outgoing_headers(technique)
            )
    except httpx.TimeoutException:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            504, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend timeout after %.1f ms", latency_ms)
        return JSONResponse(
            {"error": "gateway_timeout"},
            status_code=504,
            headers={"X-Request-ID": request_id},
        )
    except httpx.RequestError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            502, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend connection error: %s", exc)
        return JSONResponse(
            {"error": "backend_unavailable"},
            status_code=502,
            headers={"X-Request-ID": request_id},
        )

    backend_duration_s = time.monotonic() - backend_start

    if resp.status_code >= 500:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            502, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        log.error("Backend returned %d", resp.status_code)
        return JSONResponse(
            {"error": "backend_error"},
            status_code=502,
            headers={"X-Request-ID": request_id},
        )

    try:
        data = resp.json()
    except Exception:
        latency_ms = (time.monotonic() - start) * 1000
        record_metrics(
            502, latency_ms, 0, 0, technique=technique, server_profile=server_profile
        )
        return JSONResponse(
            {"error": "backend_invalid_response"},
            status_code=502,
            headers={"X-Request-ID": request_id},
        )

    BACKEND_REQUEST_DURATION_SECONDS.labels(
        backend=backend_name, technique=technique
    ).observe(backend_duration_s)
    REQUEST_SIZE_BYTES.labels(backend=backend_name, technique=technique).observe(
        len(json.dumps(body).encode())
    )

    latency_ms = (time.monotonic() - start) * 1000
    data = _normalize_response(data, request_id, latency_ms=latency_ms)
    RESPONSE_SIZE_BYTES.labels(backend=backend_name, technique=technique).observe(
        len(resp.content)
    )
    usage = data["usage"]
    completion_tokens = usage.get("completion_tokens", 0)
    record_metrics(
        resp.status_code,
        latency_ms,
        usage.get("prompt_tokens", 0),
        completion_tokens,
        technique=technique,
        server_profile=server_profile,
    )
    if backend_duration_s > 0 and completion_tokens > 0:
        TOKENS_PER_SECOND.labels(
            technique=technique, server_profile=server_profile
        ).observe(completion_tokens / backend_duration_s)
    _span_event(
        "backend.response.complete",
        backend=backend_name,
        status_code=str(resp.status_code),
        completion_tokens=str(completion_tokens),
        backend_duration_ms=f"{backend_duration_s * 1000:.1f}",
    )
    log.info(
        "POST /v1/chat/completions status=%d latency_ms=%.1f backend_ms=%.1f mode=backend",
        resp.status_code,
        latency_ms,
        backend_duration_s * 1000,
    )
    e2e_s = latency_ms / 1000.0
    tpot_avg_s = (
        (backend_duration_s / completion_tokens) if completion_tokens > 0 else 0.0
    )
    await _append_gateway_metrics_log(
        _build_metrics_log_row(
            technique=technique,
            server_profile=server_profile,
            upstream_path="/v1/chat/completions",
            streaming=False,
            status_code=resp.status_code,
            trace_id=request_id,
            e2e_s=e2e_s,
            ttft_s=e2e_s,
            tpot_avg_s=tpot_avg_s,
            inter_chunk_delays_s=[],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
        )
    )
    return JSONResponse(
        data, status_code=resp.status_code, headers={"X-Request-ID": request_id}
    )


async def _proxy_stream(
    request_id: str,
    url: str,
    body: dict[str, Any],
    start: float,
    log: logging.LoggerAdapter,
    *,
    technique: str = "baseline",
    server_profile: str = "default",
    backend_name: str = "legacy",
) -> Response:
    async def _stream_gen() -> AsyncGenerator[bytes, None]:
        prompt_tokens = 0
        completion_tokens = 0
        first_chunk = True
        last_chunk_time = time.monotonic()
        inter_chunk_delays: list[float] = []
        response_bytes = 0
        chunk_count = 0

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", url, json=body, headers=_outgoing_headers(technique)
                ) as resp:
                    if resp.status_code >= 500:
                        latency_ms = (time.monotonic() - start) * 1000
                        record_metrics(
                            502,
                            latency_ms,
                            0,
                            0,
                            technique=technique,
                            server_profile=server_profile,
                        )
                        log.error("Backend returned %d on stream", resp.status_code)
                        yield json.dumps({"error": "backend_error"}).encode()
                        return

                    ttft_s: float | None = None
                    async for line in resp.aiter_lines():
                        encoded = (line + "\n").encode()
                        yield encoded
                        response_bytes += len(encoded)
                        if line.startswith("data: ") and line != "data: [DONE]":
                            chunk_count += 1
                            now = time.monotonic()
                            if first_chunk:
                                ttft_s = now - start
                                TTFT_SECONDS.observe(ttft_s)
                                _span_event(
                                    "stream.first_chunk", ttft_s=f"{ttft_s:.3f}"
                                )
                                first_chunk = False
                            else:
                                gap = now - last_chunk_time
                                INTER_CHUNK_SECONDS.observe(gap)
                                inter_chunk_delays.append(gap)
                            last_chunk_time = now
                            try:
                                chunk = json.loads(line[6:])
                                if usage := chunk.get("usage"):
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get(
                                        "completion_tokens", 0
                                    )
                            except (json.JSONDecodeError, AttributeError):
                                pass
                    yield b"\n"
                    response_bytes += 1
        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            record_metrics(
                504,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend stream timeout after %.1f ms", latency_ms)
            yield (json.dumps({"error": "gateway_timeout"}) + "\n").encode()
            return
        except httpx.RequestError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            record_metrics(
                502,
                latency_ms,
                0,
                0,
                technique=technique,
                server_profile=server_profile,
            )
            log.error("Backend stream connection error: %s", exc)
            yield (json.dumps({"error": "backend_unavailable"}) + "\n").encode()
            return

        latency_ms = (time.monotonic() - start) * 1000
        e2e_s = latency_ms / 1000.0
        # Use TTFT as a proxy for backend request duration on streaming paths
        if ttft_s is not None:
            BACKEND_REQUEST_DURATION_SECONDS.labels(
                backend=backend_name, technique=technique
            ).observe(ttft_s)
        REQUEST_SIZE_BYTES.labels(backend=backend_name, technique=technique).observe(
            len(json.dumps(body).encode())
        )
        RESPONSE_SIZE_BYTES.labels(backend=backend_name, technique=technique).observe(
            response_bytes
        )
        STREAMING_CHUNKS_TOTAL.labels(technique=technique).inc(chunk_count)
        _span_event(
            "stream.complete",
            chunks=str(chunk_count),
            completion_tokens=str(completion_tokens),
            e2e_ms=f"{latency_ms:.1f}",
        )
        if latency_ms > 0 and completion_tokens > 0:
            TOKENS_PER_SECOND.labels(
                technique=technique, server_profile=server_profile
            ).observe(completion_tokens / (latency_ms / 1000))
        record_metrics(
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
        actual_ttft_s = ttft_s if ttft_s is not None else e2e_s
        tpot_avg_s: float
        if inter_chunk_delays:
            tpot_avg_s = sum(inter_chunk_delays) / len(inter_chunk_delays)
        elif completion_tokens > 0 and e2e_s > 0:
            tpot_avg_s = e2e_s / completion_tokens
        else:
            tpot_avg_s = 0.0
        await _append_gateway_metrics_log(
            _build_metrics_log_row(
                technique=technique,
                server_profile=server_profile,
                upstream_path="/v1/chat/completions",
                streaming=True,
                status_code=200,
                trace_id=request_id,
                e2e_s=e2e_s,
                ttft_s=actual_ttft_s,
                tpot_avg_s=tpot_avg_s,
                inter_chunk_delays_s=inter_chunk_delays,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    return StreamingResponse(
        _stream_gen(),
        media_type="text/event-stream",
        headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    _setup_otel()
    prometheus_client.start_http_server(settings.metrics_port)
    logger.info("Prometheus metrics available on port %d", settings.metrics_port)

    mode = f"backend={settings.backend_url}" if settings.backend_url else "echo mode"
    logger.info("Inference gateway listening on port %d (%s)", settings.port, mode)

    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
