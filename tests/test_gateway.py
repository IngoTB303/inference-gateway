"""Tests for the inference gateway."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import HTTPServer

import httpx
import pytest
import respx

import main
from gateway.backends.http_backend import HttpBackend
from gateway.config import GatewayConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPLETION_PAYLOAD = {
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
}


def _post(url: str, body: dict, headers: dict | None = None):
    """POST JSON body; returns (status, body_dict, response_headers)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read()), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), e.headers


def _get(url: str):
    """GET; returns (status, body_dict)."""
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gateway(monkeypatch):
    """Echo-mode gateway on a free port (legacy settings path)."""
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    monkeypatch.setattr(main, "gateway_config", None)  # use legacy path
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def backend_gateway(monkeypatch):
    """Backend-proxy gateway pointing at the test mock URL (legacy settings path)."""
    monkeypatch.setattr(main.settings, "backend_url", "http://test-backend")
    monkeypatch.setattr(main.settings, "api_key", None)
    monkeypatch.setattr(main, "gateway_config", None)  # use legacy path
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


def test_healthz(gateway):
    status, body = _get(f"{gateway}/healthz")
    assert status == 200
    assert body == {"status": "ok"}


def test_unknown_route_get(gateway):
    status, _ = _get(f"{gateway}/unknown")
    assert status == 404


def test_unknown_route_post(gateway):
    status, _, _ = _post(f"{gateway}/unknown", {})
    assert status == 404


# ---------------------------------------------------------------------------
# Echo mode — response shape
# ---------------------------------------------------------------------------


def test_echo_returns_correct_shape(gateway):
    status, body, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 200
    assert "id" in body
    assert "choices" in body
    assert "usage" in body


def test_echo_content(gateway):
    _, body, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    content = body["choices"][0]["message"]["content"]
    assert content.startswith("Echo: hello")


def test_echo_usage_fields(gateway):
    _, body, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    usage = body["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


# ---------------------------------------------------------------------------
# Request-ID
# ---------------------------------------------------------------------------


def test_request_id_from_header(gateway):
    rid = "my-request-123"
    status, body, headers = _post(
        f"{gateway}/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"X-Request-ID": rid},
    )
    assert status == 200
    assert body["id"] == rid
    assert headers.get("X-Request-ID") == rid


def test_request_id_generated(gateway):
    _, body, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    generated_id = body["id"]
    parsed = uuid.UUID(generated_id)
    assert parsed.version == 4


def test_request_id_alt_header(gateway):
    rid = "alt-request-456"
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Request-Id": rid},
    )
    assert status == 200
    assert body["id"] == rid


# ---------------------------------------------------------------------------
# Validation (400 errors)
# ---------------------------------------------------------------------------


def test_invalid_json_body(gateway):
    req = urllib.request.Request(
        f"{gateway}/v1/chat/completions",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        pytest.fail("Expected HTTPError")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert json.loads(e.read())["error"] == "invalid_json"


def test_missing_messages(gateway):
    status, body, _ = _post(f"{gateway}/v1/chat/completions", {})
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_empty_messages_list(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"model": "test", "messages": []},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


# ---------------------------------------------------------------------------
# Echo streaming (SSE)
# ---------------------------------------------------------------------------


def test_echo_streaming(gateway):
    payload = {**COMPLETION_PAYLOAD, "stream": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{gateway}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        raw = resp.read().decode()

    lines = [line for line in raw.split("\n") if line.startswith("data:")]
    assert any(line.startswith("data: {") for line in lines)
    assert any(line == "data: [DONE]" for line in lines)


# ---------------------------------------------------------------------------
# Backend proxy (mocked with respx)
# ---------------------------------------------------------------------------


@respx.mock
def test_backend_success(backend_gateway):
    mock_body = {
        "id": "backend-id",
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_body)
    )

    status, body, headers = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 200
    assert "id" in body
    assert headers.get("X-Request-ID") is not None


@respx.mock
def test_backend_5xx(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )

    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 502
    assert body["error"] == "backend_error"


@respx.mock
def test_backend_timeout(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 504
    assert body["error"] == "gateway_timeout"


@respx.mock
def test_backend_connection_error(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )

    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 502
    assert body["error"] == "backend_unavailable"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_increments(gateway):
    _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)

    _, metrics_body = _get(f"{gateway}/metrics")
    assert metrics_body["request_count"] == 2


def test_metrics_error_count(gateway):
    _post(f"{gateway}/v1/chat/completions", {})  # missing messages → 400

    _, metrics_body = _get(f"{gateway}/metrics")
    assert metrics_body["error_count"] == 1


def test_metrics_prompt_tokens_echo(gateway):
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    _post(f"{gateway}/v1/chat/completions", payload)

    _, metrics_body = _get(f"{gateway}/metrics")
    assert metrics_body["request_count"] == 1
    assert metrics_body["prompt_tokens_total"] == 2  # len("hello world".split()) == 2


# ---------------------------------------------------------------------------
# Extended validation (#1 + #6)
# ---------------------------------------------------------------------------


def test_invalid_message_missing_role(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"content": "hi"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_message_missing_content(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_message_bad_role(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "unknown", "content": "hi"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_stream_type(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "stream": "yes"},
    )
    assert status == 400
    assert body["error"] == "invalid_stream"


def test_invalid_max_tokens_type(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "fifty"},
    )
    assert status == 400
    assert body["error"] == "invalid_max_tokens"


def test_invalid_max_tokens_range(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1},
    )
    assert status == 400
    assert body["error"] == "invalid_max_tokens"


def test_invalid_temperature_range(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 5.0},
    )
    assert status == 400
    assert body["error"] == "invalid_temperature"


def test_invalid_stop_type(gateway):
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "stop": 123},
    )
    assert status == 400
    assert body["error"] == "invalid_stop"


def test_model_default_omitted(gateway):
    """Omitting model should succeed."""
    status, body, _ = _post(
        f"{gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200


@respx.mock
def test_full_contract_fields_forwarded(backend_gateway):
    """Backend receives max_tokens, temperature, stop in the forwarded request."""
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    respx.post("http://test-backend/v1/chat/completions").mock(side_effect=capture)

    _post(
        f"{backend_gateway}/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stop": ["\n"],
        },
    )

    assert captured["body"]["max_tokens"] == 100
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["stop"] == ["\n"]


# ---------------------------------------------------------------------------
# Multi-backend routing (#8)
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_backend_gateway(monkeypatch):
    """Gateway with config-based multi-backend routing (echo default + remote mock)."""
    from gateway.backends.echo import EchoBackend

    echo = EchoBackend(name="local")
    remote = HttpBackend(name="remote-modal-llama", url="http://test-backend")
    config = GatewayConfig(backends=[echo, remote], default_backend=echo)

    monkeypatch.setattr(main, "gateway_config", config)
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_routing_by_model_echo(multi_backend_gateway):
    """model: 'local' routes to echo backend and response has backend: 'local'."""
    status, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {"model": "local", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 200
    assert body["backend"] == "local"
    assert body["choices"][0]["message"]["content"].startswith("Echo:")


def test_routing_default_fallback(multi_backend_gateway):
    """Omitting model falls back to default (echo) backend."""
    status, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 200
    assert body["backend"] == "local"


def test_routing_unknown_model_fallback(multi_backend_gateway):
    """Unknown model name falls back to default backend."""
    status, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {
            "model": "nonexistent-backend",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert status == 200
    assert body["backend"] == "local"


@respx.mock
def test_routing_by_model_remote(multi_backend_gateway):
    """model: 'remote-modal-llama' routes to the HTTP backend."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
    )
    status, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {
            "model": "remote-modal-llama",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert status == 200
    assert body["backend"] == "remote-modal-llama"


@respx.mock
def test_routing_by_model_vllm(monkeypatch):
    """model: 'remote-modal-vllm' routes to the vllm backend and includes backend field."""
    from gateway.backends.echo import EchoBackend

    echo = EchoBackend(name="local")
    vllm = HttpBackend(name="remote-modal-vllm", url="http://test-vllm-backend")
    config = GatewayConfig(backends=[echo, vllm], default_backend=echo)

    monkeypatch.setattr(main, "gateway_config", config)
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    respx.post("http://test-vllm-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )
    )

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = _post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": "remote-modal-vllm",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 200
        assert body["backend"] == "remote-modal-vllm"
    finally:
        server.shutdown()


def test_backend_metadata_in_response(multi_backend_gateway):
    """Every response includes a 'backend' field."""
    status, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200
    assert "backend" in body


def test_get_backends_endpoint(multi_backend_gateway):
    """GET /v1/backends lists configured backends."""
    status, body = _get(f"{multi_backend_gateway}/v1/backends")
    assert status == 200
    names = [b["name"] for b in body["backends"]]
    assert "local" in names
    assert "remote-modal-llama" in names
    assert body["default"] == "local"


def test_config_loading():
    """load_config() correctly parses config.yaml and creates backend objects."""
    from gateway.backends.echo import EchoBackend
    from gateway.config import load_config

    cfg = load_config("config.yaml")
    assert cfg.default_backend.name == "local"
    assert isinstance(cfg.default_backend, EchoBackend)
    names = [b.name for b in cfg.all_backends]
    assert "local" in names
    assert "remote-modal-llama" in names
    assert "remote-modal-vllm" in names


def test_no_config_fallback(tmp_path, monkeypatch):
    """When config.yaml is absent, load_config falls back to BACKEND_URL env var."""
    from gateway.config import load_config

    monkeypatch.setenv("BACKEND_URL", "http://fallback-backend")
    cfg = load_config(tmp_path / "nonexistent.yaml")
    assert cfg.default_backend.name == "remote"
    assert isinstance(cfg.default_backend, HttpBackend)


# ---------------------------------------------------------------------------
# Auth and policy hooks (#2)
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_gateway(monkeypatch):
    """Gateway with API_KEY=test-secret-key."""
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", "test-secret-key")
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_auth_no_header_401(auth_gateway):
    status, body, _ = _post(f"{auth_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 401
    assert body["error"] == "unauthorized"


def test_auth_wrong_key_401(auth_gateway):
    status, body, _ = _post(
        f"{auth_gateway}/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert status == 401
    assert body["error"] == "unauthorized"


def test_auth_correct_bearer_200(auth_gateway):
    status, _, _ = _post(
        f"{auth_gateway}/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert status == 200


def test_auth_api_key_scheme(auth_gateway):
    status, _, _ = _post(
        f"{auth_gateway}/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Api-Key test-secret-key"},
    )
    assert status == 200


def test_no_auth_configured(gateway):
    """When no API_KEY is set, requests succeed without Authorization header."""
    status, _, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 200


# ---------------------------------------------------------------------------
# Response normalization (#7)
# ---------------------------------------------------------------------------


@respx.mock
def test_backend_missing_usage_normalized(backend_gateway):
    """Backend response without usage gets normalized to zeros."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 200
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["completion_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


@respx.mock
def test_backend_missing_model_normalized(backend_gateway):
    """Backend response without model gets 'unknown'."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )
    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 200
    assert body["model"] == "unknown"


@respx.mock
def test_backend_missing_object_normalized(backend_gateway):
    """Backend response without object gets 'chat.completion'."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )
    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 200
    assert body["object"] == "chat.completion"


@respx.mock
def test_backend_extra_fields_stripped(backend_gateway):
    """Extra keys from the backend (created, system_fingerprint, etc.) are stripped."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "backend-id",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "llama-3",
                "system_fingerprint": "fp_abc123",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {"prompt_n": 3, "predicted_n": 2},
            },
        )
    )
    status, body, _ = _post(
        f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD
    )
    assert status == 200

    # Only the documented top-level keys should be present
    allowed_keys = {"id", "object", "model", "choices", "usage", "backend"}
    assert set(body.keys()) <= allowed_keys, (
        f"Unexpected top-level keys: {set(body.keys()) - allowed_keys}"
    )

    # Choices should only have documented keys
    choice = body["choices"][0]
    assert set(choice.keys()) == {"index", "message", "finish_reason"}

    # Usage should only have documented keys
    assert set(body["usage"].keys()) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
    }


@respx.mock
def test_streaming_logs_usage(backend_gateway):
    """Streaming backend that sends usage in final chunk updates metrics."""
    usage_chunk = json.dumps(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    )
    sse_body = f"data: {usage_chunk}\n\ndata: [DONE]\n\n"

    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    payload = {**COMPLETION_PAYLOAD, "stream": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{backend_gateway}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()  # consume stream

    _, metrics_body = _get(f"{backend_gateway}/metrics")
    assert metrics_body["prompt_tokens_total"] == 7
    assert metrics_body["completion_tokens_total"] == 3


# ---------------------------------------------------------------------------
# Latency in usage (#10)
# ---------------------------------------------------------------------------


def test_echo_latency_in_usage(gateway):
    """Echo response includes latency_ms in usage."""
    _, body, _ = _post(f"{gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert "latency_ms" in body["usage"]
    assert body["usage"]["latency_ms"] >= 0


@respx.mock
def test_backend_latency_in_usage(backend_gateway):
    """Backend response includes latency_ms in usage, set by the gateway."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
    )
    _, body, _ = _post(f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert "latency_ms" in body["usage"]
    assert body["usage"]["latency_ms"] >= 0


@respx.mock
def test_config_backend_latency_in_usage(multi_backend_gateway):
    """Config-routed backend response includes latency_ms in usage."""
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )
    )
    _, body, _ = _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {
            "model": "remote-modal-llama",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert "latency_ms" in body["usage"]
    assert body["usage"]["latency_ms"] >= 0


@respx.mock
def test_model_field_not_forwarded_to_backend(multi_backend_gateway):
    """Gateway routing key ('model') is stripped before forwarding to HTTP backend.

    This prevents errors like vLLM's 'model does not exist' when the user
    sets model to a gateway backend name rather than an actual LLM model name.
    """
    captured = {}

    def capture(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    respx.post("http://test-backend/v1/chat/completions").mock(side_effect=capture)

    _post(
        f"{multi_backend_gateway}/v1/chat/completions",
        {
            "model": "remote-modal-llama",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert "model" not in captured["body"], (
        "Gateway backend name must not be forwarded as 'model' to the upstream server"
    )


# ---------------------------------------------------------------------------
# Live backend tests (#9) — skipped by default; run with: uv run pytest -m live
# ---------------------------------------------------------------------------


@pytest.fixture
def live_gateway(monkeypatch):
    """Gateway backed by real config.yaml on a free port."""
    from gateway.config import load_config

    cfg = load_config("config.yaml")
    monkeypatch.setattr(main, "gateway_config", cfg)
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.mark.live
def test_live_backends_endpoint(live_gateway):
    """GET /v1/backends lists all backends from config.yaml."""
    status, body = _get(f"{live_gateway}/v1/backends")
    assert status == 200
    names = [b["name"] for b in body["backends"]]
    assert "local" in names
    assert "local-llama" in names
    assert "remote-modal-llama" in names
    assert "remote-modal-vllm" in names
    assert body["default"] == "local"


def _assert_valid_completion(body: dict, backend_name: str) -> None:
    """Shared assertions for a successful completion response."""
    assert "id" in body
    assert body.get("backend") == backend_name
    assert len(body.get("choices", [])) > 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    usage = body.get("usage", {})
    assert "latency_ms" in usage
    assert usage["latency_ms"] >= 0


@pytest.mark.live
def test_live_local_llama(live_gateway):
    """Live inference against local-llama backend."""
    try:
        status, body, _ = _post(
            f"{live_gateway}/v1/chat/completions",
            {
                "model": "local-llama",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
            },
        )
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read())

    if status in (502, 504):
        pytest.skip(f"local-llama backend not reachable (status {status})")

    assert status == 200
    _assert_valid_completion(body, "local-llama")


@pytest.mark.live
def test_live_remote_modal_llama(live_gateway):
    """Live inference against remote-modal-llama backend."""
    try:
        status, body, _ = _post(
            f"{live_gateway}/v1/chat/completions",
            {
                "model": "remote-modal-llama",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
            },
        )
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read())

    if status in (502, 504):
        pytest.skip(f"remote-modal-llama backend not reachable (status {status})")

    assert status == 200
    _assert_valid_completion(body, "remote-modal-llama")


@pytest.mark.live
def test_live_remote_modal_vllm(live_gateway):
    """Live inference against remote-modal-vllm backend."""
    try:
        status, body, _ = _post(
            f"{live_gateway}/v1/chat/completions",
            {
                "model": "remote-modal-vllm",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
            },
        )
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read())

    if status in (502, 504):
        pytest.skip(f"remote-modal-vllm backend not reachable (status {status})")

    assert status == 200
    _assert_valid_completion(body, "remote-modal-vllm")


@pytest.mark.live
def test_live_backend_timeout(monkeypatch):
    """Backend that does not respond within timeout triggers a 504."""
    from gateway.backends.echo import EchoBackend as _EchoBackend

    echo = _EchoBackend(name="local")
    unreachable = HttpBackend(
        name="unreachable", url="http://127.0.0.1:19999", timeout=0.1
    )
    config = GatewayConfig(backends=[echo, unreachable], default_backend=echo)

    monkeypatch.setattr(main, "gateway_config", config)
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    for f in (
        "request_count",
        "error_count",
        "prompt_tokens_total",
        "completion_tokens_total",
    ):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    server = HTTPServer(("127.0.0.1", 0), main.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body, _ = _post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {"model": "unreachable", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 504
        assert body["error"] == "gateway_timeout"
    finally:
        server.shutdown()
