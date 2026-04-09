"""Tests for the inference gateway."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import main
from gateway.backends.http_backend import HttpBackend
from gateway.config import GatewayConfig, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPLETION_PAYLOAD = {
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
}


def _post(client, path, body: dict, headers: dict | None = None, timeout: float = 30.0):
    """POST JSON body; returns (status, body_dict, response_headers).

    client can be a TestClient (unit tests) or a full URL string (live tests).
    For live tests pass the full URL as client and omit path (or pass path="").
    """
    if isinstance(client, str):
        # live test: client is the full URL, path is the body
        with httpx.Client(timeout=timeout) as c:
            try:
                resp = c.post(client, json=path, headers={"Content-Type": "application/json", **(headers or {})})
                return resp.status_code, resp.json(), resp.headers
            except httpx.HTTPStatusError as e:
                return e.response.status_code, e.response.json(), e.response.headers
    resp = client.post(path, json=body, headers=headers or {})
    return resp.status_code, resp.json(), resp.headers


def _get(client, path: str):
    """GET; returns (status, body_dict).

    client can be a TestClient (unit tests) or a base URL string (live tests).
    """
    if isinstance(client, str):
        with httpx.Client() as c:
            try:
                resp = c.get(client + path)
                return resp.status_code, resp.json()
            except httpx.HTTPStatusError as e:
                return e.response.status_code, e.response.json()
    resp = client.get(path)
    return resp.status_code, resp.json()


def _live_timeout(backend_name: str, extra: float = 30.0) -> float:
    """Return the configured backend timeout + extra seconds for live test clients."""
    cfg = load_config("config.yaml")
    backend = cfg.get_backend_for_model(backend_name)
    if isinstance(backend, HttpBackend):
        return backend.timeout + extra
    return extra


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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


def test_healthz(gateway):
    status, body = _get(gateway, "/healthz")
    assert status == 200
    assert body == {"status": "ok"}


def test_root_endpoint(gateway):
    status, body = _get(gateway, "/")
    assert status == 200
    assert body["service"] == "inference-gateway"
    paths = [e["path"] for e in body["endpoints"]]
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
    assert "/health" in paths


def test_health_echo_mode(gateway):
    status, body = _get(gateway, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["upstream"] is None


def test_unknown_route_get(gateway):
    status, _ = _get(gateway, "/unknown")
    assert status == 404


def test_unknown_route_post(gateway):
    status, _, _ = _post(gateway, "/unknown", {})
    assert status == 404


# ---------------------------------------------------------------------------
# Echo mode — response shape
# ---------------------------------------------------------------------------


def test_echo_returns_correct_shape(gateway):
    status, body, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 200
    assert "id" in body
    assert "choices" in body
    assert "usage" in body


def test_echo_content(gateway):
    _, body, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
    content = body["choices"][0]["message"]["content"]
    assert content.startswith("Echo: hello")


def test_echo_usage_fields(gateway):
    _, body, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
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
        gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"X-Request-ID": rid},
    )
    assert status == 200
    assert body["id"] == rid
    assert headers.get("X-Request-ID") == rid


def test_request_id_generated(gateway):
    _, body, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
    generated_id = body["id"]
    parsed = uuid.UUID(generated_id)
    assert parsed.version == 4


def test_request_id_alt_header(gateway):
    rid = "alt-request-456"
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Request-Id": rid},
    )
    assert status == 200
    assert body["id"] == rid


# ---------------------------------------------------------------------------
# Validation (400 errors)
# ---------------------------------------------------------------------------


def test_invalid_json_body(gateway):
    resp = gateway.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_json"


def test_missing_messages(gateway):
    status, body, _ = _post(gateway, "/v1/chat/completions", {})
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_empty_messages_list(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"model": "test", "messages": []},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


# ---------------------------------------------------------------------------
# Echo streaming (SSE)
# ---------------------------------------------------------------------------


def test_echo_streaming(gateway):
    payload = {**COMPLETION_PAYLOAD, "stream": True}
    resp = gateway.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    raw = resp.text

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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
    )
    assert status == 502
    assert body["error"] == "backend_error"


@respx.mock
def test_backend_timeout(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    status, body, _ = _post(
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
    )
    assert status == 504
    assert body["error"] == "gateway_timeout"


@respx.mock
def test_backend_connection_error(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )

    status, body, _ = _post(
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
    )
    assert status == 502
    assert body["error"] == "backend_unavailable"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_increments(gateway):
    _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
    _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)

    _, metrics_body = _get(gateway, "/metrics")
    assert metrics_body["request_count"] == 2


def test_metrics_error_count(gateway):
    _post(
        gateway, "/v1/chat/completions",
        {})  # missing messages → 400

    _, metrics_body = _get(gateway, "/metrics")
    assert metrics_body["error_count"] == 1


def test_metrics_prompt_tokens_echo(gateway):
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    _post(gateway, "/v1/chat/completions", payload)

    _, metrics_body = _get(gateway, "/metrics")
    assert metrics_body["request_count"] == 1
    assert metrics_body["prompt_tokens_total"] == 2  # len("hello world".split()) == 2


# ---------------------------------------------------------------------------
# Extended validation (#1 + #6)
# ---------------------------------------------------------------------------


def test_invalid_message_missing_role(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"content": "hi"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_message_missing_content(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_message_bad_role(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "unknown", "content": "hi"}]},
    )
    assert status == 400
    assert body["error"] == "invalid_messages"


def test_invalid_stream_type(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "stream": "yes"},
    )
    assert status == 400
    assert body["error"] == "invalid_stream"


def test_invalid_max_tokens_type(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "fifty"},
    )
    assert status == 400
    assert body["error"] == "invalid_max_tokens"


def test_invalid_max_tokens_range(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1},
    )
    assert status == 400
    assert body["error"] == "invalid_max_tokens"


def test_invalid_temperature_range(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 5.0},
    )
    assert status == 400
    assert body["error"] == "invalid_temperature"


def test_invalid_stop_type(gateway):
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "stop": 123},
    )
    assert status == 400
    assert body["error"] == "invalid_stop"


def test_model_default_omitted(gateway):
    """Omitting model should succeed."""
    status, body, _ = _post(
        gateway, "/v1/chat/completions",
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
        backend_gateway, "/v1/chat/completions",
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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


def test_routing_by_model_echo(multi_backend_gateway):
    """model: 'local' routes to echo backend and response has backend: 'local'."""
    status, body, _ = _post(
        multi_backend_gateway, "/v1/chat/completions",
        {"model": "local", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 200
    assert body["backend"] == "local"
    assert body["choices"][0]["message"]["content"].startswith("Echo:")


def test_routing_default_fallback(multi_backend_gateway):
    """Omitting model falls back to default (echo) backend."""
    status, body, _ = _post(
        multi_backend_gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 200
    assert body["backend"] == "local"


def test_routing_unknown_model_fallback(multi_backend_gateway):
    """Unknown model name falls back to default backend."""
    status, body, _ = _post(
        multi_backend_gateway, "/v1/chat/completions",
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
        multi_backend_gateway, "/v1/chat/completions",
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

    with TestClient(main.app, raise_server_exceptions=False) as _client:
        status, body, _ = _post(
            _client, "/v1/chat/completions",
            {
                "model": "remote-modal-vllm",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 200
        assert body["backend"] == "remote-modal-vllm"
    # (inline server replaced with TestClient)


def test_backend_metadata_in_response(multi_backend_gateway):
    """Every response includes a 'backend' field."""
    status, body, _ = _post(
        multi_backend_gateway, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200
    assert "backend" in body


def test_get_backends_endpoint(multi_backend_gateway):
    """GET /v1/backends lists configured backends."""
    status, body = _get(multi_backend_gateway, "/v1/backends")
    assert status == 200
    names = [b["name"] for b in body["backends"]]
    assert "local" in names
    assert "remote-modal-llama" in names
    assert body["default"] == "local"


def test_v1_models_echo_only_404(gateway):
    """GET /v1/models returns 404 when no HTTP backend is configured."""
    status, body = _get(gateway, "/v1/models")
    assert status == 404
    assert body["error"] == "not_found"


@pytest.fixture
def http_default_gateway(monkeypatch):
    """Gateway where the default backend is an HttpBackend (for /v1/models tests)."""
    from gateway.backends.echo import EchoBackend

    remote = HttpBackend(name="remote", url="http://test-backend")
    echo = EchoBackend(name="local")
    config = GatewayConfig(backends=[remote, echo], default_backend=remote)

    monkeypatch.setattr(main, "gateway_config", config)
    monkeypatch.setattr(main.settings, "backend_url", None)
    monkeypatch.setattr(main.settings, "api_key", None)
    for f in ("request_count", "error_count", "prompt_tokens_total", "completion_tokens_total"):
        monkeypatch.setattr(main.metrics, f, 0)
    monkeypatch.setattr(main.metrics, "total_latency_ms", 0.0)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


@respx.mock
def test_v1_models_proxies_to_backend(http_default_gateway):
    """GET /v1/models is proxied to the default HTTP backend."""
    models_response = {"object": "list", "data": [{"id": "gemma-4", "object": "model"}]}
    respx.get("http://test-backend/v1/models").mock(
        return_value=httpx.Response(200, json=models_response)
    )
    status, body = _get(http_default_gateway, "/v1/models")
    assert status == 200
    assert body["object"] == "list"
    assert any(m["id"] == "gemma-4" for m in body["data"])


@respx.mock
def test_v1_models_upstream_unavailable(http_default_gateway):
    """GET /v1/models returns 502 when the upstream is unreachable."""
    respx.get("http://test-backend/v1/models").mock(
        side_effect=httpx.ConnectError("refused")
    )
    status, body = _get(http_default_gateway, "/v1/models")
    assert status == 502
    assert body["error"] == "upstream_unavailable"


def test_config_loading():
    """load_config() correctly parses config.yaml and creates backend objects."""
    from gateway.config import load_config

    cfg = load_config("config.yaml")
    names = [b.name for b in cfg.all_backends]
    assert "local" in names


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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


def test_auth_no_header_401(auth_gateway):
    status, body, _ = _post(auth_gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 401
    assert body["error"] == "unauthorized"


def test_auth_wrong_key_401(auth_gateway):
    status, body, _ = _post(
        auth_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert status == 401
    assert body["error"] == "unauthorized"


def test_auth_correct_bearer_200(auth_gateway):
    status, _, _ = _post(
        auth_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert status == 200


def test_auth_api_key_scheme(auth_gateway):
    status, _, _ = _post(
        auth_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"Authorization": "Api-Key test-secret-key"},
    )
    assert status == 200


def test_no_auth_configured(gateway):
    """When no API_KEY is set, requests succeed without Authorization header."""
    status, _, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
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
        backend_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD
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
    backend_gateway.post("/v1/chat/completions", json=payload)  # consume stream

    _, metrics_body = _get(backend_gateway, "/metrics")
    assert metrics_body["prompt_tokens_total"] == 7
    assert metrics_body["completion_tokens_total"] == 3


# ---------------------------------------------------------------------------
# Latency in usage (#10)
# ---------------------------------------------------------------------------


def test_echo_latency_in_usage(gateway):
    """Echo response includes latency_ms in usage."""
    _, body, _ = _post(gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
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
    _, body, _ = _post(backend_gateway, "/v1/chat/completions", COMPLETION_PAYLOAD)
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
        multi_backend_gateway, "/v1/chat/completions",
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
        multi_backend_gateway, "/v1/chat/completions",
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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


@pytest.mark.live
def test_live_backends_endpoint(live_gateway):
    """GET /v1/backends lists all backends from config.yaml."""
    status, body = _get(live_gateway, "/v1/backends")
    assert status == 200
    names = [b["name"] for b in body["backends"]]
    assert "local" in names
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
def test_live_remote_modal_llama(live_gateway):
    """Live inference against remote-modal-llama backend."""
    status, body, _ = _post(
        live_gateway, "/v1/chat/completions",
        {
            "model": "remote-modal-llama",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10,
        },
        timeout=_live_timeout("remote-modal-llama"),
    )

    if status in (502, 504):
        pytest.skip(f"remote-modal-llama backend not reachable (status {status})")

    assert status == 200
    _assert_valid_completion(body, "remote-modal-llama")


@pytest.mark.live
def test_live_remote_modal_vllm(live_gateway):
    """Live inference against remote-modal-vllm backend."""
    status, body, _ = _post(
        live_gateway, "/v1/chat/completions",
        {
            "model": "remote-modal-vllm",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10,
        },
        timeout=_live_timeout("remote-modal-vllm"),
    )

    if status in (502, 504):
        pytest.skip(f"remote-modal-vllm backend not reachable (status {status})")

    assert status == 200
    _assert_valid_completion(body, "remote-modal-vllm")


@pytest.mark.live
def test_live_backend_timeout(live_gateway, monkeypatch):
    """A 1-second timeout on a slow backend triggers a 504."""
    assert main.gateway_config is not None
    backend = main.gateway_config.get_backend_for_model("remote-modal-vllm")
    if not isinstance(backend, HttpBackend):
        pytest.skip("remote-modal-vllm not configured as HTTP backend")

    monkeypatch.setattr(backend, "timeout", 1.0)

    status, body, _ = _post(
        live_gateway, "/v1/chat/completions",
        {"model": "remote-modal-vllm", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10.0,
    )

    if status == 502:
        pytest.skip("remote-modal-vllm backend not reachable (502)")

    assert status == 504
    assert body["error"] == "gateway_timeout"


# ---------------------------------------------------------------------------
# Prometheus metrics (#12)
# ---------------------------------------------------------------------------


@pytest.fixture
def prom_gateway(monkeypatch):
    """Echo gateway with reset in-memory metrics."""
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

    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


def test_prom_requests_total_increments(prom_gateway):
    """gateway_requests_total counter increments after a successful request."""
    from prometheus_client import REGISTRY

    labels = {
        "status_code": "200",
        "model": "unknown",
        "technique": "baseline",
        "server_profile": "default",
    }
    before = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0

    _post(prom_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD)

    after = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0
    assert after == before + 1


def test_prom_tokens_total_increments(prom_gateway):
    """gateway_tokens_total counter increments for prompt and completion tokens."""
    from prometheus_client import REGISTRY

    prompt_before = (
        REGISTRY.get_sample_value("gateway_tokens_total", {"type": "prompt"}) or 0.0
    )
    completion_before = (
        REGISTRY.get_sample_value("gateway_tokens_total", {"type": "completion"}) or 0.0
    )

    _post(
        prom_gateway, "/v1/chat/completions",
        {"model": "test", "messages": [{"role": "user", "content": "hello world"}]},
    )

    assert (
        REGISTRY.get_sample_value("gateway_tokens_total", {"type": "prompt"}) or 0.0
    ) > prompt_before
    assert (
        REGISTRY.get_sample_value("gateway_tokens_total", {"type": "completion"}) or 0.0
    ) > completion_before


def test_prom_request_duration_observed(prom_gateway):
    """gateway_request_duration_seconds histogram records one observation per request."""
    from prometheus_client import REGISTRY

    labels = {"technique": "baseline", "server_profile": "default"}
    count_before = (
        REGISTRY.get_sample_value("gateway_request_duration_seconds_count", labels)
        or 0.0
    )

    _post(prom_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD)

    count_after = (
        REGISTRY.get_sample_value("gateway_request_duration_seconds_count", labels)
        or 0.0
    )
    assert count_after == count_before + 1


def test_prom_active_requests_gauge(prom_gateway):
    """gateway_active_requests gauge returns to 0 after request completes."""
    from prometheus_client import REGISTRY

    _post(prom_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD)

    assert (REGISTRY.get_sample_value("gateway_active_requests") or 0.0) == 0.0


def test_prom_error_counter_increments(prom_gateway):
    """gateway_errors_total increments on a 400 response."""
    from prometheus_client import REGISTRY

    labels = {
        "status_code": "400",
        "technique": "baseline",
        "server_profile": "default",
    }
    before = REGISTRY.get_sample_value("gateway_errors_total", labels) or 0.0

    _post(prom_gateway, "/v1/chat/completions",
        {"model": "test", "messages": []})

    after = REGISTRY.get_sample_value("gateway_errors_total", labels) or 0.0
    assert after == before + 1


def test_prom_x_technique_header_sets_label(prom_gateway):
    """X-Technique header value appears as the technique label in metrics."""
    from prometheus_client import REGISTRY

    labels = {
        "status_code": "200",
        "model": "unknown",
        "technique": "chunked_prefill",
        "server_profile": "default",
    }
    before = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0

    _post(
        prom_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD,
        headers={"X-Technique": "chunked_prefill"},
    )

    after = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0
    assert after == before + 1


def test_prom_missing_x_technique_defaults_to_baseline(prom_gateway):
    """Requests without X-Technique header use technique=baseline."""
    from prometheus_client import REGISTRY

    labels = {
        "status_code": "200",
        "model": "unknown",
        "technique": "baseline",
        "server_profile": "default",
    }
    before = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0

    _post(prom_gateway, "/v1/chat/completions",
        COMPLETION_PAYLOAD)

    after = REGISTRY.get_sample_value("gateway_requests_total", labels) or 0.0
    assert after == before + 1


# ---------------------------------------------------------------------------
# Nginx load balancer config (#14)
# ---------------------------------------------------------------------------


def test_nginx_config_exists():
    """nginx-gateway-lb.conf is present in the project root."""
    from pathlib import Path

    conf = Path("nginx-gateway-lb.conf")
    assert conf.exists(), "nginx-gateway-lb.conf not found in project root"


def test_nginx_config_upstream_ports():
    """nginx-gateway-lb.conf defines upstreams on :8080 and :8081."""
    content = open("nginx-gateway-lb.conf").read()
    assert "127.0.0.1:8080" in content, "upstream :8080 missing"
    assert "127.0.0.1:8081" in content, "upstream :8081 missing"


def test_nginx_config_listen_port():
    """nginx-gateway-lb.conf listens on port 8780."""
    content = open("nginx-gateway-lb.conf").read()
    assert "listen 8780" in content


def test_nginx_config_sse_buffering_off():
    """nginx-gateway-lb.conf disables proxy buffering (required for SSE)."""
    content = open("nginx-gateway-lb.conf").read()
    assert "proxy_buffering" in content and "off" in content


def test_nginx_config_stub_status():
    """nginx-gateway-lb.conf exposes /nginx_status for Prometheus exporter."""
    content = open("nginx-gateway-lb.conf").read()
    assert "stub_status" in content
    assert "/nginx_status" in content


def test_nginx_config_rootless():
    """nginx-gateway-lb.conf uses /tmp paths so it runs without root."""
    content = open("nginx-gateway-lb.conf").read()
    assert "/tmp/" in content


def test_prometheus_config_scrapes_both_gateways():
    """prometheus.yml defines scrape jobs for both gateway instances."""
    import yaml

    with open("monitoring/prometheus.yml") as f:
        cfg = yaml.safe_load(f)
    jobs = {job["job_name"] for job in cfg["scrape_configs"]}
    assert "gateway" in jobs, "gateway job missing"
    assert "gateway2" in jobs, "gateway2 job missing"


def test_prometheus_config_scrapes_nginx():
    """prometheus.yml defines a scrape job for the nginx exporter."""
    import yaml

    with open("monitoring/prometheus.yml") as f:
        cfg = yaml.safe_load(f)
    jobs = {job["job_name"] for job in cfg["scrape_configs"]}
    assert "nginx" in jobs, "nginx exporter job missing"


def test_docker_compose_nginx_exporter():
    """monitoring/docker-compose.yml includes the nginx-prometheus-exporter service."""
    import yaml

    with open("monitoring/docker-compose.yml") as f:
        cfg = yaml.safe_load(f)
    assert "nginx-exporter" in cfg["services"], "nginx-exporter service missing"
    cmd = " ".join(cfg["services"]["nginx-exporter"].get("command", []))
    assert "nginx_status" in cmd, "nginx_status scrape URI missing from exporter command"
