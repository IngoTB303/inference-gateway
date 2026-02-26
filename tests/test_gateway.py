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
    """Echo-mode gateway on a free port."""
    monkeypatch.setattr(main.settings, "backend_url", None)
    for f in ("request_count", "error_count", "prompt_tokens_total", "completion_tokens_total"):
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
    """Backend-proxy gateway pointing at the test mock URL."""
    monkeypatch.setattr(main.settings, "backend_url", "http://test-backend")
    for f in ("request_count", "error_count", "prompt_tokens_total", "completion_tokens_total"):
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


def test_models_list(gateway):
    status, body = _get(f"{gateway}/v1/models")
    assert status == 200
    assert isinstance(body.get("data"), list)
    assert len(body["data"]) > 0


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
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_body)
    )

    status, body, headers = _post(f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 200
    assert "id" in body
    assert headers.get("X-Request-ID") is not None


@respx.mock
def test_backend_5xx(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )

    status, body, _ = _post(f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 502
    assert body["error"] == "backend_error"


@respx.mock
def test_backend_timeout(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    status, body, _ = _post(f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
    assert status == 504
    assert body["error"] == "gateway_timeout"


@respx.mock
def test_backend_connection_error(backend_gateway):
    respx.post("http://test-backend/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )

    status, body, _ = _post(f"{backend_gateway}/v1/chat/completions", COMPLETION_PAYLOAD)
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
