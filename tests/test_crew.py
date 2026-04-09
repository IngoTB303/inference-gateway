"""Unit tests for crew.py — gateway health check, vLLM wait, crew builder.

These tests require the crew dependency group:
    uv sync --group crew
    uv run pytest tests/test_crew.py

They are skipped automatically when crewai/litellm are not installed.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

# Skip the entire module if crew dependencies are not installed
pytest.importorskip("litellm", reason="crew group not installed — run: uv sync --group crew")
pytest.importorskip("crewai", reason="crew group not installed — run: uv sync --group crew")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_stub_server(handler_class) -> tuple[HTTPServer, str]:
    """Start an HTTPServer on a free port; return (server, base_url)."""
    server = HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Gateway health check (_check_gateway_health)
# ---------------------------------------------------------------------------


class _HealthOK(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def log_message(self, *args):
        pass


class _HealthMisconfigured(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"status": "misconfigured", "warning": "upstream refused"}).encode()
        )

    def log_message(self, *args):
        pass


class _Health502(BaseHTTPRequestHandler):
    """Simulates Nginx returning 502 when no gateway instance is running."""

    def do_GET(self):
        self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"error": "backend_unavailable", "detail": "no upstream"}).encode()
        )

    def log_message(self, *args):
        pass


def test_check_gateway_health_ok(monkeypatch):
    """_check_gateway_health does not exit when gateway reports status=ok."""
    server, base_url = _start_stub_server(_HealthOK)
    try:
        monkeypatch.setattr("crew.GATEWAY", base_url + "/v1")
        import crew
        crew._check_gateway_health()  # should not raise
    finally:
        server.shutdown()


def test_check_gateway_health_misconfigured(monkeypatch):
    """_check_gateway_health calls sys.exit(2) when gateway reports misconfigured."""
    server, base_url = _start_stub_server(_HealthMisconfigured)
    try:
        monkeypatch.setattr("crew.GATEWAY", base_url + "/v1")
        import crew
        with pytest.raises(SystemExit) as exc_info:
            crew._check_gateway_health()
        assert exc_info.value.code == 2
    finally:
        server.shutdown()


def test_check_gateway_health_unreachable(monkeypatch):
    """_check_gateway_health does not exit when the gateway is simply unreachable."""
    monkeypatch.setattr("crew.GATEWAY", "http://127.0.0.1:19999/v1")
    import crew
    crew._check_gateway_health()  # should return silently, not exit


def test_check_gateway_health_502_exits(monkeypatch):
    """_check_gateway_health exits with code 2 when LB returns 502 (no gateway running)."""
    server, base_url = _start_stub_server(_Health502)
    try:
        monkeypatch.setattr("crew.GATEWAY", base_url + "/v1")
        import crew
        with pytest.raises(SystemExit) as exc_info:
            crew._check_gateway_health()
        assert exc_info.value.code == 2
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# vLLM readiness wait (_wait_for_vllm)
# ---------------------------------------------------------------------------


class _ModelsReady(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"object": "list", "data": [{"id": "gemma-4", "object": "model"}]}).encode()
        )

    def log_message(self, *args):
        pass


class _ModelsEmpty(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": []}).encode())

    def log_message(self, *args):
        pass


def test_wait_for_vllm_skip_when_zero(monkeypatch):
    """_wait_for_vllm returns immediately when CREW_VLLM_WAIT_S=0."""
    monkeypatch.setenv("CREW_VLLM_WAIT_S", "0")
    import crew
    crew._wait_for_vllm()  # should return without doing anything


def test_wait_for_vllm_returns_when_ready(monkeypatch):
    """_wait_for_vllm returns as soon as /v1/models responds with data."""
    server, base_url = _start_stub_server(_ModelsReady)
    try:
        monkeypatch.setattr("crew.GATEWAY", base_url + "/v1")
        monkeypatch.setenv("CREW_VLLM_WAIT_S", "10")
        monkeypatch.setenv("CREW_VLLM_POLL_S", "1")
        import crew
        crew._wait_for_vllm()  # should return without timeout
    finally:
        server.shutdown()


def test_wait_for_vllm_times_out(monkeypatch):
    """_wait_for_vllm calls sys.exit(3) when /v1/models never returns data."""
    server, base_url = _start_stub_server(_ModelsEmpty)
    try:
        monkeypatch.setattr("crew.GATEWAY", base_url + "/v1")
        monkeypatch.setenv("CREW_VLLM_WAIT_S", "2")
        monkeypatch.setenv("CREW_VLLM_POLL_S", "1")
        import crew
        with pytest.raises(SystemExit) as exc_info:
            crew._wait_for_vllm()
        assert exc_info.value.code == 3
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# build_crew
# ---------------------------------------------------------------------------


def test_build_crew_returns_crew_object(monkeypatch):
    """build_crew() returns a Crew with two agents and two tasks."""
    pytest.importorskip("crewai")
    import crew
    from crewai import Crew

    c = crew.build_crew(technique="baseline", topic="test topic")
    assert isinstance(c, Crew)
    assert len(c.agents) == 2
    assert len(c.tasks) == 2


def test_build_crew_agent_roles(monkeypatch):
    """Crew has a Researcher and a Writer agent."""
    pytest.importorskip("crewai")
    import crew

    c = crew.build_crew(technique="baseline", topic="test topic")
    roles = {a.role for a in c.agents}
    assert "Researcher" in roles
    assert "Writer" in roles


def test_build_crew_topic_in_task_description(monkeypatch):
    """The custom topic appears in the Researcher task description."""
    pytest.importorskip("crewai")
    import crew

    topic = "speculative decoding benefits"
    c = crew.build_crew(technique="baseline", topic=topic)
    # The first task belongs to the Researcher
    assert topic in c.tasks[0].description


def test_build_crew_sets_technique_header(monkeypatch):
    """build_crew sets litellm.headers with the X-Technique value."""
    pytest.importorskip("crewai")
    import litellm
    import crew

    crew.build_crew(technique="chunked_prefill", topic="test")
    assert litellm.headers.get("X-Technique") == "chunked_prefill"


def test_build_crew_beam_search_extra_body(monkeypatch):
    """beam_search technique sets litellm.extra_body with use_beam_search."""
    pytest.importorskip("crewai")
    import litellm
    import crew

    crew.build_crew(technique="beam_search", topic="test")
    assert litellm.extra_body.get("use_beam_search") is True


def test_build_crew_default_topic():
    """build_crew uses a sensible default topic when none is supplied."""
    pytest.importorskip("crewai")
    import crew

    c = crew.build_crew()
    assert "chunked prefill" in c.tasks[0].description.lower()


# ---------------------------------------------------------------------------
# Gateway root URL helper
# ---------------------------------------------------------------------------


def test_gateway_root_strips_v1():
    """_gateway_root removes the /v1 suffix from GATEWAY."""
    import crew

    original = crew.GATEWAY
    try:
        crew.GATEWAY = "http://127.0.0.1:8780/v1"
        assert crew._gateway_root() == "http://127.0.0.1:8780"
    finally:
        crew.GATEWAY = original


def test_gateway_root_no_v1():
    """_gateway_root works when GATEWAY has no /v1 suffix."""
    import crew

    original = crew.GATEWAY
    try:
        crew.GATEWAY = "http://127.0.0.1:8080"
        assert crew._gateway_root() == "http://127.0.0.1:8080"
    finally:
        crew.GATEWAY = original
