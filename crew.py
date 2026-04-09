"""CrewAI agentic client — Researcher → Writer crew routed through the inference gateway.

Usage:
    python crew.py [--topic "..."] [--technique baseline]

Environment variables (can be set in .env):
    GATEWAY_OPENAI_BASE     Full gateway base URL including /v1
                            Defaults to LB at http://127.0.0.1:8780/v1
    GATEWAY_USE_LOAD_BALANCER  Set to false to bypass Nginx and hit :8080 directly
    MODEL_NAME              Model to request (default: local)
    OPENAI_API_KEY          API key forwarded to gateway (default: dummy)
    API_KEY                 Alternative env var for the gateway API key
    CREW_VLLM_WAIT_S        Seconds to wait for vLLM via /v1/models (0 to skip, default: 0)
    CREW_LLM_STREAM         Enable streaming (default: true)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import litellm  # noqa: E402
from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.llm import LLM  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "modal-gemma4-optimized")

_use_lb = os.environ.get("GATEWAY_USE_LOAD_BALANCER", "true").strip().lower() not in (
    "0", "false", "no", "off",
)
GATEWAY: str = (
    os.environ.get("GATEWAY_OPENAI_BASE")
    or os.environ.get("OPENAI_API_BASE")
    or (
        f"http://{os.environ.get('GATEWAY_LB_HOST', '127.0.0.1')}:{os.environ.get('GATEWAY_LB_PORT', '8780')}/v1"
        if _use_lb
        else f"http://{os.environ.get('GATEWAY_HOST', '127.0.0.1')}:{os.environ.get('GATEWAY_PORT', '8080')}/v1"
    )
)
if not GATEWAY.rstrip("/").endswith("/v1"):
    GATEWAY = GATEWAY.rstrip("/") + "/v1"

_API_KEY: str = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "dummy")


def _gateway_root() -> str:
    g = GATEWAY.rstrip("/")
    return g[: -len("/v1")] if g.endswith("/v1") else g


# ---------------------------------------------------------------------------
# OpenTelemetry (optional — enabled when OTEL_TRACES_EXPORTER=otlp)
# ---------------------------------------------------------------------------


def _setup_otel() -> "Any | None":
    """Configure TracerProvider. Returns the provider, or None if disabled/unavailable."""
    if os.environ.get("OTEL_TRACES_EXPORTER", "none").lower() != "otlp":
        return None
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry import trace
    except ImportError:
        print("opentelemetry packages not installed; tracing disabled.", file=sys.stderr)
        return None
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    resource = Resource({SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "crew")})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def _otel_inject_headers(headers: dict) -> None:
    """Inject current W3C trace context into headers. No-op if OTel is unavailable."""
    try:
        from opentelemetry.propagate import inject
        inject(headers)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# vLLM readiness check
# ---------------------------------------------------------------------------


def _wait_for_vllm() -> None:
    """Poll GET /v1/models until vLLM reports a loaded model, or time out."""
    raw = os.environ.get("CREW_VLLM_WAIT_S", "0").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    try:
        max_wait = float(raw)
    except ValueError:
        max_wait = 240.0
    if max_wait <= 0:
        return

    interval = max(2.0, min(float(os.environ.get("CREW_VLLM_POLL_S", "8") or "8"), 60.0))
    models_url = GATEWAY.rstrip("/") + "/models"
    deadline = time.monotonic() + max_wait
    last_note = ""

    print(
        f"Waiting for vLLM via gateway (GET /v1/models, up to {int(max_wait)}s; "
        "CREW_VLLM_WAIT_S=0 to skip)…",
        file=sys.stderr,
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(models_url, method="GET"), timeout=90
            ) as resp:
                payload = json.loads(resp.read().decode())
                data = payload.get("data")
                if isinstance(data, list) and data:
                    print("vLLM is ready — running crew.\n", file=sys.stderr)
                    return
                last_note = "empty /v1/models data"
        except urllib.error.HTTPError as e:
            last_note = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_note = str(e)
        except (json.JSONDecodeError, TypeError, ValueError):
            last_note = "invalid JSON from /v1/models"
        time.sleep(interval)

    print(
        f"\nTimed out waiting for vLLM ({int(max_wait)}s). Last status: {last_note!r}\n"
        "Set CREW_VLLM_WAIT_S=0 to skip this check, or raise the value.\n",
        file=sys.stderr,
    )
    sys.exit(3)


def _check_gateway_health() -> None:
    """Check gateway reachability and upstream configuration before starting the crew."""
    url = _gateway_root() + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (502, 503, 504):
            # Nginx is up but no gateway instance is listening
            print(
                f"\n✗ Load balancer returned HTTP {exc.code} — no gateway instance is running.\n"
                "  Start one (or two) gateway instances first:\n"
                "    Terminal 1:  PORT=8080 GATEWAY_METRICS_PORT=9101 uv run python main.py\n"
                "    Terminal 2:  PORT=8081 GATEWAY_METRICS_PORT=9102 uv run python main.py\n",
                file=sys.stderr,
            )
            sys.exit(2)
        return  # other HTTP errors — proceed and let the crew surface them
    except Exception:
        return  # gateway not running at all — proceed, crew will fail with a clear message

    if body.get("status") == "misconfigured":
        warn = body.get("warning", "gateway reports upstream is misconfigured")
        print(f"\n✗ Gateway /health warning: {warn}", file=sys.stderr)
        print("  Check GATEWAY_OPENAI_BASE and that the backend is reachable.\n", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Crew builder
# ---------------------------------------------------------------------------


def build_crew(
    technique: str = "baseline",
    topic: str = "benefits of chunked prefill in LLM serving",
    gateway: str | None = None,
    model: str | None = None,
) -> Crew:
    """Build and return a Researcher → Writer crew targeting the gateway.

    Args:
        technique:  X-Technique header value (Prometheus label).
        topic:      Research subject for the Researcher agent.
        gateway:    Override GATEWAY base URL (defaults to module-level GATEWAY).
        model:      Override MODEL_NAME (defaults to module-level MODEL_NAME).
    """
    _gateway = gateway or GATEWAY
    _model = model or MODEL_NAME

    litellm.drop_params = True
    litellm.headers = {"X-Technique": technique}
    if technique == "beam_search":
        litellm.extra_body = {"use_beam_search": True, "best_of": 4}
    else:
        litellm.extra_body = {}

    llm_kwargs: dict[str, Any] = {
        "model": f"openai/{_model}",
        "api_key": _API_KEY,
        "base_url": _gateway,
        "temperature": 0.2,
    }
    if os.environ.get("CREW_LLM_STREAM", "true").strip().lower() not in ("0", "false", "no", "off"):
        llm_kwargs["stream"] = True
    llm = LLM(**llm_kwargs)

    researcher = Agent(
        role="Researcher",
        goal="Collect concise facts for a short brief.",
        backstory="You summarize sources clearly in 3–5 bullet points.",
        llm=llm,
        verbose=True,
    )
    writer = Agent(
        role="Writer",
        goal="Turn research into a tight paragraph.",
        backstory="You write clear prose under 120 words.",
        llm=llm,
        verbose=True,
    )

    task_research = Task(
        description=(
            f"Topic: {topic}. "
            "List the key ideas only — no preamble, no conclusion."
        ),
        expected_output="Bullet list of 3–5 concise points.",
        agent=researcher,
    )
    task_write = Task(
        description="Using only the research above, write one short paragraph for a student.",
        expected_output="One paragraph, maximum 120 words.",
        agent=writer,
        context=[task_research],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task_research, task_write],
        process=Process.sequential,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Researcher→Writer CrewAI crew through the inference gateway."
    )
    parser.add_argument(
        "--technique",
        default="baseline",
        help="X-Technique label sent on every LLM request (default: baseline).",
    )
    parser.add_argument(
        "--topic",
        default="benefits of chunked prefill in LLM serving",
        help="Research topic passed to the Researcher agent.",
    )
    args = parser.parse_args()

    _check_gateway_health()
    _wait_for_vllm()

    print(f"Gateway : {GATEWAY}", file=sys.stderr)
    print(f"Model   : {MODEL_NAME}", file=sys.stderr)
    print(f"Topic   : {args.topic}", file=sys.stderr)
    print(f"Technique: {args.technique}\n", file=sys.stderr)

    _otel_provider = _setup_otel()
    crew = build_crew(technique=args.technique, topic=args.topic)

    # Wrap kickoff in a span and propagate W3C trace context to the gateway
    if _otel_provider is not None:
        import contextlib
        from opentelemetry import trace as _otel_trace
        _span_cm = _otel_trace.get_tracer("crew").start_as_current_span("crew.run")
    else:
        import contextlib
        _span_cm = contextlib.nullcontext()

    with _span_cm as span:
        if span is not None:
            span.set_attribute("llm.technique", args.technique)
        _otel_inject_headers(litellm.headers)
        try:
            result = crew.kickoff()
        except Exception as exc:
            print(f"\nCrew failed: {exc}", file=sys.stderr)
            if _otel_provider is not None:
                _otel_provider.shutdown()
            sys.exit(1)

    if _otel_provider is not None:
        _otel_provider.shutdown()

    print("\n--- Crew output ---")
    print(result)


if __name__ == "__main__":
    main()
