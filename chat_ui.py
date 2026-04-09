"""Gradio chat UI for the inference gateway.

Tabs:
  💬 Chat   — streaming chat directly against the gateway's /v1/chat/completions
  🤖 CrewAI — editable topic + technique runs the Researcher→Writer crew

Usage:
    uv run --group crew python chat_ui.py
    # then open http://localhost:7860

Environment variables (inherit from .env / shell):
    GATEWAY_OPENAI_BASE   default gateway base URL (default: http://127.0.0.1:8780/v1 via LB)
    MODEL_NAME            default model (default: modal-gemma4-optimized)
    API_KEY / OPENAI_API_KEY  gateway API key (omit if auth is disabled)
    CHAT_UI_PORT          Gradio port (default: 7860)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from crew import GATEWAY, MODEL_NAME, _API_KEY, build_crew  # noqa: E402

# ---------------------------------------------------------------------------
# Gateway options
# ---------------------------------------------------------------------------

_GATEWAY_OPTIONS = [
    ("Load Balancer  :8780", "http://127.0.0.1:8780/v1"),
    ("Gateway 1  :8080",     "http://127.0.0.1:8080/v1"),
    ("Gateway 2  :8081",     "http://127.0.0.1:8081/v1"),
]
_DEFAULT_GATEWAY = next(
    (v for _, v in _GATEWAY_OPTIONS if v == GATEWAY),
    _GATEWAY_OPTIONS[0][1],
)

TECHNIQUES = ["baseline", "chunked_prefill", "prefix_caching", "beam_search"]

_HEADERS_BASE = {"Content-Type": "application/json"}
if _API_KEY and _API_KEY != "dummy":
    _HEADERS_BASE["Authorization"] = f"Bearer {_API_KEY}"


# ---------------------------------------------------------------------------
# Model list helpers
# ---------------------------------------------------------------------------


def _fetch_models(gateway_url: str) -> list[str]:
    """Return model IDs from /v1/models, falling back to backend names from /v1/backends."""
    base = gateway_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    # Try /v1/models first (works when a real vLLM backend is configured)
    try:
        resp = httpx.get(f"{base}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            models = [m["id"] for m in data if isinstance(m, dict)]
            if models:
                return models
    except Exception:
        pass

    # Fall back to gateway backend names from /v1/backends
    try:
        resp = httpx.get(f"{base}/v1/backends", timeout=5.0)
        if resp.status_code == 200:
            backends = [b["name"] for b in resp.json().get("backends", [])]
            if backends:
                return backends
    except Exception:
        pass

    return [MODEL_NAME]


def _refresh_models(gateway_url: str) -> gr.Dropdown:
    """Event handler: refresh the model dropdown when the gateway changes."""
    models = _fetch_models(gateway_url)
    # Prefer the configured default if it's in the list, otherwise first entry
    default = MODEL_NAME if MODEL_NAME in models else models[0]
    return gr.Dropdown(choices=models, value=default)


# ---------------------------------------------------------------------------
# Chat tab — streaming completions
# ---------------------------------------------------------------------------


def _chat_stream(message: str, history: list, technique: str, gateway_url: str, model: str):
    """Generator: stream tokens from /v1/chat/completions, yield growing text."""
    base = gateway_url.rstrip("/")
    completions_url = base + "/chat/completions"

    messages = []
    for entry in history:
        if isinstance(entry, dict):
            messages.append(entry)
        else:
            user_msg, assistant_msg = entry
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    partial = ""
    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                completions_url,
                json={"model": model, "messages": messages, "stream": True},
                headers={**_HEADERS_BASE, "X-Technique": technique},
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        partial += delta
                        yield partial
                    except (KeyError, json.JSONDecodeError):
                        continue
    except httpx.RequestError as exc:
        yield f"⚠️ Gateway unreachable: {exc}\n\nMake sure a gateway instance is running."
    except httpx.HTTPStatusError as exc:
        yield f"⚠️ Gateway returned {exc.response.status_code}: {exc.response.text}"


# ---------------------------------------------------------------------------
# CrewAI tab — Researcher → Writer
# ---------------------------------------------------------------------------


def _run_crew(topic: str, technique: str, gateway_url: str, model: str, progress=gr.Progress()) -> str:
    """Run the Researcher→Writer crew and return the final output."""
    if not topic.strip():
        return "⚠️ Please enter a topic."
    progress(0.0, desc="Building crew…")
    try:
        crew = build_crew(technique=technique, topic=topic, gateway=gateway_url, model=model)
        progress(0.2, desc="Running Researcher agent…")
        result = crew.kickoff()
        progress(1.0, desc="Done.")
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Crew error: {exc}"


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Inference Gateway") as demo:
    gr.Markdown("# 🚀 Inference Gateway")

    # ── Global controls (shared across tabs) ────────────────────────────────
    with gr.Row():
        gateway_selector = gr.Dropdown(
            choices=_GATEWAY_OPTIONS,
            value=_DEFAULT_GATEWAY,
            label="Gateway",
            info="Route requests through the Nginx LB or hit a gateway instance directly.",
            scale=2,
        )
        model_selector = gr.Dropdown(
            choices=_fetch_models(_DEFAULT_GATEWAY),
            value=MODEL_NAME if MODEL_NAME in _fetch_models(_DEFAULT_GATEWAY) else None,
            label="Model",
            info="Fetched from /v1/models (or /v1/backends in echo mode).",
            allow_custom_value=True,
            scale=2,
        )
        refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

    gateway_selector.change(fn=_refresh_models, inputs=gateway_selector, outputs=model_selector)
    refresh_btn.click(fn=_refresh_models, inputs=gateway_selector, outputs=model_selector)

    # ── Chat tab ─────────────────────────────────────────────────────────────
    with gr.Tab("💬 Chat"):
        technique_chat = gr.Dropdown(
            choices=TECHNIQUES,
            value="baseline",
            label="X-Technique",
            info="Sets the X-Technique header — visible as a label in Prometheus metrics.",
        )
        gr.ChatInterface(
            fn=_chat_stream,
            additional_inputs=[technique_chat, gateway_selector, model_selector],
            fill_height=False,
            chatbot=gr.Chatbot(height=340),
        )

    # ── CrewAI tab ───────────────────────────────────────────────────────────
    with gr.Tab("🤖 CrewAI — Researcher → Writer"):
        gr.Markdown(
            "Enter a research topic. The **Researcher** agent collects 3–5 bullet points; "
            "the **Writer** agent turns them into a short paragraph (≤120 words)."
        )
        with gr.Row():
            topic_input = gr.Textbox(
                label="Research topic",
                value="benefits of chunked prefill in LLM serving",
                placeholder="e.g. vLLM speculative decoding, prefix caching for RAG…",
                scale=3,
            )
            technique_crew = gr.Dropdown(
                choices=TECHNIQUES,
                value="baseline",
                label="X-Technique",
                scale=1,
            )
        run_btn = gr.Button("▶ Run Crew", variant="primary")
        crew_output = gr.Textbox(
            label="Crew output",
            lines=12,
            interactive=False,
            placeholder="Crew output will appear here…",
        )
        run_btn.click(
            fn=_run_crew,
            inputs=[topic_input, technique_crew, gateway_selector, model_selector],
            outputs=crew_output,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("CHAT_UI_PORT", "7860")),
        show_error=True,
        theme=gr.themes.Soft(),
    )
