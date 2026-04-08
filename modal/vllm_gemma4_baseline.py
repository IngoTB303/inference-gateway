"""Modal app — vLLM serving google/gemma-4-e2b-it (baseline flags).

Deploy:  modal deploy modal/vllm_gemma4_baseline.py
URL:     printed after deploy; add to config.yaml as backend 'modal-baseline'

Requires a Modal secret named 'huggingface' with HF_TOKEN set:
  modal secret create huggingface HF_TOKEN=hf_...
"""
from __future__ import annotations

import subprocess

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-e2b-it"
SERVED_MODEL_NAME = "gemma-4-e2b-it"   # name clients send in the "model" field
GPU_TYPE = "A10G"
VLLM_PORT = 8000
MINUTES = 60

# A10G has 24 GB VRAM. 128K native ctx requires ~60 GB; cap at 8192 for safety.
# Raise to 16384 if you need longer contexts and memory allows.
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90

# ---------------------------------------------------------------------------
# Image — CUDA 12.4 + latest vLLM from PyPI
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])                          # clear default CUDA entrypoint
    .pip_install("vllm", "huggingface-hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Persistent volumes — survive container restarts; avoid re-downloading weights
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gemma4-vllm-cache", create_if_missing=True)

app = modal.App("vllm-gemma4-e2b-baseline")

# ---------------------------------------------------------------------------
# vLLM server function
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    timeout=20 * MINUTES,            # covers image build + model download on first run
    scaledown_window=15 * MINUTES,   # keep warm between requests
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=16)     # up to 16 in-flight requests per container
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Start vLLM OpenAI-compatible server — baseline flags only."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--trust-remote-code",
        # Baseline: no chunked prefill, no prefix caching, no speculative decoding
    ]
    print("Starting vLLM (baseline):", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url()
    print(f"\nvLLM baseline URL: {url}")
    print(f"  Add to config.yaml:")
    print(f"    - name: modal-baseline")
    print(f"      type: http")
    print(f"      url: {url}")
    print(f"      timeout: 120")
    print(f"      model: {SERVED_MODEL_NAME}")
