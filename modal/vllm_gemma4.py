"""Modal app — vLLM serving google/gemma-4-e2b-it, standard flags (one warm container).

Deploy:  modal deploy modal/vllm_gemma4.py
         bash scripts/deploy_modal_vllm.sh standard

URL:     printed after deploy; add to config.yaml as backend 'modal-gemma4-standard'

Requires a Modal secret named 'huggingface-secret' with HF_TOKEN set:
  modal secret create huggingface-secret HF_TOKEN=hf_...

GPU note: A10G has 24 GB VRAM. Gemma 4 E2B native context is 128k, which needs
~60 GB; max_model_len is capped at 8192 for A10G memory safety.
"""

from __future__ import annotations

import json
import subprocess

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-e2b-it"
SERVED_MODEL_NAME = "gemma-4-e2b-it"
GPU_TYPE = "A10G"
VLLM_PORT = 8000
MINUTES = 60

MAX_MODEL_LEN = 8192  # capped for 24 GB VRAM; raise to 16384 if headroom allows
GPU_MEMORY_UTILIZATION = 0.90

# ---------------------------------------------------------------------------
# Image — CUDA 12.9 + latest vLLM from OpenAI
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")  # step 1: vllm (pulls transformers<5)
    .uv_pip_install("transformers==5.5.0")  # step 2: override to transformers 5.5.0
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# ---------------------------------------------------------------------------
# Persistent volumes — weights survive container restarts
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gemma4-vllm-cache", create_if_missing=True)

app = modal.App("vllm-gemma4-standard")


# ---------------------------------------------------------------------------
# vLLM server — scale-to-zero (min_containers=0 saves GPU cost when idle)
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    timeout=20 * MINUTES,  # covers image build + weight download on cold start
    min_containers=0,  # scale to zero when idle; cold start ~2-3 min (weights cached in volume)
    max_containers=1,  # hard cap — no autoscaling; cost-controlled for testing
    scaledown_window=5 * MINUTES,  # scale down after 5 min of no requests
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=16)  # up to 16 in-flight requests per container
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Start vLLM OpenAI-compatible server with standard flags."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0, "audio": 0}),
        "--async-scheduling",
        # Standard profile: no chunked prefill, no prefix caching
        "--max-num-seqs",
        "64",  # explicit to avoid vLLM's default of 256, which risks sampler warmup OOM
    ]
    print("Starting vLLM (standard):", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url()
    print(f"\nvLLM standard URL: {url}")
    print("Add to config.yaml:")
    print("  - name: modal-gemma4-standard")
    print("    type: http")
    print(f"    url: {url}")
    print("    timeout: 120")
    print(f"    model: {SERVED_MODEL_NAME}")
