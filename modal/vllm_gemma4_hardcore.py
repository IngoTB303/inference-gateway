"""Modal app — vLLM serving google/gemma-4-e2b-it, hardcore-optimized flags.

Deploy:  modal deploy modal/vllm_gemma4_hardcore.py
         bash scripts/deploy_modal_vllm.sh hardcore

URL:     printed after deploy; add to config.yaml as backend 'modal-gemma4-hardcore'

Requires a Modal secret named 'huggingface-secret' with HF_TOKEN set:
  modal secret create huggingface-secret HF_TOKEN=hf_...

Optimizations applied on top of the 'optimized' profile:
  --kv-cache-dtype fp8          quantises KV cache entries to FP8, cutting KV
                                memory by ~50% and allowing more sequences in
                                flight on A10G's 24 GB; stable for Gemma 4 per
                                vLLM 0.19 release notes.
  --max-num-batched-tokens 4096 larger chunks than the 'optimized' profile (512)
                                improve single-request decode throughput while
                                the smaller KV footprint keeps VRAM safe.
  --max-num-seqs 128            double the 'optimized' profile; fp8 KV cache
                                creates enough headroom for this on A10G.

Gemma 4 note: heterogeneous attention head dims (256 / 512) force vLLM to use
the Triton attention backend instead of FlashAttention — this is a model
architecture constraint, not a flag issue.  Expect ~20-40 tok/s on A10G.
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

MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.88  # slightly lower than other profiles; fp8 needs headroom

CHUNKED_PREFILL_TOKENS = 4096  # much larger than optimised profile → better throughput
MAX_NUM_SEQS = 128  # fp8 KV cache halves per-seq VRAM cost → safe to double

# ---------------------------------------------------------------------------
# Image — CUDA 12.9 + vLLM 0.19.0 (guide-prescribed install for Gemma 4)
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ---------------------------------------------------------------------------
# Persistent volumes — weights shared with standard and optimized apps
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gemma4-vllm-cache", create_if_missing=True)

app = modal.App("vllm-gemma4-hardcore")


# ---------------------------------------------------------------------------
# vLLM server — one warm container
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    timeout=20 * MINUTES,
    min_containers=1,
    max_containers=1,
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Start vLLM with all safe throughput optimisations for Gemma 4 on A10G."""
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
        # --- fp8 KV cache (core of the hardcore profile) ---
        "--kv-cache-dtype",
        "fp8",
        # --- Chunked prefill + prefix caching (from optimized profile) ---
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(CHUNKED_PREFILL_TOKENS),
        "--enable-prefix-caching",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]
    print("Starting vLLM (hardcore):", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url()
    print(f"\nvLLM hardcore URL: {url}")
    print("Add to config.yaml:")
    print("  - name: modal-gemma4-hardcore")
    print("    type: http")
    print(f"    url: {url}")
    print("    timeout: 120")
    print(f"    model: {SERVED_MODEL_NAME}")
