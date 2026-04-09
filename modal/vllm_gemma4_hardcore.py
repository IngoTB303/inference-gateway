"""Modal app — vLLM serving google/gemma-4-e2b-it, hardcore-optimized flags.

Deploy:  modal deploy modal/vllm_gemma4_hardcore.py
         bash scripts/deploy_modal_vllm.sh hardcore

URL:     printed after deploy; add to config.yaml as backend 'modal-gemma4-hardcore'

Requires a Modal secret named 'huggingface-secret' with HF_TOKEN set:
  modal secret create huggingface-secret HF_TOKEN=hf_...

Optimizations applied on top of the 'optimized' profile:
  --max-num-batched-tokens 8192   16× larger chunks than the 'optimized' profile
                                  (512 tokens); allows much longer prompts to be
                                  prefilled in a single pass, reducing TTFT for
                                  large-context requests.
  --max-num-seqs 256              4× the 'optimized' profile; Gemma 4 E2B weights
                                  (~4 GB bfloat16) leave ample KV headroom on A10G
                                  at 95% memory utilisation.
  --gpu-memory-utilization 0.95   push more of the 24 GB VRAM into KV cache;
                                  the profile run stays below OOM on A10G E2B.

Note on FP8 KV cache: --kv-cache-dtype fp8 requires the fp8e4nv format which is
only supported on Hopper (H100/H200, compute capability ≥ 9.0).  A10G is Ampere
(cc 8.6) — fp8 KV cache is NOT used here.

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
GPU_MEMORY_UTILIZATION = 0.95  # push VRAM harder — safe for E2B on A10G

CHUNKED_PREFILL_TOKENS = 8192  # max batch budget: single-pass prefill for long prompts
MAX_NUM_SEQS = 256  # 4× the optimized profile; E2B weights leave KV headroom

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
    """Start vLLM with maximum safe throughput flags for Gemma 4 on A10G."""
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
        # --- Chunked prefill + prefix caching (from optimized profile) ---
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(CHUNKED_PREFILL_TOKENS),
        "--enable-prefix-caching",
        # --- Increased concurrency (core of the hardcore profile) ---
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
