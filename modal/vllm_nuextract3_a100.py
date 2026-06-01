"""Modal app — vLLM serving numind/NuExtract3 on A100 GPU.

Deploy:  modal deploy modal/vllm_nuextract3_a100.py

URL:     printed after deploy

Requires a Modal secret named 'huggingface-secret' with HF_TOKEN set:
  modal secret create huggingface-secret HF_TOKEN=hf_...

Model: NuExtract3 — 4B multimodal vision-language model for structured extraction
       and document-to-Markdown conversion. Based on Qwen3.5-4B.

KV cache optimizations (A100, 80 GB VRAM):
  --gpu-memory-utilization 0.95   Model is ~10 GB in BF16; leaves ~66 GB for KV
                                  cache — enough for multiple concurrent 128K
                                  context requests.
  --kv-cache-dtype fp8_e5m2       Halves KV cache memory (supported on sm80+/A100),
                                  effectively doubling concurrent capacity or
                                  context length headroom.
  --enable-prefix-caching         Reuses KV cache across requests sharing common
                                  prefixes (e.g. same template/instructions).
  --enable-chunked-prefill        Efficient prefill for long multimodal inputs.

Speculative decoding:
  --speculative-config            Uses Qwen3 Multi-Token Prediction (MTP) for ~2×
                                  faster decoding with 2 speculative tokens.
"""

from __future__ import annotations

import json
import subprocess

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "numind/NuExtract3"
SERVED_MODEL_NAME = "NuExtract3"
GPU_TYPE = "A100"
VLLM_PORT = 8000
MINUTES = 60

MAX_MODEL_LEN = 131072  # Full 128K context window
GPU_MEMORY_UTILIZATION = 0.95  # 4B model leaves ~66 GB for KV cache on 80 GB A100

CHUNKED_PREFILL_TOKENS = 16384  # large prefill budget for multimodal inputs
MAX_NUM_SEQS = 64  # conservative concurrency for 128K context

# ---------------------------------------------------------------------------
# Image — latest vLLM OpenAI-compatible server (includes CUDA + vLLM)
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ---------------------------------------------------------------------------
# Persistent volumes — model weights cache
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("nuextract3-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("nuextract3-vllm-cache", create_if_missing=True)

app = modal.App("vllm-nuextract3-a100")


# ---------------------------------------------------------------------------
# vLLM server — scale-to-zero (min_containers=0 saves GPU cost when idle)
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    timeout=20 * MINUTES,
    min_containers=0,  # scale to zero when idle; cold start ~3-5 min (weights cached in volume)
    max_containers=1,
    scaledown_window=15 * MINUTES,  # scale down after 15 min of no requests
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    """Start vLLM serving NuExtract3 on A100 with KV cache optimizations."""
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
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        # --- Multimodal settings ---
        "--limit-mm-per-prompt",
        json.dumps({"image": 99, "video": 0}),
        "--chat-template-content-format",
        "openai",
        "--generation-config",
        "vllm",
        # --- KV cache optimizations ---
        "--kv-cache-dtype",
        "fp8_e5m2",  # halves KV cache memory on A100 (sm80+)
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(CHUNKED_PREFILL_TOKENS),
        # --- Speculative decoding (Multi-Token Prediction) ---
        "--speculative-config",
        json.dumps({"method": "qwen3_next_mtp", "num_speculative_tokens": 2}),
        # --- Concurrency ---
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
    ]
    print("Starting vLLM (NuExtract3 on A100):", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    url = serve.get_web_url()
    print(f"\nvLLM NuExtract3 URL: {url}")
    print(f"Model: {SERVED_MODEL_NAME}")
    print(f"OpenAI-compatible API: {url}/v1")