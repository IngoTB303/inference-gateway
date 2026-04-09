#!/usr/bin/env bash
# Shared experiment configuration — source this at the top of experiment scripts.

# Gateway base URL (including /v1); defaults to Nginx LB
GATEWAY_BASE="${GATEWAY_OPENAI_BASE:-http://127.0.0.1:8780/v1}"

# Research topic passed to crew.py
TOPIC="${TOPIC:-vLLM optimization techniques on A10G GPUs}"

# Number of crew runs per technique label
N_RUNS="${N_RUNS:-3}"

# Seconds to wait for a Modal backend to warm up after deploy
# (cold start + weight download on A10G typically takes 3-8 min)
BACKEND_WAIT_S="${BACKEND_WAIT_S:-480}"

# Polling interval when waiting for a backend (seconds)
POLL_INTERVAL=15
