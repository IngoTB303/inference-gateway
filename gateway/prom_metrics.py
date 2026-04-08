"""Prometheus metric definitions for the inference gateway."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "gateway_requests",
    "Total number of requests received",
    ["status_code", "model", "technique", "server_profile"],
)

ERRORS_TOTAL = Counter(
    "gateway_errors",
    "Total number of error responses (4xx/5xx)",
    ["status_code", "technique", "server_profile"],
)

TOKENS_TOTAL = Counter(
    "gateway_tokens",
    "Total tokens processed",
    ["type"],  # prompt | completion
)

GPU_COST_USD_TOTAL = Counter(
    "gateway_gpu_cost_usd",
    "Estimated GPU cost in USD based on request duration and GPU_HOURLY_COST_USD",
    ["technique", "server_profile"],
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

REQUEST_DURATION_SECONDS = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency in seconds",
    ["technique", "server_profile"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TTFT_SECONDS = Histogram(
    "gateway_ttft_seconds",
    "Streaming time-to-first-token in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INTER_CHUNK_SECONDS = Histogram(
    "gateway_inter_chunk_seconds",
    "Time between consecutive SSE chunks in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Number of requests currently being processed",
)
