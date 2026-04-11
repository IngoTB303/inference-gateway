# Inference Gateway — Submission

**Author:** Ingo Villnow  
**Stack:** FastAPI + uvicorn gateway · Nginx LB · Modal vLLM (A10G, Gemma 4 E2B-IT) · CrewAI Researcher→Writer · Prometheus + Grafana

This document gives a reviewer everything needed to clone, configure, and run the full stack from scratch.  
For architecture details, API reference, and test instructions see [README.md](README.md).  
For experiment analysis, SLIs/SLOs, and hardware justification see [submission.ipynb](submission.ipynb) / [submission.pdf](submission.pdf).

---

## Rubric Quick-Reference

| Rubric Area (Weight) | Where to Find It |
|---|---|
| **Agent Framing & SLOs** (10%) | Notebook Section 2, `crew.py` (Researcher→Writer pipeline, SLI/SLO table) |
| **End-to-End Path & Diagnosis Story** (20%) | Notebook Section 1.2 (layer-by-layer triage table), Grafana dashboards, Reflection below |
| **Experimental Rigor — Deltas & Controls** (25%) | Notebook Section 3, `data/experiments.csv`, `scripts/run_experiments.sh` |
| **Model / Instance Justification** (20%) | Notebook Section 4 (Gemma 4 + A10G choice, GPU comparison table, fallback analysis) |
| **Dashboard — Full-path** (15%) | Notebook Section 5, `monitoring/grafana_dashboards/` (4 JSON dashboards auto-provisioned) |
| **Presentation** (10%) | This document + notebook narrative flow |

---

## Submission Checklist

- [x] **Architecture:** Nginx :8780 → Gateway :8080/:8081 → Modal vLLM (A10G) — verified with curl and Grafana
- [x] **Metrics:** SLIs defined in notebook Section 2 (p50/p95 latency, success rate ≥90%, cost ≤$0.05/request)
- [x] **Reproducibility:** All scripts documented, `.env.example` provided, no secrets committed
- [x] **Results:** Summary table + 2 plot figures in notebook Section 3; version pinning in Modal deploy files
- [x] **Memo:** Data-backed GPU comparison table (A10G vs T4 vs L4 vs H100) in notebook Section 4
- [x] **Dashboard:** 4 Grafana dashboards covering gateway, Nginx, vLLM, and per-technique cost; JSON exported in `monitoring/grafana_dashboards/`
- [x] **Reflection:** Below (<300 words) covering surprises, debugging, and next steps

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | ≥ 0.5 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | ≥ 24 | docker.com/get-docker |
| [modal CLI](https://modal.com/docs/guide) | latest | `uv tool install modal` |
| nginx | any (with `ngx_http_stub_status_module`) | `sudo apt install nginx` |
| Hugging Face token | — | [hf.co/settings/tokens](https://huggingface.co/settings/tokens) (read access to `google/gemma-4-e2b-it`) |

---

## Step 1 — Clone & configure

```bash
git clone https://github.com/IngoTB303/inference-gateway.git
cd inference-gateway

# Install all dependency groups
uv sync --group crew --group otel --group notebook

# Create your local env file
cp .env.example .env
```

Open `.env` and set at minimum:

| Variable | What to set |
|----------|-------------|
| `API_KEY` | Any string (e.g. `test-key`) — required if you want auth |
| `OPENAI_API_KEY` | Same value as `API_KEY` (forwarded by crew.py) |
| `GPU_HOURLY_COST_USD` | Your Modal A10G rate (default: `1.10`) |

---

## Step 2 — Create Modal HuggingFace secret

```bash
modal secret create huggingface-secret HF_TOKEN=hf_<your_token>
```

This is required for the vLLM containers to download `google/gemma-4-e2b-it` from HuggingFace.

---

## Step 3 — Deploy Modal vLLM containers

Deploy one or more profiles. Each runs `google/gemma-4-e2b-it` on an A10G GPU:

```bash
# Standard flags only (baseline)
bash scripts/deploy_modal_vllm.sh standard

# Chunked prefill + prefix caching (optimized)
bash scripts/deploy_modal_vllm.sh optimized

# Maximum batch budget (hardcore)
bash scripts/deploy_modal_vllm.sh hardcore
```

After each deploy, the script prints the container URL. Paste it into `config.yaml`:

```yaml
# config.yaml
backends:
  - name: modal-gemma4-standard
    type: http
    url: https://ingo-villnow--vllm-gemma4-standard-serve.modal.run/
    timeout: 120
    model: gemma-4-e2b-it

  - name: modal-gemma4-optimized
    type: http
    url: https://ingo-villnow--vllm-gemma4-optimized-serve.modal.run/
    timeout: 120
    model: gemma-4-e2b-it

default_backend: local
```

---

## Step 4 — Start the gateway

```bash
# Instance 1 (terminal 1)
PORT=8080 GATEWAY_METRICS_PORT=9101 uv run python main.py

# Optional — instance 2 for load balancing (terminal 2)
PORT=8081 GATEWAY_METRICS_PORT=9102 uv run python main.py
```

Health check:

```bash
curl http://localhost:8080/health
```

---

## Step 5 — Start Nginx load balancer

```bash
# Start (rootless, no sudo required)
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"

# Stop
nginx -p /tmp -s stop
```

Nginx listens on `:8780` and round-robins between `:8080` and `:8081`.

**Quick alternative** — start both gateways + Nginx in one command:

```bash
API_KEY=test-key bash scripts/start_stack.sh
```

---

## Step 6 — Start the monitoring stack

```bash
cd monitoring
docker compose up -d
```

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |
| Jaeger | http://localhost:16686 |

Four dashboards are auto-provisioned: **Gateway Proxy**, **Overview**, **Technique Cost**, \*\*vLLM Ops\*\*.

---

## Step 7 — Run the crew agent

```bash
# Uses default topic and baseline technique
uv run --group crew python crew.py

# Specify technique and topic
uv run --group crew python crew.py --technique optimized --topic "benefits of chunked prefill"
```

The Researcher→Writer crew makes 2 sequential LLM calls through the gateway. Output is printed to stdout; metrics are visible immediately in Grafana.

---

## Step 8 — Run the experiment suite

```bash
# 3 crew runs × 3 profiles (standard / optimized / hardcore)
bash scripts/run_experiments.sh

# Deploy containers first, then run
bash scripts/run_experiments.sh --deploy

# Custom subset, runs, and topic
bash scripts/run_experiments.sh --profiles standard,optimized --runs 5 --topic "prefix caching"
```

Results are written to `data/experiments.csv`. A comparison table is printed to stdout when the suite completes.

---

## Step 9 — View dashboards & analyse results

```bash
# Open Grafana
open http://localhost:3000   # or xdg-open on Linux

# Analyse experiment results in the notebook
uv sync --group notebook
# Open submission.ipynb in VS Code (select the project venv as kernel)
code submission.ipynb

# Regenerate submission.pdf
bash scripts/export_pdf.sh
```

Key PromQL queries:

```promql
# Request rate per technique
rate(gateway_requests_total[1m])

# p95 end-to-end latency
histogram_quantile(0.95, rate(gateway_request_duration_seconds_bucket[5m]))

# Cumulative GPU cost by technique
increase(gateway_gpu_cost_usd_total[1h])
```

---

## Key Design Decisions

### Why Modal instead of SSH tunnels

The deliverables mention an SSH tunnel in the reference stack. I chose Modal instead because it provides **reproducible GPU containers** with version-pinned CUDA/vLLM images (`vllm==0.19.0`, `transformers==5.5.0`, CUDA 12.9). An SSH tunnel to a shared lab server would add a fragile network hop whose latency is outside my control and make the setup non-reproducible for reviewers. With Modal, `bash scripts/deploy_modal_vllm.sh optimized` gives any reviewer the same environment I tested against.

### Why two gateway instances behind Nginx

A single gateway instance would satisfy the functional requirement, but running two instances demonstrates that the gateway is **stateless and horizontally scalable**. It also exercises the Nginx round-robin path that a production deployment would use, and lets Prometheus show per-instance metrics (`gateway` vs `gateway2` jobs) — proving the dashboards work at the infrastructure level, not just the application level.

### Why X-Technique as a custom header

I needed a way to segment Prometheus metrics by engine configuration (baseline vs optimized vs hardcore) without changing the OpenAI request schema. A custom `X-Technique` header flows through Nginx transparently, lands as a Prometheus label on every gateway metric, and lets Grafana filter dashboards by technique. The alternative — encoding it in the `model` field — would have conflated routing and labeling.

### Why `gpu_memory_utilization` differs across profiles

The three profiles use 0.90 (standard, optimised) and 0.92 (hardcore) rather than the intuitive 0.95 maximum. vLLM's startup includes a `_dummy_sampler_run` that allocates logit tensors for `max_num_seqs` dummy sequences simultaneously. On an A10G the usable VRAM after driver overhead is ~22 GiB, not the marketed 24 GiB. At 0.95 only 253 MiB remained — less than the 256 MiB the warmup needed at `max_num_seqs=256`, causing a crash that surfaced as a Modal port-timeout rather than an obvious OOM. Setting `gpu_memory_utilization=0.92` and `max_num_seqs=128` for the hardcore profile leaves ~1.7 GiB of headroom while still allocating more KV cache than the 0.90 profiles.

### Why echo mode as the default

When `config.yaml` points `default_backend` to the `local` echo backend, every Gateway endpoint works without any external dependency. This means reviewers can `uv run python main.py` and immediately run the full test suite (144 tests pass) without deploying a GPU container or setting up credentials. Real backends are one `config.yaml` edit away.

---

## Reflection

**Biggest surprise:** Gemma 4's heterogeneous attention head dimensions (256 for local layers, 512 for global layers) forced vLLM to fall back from FlashAttention to the Triton attention backend. This was undocumented for vLLM 0.19.0 and only surfaced at serve-time via a log line: `Using Triton attention backend`. The practical impact was lower throughput (~20–40 tok/s on A10G) than comparably sized models with uniform head dimensions — roughly half what I projected from published benchmarks. The workaround was `--async-scheduling` combined with tuned batch budgets (`--max-num-batched-tokens`) to pipeline kernel calls and prevent the Triton backend from becoming the bottleneck.

**What broke first (and how I diagnosed it):** The first experiment run timed out. I saw `gateway_errors_total{status_code="504"}` spike to 1 in Grafana immediately. Checking `gateway_request_duration_seconds`, latency was flatlined at exactly 120 seconds — the backend timeout configured in `config.yaml`. This ruled out a vLLM OOM or inference error (which would have returned 5xx faster) and pointed to Modal cold-start: image pull, CUDA initialisation, and weight download exceeding the timeout. The fix was two-fold: raise `scaledown_window` to 15 minutes (so the container stays warm between runs) and increase `timeout` in `config.yaml` to 300 seconds for the initial deploy. All subsequent runs completed in 4–16 seconds.

**The hardcore container OOM (and why 0.95 is too aggressive on A10G):** After the cold-start issue was resolved, deploying the hardcore profile (`--gpu-memory-utilization 0.95 --max-num-seqs 256`) crashed during startup every time. The error was `torch.OutOfMemoryError` inside `_dummy_sampler_run` — vLLM's sampler warmup, which allocates dummy logit tensors for `max_num_seqs` sequences before accepting any requests. At 0.95 utilisation vLLM pre-allocated 20.96 GiB of the A10G's 22.06 GiB, leaving only 253 MiB free. The softmax over 256 dummy sequences needed 256 MiB — 3 MiB over budget. The Modal container then timed out waiting for port 8000 to accept connections, masking the root cause in the container logs. The fix was to drop `gpu_memory_utilization` to 0.92 (freeing ~660 MiB of headroom) and `max_num_seqs` to 128 (warmup memory scales linearly with sequence count). The result is still more aggressive than the optimised profile (0.90, 64 seqs) and the container now starts cleanly. The lesson: published GPU specs say "24 GB" but usable VRAM after CUDA context and driver overhead is closer to 22 GiB — always budget for at least 500 MiB of overhead on top of model weights + KV cache + warmup tensors.

**What I'd instrument next:** The highest-value addition would be a **`gateway_backend_request_duration_seconds`** histogram that measures only the time waiting for the backend, excluding gateway overhead (validation, auth, normalization). Currently `gateway_request_duration_seconds` is end-to-end, so when latency regresses I can't immediately tell whether the problem is in my code or in vLLM. I'd also add per-agent task timing in `crew.py` to decompose the crew's wall-clock into Researcher vs Writer phases — understanding which agent dominates latency is critical for optimising the agentic use case.

**Next steps for production:** Three changes would have the most impact:
1. **FP8 KV cache on H100** — moving to Hopper (compute capability ≥ 9.0) unlocks `--kv-cache-dtype fp8`, roughly halving KV memory and allowing `max_model_len=32768` or much higher concurrent sequences on the same VRAM budget.
2. **Modal autoscaling** — replacing `min_containers=max_containers=1` with a proper autoscaler that scales to zero when idle and bursts to 3–5 containers under load would make the cost SLO self-enforcing.
3. **Structured crew output** — the Writer agent occasionally exceeds the 120-word limit; a Pydantic output schema on the CrewAI task would enforce the contract at the framework level and eliminate silent application-layer SLO violations.
