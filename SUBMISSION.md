# Inference Gateway — Submission

**Author:** Ingo Villnow  
**Stack:** Python gateway · Nginx LB · Modal vLLM (A10G, Gemma 4 E2B-IT) · CrewAI Researcher→Writer · Prometheus + Grafana

This document gives a reviewer everything needed to clone, configure, and run the full stack from scratch.  
For architecture details, API reference, and test instructions see [README.md](README.md).  
For experiment analysis, SLIs/SLOs, and hardware justification see [submission.ipynb](submission.ipynb) / [submission.pdf](submission.pdf).

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

Four dashboards are auto-provisioned: **Gateway Proxy**, **Overview**, **Technique Cost**, **TinyLlama Ops**.

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

## Reflection

**Biggest surprise:** Gemma 4 uses heterogeneous attention head dimensions (256 for local layers, 512 for global layers), which forces vLLM to fall back from FlashAttention to the Triton attention backend. This was not documented for vLLM 0.19.0 and only surfaced at serve-time. The practical impact was lower throughput (~20–40 tok/s on A10G) than similar-sized models with uniform head dimensions. The workaround was `--async-scheduling` combined with tuned batch budgets to prevent the Triton backend from becoming the bottleneck.

**What broke first (and which metric caught it):** The first experiment run timed out because the Modal cold-start — image pull, CUDA setup, and weight download — exceeded the gateway's 120 s backend timeout. The metric that caught it was `gateway_errors_total{status_code="504"}` spiking to 1 immediately in Grafana. Increasing `scaledown_window` to 15 minutes (so the container stays warm between runs) and raising `timeout` in `config.yaml` fixed the issue for all subsequent runs.

**Next steps for production:** Three changes would have the most impact:
1. **FP8 KV cache on H100** — moving to Hopper (compute capability ≥ 9.0) unlocks `--kv-cache-dtype fp8`, roughly halving KV memory usage and allowing `max_model_len=32768` or much higher concurrent sequences on the same VRAM budget.
2. **Modal autoscaling** — replacing `min_containers=max_containers=1` with a proper autoscaler that scales to zero when idle and bursts to 3–5 containers under load would make the cost SLO self-enforcing without manual intervention.
3. **Structured crew output + retry** — the Writer agent occasionally produces output slightly over the 120-word limit; adding a Pydantic output schema to the CrewAI task would enforce the contract at the framework level and eliminate silent application-layer SLO violations.
