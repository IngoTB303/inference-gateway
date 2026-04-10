# ✍️ Project Deliverable: InferenceOps Assignment

**Deadline:** Sunday, April 12, 2026
**Scope:** You own one agentic use case and the entire path from the first request byte until the response is served. You will experiment, justify, and dashboard the system like a professional production review.

---

## 📁 Submission Requirements
Submit a **GitHub repository URL** containing the following:

* **`submission.ipynb`**: Runnable top-to-bottom with committed sample data.
* **`submission.pdf`**: Created via `nbconvert` as a failsafe for rendering.
* **README.md (or SUBMISSION.md)**: Instructions to clone, configure `.env` (using `.env.example`), start the stack, and run the agent/load script. 
    > ⚠️ **IMPORTANT:** Do not include secrets in the repository.

### GitHub Repo must include:
* **Runnable Path:** Nginx (if required) → Gateway → Tunnel → vLLM (as configured in class).
* **Sample Data:** Committed data for notebook processing.

---

## 1. Assignment Objectives
* **Define an Agentic Application:** (Multi-step / tools / Crew-style) with success metrics and failure modes.
* **End-to-End Tracing:** LB → Gateway → Tunnel → vLLM. Identify which metrics belong to which hop.
* **Run Inference Experiments:** At least two distinct `vllm serve` configurations with pinned versions/flags.
* **Hardware Justification:** Choose model and GPU SKU based on measured latency, throughput, errors, and cost.
* **Stakeholder Dashboard:** Build a dashboard showing full-path health, not just GPU charts.

---

## 2. End-to-End Stack (System Diagram)
Include a **Mermaid diagram** or image in your notebook representing this path:

* **Client:** `crew.py` or `curl` (OpenAI-compatible).
* **Load Balancer:** Nginx or local equivalent (e.g., `127.0.0.1:8780`).
* **Inference Gateway:** `gateway.py` handling routing, `X-Technique` headers, and Prometheus metrics.
* **Tunnel:** `ssh -L` mapping local ports to the remote instance.
* **Inference Engine:** Your choice of `vllm serve`, `SGLang`, or `Llama.cpp`.

> **💡 InferenceOps Challenge Question:** > *Answer in your notebook:* If latency or errors spike, which layer do you check first (LB 502 vs. Gateway upstream vs. Tunnel vs. vLLM OOM), and what specific metric do you use?

---

## 3. Part A — Agentic Use Case and Metrics
Pick one pattern:
1.  **Research/RAG Agent:** Retrieve → Synthesize (defined via golden prompts).
2.  **Ops/Triage Agent:** Classify + Action steps (mocked tools allowed).
3.  **Code Agent:** Multi-step review/edit loop with a fixed benchmark set.

### Requirements for the Notebook:
* **User Journey:** Map the path and identify failure modes (timeouts, bad tool args).
* **SLIs / SLOs:** Define p50/p95 for completed tasks, success rates, and cost per successful task.
* **Hypothesis:** State which engine settings (e.g., prefix caching, chunked prefill) should help your specific traffic pattern.

---

## 4. Part B — Experiments (Engine + Gateway)
Implement a minimal path that hits the entire stack (Nginx → Gateway).
* **Matrix:** * $\ge 2$ distinct `vllm serve` engine deltas (e.g., *Baseline* vs. *Chunked Prefill*).
    * $\ge 1$ extra dimension (e.g., *Beam Search* vs. *Eagle2* or variations in generation params).
* **Execution:** Scripted $N$ runs. Record vLLM version, Model ID, Serve flags, and GPU SKU.
* **Evidence:** Produce tables and at least two plots (latency, error rates).

---

## 5. Part C — Model and Instance Memo
Provide a clear justification for your technical choices:
* **Model:** Why this specific model (context window, behavior, tool use)? What is the fallback if cost/latency fails SLO?
* **Instance:** Why this specific GPU? What breaks if you downgrade VRAM or tier?
* **Production Knobs:** When would you change Gateway routing vs. Engine args vs. Scaling out?

---

## 6. Part D — Dashboard (Agent + InferenceOps)
Build an end-to-end dashboard (Grafana/Prometheus).
* **Request Volume & Errors:** View failures via Gateway/LB.
* **Latency:** p50/p95 broken down by technique or `server_profile`.
* **Layer Coverage:** Panels for Gateway (9101) and vLLM (/metrics).
* **Efficiency:** Tokens/sec, cost proxy, or engine utilization.
* **Agent Success:** (Optional) Golden-set pass rates over time.

---

## 7. Submission Checklist
* [ ] **Architecture:** LB → Gateway → Tunnel → vLLM path verified.
* [ ] **Metrics:** SLIs tied to the user-visible path.
* [ ] **Reproducibility:** All scripts and `.env` keys documented.
* [ ] **Results:** Tables, plots, and version pinning included.
* [ ] **Memo:** Data-backed hardware/model justification.
* [ ] **Dashboard:** Labeled layers and JSON export included.
* [ ] **Reflection:** README includes a <300-word reflection on surprises and next steps.

---

## 8. Grading Rubric

| Area | Weight |
| :--- | :--- |
| **Agent Framing & SLOs** | 10% |
| **End-to-End Path & Diagnosis Story** | 20% |
| **Experimental Rigor (Deltas & Controls)** | 25% |
| **Model / Instance Justification** | 20% |
| **Dashboard (Full-path, not just GPU)** | 15% |
| **Presentation** | 10% |

---

## 9. Stretch Goals (Optional)
* **Multi-backend:** Nginx routing to multiple gateways.
* **Speculative Decoding:** Implementation with a draft model and failure handling.
* **OTLP Traces:** Integrated Jaeger/OpenTelemetry tracing.

---
*Scoring will be communicated privately.*