## Plan: Latency in responses + real-backend tests

**Issue #10** adds `latency_ms` to the `usage` object of every non-streaming `/v1/chat/completions` response. The latency is already measured; it just needs to flow into the response body. **Issue #9** adds integration tests (pytest + Bruno) that talk to the real backends declared in `config.yaml`, discovering them first via `GET /v1/backends`.

---

### Issue #10 — `latency_ms` in `usage`

**Steps**

1. **`main.py` `build_response()`** — add optional `latency_ms: float = 0.0` parameter; include `"latency_ms": round(latency_ms, 2)` in the `usage` dict.

2. **`main.py` `_normalize_response()`** — add optional `latency_ms: float = 0.0` parameter; set `usage["latency_ms"] = round(latency_ms, 2)` (overwriting any value from the backend so our gateway measurement is authoritative).

3. **`main.py` `_handle_echo()`** — move `latency_ms = (time.monotonic() - start) * 1000` to before the `build_response()` call; pass it as `latency_ms=latency_ms`. (Currently timing happens after the response is sent.)

4. **`main.py` `_handle_with_config()` call site** — compute `latency_ms` before the `_normalize_response()` call (currently computed two lines after), then pass it.

5. **`main.py` `_handle_backend()` / `_proxy_non_stream()` call site** — same reorder: compute `latency_ms` before `_normalize_response()`, then pass it.

6. **Tests in `tests/test_gateway.py`**:
   - `test_echo_latency_in_usage` — assert `usage["latency_ms"] >= 0` in echo response.
   - `test_backend_latency_in_usage` — assert `usage["latency_ms"] >= 0` in mocked-backend response (add to existing `@respx.mock` section).

7. **Bruno** — add a `latency_ms` assertion to `tests/bruno/POST endpoints/backend_success.bru` and one of the echo tests.

---

### Issue #9 — Tests for real backends

**Steps**

1. **Register custom marker** — add `markers = ["live: mark test as requiring a live backend"]` to `pyproject.toml` under `[tool.pytest.ini_options]`; add `addopts = "-m not live"` so live tests are skipped by default and run with `uv run pytest -m live`.

2. **Shared live-gateway fixture** — add a `live_gateway` fixture in `tests/test_gateway.py` that starts the gateway using the real `load_config("config.yaml")` on a free port (same pattern as existing fixtures).

3. **Backend-discovery helper test** — `test_live_backends_endpoint` (`@pytest.mark.live`) — GET `{live_gateway}/v1/backends`, assert status 200, assert `local` and at least one HTTP backend are in the list. Mark the discovered HTTP backend URLs for the following tests to use via `pytest` module state or a shared fixture.

4. **Per-backend inference tests** (`@pytest.mark.live`):
   - `test_live_local_llama` — POST to `{live_gateway}/v1/chat/completions` with `"model": "local-llama"`; assert 200, valid shape, `backend == "local-llama"`, `latency_ms >= 0` in `usage`.
   - `test_live_remote_modal_llama` — same but `"model": "remote-modal-llama"`.
   - `test_live_remote_modal_vllm` — same but `"model": "remote-modal-vllm"`.
   - Each wrapped in a `try/except urllib.error.HTTPError` to surface a clear skip-vs-fail distinction via `pytest.skip()` when the backend is not reachable (connection refused / timeout → skip; wrong shape → fail).

5. **Timeout test** (`@pytest.mark.live`) — create a `GatewayConfig` with an `HttpBackend` pointing at an unreachable URL (`http://127.0.0.1:19999`) and `timeout=0.1`; fire the request; assert 504.

6. **Bruno tests** — add three new files in `tests/bruno/POST endpoints/`:
   - `real_backend_local_llama.bru` — POST with `model: local-llama`, assert 200 + `backend == "local-llama"` + `usage.latency_ms >= 0`.
   - `real_backend_modal_llama.bru` — same for `remote-modal-llama`.
   - `real_backend_modal_vllm.bru` — same for `remote-modal-vllm`.
   - `backend_timeout.bru` — document only (cannot configure timeout via HTTP); note to run programmatically.

---

**Verification**

```bash
# All existing tests still pass
uv run pytest

# Live tests (requires real backends available)
uv run pytest -m live -v

# Lint
uv run ruff check . && uv run ruff format --check .

# Bruno (echo + latency assertion)
# Run Bruno collection against a running gateway
```

**Decisions**
- `latency_ms` goes inside `usage` (per issue #10 spec), not as a header.
- `latency_ms` is the gateway-measured wall-clock time (ms, 2 decimal places), always overwriting any backend-supplied value — so clients get a consistent, trustworthy figure.
- Live tests default to skipped (`-m not live`) to keep CI green when no real backends are available.
- create and work in a feature branch named `feature/latency-and-live-tests` (or similar) for these changes, then merge back to main when tests are successful, but create pull requests for review first. 
