"""YAML-based backend configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from gateway.backends.base import BackendBase
from gateway.backends.echo import EchoBackend
from gateway.backends.http_backend import HttpBackend


def _make_backend(entry: dict[str, Any]) -> BackendBase:
    """Instantiate a backend from a config dict entry."""
    name = entry["name"]
    backend_type = entry.get("type", "echo")

    if backend_type == "echo":
        return EchoBackend(name=name)
    if backend_type in ("http", "llamacpp", "vllm"):
        url = entry["url"]
        timeout = float(entry.get("timeout", 60.0))
        model = entry.get("model")  # optional: forwarded as "model" in request body
        max_model_len = entry.get("max_model_len")  # optional: token limit guard
        if max_model_len is not None:
            max_model_len = int(max_model_len)
        return HttpBackend(
            name=name, url=url, timeout=timeout, model=model, max_model_len=max_model_len
        )

    raise ValueError(f"Unknown backend type: {backend_type!r}")


class GatewayConfig:
    """Holds all configured backends and routes requests by model name."""

    def __init__(
        self, backends: list[BackendBase], default_backend: BackendBase
    ) -> None:
        self._backends: dict[str, BackendBase] = {b.name: b for b in backends}
        self._default = default_backend

    def get_backend_for_model(self, model: str | None) -> BackendBase:
        """Return the backend for the given model name, or the default backend."""
        if model and model in self._backends:
            return self._backends[model]
        return self._default

    @property
    def all_backends(self) -> list[BackendBase]:
        return list(self._backends.values())

    @property
    def default_backend(self) -> BackendBase:
        return self._default


def load_config(path: str | Path = "config.yaml") -> GatewayConfig:
    """Load backend config from a YAML file.

    Falls back gracefully: if the file is absent, reads BACKEND_URL from the
    environment (legacy behaviour). If that is also absent, returns an echo-only
    config.
    """
    config_path = Path(path)

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        backend_entries = raw.get("backends", [])
        backends = [_make_backend(entry) for entry in backend_entries]

        default_name = raw.get("default_backend")
        backend_map = {b.name: b for b in backends}

        if default_name and default_name in backend_map:
            default_backend = backend_map[default_name]
        elif backends:
            default_backend = backends[0]
        else:
            default_backend = EchoBackend(name="local")

        if not backends:
            backends = [default_backend]

        return GatewayConfig(backends=backends, default_backend=default_backend)

    # Legacy fallback: BACKEND_URL env var
    backend_url = os.environ.get("BACKEND_URL") or None
    if backend_url:
        http_backend = HttpBackend(name="remote", url=backend_url)
        echo_backend = EchoBackend(name="local")
        return GatewayConfig(
            backends=[echo_backend, http_backend],
            default_backend=http_backend,
        )

    # Pure echo mode
    echo_backend = EchoBackend(name="local")
    return GatewayConfig(backends=[echo_backend], default_backend=echo_backend)
