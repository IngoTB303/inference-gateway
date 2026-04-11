"""HTTP backend — proxies requests to a remote OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from gateway.backends.base import BackendBase


class HttpBackend(BackendBase):
    """POSTs to a configured URL that speaks the OpenAI API."""

    def __init__(
        self, name: str, url: str, timeout: float = 60.0, model: str | None = None
    ) -> None:
        self.name = name
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.model = model  # if set, injected as "model" in the forwarded body

    @property
    def completions_url(self) -> str:
        return self.base_url + "/v1/chat/completions"

    async def generate(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        """Forward the request to the remote backend and return the response dict.

        Raises:
            httpx.TimeoutException: backend did not respond in time
            httpx.RequestError: connection-level failure
            RuntimeError: backend returned 5xx or unparseable response
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.completions_url, json=body)

        if resp.status_code >= 500:
            raise RuntimeError(f"backend_error:{resp.status_code}")

        try:
            return resp.json()
        except Exception as exc:
            raise RuntimeError("backend_invalid_response") from exc
