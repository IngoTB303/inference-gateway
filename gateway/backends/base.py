"""Abstract base class for all inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendBase(ABC):
    """Interface that all backends must implement.

    The router calls only ``generate()`` — no if/else on backend type in handler code.
    Adding a new backend means writing one subclass and one config entry.
    """

    name: str  # unique backend name, set per instance

    @abstractmethod
    async def generate(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        """Non-streaming completion. Returns an OpenAI-shaped response dict."""
        ...
