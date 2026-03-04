"""Echo backend — returns prompt text without calling any external service."""

from __future__ import annotations

from typing import Any

from gateway.backends.base import BackendBase


class EchoBackend(BackendBase):
    """Returns 'Echo: <prompt>' without calling any external service."""

    def __init__(self, name: str = "local") -> None:
        self.name = name

    def generate(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        messages = body.get("messages", [])
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        content = f"Echo: {prompt}"
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())

        return {
            "id": request_id,
            "object": "chat.completion",
            "model": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
