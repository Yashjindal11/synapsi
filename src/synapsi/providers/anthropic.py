from __future__ import annotations

import time
from typing import Any

import httpx

from synapsi.core.errors import ProviderError
from synapsi.core.usage import TokenUsage
from synapsi.providers._http import HTTPProviderMixin, resolve_api_key
from synapsi.providers.base import Completion, CompletionRequest, ModelProvider


class AnthropicProvider(HTTPProviderMixin, ModelProvider):
    """Anthropic Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        default_max_tokens: int = 2048,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        self._api_key = resolve_api_key(api_key, api_key_env, required=True)
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.default_max_tokens = default_max_tokens
        self._init_client(client, timeout)

    async def complete(self, request: CompletionRequest) -> Completion:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages = [m.model_dump() for m in request.messages if m.role != "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.default_max_tokens,
            "temperature": min(request.temperature, 1.0),
        }
        if system:
            payload["system"] = system
        if request.stop:
            payload["stop_sequences"] = request.stop
        headers = {
            "x-api-key": self._api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        start = time.perf_counter()
        data = await self._post(f"{self.base_url}/messages", payload, headers)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.id}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=self.id,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("input_tokens") or 0),
                completion_tokens=int(usage.get("output_tokens") or 0),
            ),
            latency_s=time.perf_counter() - start,
            finish_reason=data.get("stop_reason"),
        )
