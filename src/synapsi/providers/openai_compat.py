from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from synapsi.core.errors import ConfigError, ProviderError
from synapsi.core.usage import TokenUsage
from synapsi.providers._http import HTTPProviderMixin, resolve_api_key
from synapsi.providers.base import Completion, CompletionRequest, ModelProvider


class OpenAICompatibleProvider(HTTPProviderMixin, ModelProvider):
    """Any server implementing ``POST /chat/completions`` (OpenAI, Ollama,
    vLLM, LM Studio, Hugging Face router, Together, Groq, ...)."""

    provider = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        require_key: bool = False,
        json_mode_supported: bool = True,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(model)
        if provider_name:
            self.provider = provider_name
        self.base_url = base_url.rstrip("/")
        self._api_key = resolve_api_key(api_key, api_key_env, required=require_key)
        self.json_mode_supported = json_mode_supported
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self._init_client(client, timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, request: CompletionRequest, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            **self.extra_body,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.json_mode and self.json_mode_supported:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
        return payload

    async def complete(self, request: CompletionRequest) -> Completion:
        start = time.perf_counter()
        data = await self._post(
            f"{self.base_url}/chat/completions", self._payload(request), self._headers()
        )
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.id}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=self.id,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
            latency_s=time.perf_counter() - start,
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        async with self.client.stream(
            "POST", url, json=self._payload(request, stream=True), headers=self._headers()
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:300]
                raise ProviderError(
                    f"{self.id} returned HTTP {response.status_code}: {body}",
                    status=response.status_code,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta


def openai_provider(model: str, **kw: Any) -> OpenAICompatibleProvider:
    kw.setdefault("base_url", "https://api.openai.com/v1")
    kw.setdefault("api_key_env", "OPENAI_API_KEY")
    kw.setdefault("require_key", True)
    return OpenAICompatibleProvider(model, provider_name="openai", **kw)


def ollama_provider(model: str, **kw: Any) -> OpenAICompatibleProvider:
    kw.setdefault("base_url", "http://localhost:11434/v1")
    return OpenAICompatibleProvider(model, provider_name="ollama", **kw)


def huggingface_provider(model: str, **kw: Any) -> OpenAICompatibleProvider:
    kw.setdefault("base_url", "https://router.huggingface.co/v1")
    kw.setdefault("api_key_env", "HF_TOKEN")
    kw.setdefault("require_key", True)
    kw.setdefault("json_mode_supported", False)
    return OpenAICompatibleProvider(model, provider_name="huggingface", **kw)


def compatible_provider(model: str, **kw: Any) -> OpenAICompatibleProvider:
    if "base_url" not in kw:
        raise ConfigError("openai-compatible provider requires base_url")
    kw.setdefault("api_key_env", "OPENAI_COMPATIBLE_API_KEY")
    return OpenAICompatibleProvider(model, **kw)
