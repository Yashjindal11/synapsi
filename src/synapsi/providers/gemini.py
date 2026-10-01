from __future__ import annotations

import time
from typing import Any

import httpx

from synapsi.core.errors import ProviderError
from synapsi.core.usage import TokenUsage
from synapsi.providers._http import HTTPProviderMixin, resolve_api_key
from synapsi.providers.base import Completion, CompletionRequest, ModelProvider


class GeminiProvider(HTTPProviderMixin, ModelProvider):
    """Google Gemini ``generateContent`` API."""

    provider = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        self._api_key = resolve_api_key(api_key, api_key_env, required=True)
        self.base_url = base_url.rstrip("/")
        self._init_client(client, timeout)

    async def complete(self, request: CompletionRequest) -> Completion:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens:
            config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            config["stopSequences"] = request.stop
        if request.json_mode:
            config["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"x-goog-api-key": self._api_key or "", "content-type": "application/json"}
        start = time.perf_counter()
        data = await self._post(
            f"{self.base_url}/models/{self.model}:generateContent", payload, headers
        )
        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.id}") from exc
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            model=self.id,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("promptTokenCount") or 0),
                completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            ),
            latency_s=time.perf_counter() - start,
            finish_reason=candidate.get("finishReason"),
        )
