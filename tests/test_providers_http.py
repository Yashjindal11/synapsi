import json
from typing import Any

import httpx
import pytest

from synapsi.core.errors import ConfigError, ProviderError
from synapsi.providers import CompletionRequest, Message, create_provider
from synapsi.providers.anthropic import AnthropicProvider
from synapsi.providers.gemini import GeminiProvider
from synapsi.providers.openai_compat import OpenAICompatibleProvider


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


REQ = CompletionRequest(
    messages=[Message(role="system", content="sys"), Message(role="user", content="hi")],
    temperature=0.2,
    max_tokens=50,
    json_mode=True,
)


async def test_openai_compatible_payload_and_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    p = OpenAICompatibleProvider("m", base_url="http://x/v1/", api_key="k", client=_client(handler))
    out = await p.complete(REQ)
    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert out.text == '{"ok": true}' and out.usage.prompt_tokens == 11


async def test_openai_compatible_streaming() -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    p = OpenAICompatibleProvider(
        "m", base_url="http://x", client=_client(lambda r: httpx.Response(200, text=body))
    )
    chunks = [c async for c in p.stream(REQ)]
    assert "".join(chunks) == "Hello"


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (400, False)])
async def test_http_errors_are_classified(status: int, retryable: bool) -> None:
    p = OpenAICompatibleProvider(
        "m", base_url="http://x", client=_client(lambda r: httpx.Response(status, text="err"))
    )
    with pytest.raises(ProviderError) as info:
        await p.complete(REQ)
    assert info.value.retryable is retryable
    assert info.value.status == status


async def test_anthropic_moves_system_prompt() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hey"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
                "stop_reason": "end_turn",
            },
        )

    p = AnthropicProvider("claude-x", api_key="secret", client=_client(handler))
    out = await p.complete(REQ)
    assert seen["body"]["system"] == "sys"
    assert all(m["role"] != "system" for m in seen["body"]["messages"])
    assert seen["key"] == "secret"
    assert out.text == "hey" and out.usage.completion_tokens == 1


async def test_gemini_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "yo"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1},
            },
        )

    p = GeminiProvider("gemini-x", api_key="g", client=_client(handler))
    out = await p.complete(REQ)
    assert seen["url"].endswith("/models/gemini-x:generateContent")
    assert "key=" not in seen["url"]
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert out.text == "yo"


def test_missing_key_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_provider("openai:gpt-test")


def test_registry_builds_local_providers() -> None:
    assert create_provider("ollama:llama3.1:8b").id == "ollama:llama3.1:8b"
    p = create_provider("openai-compatible:m", base_url="http://localhost:8000/v1")
    assert p.id == "openai-compatible:m"
    with pytest.raises(ConfigError):
        create_provider("openai-compatible:m")
