from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from synapsi.core.usage import TokenUsage


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int | None = None
    json_mode: bool = False
    response_schema: dict[str, Any] | None = None
    stop: list[str] | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Completion(BaseModel):
    text: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = 0.0
    finish_reason: str | None = None
    cached: bool = False


class ModelProvider(ABC):
    """Minimal interface every model backend implements.

    Implementations must be safe to call concurrently from one event loop.
    """

    provider: str = "custom"

    def __init__(self, model: str):
        self.model = model

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> Completion:
        """Return a single completion for ``request``."""

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Yield text chunks. Default: one chunk from :meth:`complete`."""
        completion = await self.complete(request)
        yield completion.text

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources, if any."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"
