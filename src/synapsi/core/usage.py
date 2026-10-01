from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token counts reported by a provider for a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Usage(BaseModel):
    """Aggregated usage over many calls.

    ``cost_usd`` is ``None`` when any contributing call had no known price;
    SynapSI never invents prices.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_s: float = 0.0
    cached_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(
        self,
        tokens: TokenUsage,
        *,
        cost_usd: float | None,
        latency_s: float,
        cached: bool = False,
    ) -> None:
        self.calls += 1
        self.prompt_tokens += tokens.prompt_tokens
        self.completion_tokens += tokens.completion_tokens
        self.latency_s += latency_s
        if cached:
            self.cached_calls += 1
        if self.cost_usd is not None:
            self.cost_usd = None if cost_usd is None else self.cost_usd + cost_usd

    def merge(self, other: Usage) -> Usage:
        cost = (
            None
            if self.cost_usd is None or other.cost_usd is None
            else self.cost_usd + other.cost_usd
        )
        return Usage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=cost,
            latency_s=self.latency_s + other.latency_s,
            cached_calls=self.cached_calls + other.cached_calls,
        )
