import asyncio
import time

import pytest

from synapsi.core.errors import ProviderError
from synapsi.core.pricing import PricingTable
from synapsi.core.usage import TokenUsage
from synapsi.providers import Completion, CompletionRequest, Message, MockProvider, ModelProvider
from synapsi.providers.wrappers import (
    CachedProvider,
    RateLimitedProvider,
    RetryingProvider,
    managed,
)


def _req(temperature: float = 0.0, text: str = "q") -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], temperature=temperature)


class Flaky(ModelProvider):
    provider = "flaky"

    def __init__(self, failures: int, retryable: bool = True):
        super().__init__("f")
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("boom", retryable=self.retryable)
        return Completion(text="ok", model=self.id)


async def test_retry_recovers_from_transient_errors() -> None:
    inner = Flaky(failures=2)
    out = await RetryingProvider(inner, max_retries=2, base_delay=0.001).complete(_req())
    assert out.text == "ok" and inner.calls == 3


async def test_retry_does_not_retry_permanent_errors() -> None:
    inner = Flaky(failures=1, retryable=False)
    with pytest.raises(ProviderError):
        await RetryingProvider(inner, base_delay=0.001).complete(_req())
    assert inner.calls == 1


async def test_cache_only_deterministic_requests_by_default(tmp_path: object) -> None:
    inner = MockProvider()
    cached = CachedProvider(inner, directory=str(tmp_path))
    first = await cached.complete(_req(0.0))
    second = await cached.complete(_req(0.0))
    assert second.cached and second.text == first.text
    await cached.complete(_req(0.7))
    await cached.complete(_req(0.7))
    assert len(inner.requests) == 3
    fresh = CachedProvider(MockProvider(), directory=str(tmp_path))
    assert (await fresh.complete(_req(0.0))).cached


async def test_rate_limiter_bounds_concurrency() -> None:
    active = 0
    peak = 0

    class Slow(ModelProvider):
        provider = "slow"

        async def complete(self, request: CompletionRequest) -> Completion:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return Completion(text="", model=self.id)

    limited = RateLimitedProvider(Slow("s"), max_concurrency=2)
    await asyncio.gather(*(limited.complete(_req()) for _ in range(6)))
    assert peak == 2


async def test_rate_limiter_spaces_requests() -> None:
    limited = RateLimitedProvider(MockProvider(), requests_per_minute=60 * 50)
    start = time.monotonic()
    await asyncio.gather(*(limited.complete(_req()) for _ in range(5)))
    assert time.monotonic() - start >= 0.07


def test_managed_keeps_identity() -> None:
    assert managed(MockProvider("x"), cache=True).id == "mock:x"


def test_pricing_is_opt_in() -> None:
    table = PricingTable({"openai:m": (1.0, 2.0)})
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert table.cost("openai:m", usage) == pytest.approx(2.0)
    assert table.cost("openai:other", usage) is None
    assert table.cost("mock:mock", usage) == 0.0
