"""Composable provider wrappers: retries, rate limiting, caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path

from synapsi.core.errors import ProviderError
from synapsi.providers.base import Completion, CompletionRequest, ModelProvider


class ProviderWrapper(ModelProvider):
    def __init__(self, inner: ModelProvider):
        super().__init__(inner.model)
        self.inner = inner
        self.provider = inner.provider

    @property
    def id(self) -> str:
        return self.inner.id

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        async for chunk in self.inner.stream(request):
            yield chunk

    async def aclose(self) -> None:
        await self.inner.aclose()


class RetryingProvider(ProviderWrapper):
    """Retry transient failures with exponential backoff and jitter."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        super().__init__(inner)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def complete(self, request: CompletionRequest) -> Completion:
        attempt = 0
        while True:
            try:
                return await self.inner.complete(request)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = min(self.max_delay, self.base_delay * 2**attempt)
                await asyncio.sleep(delay * (0.5 + random.random() / 2))
                attempt += 1


class RateLimitedProvider(ProviderWrapper):
    """Bound concurrency and (optionally) requests per minute."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_concurrency: int = 8,
        requests_per_minute: float | None = None,
    ):
        super().__init__(inner)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def _wait_for_slot(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def complete(self, request: CompletionRequest) -> Completion:
        async with self._semaphore:
            await self._wait_for_slot()
            return await self.inner.complete(request)


def cache_key(provider_id: str, request: CompletionRequest) -> str:
    payload = request.model_dump(exclude={"metadata"})
    payload["provider"] = provider_id
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachedProvider(ProviderWrapper):
    """Cache completions by request content.

    By default only deterministic requests (``temperature == 0``) are cached,
    because caching sampled outputs silently removes the diversity that
    multi-agent experiments depend on. ``directory`` enables an on-disk cache
    shared across runs.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        directory: str | Path | None = None,
        max_entries: int = 1024,
        cache_sampled: bool = False,
    ):
        super().__init__(inner)
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.cache_sampled = cache_sampled
        self._memory: OrderedDict[str, Completion] = OrderedDict()

    def _cacheable(self, request: CompletionRequest) -> bool:
        return self.cache_sampled or request.temperature == 0

    async def complete(self, request: CompletionRequest) -> Completion:
        if not self._cacheable(request):
            return await self.inner.complete(request)
        key = cache_key(self.inner.id, request)
        hit = self._get(key)
        if hit is not None:
            return hit.model_copy(update={"cached": True, "latency_s": 0.0})
        completion = await self.inner.complete(request)
        self._put(key, completion)
        return completion

    def _get(self, key: str) -> Completion | None:
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        if self.directory:
            path = self.directory / f"{key}.json"
            if path.exists():
                completion = Completion.model_validate_json(path.read_text())
                self._memory[key] = completion
                return completion
        return None

    def _put(self, key: str, completion: Completion) -> None:
        self._memory[key] = completion
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)
        if self.directory:
            (self.directory / f"{key}.json").write_text(completion.model_dump_json())


def managed(
    provider: ModelProvider,
    *,
    retries: int = 2,
    max_concurrency: int = 8,
    requests_per_minute: float | None = None,
    cache: bool | str | Path = False,
) -> ModelProvider:
    """Apply the standard wrapper stack: cache -> rate limit -> retry -> provider."""
    wrapped: ModelProvider = provider
    if retries:
        wrapped = RetryingProvider(wrapped, max_retries=retries)
    wrapped = RateLimitedProvider(
        wrapped, max_concurrency=max_concurrency, requests_per_minute=requests_per_minute
    )
    if cache:
        directory = None if cache is True else cache
        wrapped = CachedProvider(wrapped, directory=directory)
    return wrapped
