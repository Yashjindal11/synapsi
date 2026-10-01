from __future__ import annotations

import os
from typing import Any

import httpx

from synapsi.core.errors import ConfigError, ProviderError

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def resolve_api_key(api_key: str | None, env_var: str | None, *, required: bool) -> str | None:
    if api_key:
        return api_key
    value = os.environ.get(env_var) if env_var else None
    if required and not value:
        raise ConfigError(f"missing API key: set the {env_var} environment variable")
    return value


class HTTPProviderMixin:
    """Shared httpx client handling and error mapping for HTTP providers."""

    _client: httpx.AsyncClient | None = None
    _owns_client: bool = True
    timeout: float = 120.0

    def _init_client(self, client: httpx.AsyncClient | None, timeout: float) -> None:
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        try:
            response = await self.client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"timeout calling {_host(url)}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"network error calling {_host(url)}: {exc}", retryable=True
            ) from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"{_host(url)} returned HTTP {response.status_code}: {response.text[:300]}",
                retryable=response.status_code in RETRYABLE_STATUS,
                status=response.status_code,
            )
        return response.json()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _host(url: str) -> str:
    return httpx.URL(url).host or url
