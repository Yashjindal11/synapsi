"""Web search and page fetching with pluggable backends.

Search results become ``web`` evidence with the URL as provenance. A model's
claim that "a website says X" is never treated as web evidence.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from typing import Protocol

import httpx

from synapsi.core.errors import ConfigError
from synapsi.evidence.models import SourceKind
from synapsi.providers._http import resolve_api_key
from synapsi.tools.base import Tool, ToolResult, ToolSource
from synapsi.tools.documents import tokenize


class SearchBackend(Protocol):
    async def search(self, query: str, k: int) -> list[ToolSource]: ...


class StaticSearch:
    """Offline backend over a fixed corpus ``{url: text}`` (tests, reproducible demos)."""

    def __init__(self, corpus: dict[str, str]):
        self.corpus = corpus

    async def search(self, query: str, k: int) -> list[ToolSource]:
        terms = set(tokenize(query))
        scored = sorted(
            ((len(terms & set(tokenize(text))), url) for url, text in self.corpus.items()),
            reverse=True,
        )
        return [
            ToolSource(content=self.corpus[url], source_id=url, title=url)
            for score, url in scored[:k]
            if score > 0
        ]


class TavilySearch:
    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None):
        self._key = resolve_api_key(api_key, "TAVILY_API_KEY", required=True)
        self._client = client or httpx.AsyncClient(timeout=30)

    async def search(self, query: str, k: int) -> list[ToolSource]:
        response = await self._client.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": k},
            headers={"Authorization": f"Bearer {self._key}"},
        )
        response.raise_for_status()
        return [
            ToolSource(content=r.get("content", ""), source_id=r.get("url"), title=r.get("title"))
            for r in response.json().get("results", [])
        ]


class BraveSearch:
    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None):
        self._key = resolve_api_key(api_key, "BRAVE_API_KEY", required=True)
        self._client = client or httpx.AsyncClient(timeout=30)

    async def search(self, query: str, k: int) -> list[ToolSource]:
        response = await self._client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": k},
            headers={"X-Subscription-Token": self._key or "", "Accept": "application/json"},
        )
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        return [
            ToolSource(
                content=r.get("description", ""), source_id=r.get("url"), title=r.get("title")
            )
            for r in results
        ]


def search_backend(name: str) -> SearchBackend:
    if name == "tavily":
        return TavilySearch()
    if name == "brave":
        return BraveSearch()
    raise ConfigError(f"unknown search backend {name!r}; use 'tavily' or 'brave'")


def web_search_tool(backend: SearchBackend, k: int = 4) -> Tool:
    async def web_search(query: str) -> list[ToolSource]:
        return await backend.search(query, k)

    return Tool(
        "web_search",
        "Search the web. Returns snippets with URLs.",
        web_search,
        parameters={"query": "str"},
        evidence_kind=SourceKind.WEB,
        capabilities=("search",),
    )


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


async def _is_public_host(host: str) -> bool:
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def fetch_url_tool(
    *,
    allow_private: bool = False,
    max_chars: int = 6000,
    max_bytes: int = 2_000_000,
    client: httpx.AsyncClient | None = None,
) -> Tool:
    """Fetch a public web page as text.

    Private, loopback, and link-local addresses are refused unless
    ``allow_private`` is set, so model-chosen URLs cannot probe internal
    networks. Redirects are not followed automatically for the same reason.
    """
    http = client or httpx.AsyncClient(timeout=20, follow_redirects=False)

    async def fetch_url(url: str) -> ToolResult:
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host:
            return ToolResult(tool="fetch_url", error="only http(s) URLs are allowed")
        if not allow_private and not await _is_public_host(parsed.host):
            return ToolResult(tool="fetch_url", error="refusing non-public address")
        async with http.stream("GET", url) as response:
            if response.is_redirect:
                location = response.headers.get("location", "")
                return ToolResult(
                    tool="fetch_url", error=f"redirected to {location}; fetch it explicitly"
                )
            if response.status_code >= 400:
                return ToolResult(tool="fetch_url", error=f"HTTP {response.status_code}")
            body = b""
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > max_bytes:
                    break
        text = body.decode(response.encoding or "utf-8", errors="replace")
        if "html" in response.headers.get("content-type", ""):
            text = html_to_text(text)
        return ToolResult(
            tool="fetch_url",
            sources=[ToolSource(content=text[:max_chars], source_id=url, title=url)],
        )

    return Tool(
        "fetch_url",
        "Fetch the text of a public web page by URL.",
        fetch_url,
        parameters={"url": "str"},
        evidence_kind=SourceKind.WEB,
    )
