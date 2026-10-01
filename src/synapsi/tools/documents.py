"""Local document retrieval with a small, dependency-free BM25 index."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from synapsi.evidence.models import SourceKind
from synapsi.tools.base import Tool, ToolSource

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Chunk(BaseModel):
    source_id: str
    title: str
    locator: str
    text: str


class DocumentStore:
    def __init__(self, chunk_chars: int = 800, k1: float = 1.5, b: float = 0.75):
        self.chunk_chars = chunk_chars
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._tokens: list[Counter[str]] = []
        self._df: Counter[str] = Counter()

    def add(self, text: str, *, source_id: str, title: str | None = None) -> int:
        added = 0
        for i, piece in enumerate(self._split(text)):
            chunk = Chunk(
                source_id=source_id, title=title or source_id, locator=f"chunk {i + 1}", text=piece
            )
            counts = Counter(tokenize(piece))
            self.chunks.append(chunk)
            self._tokens.append(counts)
            self._df.update(counts.keys())
            added += 1
        return added

    def add_file(self, path: str | Path) -> int:
        p = Path(path)
        return self.add(
            p.read_text(encoding="utf-8", errors="replace"), source_id=str(p), title=p.name
        )

    def add_directory(self, path: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt")) -> int:
        return sum(self.add_file(f) for pat in patterns for f in sorted(Path(path).rglob(pat)))

    def _split(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) > self.chunk_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(current)
        return chunks

    def search(self, query: str, k: int = 3) -> list[ToolSource]:
        terms = tokenize(query)
        if not self.chunks or not terms:
            return []
        n = len(self.chunks)
        avg = sum(sum(c.values()) for c in self._tokens) / n
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._tokens):
            length = sum(counts.values())
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += (
                    idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / avg))
                )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [
            ToolSource(
                content=self.chunks[i].text,
                source_id=self.chunks[i].source_id,
                title=self.chunks[i].title,
                locator=self.chunks[i].locator,
            )
            for _, i in scored[:k]
        ]


def document_search_tool(store: DocumentStore, k: int = 3) -> Tool:
    def document_search(query: str) -> list[ToolSource]:
        return store.search(query, k=k)

    return Tool(
        "document_search",
        "Search the provided documents. Returns the most relevant passages.",
        document_search,
        parameters={"query": "str"},
        evidence_kind=SourceKind.DOCUMENT,
        capabilities=("search",),
    )
