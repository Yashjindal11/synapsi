from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator

from pydantic import BaseModel, Field

from synapsi.evidence.models import Evidence


class SharedSource(BaseModel):
    fingerprint: str
    evidence_ids: list[str]
    agents: list[str]
    claim_ids: list[str]


class SourceDependence(BaseModel):
    """How much apparent agreement rests on the same underlying sources.

    Three agents citing one web page are one source, not three confirmations.
    """

    distinct_sources: int = 0
    shared_sources: list[SharedSource] = Field(default_factory=list)
    shared_source_ratio: float = 0.0


class EvidencePool:
    """All evidence gathered in a run, with sequential ids (E1, E2, ...)."""

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._by_key: dict[tuple[str, str], str] = {}

    @classmethod
    def from_items(cls, items: Iterable[Evidence]) -> EvidencePool:
        """Rebuild a pool from serialised evidence (ids are kept as-is)."""
        pool = cls()
        for ev in items:
            pool._items[ev.id] = ev
            pool._by_key[(ev.fingerprint, ev.content.strip())] = ev.id
        return pool

    def add(self, evidence: Evidence) -> Evidence:
        key = (evidence.fingerprint, evidence.content.strip())
        if key in self._by_key:
            return self._items[self._by_key[key]]
        evidence = evidence.model_copy(update={"id": f"E{len(self._items) + 1}"})
        self._items[evidence.id] = evidence
        self._by_key[key] = evidence.id
        return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._items

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def valid_ids(self, ids: Iterable[str]) -> list[str]:
        """Keep only ids that exist, preserving order and dropping duplicates."""
        return list(dict.fromkeys(i for i in ids if i in self._items))

    def link(self, evidence_id: str, claim_id: str, *, contradicts: bool = False) -> None:
        evidence = self._items.get(evidence_id)
        if evidence is None:
            return
        target = evidence.contradicts if contradicts else evidence.supports
        if claim_id not in target:
            target.append(claim_id)

    def dependence(self, citations: Iterable[tuple[str, str, str]]) -> SourceDependence:
        """``citations`` are ``(claim_id, agent, evidence_id)`` triples."""
        agents: defaultdict[str, set[str]] = defaultdict(set)
        claims: defaultdict[str, set[str]] = defaultdict(set)
        evidence_ids: defaultdict[str, set[str]] = defaultdict(set)
        for claim_id, agent, evidence_id in citations:
            evidence = self._items.get(evidence_id)
            if evidence is None or not evidence.is_external:
                continue
            fp = evidence.fingerprint
            agents[fp].add(agent)
            claims[fp].add(claim_id)
            evidence_ids[fp].add(evidence_id)
        shared = [
            SharedSource(
                fingerprint=fp,
                evidence_ids=sorted(evidence_ids[fp]),
                agents=sorted(agents[fp]),
                claim_ids=sorted(claims[fp]),
            )
            for fp in sorted(agents)
            if len(agents[fp]) > 1
        ]
        total = len(agents)
        return SourceDependence(
            distinct_sources=total,
            shared_sources=shared,
            shared_source_ratio=len(shared) / total if total else 0.0,
        )
