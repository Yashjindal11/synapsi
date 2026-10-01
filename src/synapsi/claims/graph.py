from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from synapsi.claims.models import Claim, ClaimRelation, ClaimStatus, RelationKind
from synapsi.deliberation.models import Challenge, ChallengeStatus
from synapsi.evidence.pool import EvidencePool


class ClaimGraph:
    """Claims (C1, C2, ...) and typed relations between them.

    Status is computed from evidence and challenges, never from how many agents
    assert a claim. See ``docs/protocol.md`` for the rules.
    """

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        self.relations: list[ClaimRelation] = []

    @classmethod
    def from_parts(cls, claims: Iterable[Claim], relations: Iterable[ClaimRelation]) -> ClaimGraph:
        """Rebuild a graph from serialised claims (ids are kept as-is)."""
        graph = cls()
        graph._claims = {c.id: c for c in claims}
        graph.relations = list(relations)
        return graph

    # -- construction -----------------------------------------------------
    def add(self, claim: Claim) -> Claim:
        claim = claim.model_copy(update={"id": f"C{len(self._claims) + 1}"})
        self._claims[claim.id] = claim
        return claim

    def relate(
        self, source: str, target: str, kind: RelationKind, *, agent: str = "", note: str = ""
    ) -> ClaimRelation | None:
        if source == target or source not in self._claims or target not in self._claims:
            return None
        for rel in self.relations:
            if (rel.source, rel.target, rel.kind) == (source, target, kind):
                return rel
        relation = ClaimRelation(source=source, target=target, kind=kind, agent=agent, note=note)
        self.relations.append(relation)
        return relation

    def withdraw(self, claim_id: str) -> None:
        if claim_id in self._claims:
            self._claims[claim_id].withdrawn = True

    # -- access -------------------------------------------------------------
    def get(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def __contains__(self, claim_id: object) -> bool:
        return claim_id in self._claims

    def __iter__(self) -> Iterator[Claim]:
        return iter(self._claims.values())

    def __len__(self) -> int:
        return len(self._claims)

    def all(self) -> list[Claim]:
        return list(self._claims.values())

    def active(self) -> list[Claim]:
        return [c for c in self._claims.values() if not c.withdrawn]

    def by_agent(self, agent: str, *, active_only: bool = True) -> list[Claim]:
        return [
            c
            for c in self._claims.values()
            if c.agent == agent and not (active_only and c.withdrawn)
        ]

    def valid_ids(self, ids: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(i for i in ids if i in self._claims))

    def citations(self) -> list[tuple[str, str, str]]:
        return [(c.id, c.agent, e) for c in self.active() for e in c.evidence_ids]

    # -- assessment -----------------------------------------------------------
    def external_support(self, claim: Claim, pool: EvidencePool) -> list[str]:
        ids = list(claim.evidence_ids) + [e.id for e in pool if claim.id in e.supports]
        return [i for i in dict.fromkeys(ids) if (ev := pool.get(i)) and ev.is_external]

    def external_contradiction(self, claim: Claim, pool: EvidencePool) -> list[str]:
        ids = list(claim.contradicting_evidence_ids) + [
            e.id for e in pool if claim.id in e.contradicts
        ]
        return [i for i in dict.fromkeys(ids) if (ev := pool.get(i)) and ev.is_external]

    def assess(self, pool: EvidencePool, challenges: Iterable[Challenge]) -> None:
        by_target: dict[str, list[Challenge]] = {}
        for ch in challenges:
            by_target.setdefault(ch.target_claim_id, []).append(ch)

        for claim in self._claims.values():
            if claim.status_reason.startswith("judge:"):
                continue
            status, reason = self._base_status(claim, pool, by_target.get(claim.id, []))
            claim.status, claim.status_reason = status, reason

        supported = {c.id for c in self._claims.values() if c.status is ClaimStatus.SUPPORTED}
        for rel in self.relations:
            target = self._claims[rel.target]
            if (
                rel.kind is RelationKind.CONTRADICTS
                and rel.source in supported
                and not target.status_reason.startswith("judge:")
                and not self.external_support(target, pool)
            ):
                target.status = ClaimStatus.CONTRADICTED
                target.status_reason = f"contradicted by supported claim {rel.source}"

    def _base_status(
        self, claim: Claim, pool: EvidencePool, challenges: list[Challenge]
    ) -> tuple[ClaimStatus, str]:
        support = self.external_support(claim, pool)
        contra = self.external_contradiction(claim, pool)
        model_support = [
            i for i in claim.evidence_ids if (ev := pool.get(i)) and not ev.is_external
        ]
        open_ = [c.id for c in challenges if c.status is ChallengeStatus.OPEN]
        conceded = [c.id for c in challenges if c.status is ChallengeStatus.CONCEDED]

        if contra and not support:
            return ClaimStatus.CONTRADICTED, f"contradicting evidence {contra}"
        if support and (contra or open_ or conceded):
            detail = contra or open_ or conceded
            return ClaimStatus.PARTIALLY_SUPPORTED, f"external support {support}; disputed {detail}"
        if support:
            return ClaimStatus.SUPPORTED, f"external support {support}"
        if conceded:
            return ClaimStatus.UNCERTAIN, f"owner conceded challenge {conceded}"
        if open_:
            return ClaimStatus.UNCERTAIN, f"unanswered challenge {open_}"
        if model_support:
            return ClaimStatus.UNVERIFIED, "only model-knowledge support"
        return ClaimStatus.UNSUPPORTED, "no evidence offered"

    def apply_assessment(
        self, claim_id: str, status: ClaimStatus, note: str, pool: EvidencePool
    ) -> ClaimStatus | None:
        """Apply an external (judge) assessment, capped by available evidence.

        A judge's opinion is itself model output, so it cannot make a claim
        ``supported`` or ``partially_supported`` without external evidence.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            return None
        if status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED) and not (
            self.external_support(claim, pool)
        ):
            status = ClaimStatus.UNVERIFIED
            note = f"{note} (capped: no external evidence)".strip()
        claim.status = status
        claim.status_reason = f"judge: {note}" if note else "judge"
        return status

    # -- export ---------------------------------------------------------------
    def to_graph(self, pool: EvidencePool | None = None) -> dict[str, Any]:
        """Nodes and edges for visualisation (claims, evidence, relations)."""
        nodes: list[dict[str, Any]] = [
            {
                "id": c.id,
                "kind": "claim",
                "label": c.statement,
                "agent": c.agent,
                "status": c.status.value,
                "confidence": c.confidence,
                "withdrawn": c.withdrawn,
            }
            for c in self._claims.values()
        ]
        edges: list[dict[str, Any]] = [
            {"source": r.source, "target": r.target, "kind": r.kind.value} for r in self.relations
        ]
        if pool is not None:
            for ev in pool:
                nodes.append(
                    {
                        "id": ev.id,
                        "kind": "evidence",
                        "label": ev.content[:160],
                        "source_kind": ev.provenance.source_kind.value,
                    }
                )
            for claim in self._claims.values():
                for eid in claim.evidence_ids:
                    edges.append({"source": eid, "target": claim.id, "kind": "supports"})
                for eid in claim.contradicting_evidence_ids:
                    edges.append({"source": eid, "target": claim.id, "kind": "contradicts"})
            for ev in pool:
                for cid in ev.supports:
                    edges.append({"source": ev.id, "target": cid, "kind": "supports"})
                for cid in ev.contradicts:
                    edges.append({"source": ev.id, "target": cid, "kind": "contradicts"})
        unique = {(e["source"], e["target"], e["kind"]): e for e in edges}
        return {"nodes": nodes, "edges": list(unique.values())}
