from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapsi._version import __version__
from synapsi.agents.prompts import PROMPT_VERSION
from synapsi.claims.graph import ClaimGraph
from synapsi.claims.models import Claim, ClaimRelation
from synapsi.core.problem import Problem
from synapsi.deliberation.models import (
    Challenge,
    Disagreement,
    Perspective,
    PositionChange,
    Rebuttal,
    Review,
    Vote,
)
from synapsi.evidence.models import Evidence
from synapsi.evidence.pool import EvidencePool, SourceDependence
from synapsi.judgment.models import Judgment
from synapsi.observability.usage import UsageReport
from synapsi.synthesis.models import Synthesis


class UncertaintyReport(BaseModel):
    """Signals about how much to trust the result. None of them is a proof."""

    verdict: str | None = None
    final_confidence: float | None = None
    answer_distribution: dict[str, int] = Field(default_factory=dict)
    initial_answer_distribution: dict[str, int] = Field(default_factory=dict)
    agreement_rate: float | None = None
    initial_agreement_rate: float | None = None
    position_changes: int = 0
    unsupported_position_changes: int = 0
    claim_status_counts: dict[str, int] = Field(default_factory=dict)
    external_evidence: int = 0
    model_knowledge_items: int = 0
    source_dependence: SourceDependence = Field(default_factory=SourceDependence)
    unresolved_disagreements: int = 0
    invalid_references: int = 0


class SourceRecord(BaseModel):
    evidence_id: str
    source_kind: str
    source_id: str | None = None
    title: str | None = None
    tool: str | None = None
    produced_by: str | None = None


class RunMetadata(BaseModel):
    synapsi_version: str = __version__
    prompt_version: str = PROMPT_VERSION
    strategy: str
    mode: str
    settings: dict[str, Any]
    workflow: dict[str, Any]
    agents: list[dict[str, Any]]
    judge: dict[str, Any] | None = None
    synthesizer: dict[str, Any] | None = None
    started_at: datetime
    finished_at: datetime
    latency_s: float
    usage: UsageReport
    stopped_reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SynapSIResult(BaseModel):
    """Everything a run produced, in a stable, serialisable form."""

    run_id: str
    problem: Problem
    interpretation: str | None = None
    perspectives: list[Perspective]
    final_perspectives: dict[str, Perspective]
    claims: list[Claim]
    claim_relations: list[ClaimRelation]
    evidence: list[Evidence]
    challenges: list[Challenge]
    rebuttals: list[Rebuttal]
    reviews: list[Review]
    votes: list[Vote]
    position_changes: list[PositionChange]
    disagreements: list[Disagreement]
    judgment: Judgment | None
    synthesis: Synthesis | None
    uncertainty: UncertaintyReport
    provenance: list[SourceRecord]
    metadata: RunMetadata

    @property
    def answer(self) -> str | None:
        return self.judgment.answer if self.judgment else None

    @property
    def summary(self) -> str:
        return self.synthesis.summary if self.synthesis else ""

    def claim_graph(self) -> dict[str, Any]:
        """Nodes and edges (claims, evidence, relations) for visualisation."""
        graph = ClaimGraph.from_parts(self.claims, self.claim_relations)
        return graph.to_graph(EvidencePool.from_items(self.evidence))

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        from synapsi.reports import to_markdown

        return to_markdown(self)

    def to_html(self) -> str:
        from synapsi.reports import to_html

        return to_html(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> SynapSIResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _agreement(dist: Counter[str]) -> float | None:
    total = sum(dist.values())
    return round(max(dist.values()) / total, 3) if total else None


def uncertainty_report(
    *,
    perspectives: list[Perspective],
    initial: list[Perspective],
    position_changes: list[PositionChange],
    claims: list[Claim],
    evidence: list[Evidence],
    dependence: SourceDependence,
    disagreements: list[Disagreement],
    judgment: Judgment | None,
    invalid_references: int,
) -> UncertaintyReport:
    current = Counter(p.answer for p in perspectives if p.answer is not None)
    first = Counter(p.answer for p in initial if p.answer is not None)
    return UncertaintyReport(
        verdict=judgment.verdict.value if judgment else None,
        final_confidence=judgment.confidence if judgment else None,
        answer_distribution=dict(current),
        initial_answer_distribution=dict(first),
        agreement_rate=_agreement(current),
        initial_agreement_rate=_agreement(first),
        position_changes=len(position_changes),
        unsupported_position_changes=sum(
            1 for c in position_changes if not c.cited_new_evidence and not c.after_concession
        ),
        claim_status_counts=dict(Counter(c.status.value for c in claims if not c.withdrawn)),
        external_evidence=sum(1 for e in evidence if e.is_external),
        model_knowledge_items=sum(1 for e in evidence if not e.is_external),
        source_dependence=dependence,
        unresolved_disagreements=sum(1 for d in disagreements if d.status == "unresolved"),
        invalid_references=invalid_references,
    )
