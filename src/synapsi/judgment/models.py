from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from synapsi.claims.models import ClaimStatus


class Verdict(StrEnum):
    DECIDED = "decided"
    INCONCLUSIVE = "inconclusive"
    SPLIT = "split"


class EvidenceStrength(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class ClaimAssessment(BaseModel):
    claim_id: str
    status: ClaimStatus
    note: str = ""


class Judgment(BaseModel):
    judge: str
    method: Literal["model", "structural", "vote", "single"]
    decision: str
    answer: str | None = None
    verdict: Verdict = Verdict.DECIDED
    supporting_claim_ids: list[str] = Field(default_factory=list)
    contradicting_claim_ids: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    unresolved_issues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = ""
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    minority_positions: list[str] = Field(default_factory=list)
