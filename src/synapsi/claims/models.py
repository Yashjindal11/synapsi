from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    FACTUAL = "factual"
    CAUSAL = "causal"
    STATISTICAL = "statistical"
    PREDICTIVE = "predictive"
    NORMATIVE = "normative"
    DEFINITIONAL = "definitional"
    METHODOLOGICAL = "methodological"
    OTHER = "other"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    UNVERIFIED = "unverified"


class RelationKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"


class Claim(BaseModel):
    id: str = ""
    statement: str
    type: ClaimType = ClaimType.OTHER
    agent: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    status_reason: str = ""
    round: int = 0
    withdrawn: bool = False
    revised_from: str | None = None


class ClaimRelation(BaseModel):
    source: str
    target: str
    kind: RelationKind
    agent: str = ""
    note: str = ""
