"""Schemas models must return. They are drafts: the framework assigns ids,
validates references, and converts them into protocol objects."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from synapsi.claims.models import ClaimType
from synapsi.deliberation.models import ChallengeKind, ResponseType


class ClaimDraft(BaseModel):
    statement: str = Field(min_length=1)
    type: ClaimType = ClaimType.OTHER
    evidence_ids: list[str] = Field(default_factory=list)
    basis: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)


class PerspectiveDraft(BaseModel):
    position: str = Field(min_length=1)
    answer: str | None = None
    claims: list[ClaimDraft] = Field(default_factory=list, max_length=8)
    assumptions: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    uncertainty: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    counterarguments: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ChallengeDraft(BaseModel):
    target_claim_id: str
    kind: ChallengeKind = ChallengeKind.EVIDENCE
    problem: str = Field(min_length=1)
    question: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ChallengeSet(BaseModel):
    challenges: list[ChallengeDraft] = Field(default_factory=list, max_length=8)


class ReviewItem(BaseModel):
    target: str
    agreements: list[str] = Field(default_factory=list)
    challenges: list[ChallengeDraft] = Field(default_factory=list, max_length=6)
    assessment: str = ""
    quality: int = Field(default=3, ge=1, le=5)


class ReviewSet(BaseModel):
    reviews: list[ReviewItem] = Field(default_factory=list)


class RebuttalDraft(BaseModel):
    challenge_id: str
    response_type: ResponseType
    response: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    revised_statement: str | None = None


class RebuttalSet(BaseModel):
    rebuttals: list[RebuttalDraft] = Field(default_factory=list)


class RevisionDraft(BaseModel):
    position: str = Field(min_length=1)
    answer: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    keep_claim_ids: list[str] = Field(default_factory=list)
    withdraw_claim_ids: list[str] = Field(default_factory=list)
    new_claims: list[ClaimDraft] = Field(default_factory=list, max_length=6)
    reasoning_summary: str = ""
    reason_for_change: str = ""
    uncertainty: str = ""
    open_questions: list[str] = Field(default_factory=list)


class ToolCallDraft(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    tool_calls: list[ToolCallDraft] = Field(default_factory=list, max_length=4)


class QueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=6)


class EvidenceLinkDraft(BaseModel):
    evidence_id: str
    claim_id: str
    relation: Literal["supports", "contradicts", "irrelevant"]
    note: str = ""


class EvidenceLinkSet(BaseModel):
    links: list[EvidenceLinkDraft] = Field(default_factory=list)
