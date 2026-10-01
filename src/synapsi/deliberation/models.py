from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Perspective(BaseModel):
    """One agent's structured position at one point in the deliberation."""

    agent: str
    role: str
    model: str
    position: str
    answer: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    uncertainty: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    counterarguments: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    round: int = 0
    independent: bool = True
    stance: Literal["own", "adversarial"] = "own"


class ChallengeKind(StrEnum):
    EVIDENCE = "evidence"
    LOGIC = "logic"
    ASSUMPTION = "assumption"
    SCOPE = "scope"
    ALTERNATIVE = "alternative"
    FACTUAL = "factual"


class ChallengeStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"
    CONCEDED = "conceded"
    REVISED = "revised"


class Challenge(BaseModel):
    id: str = ""
    challenger: str
    target_claim_id: str
    target_agent: str
    kind: ChallengeKind = ChallengeKind.EVIDENCE
    problem: str
    question: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    round: int = 0
    status: ChallengeStatus = ChallengeStatus.OPEN


class ResponseType(StrEnum):
    DEFEND = "defend"
    CONCEDE = "concede"
    REVISE = "revise"
    CLARIFY = "clarify"


class Rebuttal(BaseModel):
    id: str = ""
    challenge_id: str
    agent: str
    response_type: ResponseType
    response: str
    evidence_ids: list[str] = Field(default_factory=list)
    revised_claim_id: str | None = None
    round: int = 0


class Review(BaseModel):
    reviewer: str
    target_agent: str
    agreements: list[str] = Field(default_factory=list)
    challenge_ids: list[str] = Field(default_factory=list)
    assessment: str = ""
    quality: int | None = Field(default=None, ge=1, le=5)
    blind: bool = True
    round: int = 0


class Vote(BaseModel):
    agent: str
    answer: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PositionChange(BaseModel):
    """Records when and why an agent changed its answer or position.

    Changes not backed by new evidence or a conceded challenge are a useful
    signal of conformity pressure.
    """

    agent: str
    round: int
    from_answer: str | None
    to_answer: str | None
    from_confidence: float
    to_confidence: float
    reason: str = ""
    cited_new_evidence: bool = False
    after_concession: bool = False


class DisagreementSide(BaseModel):
    position: str
    agents: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    external_evidence_count: int = 0


class Disagreement(BaseModel):
    id: str = ""
    topic: str
    kind: Literal["answer", "claim"]
    sides: list[DisagreementSide]
    claim_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    status: Literal["resolved", "unresolved"] = "unresolved"
    evidence_balance: str = ""
