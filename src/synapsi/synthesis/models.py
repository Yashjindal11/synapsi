from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    statement: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    confidence: float | None = None
    note: str = ""


class Synthesis(BaseModel):
    """Final result, separating what is known from what is not.

    ``established``: strongly supported by external evidence and unchallenged.
    ``probable``: supported with meaningful uncertainty.
    ``disputed``: agents or evidence disagree.
    ``unknown``: cannot be determined from what was gathered.
    """

    summary: str
    answer: str | None = None
    established: list[Finding] = Field(default_factory=list)
    probable: list[Finding] = Field(default_factory=list)
    disputed: list[Finding] = Field(default_factory=list)
    unknown: list[Finding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float | None = None
    method: Literal["structural", "model"] = "structural"
