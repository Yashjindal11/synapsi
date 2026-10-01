from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    USER_PROVIDED = "user_provided"
    DOCUMENT = "document"
    WEB = "web"
    DATABASE = "database"
    API = "api"
    CALCULATION = "calculation"
    EXPERIMENT = "experiment"
    TOOL = "tool"
    MODEL_KNOWLEDGE = "model_knowledge"


class Provenance(BaseModel):
    """Where a piece of evidence came from.

    ``claimed_source`` holds a source an LLM *said* it relied on. It is kept for
    transparency but never verified, so it never upgrades the source kind.
    """

    source_kind: SourceKind
    source_id: str | None = None
    title: str | None = None
    tool: str | None = None
    tool_call_id: str | None = None
    query: str | None = None
    produced_by: str | None = None
    locator: str | None = None
    claimed_source: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Evidence(BaseModel):
    id: str = ""
    content: str
    provenance: Provenance
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)

    @property
    def is_external(self) -> bool:
        """True when the evidence did not originate from a model's own assertion."""
        return self.provenance.source_kind is not SourceKind.MODEL_KNOWLEDGE

    @property
    def fingerprint(self) -> str:
        """Identifies the underlying source, used to detect shared dependence."""
        p = self.provenance
        if p.source_id:
            return f"{p.source_kind.value}:{p.source_id}"
        digest = hashlib.sha256(self.content.strip().lower().encode()).hexdigest()[:12]
        return f"{p.source_kind.value}:#{digest}"
