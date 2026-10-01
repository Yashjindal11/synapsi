from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Problem(BaseModel):
    """The question put to a council.

    ``facts`` are user-provided statements; they enter the evidence pool as
    ``user_provided`` evidence. ``context`` is background text shown to agents
    but not treated as evidence. ``answer`` is the gold label, used only by the
    experiment engine and never shown to agents.
    """

    question: str = Field(min_length=1)
    id: str = ""
    context: str | None = None
    options: list[str] | None = None
    facts: list[str] = Field(default_factory=list)
    answer: str | None = Field(default=None, exclude=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_id(self) -> Problem:
        if not self.id:
            digest = hashlib.sha256(self.question.encode()).hexdigest()[:10]
            self.id = f"p_{digest}"
        return self

    def public(self) -> Problem:
        """Copy without the gold answer, safe to show to agents and reports."""
        return self.model_copy(update={"answer": None})
