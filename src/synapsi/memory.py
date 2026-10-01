"""Optional long-term memory across runs. Disabled unless a store is passed.

Only the question, final answer, verdict, and synthesis summary are kept. Problem
context, facts, evidence, and agent outputs are never persisted, to limit
retention of sensitive material. Recalled records are shown to agents as
"prior conclusions", explicitly not as evidence.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from synapsi.result import SynapSIResult

_WORD = re.compile(r"[a-z0-9]+")


class MemoryRecord(BaseModel):
    run_id: str
    question: str
    answer: str | None = None
    verdict: str | None = None
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def as_note(self) -> str:
        outcome = self.answer or self.verdict or "no answer"
        return f"[{self.created_at:%Y-%m-%d}] {self.question} -> {outcome}: {self.summary[:300]}"


def _similarity(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


class MemoryStore(ABC):
    def __init__(self, *, k: int = 3, min_similarity: float = 0.3):
        self.k = k
        self.min_similarity = min_similarity

    @abstractmethod
    def records(self) -> list[MemoryRecord]: ...

    @abstractmethod
    def add(self, record: MemoryRecord) -> None: ...

    def recall(self, question: str) -> list[MemoryRecord]:
        scored = [(_similarity(question, r.question), r) for r in self.records()]
        relevant = [(s, r) for s, r in scored if s >= self.min_similarity]
        relevant.sort(key=lambda sr: (sr[0], sr[1].created_at), reverse=True)
        return [r for _, r in relevant[: self.k]]

    def remember(self, result: SynapSIResult) -> None:
        self.add(
            MemoryRecord(
                run_id=result.run_id,
                question=result.problem.question,
                answer=result.answer,
                verdict=result.judgment.verdict.value if result.judgment else None,
                summary=result.summary,
            )
        )


class InMemoryStore(MemoryStore):
    def __init__(self, **kw: float | int):
        super().__init__(**kw)  # type: ignore[arg-type]
        self._records: list[MemoryRecord] = []

    def records(self) -> list[MemoryRecord]:
        return list(self._records)

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)


class JSONLMemoryStore(MemoryStore):
    def __init__(self, path: str | Path, **kw: float | int):
        super().__init__(**kw)  # type: ignore[arg-type]
        self.path = Path(path)

    def records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [MemoryRecord.model_validate_json(line) for line in lines if line.strip()]

    def add(self, record: MemoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
