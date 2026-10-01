from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("synapsi")


class EventType(StrEnum):
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CLAIM_CREATED = "claim_created"
    EVIDENCE_CREATED = "evidence_created"
    CHALLENGE_CREATED = "challenge_created"
    REBUTTAL_CREATED = "rebuttal_created"
    REVIEW_CREATED = "review_created"
    POSITION_CHANGED = "position_changed"
    DISAGREEMENT_DETECTED = "disagreement_detected"
    JUDGMENT_CREATED = "judgment_created"
    SYNTHESIS_CREATED = "synthesis_created"
    BUDGET_EXCEEDED = "budget_exceeded"
    WARNING = "warning"
    ERROR = "error"


class Event(BaseModel):
    type: EventType
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step: str | None = None
    agent: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous fan-out. Subscriber failures are logged, never propagated."""

    def __init__(self, subscribers: list[Subscriber] | None = None):
        self._subscribers: list[Subscriber] = list(subscribers or [])

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(self, event: Event) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                logger.exception("event subscriber failed for %s", event.type)


class Recorder:
    """Keeps every event in memory (useful for tests and the web API)."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of_type(self, kind: EventType) -> list[Event]:
        return [e for e in self.events if e.type is kind]


class JSONLSink:
    """Append events to a JSON Lines trace file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")


def log_subscriber(event: Event) -> None:
    """Human-readable progress via the ``synapsi`` logger."""
    who = f" [{event.agent}]" if event.agent else ""
    where = f" {event.step}" if event.step else ""
    logger.info("%s%s%s", event.type.value, where, who)
