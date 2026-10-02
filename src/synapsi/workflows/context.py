from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

from synapsi.claims.graph import ClaimGraph
from synapsi.core.errors import BudgetExceeded, ConfigError
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
from synapsi.evidence.pool import EvidencePool
from synapsi.judgment.models import Judgment
from synapsi.observability.events import Event, EventBus, EventType
from synapsi.observability.usage import UsageTracker
from synapsi.providers.base import Completion, CompletionRequest, Message, ModelProvider
from synapsi.providers.structured import generate_structured
from synapsi.synthesis.models import Synthesis

if TYPE_CHECKING:
    from synapsi.agents.base import Agent

T = TypeVar("T", bound=BaseModel)


class JudgeLike(Protocol):
    name: str

    async def judge(self, ctx: RunContext) -> Judgment: ...


class SynthesizerLike(Protocol):
    async def synthesize(self, ctx: RunContext) -> Synthesis: ...


class Budget(BaseModel):
    """Hard limits for one run. ``None`` means unlimited."""

    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_model_calls: int | None = None


class RunSettings(BaseModel):
    seed: int | None = None
    rounds: int = 2
    max_challenges_per_agent: int = 3
    verify_evidence: bool = False
    blind_review: bool = True
    randomize_order: bool = True
    max_tool_rounds: int = 2
    early_stop: bool = True
    fail_fast: bool = False
    repair_attempts: int = 1
    budget: Budget = Budget()


@dataclass
class DeliberationState:
    """Short-term memory of one deliberation. Discarded after the run."""

    claims: ClaimGraph = field(default_factory=ClaimGraph)
    evidence: EvidencePool = field(default_factory=EvidencePool)
    perspectives: dict[str, Perspective] = field(default_factory=dict)
    history: list[Perspective] = field(default_factory=list)
    challenges: list[Challenge] = field(default_factory=list)
    rebuttals: list[Rebuttal] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    position_changes: list[PositionChange] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    judgment: Judgment | None = None
    synthesis: Synthesis | None = None
    round: int = 0
    invalid_references: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stopped_reason: str | None = None

    def add_challenge(self, challenge: Challenge) -> Challenge:
        challenge = challenge.model_copy(update={"id": f"X{len(self.challenges) + 1}"})
        self.challenges.append(challenge)
        return challenge

    def add_rebuttal(self, rebuttal: Rebuttal) -> Rebuttal:
        rebuttal = rebuttal.model_copy(update={"id": f"R{len(self.rebuttals) + 1}"})
        self.rebuttals.append(rebuttal)
        return rebuttal

    def open_challenges(self, agent: str | None = None) -> list[Challenge]:
        return [
            c
            for c in self.challenges
            if c.status == "open" and (agent is None or c.target_agent == agent)
        ]

    def initial_perspectives(self) -> list[Perspective]:
        """First-round, independent perspectives (one per agent)."""
        seen: dict[str, Perspective] = {}
        for p in self.history:
            if p.independent and p.stance == "own" and p.agent not in seen:
                seen[p.agent] = p
        return list(seen.values())


class RunContext:
    """Everything a step can see and change during one run."""

    def __init__(
        self,
        problem: Problem,
        agents: list[Agent],
        *,
        judge: JudgeLike | None = None,
        synthesizer: SynthesizerLike | None = None,
        settings: RunSettings | None = None,
        events: EventBus | None = None,
        usage: UsageTracker | None = None,
        run_id: str | None = None,
        background: list[str] | None = None,
    ):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ConfigError(f"agent names must be unique: {names}")
        self.problem = problem.public()
        self.agents = list(agents)
        self.judge = judge
        self.synthesizer = synthesizer
        self.settings = settings or RunSettings()
        self.events = events or EventBus()
        self.usage = usage or UsageTracker()
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.background = background or []
        self.state = DeliberationState()
        self.rng = random.Random(self.settings.seed)
        self.current_step: str | None = None
        shuffled = names[:]
        self.rng.shuffle(shuffled)
        self._labels = {name: f"Analyst {i + 1}" for i, name in enumerate(shuffled)}

    # -- agents -------------------------------------------------------------
    def agent(self, name: str) -> Agent:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(name)

    def select(self, names: list[str] | None = None, *, level: int | None = None) -> list[Agent]:
        selected = self.agents if names is None else [self.agent(n) for n in names]
        if level is not None:
            selected = [a for a in selected if a.level == level]
        return selected

    def label(self, agent: str) -> str:
        """Display name for an agent in prompts shown to *other* agents."""
        if not self.settings.blind_review:
            return agent
        return self._labels.get(agent, agent)

    def unlabel(self, label: str) -> str | None:
        if not self.settings.blind_review:
            return label if any(a.name == label for a in self.agents) else None
        for name, lab in self._labels.items():
            if lab.lower() == label.strip().lower():
                return name
        return None

    def ordered(self, items: list[Any]) -> list[Any]:
        """Randomise presentation order (seeded) to reduce positional bias."""
        items = list(items)
        if self.settings.randomize_order:
            self.rng.shuffle(items)
        return items

    # -- events -------------------------------------------------------------
    def emit(self, kind: EventType, /, *, agent: str | None = None, **data: Any) -> None:
        self.events.emit(
            Event(type=kind, run_id=self.run_id, step=self.current_step, agent=agent, data=data)
        )

    def warn(self, message: str) -> None:
        self.state.warnings.append(message)
        self.emit(EventType.WARNING, message=message)

    # -- evidence -----------------------------------------------------------
    def visible_evidence(self, agent: str, snapshot: set[str] | None = None) -> list[Evidence]:
        """Evidence an agent may see in its prompts.

        External evidence (restricted to ``snapshot`` plus the agent's own tool
        results when given) and the agent's *own* model-knowledge notes. Other
        agents' recollections are opinions, so they are never shown as evidence.
        """
        out = []
        for e in self.state.evidence:
            own = e.provenance.produced_by == agent
            if (e.is_external and (snapshot is None or e.id in snapshot or own)) or (
                not e.is_external and own
            ):
                out.append(e)
        return out

    def add_evidence(self, evidence: Evidence) -> Evidence:
        before = len(self.state.evidence)
        stored = self.state.evidence.add(evidence)
        if len(self.state.evidence) > before:
            self.emit(
                EventType.EVIDENCE_CREATED,
                agent=evidence.provenance.produced_by,
                evidence_id=stored.id,
                source_kind=stored.provenance.source_kind.value,
                source_id=stored.provenance.source_id,
            )
        return stored

    # -- model calls --------------------------------------------------------
    def check_budget(self) -> None:
        budget, total = self.settings.budget, self.usage.total
        reason = None
        if budget.max_model_calls is not None and total.calls >= budget.max_model_calls:
            reason = f"model call budget {budget.max_model_calls} reached"
        elif budget.max_tokens is not None and total.total_tokens >= budget.max_tokens:
            reason = f"token budget {budget.max_tokens} reached"
        elif (
            budget.max_cost_usd is not None
            and total.cost_usd is not None
            and total.cost_usd >= budget.max_cost_usd
        ):
            reason = f"cost budget ${budget.max_cost_usd} reached"
        if reason:
            self.emit(EventType.BUDGET_EXCEEDED, reason=reason)
            raise BudgetExceeded(reason)

    def call_seed(self, agent: str, task: str) -> int | None:
        """Provider seed for one call: reproducible per run, distinct per agent and task.

        A single shared seed would make identical agents on one model return
        identical samples, silently destroying independence.
        """
        if self.settings.seed is None:
            return None
        key = f"{self.settings.seed}:{agent}:{task}:{self.state.round}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

    async def generate(
        self,
        *,
        provider: ModelProvider,
        schema: type[T],
        messages: list[Message],
        agent: str,
        task: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        hints: dict[str, Any] | None = None,
    ) -> T:
        """Structured model call with budget checks, usage accounting, and events."""
        metadata = {
            "agent": agent,
            "task": task,
            "round": self.state.round,
            "run_id": self.run_id,
            "question": self.problem.question,
            "options": self.problem.options or [],
            **(hints or {}),
        }
        request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.call_seed(agent, task),
            metadata=metadata,
        )

        async def complete(req: CompletionRequest) -> Completion:
            self.check_budget()
            completion = await provider.complete(req)
            cost = self.usage.record(completion, agent=agent, task=task)
            self.emit(
                EventType.MODEL_CALL,
                agent=agent,
                task=task,
                model=completion.model,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                cost_usd=cost,
                latency_s=round(completion.latency_s, 4),
                cached=completion.cached,
            )
            return completion

        result, _ = await generate_structured(
            complete, request, schema, repair_attempts=self.settings.repair_attempts
        )
        return result
