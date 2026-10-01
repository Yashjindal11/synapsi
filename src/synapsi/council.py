from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synapsi.agents.base import Agent
from synapsi.core.errors import ConfigError
from synapsi.core.pricing import PricingTable
from synapsi.core.problem import Problem
from synapsi.evidence.models import Evidence, Provenance, SourceKind
from synapsi.judgment.judge import Judge
from synapsi.judgment.structural import StructuralJudge
from synapsi.modes import PRESET_ROLES, mode_settings
from synapsi.observability.events import Event, EventBus, JSONLSink
from synapsi.observability.usage import UsageTracker
from synapsi.providers.base import ModelProvider
from synapsi.providers.registry import create_provider
from synapsi.providers.wrappers import managed
from synapsi.result import RunMetadata, SourceRecord, SynapSIResult, uncertainty_report
from synapsi.strategies import build_strategy
from synapsi.synthesis.synthesizer import Synthesizer
from synapsi.workflows.base import Workflow
from synapsi.workflows.context import Budget, JudgeLike, RunContext, RunSettings

JudgeSpec = Judge | StructuralJudge | JudgeLike | str | None


class Council:
    """A panel of agents plus a deliberation strategy.

    >>> council = Council([Agent("A", "statistician", "openai:gpt-4o-mini"), ...],
    ...                   strategy="debate", mode="balanced")
    >>> result = await council.run("Should we adopt approach X?")

    ``judge="auto"`` uses a model judge on the first agent's model (in a fresh,
    isolated context). Pass ``"structural"`` for the deterministic judge, a
    model spec string, or any object with ``async judge(ctx)``.
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        strategy: str | Workflow | Callable[[], Workflow] = "independent_panel",
        judge: JudgeSpec = "auto",
        synthesizer: Synthesizer | None = None,
        mode: str = "balanced",
        settings: RunSettings | None = None,
        seed: int | None = None,
        budget: Budget | None = None,
        pricing: PricingTable | dict[str, tuple[float, float]] | None = None,
        memory: Any | None = None,
        event_handlers: Sequence[Callable[[Event], None]] = (),
        trace_dir: str | Path | None = None,
    ):
        if not agents:
            raise ConfigError("a council needs at least one agent")
        self.agents = list(agents)
        self.strategy = strategy
        self.mode = mode
        self.settings = settings or mode_settings(mode)
        if seed is not None:
            self.settings.seed = seed
        if budget is not None:
            self.settings.budget = budget
        self.judge = self._resolve_judge(judge)
        self.synthesizer = synthesizer or Synthesizer()
        self.pricing = pricing if isinstance(pricing, PricingTable) else PricingTable(pricing)
        self.memory = memory
        self.event_handlers = list(event_handlers)
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def _resolve_judge(self, judge: JudgeSpec) -> JudgeLike | None:
        if judge == "auto":
            return Judge(self.agents[0].provider)
        if judge is None or judge == "structural" or isinstance(judge, StructuralJudge):
            return None
        if isinstance(judge, str):
            return Judge(judge)
        return judge

    @classmethod
    def preset(
        cls,
        mode: str = "balanced",
        model: str | ModelProvider = "mock",
        **kwargs: Any,
    ) -> Council:
        """Default role panel for a mode (3 / 5 / 8 agents), all on ``model``."""
        if mode not in PRESET_ROLES:
            raise ConfigError(f"no preset for mode {mode!r}")
        provider = model if isinstance(model, ModelProvider) else managed(create_provider(model))
        agents = [Agent.from_role(role, provider) for role in PRESET_ROLES[mode]]
        return cls(agents, mode=mode, **kwargs)

    def workflow(self) -> Workflow:
        if isinstance(self.strategy, Workflow):
            return self.strategy
        if isinstance(self.strategy, str):
            return build_strategy(self.strategy)
        return self.strategy()

    def describe(self) -> dict[str, Any]:
        judge = self.judge
        return {
            "strategy": self.workflow().name,
            "mode": self.mode,
            "agents": [a.describe() for a in self.agents],
            "judge": judge.describe()
            if isinstance(judge, Judge)
            else {"name": "structural"}
            if judge is None
            else {"name": getattr(judge, "name", type(judge).__name__)},
            "settings": self.settings.model_dump(),
        }

    async def run(
        self,
        problem: str | Problem,
        *,
        context: str | None = None,
        options: Sequence[str] | None = None,
        facts: Sequence[str] = (),
        run_id: str | None = None,
    ) -> SynapSIResult:
        if isinstance(problem, str):
            problem = Problem(
                question=problem,
                context=context,
                options=list(options) if options else None,
                facts=list(facts),
            )
        workflow = self.workflow()
        bus = EventBus(list(self.event_handlers))
        background = self.memory.recall(problem.question) if self.memory is not None else []
        ctx = RunContext(
            problem,
            self.agents,
            judge=self.judge,
            synthesizer=self.synthesizer,
            settings=self.settings.model_copy(deep=True),
            events=bus,
            usage=UsageTracker(self.pricing),
            run_id=run_id,
            background=[r.as_note() for r in background],
        )
        if self.trace_dir is not None:
            bus.subscribe(JSONLSink(self.trace_dir / f"{ctx.run_id}.jsonl"))
        for fact in problem.facts:
            ctx.add_evidence(
                Evidence(
                    content=fact,
                    provenance=Provenance(source_kind=SourceKind.USER_PROVIDED, produced_by="user"),
                )
            )
        started = datetime.now(UTC)
        await workflow.execute(ctx)
        finished = datetime.now(UTC)
        result = self._result(ctx, workflow, started, finished)
        if self.memory is not None:
            self.memory.remember(result)
        return result

    def run_sync(self, problem: str | Problem, **kwargs: Any) -> SynapSIResult:
        return asyncio.run(self.run(problem, **kwargs))

    def _result(
        self, ctx: RunContext, workflow: Workflow, started: datetime, finished: datetime
    ) -> SynapSIResult:
        state = ctx.state
        claims = state.claims.all()
        evidence = state.evidence.all()
        info = self.describe()
        return SynapSIResult(
            run_id=ctx.run_id,
            problem=ctx.problem,
            perspectives=list(state.history),
            final_perspectives=dict(state.perspectives),
            claims=claims,
            claim_relations=list(state.claims.relations),
            evidence=evidence,
            challenges=list(state.challenges),
            rebuttals=list(state.rebuttals),
            reviews=list(state.reviews),
            votes=list(state.votes),
            position_changes=list(state.position_changes),
            disagreements=list(state.disagreements),
            judgment=state.judgment,
            synthesis=state.synthesis,
            uncertainty=uncertainty_report(
                perspectives=list(state.perspectives.values()),
                initial=state.initial_perspectives(),
                position_changes=state.position_changes,
                claims=claims,
                evidence=evidence,
                dependence=state.evidence.dependence(state.claims.citations()),
                disagreements=state.disagreements,
                judgment=state.judgment,
                invalid_references=sum(state.invalid_references.values()),
            ),
            provenance=[
                SourceRecord(
                    evidence_id=e.id,
                    source_kind=e.provenance.source_kind.value,
                    source_id=e.provenance.source_id,
                    title=e.provenance.title,
                    tool=e.provenance.tool,
                    produced_by=e.provenance.produced_by,
                )
                for e in evidence
                if e.is_external
            ],
            metadata=RunMetadata(
                strategy=workflow.name,
                mode=self.mode,
                settings=ctx.settings.model_dump(),
                workflow=workflow.describe(),
                agents=info["agents"],
                judge=info["judge"],
                synthesizer=self.synthesizer.describe(),
                started_at=started,
                finished_at=finished,
                latency_s=round((finished - started).total_seconds(), 4),
                usage=ctx.usage.report(),
                stopped_reason=state.stopped_reason,
                errors=list(state.errors),
                warnings=list(state.warnings),
            ),
        )
