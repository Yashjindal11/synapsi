from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from synapsi.core.errors import BudgetExceeded, StopWorkflow
from synapsi.observability.events import EventType
from synapsi.workflows.context import RunContext

Predicate = Callable[[RunContext], bool]


class Step(ABC):
    """A unit of work in a workflow. Subclass and implement :meth:`run`."""

    name: str = ""

    def __init__(self, name: str | None = None):
        self.name = name or self.name or type(self).__name__

    @abstractmethod
    async def run(self, ctx: RunContext) -> None: ...

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description recorded in run metadata."""
        return {"step": self.name}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


async def run_step(step: Step, ctx: RunContext) -> None:
    previous = ctx.current_step
    ctx.current_step = step.name
    ctx.emit(EventType.STEP_STARTED)
    start = time.perf_counter()
    try:
        await step.run(ctx)
    finally:
        ctx.emit(EventType.STEP_COMPLETED, latency_s=round(time.perf_counter() - start, 4))
        ctx.current_step = previous


class Sequential(Step):
    def __init__(self, steps: Sequence[Step], name: str | None = None):
        super().__init__(name or "sequential")
        self.steps = list(steps)

    async def run(self, ctx: RunContext) -> None:
        for step in self.steps:
            await run_step(step, ctx)

    def describe(self) -> dict[str, Any]:
        return {"step": self.name, "sequential": [s.describe() for s in self.steps]}


class Parallel(Step):
    """Run independent steps concurrently. They share (and may mutate) state."""

    def __init__(self, steps: Sequence[Step], name: str | None = None):
        super().__init__(name or "parallel")
        self.steps = list(steps)

    async def run(self, ctx: RunContext) -> None:
        await asyncio.gather(*(run_step(s, ctx) for s in self.steps))

    def describe(self) -> dict[str, Any]:
        return {"step": self.name, "parallel": [s.describe() for s in self.steps]}


class Loop(Step):
    """Repeat ``body`` up to ``max_iterations`` times, stopping when ``until`` holds.

    ``ctx.state.round`` is incremented before each iteration.
    """

    def __init__(
        self,
        body: Step | Sequence[Step],
        *,
        max_iterations: int | Callable[[RunContext], int],
        until: Predicate | None = None,
        name: str | None = None,
    ):
        super().__init__(name or "loop")
        self.body = body if isinstance(body, Step) else Sequential(body, name=f"{self.name}.body")
        self.max_iterations = max_iterations
        self.until = until

    async def run(self, ctx: RunContext) -> None:
        limit = self.max_iterations(ctx) if callable(self.max_iterations) else self.max_iterations
        for _ in range(limit):
            ctx.state.round += 1
            await run_step(self.body, ctx)
            if self.until is not None and self.until(ctx):
                ctx.emit(EventType.WARNING, message=f"{self.name}: stopping condition met")
                break

    def describe(self) -> dict[str, Any]:
        limit = "dynamic" if callable(self.max_iterations) else self.max_iterations
        return {"step": self.name, "loop": self.body.describe(), "max_iterations": limit}


class Conditional(Step):
    def __init__(
        self,
        predicate: Predicate,
        then: Step,
        otherwise: Step | None = None,
        name: str | None = None,
    ):
        super().__init__(name or "conditional")
        self.predicate = predicate
        self.then = then
        self.otherwise = otherwise

    async def run(self, ctx: RunContext) -> None:
        branch = self.then if self.predicate(ctx) else self.otherwise
        if branch is not None:
            await run_step(branch, ctx)

    def describe(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "then": self.then.describe(),
            "otherwise": self.otherwise.describe() if self.otherwise else None,
        }


class FunctionStep(Step):
    """Wrap a plain (async) function ``fn(ctx)`` as a step."""

    def __init__(self, fn: Callable[[RunContext], Awaitable[None] | None], name: str | None = None):
        super().__init__(name or getattr(fn, "__name__", "function"))
        self.fn = fn

    async def run(self, ctx: RunContext) -> None:
        out = self.fn(ctx)
        if out is not None:
            await out


class Workflow:
    """An ordered list of steps plus optional stopping conditions.

    ``stop_when`` is checked after every top-level step. A step can also end
    the run early by raising :class:`StopWorkflow`. Budget exhaustion stops
    the run gracefully and is recorded in ``ctx.state.stopped_reason``; the
    remaining steps that do not need a model (e.g. structural synthesis) can
    still be run via ``finalize``.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        *,
        name: str = "custom",
        stop_when: Predicate | None = None,
        finalize: Sequence[Step] = (),
    ):
        self.steps = list(steps)
        self.name = name
        self.stop_when = stop_when
        self.finalize = list(finalize)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [s.describe() for s in self.steps],
            "finalize": [s.describe() for s in self.finalize],
        }

    async def execute(self, ctx: RunContext) -> None:
        ctx.emit(EventType.WORKFLOW_STARTED, workflow=self.name)
        start = time.perf_counter()
        try:
            for step in self.steps:
                await run_step(step, ctx)
                if self.stop_when is not None and self.stop_when(ctx):
                    ctx.state.stopped_reason = f"stop condition met after {step.name}"
                    break
        except StopWorkflow as stop:
            ctx.state.stopped_reason = stop.reason
        except BudgetExceeded as exc:
            ctx.state.stopped_reason = str(exc)
            ctx.state.errors.append(f"budget: {exc}")
        for step in self.finalize:
            await run_step(step, ctx)
        ctx.emit(
            EventType.WORKFLOW_COMPLETED,
            workflow=self.name,
            latency_s=round(time.perf_counter() - start, 4),
            stopped_reason=ctx.state.stopped_reason,
        )
