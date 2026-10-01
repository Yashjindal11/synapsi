from __future__ import annotations

from typing import Any, Literal

from synapsi.judgment.models import Judgment
from synapsi.judgment.structural import StructuralJudge, majority_judgment, single_judgment
from synapsi.observability.events import EventType
from synapsi.workflows.base import Step
from synapsi.workflows.context import RunContext


class JudgeStep(Step):
    """Produce the run's judgment.

    ``method="auto"`` uses the council's judge if configured, otherwise the
    deterministic :class:`StructuralJudge`.
    """

    name = "judgment"

    def __init__(self, method: Literal["auto", "structural", "vote", "single"] = "auto"):
        super().__init__()
        self.method = method

    async def run(self, ctx: RunContext) -> None:
        judgment: Judgment
        if self.method == "vote":
            judgment = majority_judgment(ctx)
        elif self.method == "single":
            judgment = single_judgment(ctx)
        elif self.method == "structural" or ctx.judge is None:
            judgment = await StructuralJudge().judge(ctx)
        else:
            try:
                judgment = await ctx.judge.judge(ctx)
            except Exception as exc:  # budget, provider, or schema failure
                ctx.state.errors.append(f"judge failed: {type(exc).__name__}: {exc}")
                ctx.warn("model judge failed; falling back to structural judge")
                judgment = await StructuralJudge().judge(ctx)
        ctx.state.judgment = judgment
        ctx.emit(
            EventType.JUDGMENT_CREATED,
            agent=judgment.judge,
            method=judgment.method,
            verdict=judgment.verdict,
            answer=judgment.answer,
            confidence=judgment.confidence,
        )

    def describe(self) -> dict[str, Any]:
        return {"step": self.name, "method": self.method}
