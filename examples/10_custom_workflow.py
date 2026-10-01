"""10. Custom workflow: compose built-in steps with your own.

- a custom `Step` that stops early when independent agents already agree
  (skipping debate cost when there is nothing to debate),
- a debate loop capped at three rounds with early stopping,
- the standard model-free finalisation (claim status, disagreements, judge, synthesis).
"""

import asyncio
from collections import Counter

from _common import model, show

from synapsi import Agent, Council, Step, Workflow
from synapsi.core.errors import StopWorkflow
from synapsi.deliberation.steps import (
    CrossExamination,
    IndependentAnalysis,
    RespondToChallenges,
    Revision,
    converged,
)
from synapsi.strategies import finalize
from synapsi.workflows import Loop, RunContext


class StopIfUnanimous(Step):
    name = "stop_if_unanimous"

    def __init__(self, min_confidence: float = 0.7):
        super().__init__()
        self.min_confidence = min_confidence

    async def run(self, ctx: RunContext) -> None:
        perspectives = list(ctx.state.perspectives.values())
        answers = Counter(p.answer for p in perspectives)
        confident = all(p.confidence >= self.min_confidence for p in perspectives)
        if len(answers) == 1 and None not in answers and confident:
            raise StopWorkflow("independent agents agreed with high confidence")


def build() -> Workflow:
    return Workflow(
        [
            IndependentAnalysis(),
            StopIfUnanimous(),
            Loop(
                [CrossExamination(), RespondToChallenges(), Revision(see_others=False)],
                max_iterations=3,
                until=converged,
                name="debate_round",
            ),
        ],
        name="debate_if_needed",
        finalize=finalize(),
    )


async def main() -> None:
    spec = model()
    agents = [Agent.from_role(r, spec) for r in ("analyst", "statistician", "skeptic")]
    council = Council(agents, strategy=build, mode="custom", seed=12)
    result = await council.run(
        "Is a 2% week-over-week drop in app sessions likely to be noise?",
        options=["noise", "real decline", "cannot tell"],
    )
    show(result)
    print("stopped early:", result.metadata.stopped_reason)


if __name__ == "__main__":
    asyncio.run(main())
