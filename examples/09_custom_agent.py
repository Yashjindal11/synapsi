"""9. Custom agents: a subclassed LLM agent and a deterministic, model-free agent.

`BaseRateAgent` overrides `analyze` and computes the posterior exactly, so its
claim is backed by `calculation` evidence. LLM agents in the same council can
cite or challenge it like any other participant.
"""

import asyncio
import re

from _common import model, show

from synapsi import Agent, Council, RoleSpec, register_role
from synapsi.agents.drafts import (
    ClaimDraft,
    PerspectiveDraft,
    RebuttalDraft,
    RebuttalSet,
    RevisionDraft,
)
from synapsi.evidence import Evidence, Provenance, SourceKind
from synapsi.experiments.suites import base_rates
from synapsi.workflows import RunContext

ECONOMIST = register_role(
    RoleSpec(
        key="behavioral_economist",
        title="Behavioral Economist",
        description="Explains decisions through incentives and cognitive biases.",
        instructions="Name the specific bias or incentive and how it would show up in data.",
    )
)


class EconomistAgent(Agent):
    def system_prompt(self) -> str:
        return super().system_prompt() + "\n\nAlways state one testable prediction."


class BaseRateAgent(Agent):
    """Answers Bayesian base-rate questions by calculation instead of a model call."""

    async def analyze(self, ctx: RunContext, **_: object) -> PerspectiveDraft:
        nums = [float(x) / 100 for x in re.findall(r"(\d+(?:\.\d+)?)%", ctx.problem.question)]
        prevalence, sensitivity, fpr, threshold = nums[:4]
        posterior = sensitivity * prevalence / (sensitivity * prevalence + fpr * (1 - prevalence))
        ev = ctx.add_evidence(
            Evidence(
                content=f"P(condition | positive) = {posterior:.4f} (Bayes' rule)",
                provenance=Provenance(
                    source_kind=SourceKind.CALCULATION, produced_by=self.name, tool="bayes"
                ),
            )
        )
        answer = "yes" if posterior > threshold else "no"
        return PerspectiveDraft(
            position=f"The posterior is {posterior:.1%}, so the answer is {answer}.",
            answer=answer,
            claims=[
                ClaimDraft(
                    statement=f"The posterior probability is {posterior:.1%}.",
                    type="statistical",
                    evidence_ids=[ev.id],
                    confidence=0.99,
                )
            ],
            confidence=0.99,
            reasoning_summary="Exact application of Bayes' rule to the stated rates.",
        )

    async def respond(self, ctx: RunContext, challenges) -> RebuttalSet:  # type: ignore[no-untyped-def]
        calc = [e.id for e in ctx.state.evidence if e.provenance.produced_by == self.name]
        return RebuttalSet(
            rebuttals=[
                RebuttalDraft(
                    challenge_id=c.id,
                    response_type="defend",
                    response="The figure follows directly from Bayes' rule.",
                    evidence_ids=calc,
                )
                for c in challenges
            ]
        )

    async def revise(self, ctx: RunContext, own, **_: object) -> RevisionDraft:  # type: ignore[no-untyped-def]
        # A calculation does not change because others disagree.
        return RevisionDraft(
            position=own.position,
            answer=own.answer,
            confidence=own.confidence,
            keep_claim_ids=own.claim_ids,
            reason_for_change="",
        )


async def main() -> None:
    spec = model()
    problem = next(iter(base_rates(n=1, seed=8)))
    agents = [
        BaseRateAgent("Calculator", "statistician", spec),
        EconomistAgent("Economist", ECONOMIST, spec),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="debate", judge="structural", mode="fast", seed=6)
    result = await council.run(problem)
    show(result)
    print(f"gold answer: {problem.answer}")


if __name__ == "__main__":
    asyncio.run(main())
