"""Evidence gathering and claim verification steps.

Evidence enters the run only through tools. These steps decide *what* to look
up; the provenance of every result is recorded by the tool layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synapsi.claims.models import ClaimStatus, ClaimType
from synapsi.deliberation.steps import for_each_agent
from synapsi.evidence.models import Evidence
from synapsi.workflows.base import Step
from synapsi.workflows.context import RunContext

if TYPE_CHECKING:
    from synapsi.agents.base import Agent

VERIFIABLE = {ClaimType.FACTUAL, ClaimType.STATISTICAL, ClaimType.CAUSAL, ClaimType.OTHER}


def search_tools(agent: Agent) -> list[str]:
    return [name for name, t in agent.tools.items() if "search" in t.capabilities]


def _researchers(ctx: RunContext, names: list[str] | None, roles: set[str]) -> list[Agent]:
    if names is not None:
        return [a for a in ctx.select(names) if search_tools(a)]
    preferred = [a for a in ctx.agents if a.role.key in roles and search_tools(a)]
    return preferred or [a for a in ctx.agents if search_tools(a)]


class EvidenceCollection(Step):
    """Researchers plan queries and run their search tools before analysis."""

    name = "evidence_collection"

    def __init__(self, agents: list[str] | None = None, *, max_queries: int = 3):
        super().__init__()
        self.agent_names = agents
        self.max_queries = max_queries

    async def run(self, ctx: RunContext) -> None:
        researchers = _researchers(ctx, self.agent_names, {"researcher", "fact_checker"})
        if not researchers:
            ctx.warn("evidence collection skipped: no agent has a search tool")
            return

        async def collect(agent: Agent) -> list[Evidence]:
            plan = await agent.plan_queries(ctx, max_queries=self.max_queries)
            found: list[Evidence] = []
            for query in plan.queries[: self.max_queries]:
                for tool in search_tools(agent):
                    found.extend(await agent.call_tool(ctx, tool, {"query": query}))
            return found

        await for_each_agent(ctx, researchers, collect)

    def describe(self) -> dict[str, Any]:
        return {"step": self.name, "agents": self.agent_names, "max_queries": self.max_queries}


class VerifyClaims(Step):
    """Look up evidence for unverified claims and link it as support or contradiction.

    Verifiers never check their own claims. The model only classifies the
    relation between retrieved text and a claim; it cannot create evidence.
    """

    name = "verify_claims"

    def __init__(self, agents: list[str] | None = None, *, max_claims: int = 6):
        super().__init__()
        self.agent_names = agents
        self.max_claims = max_claims

    async def run(self, ctx: RunContext) -> None:
        verifiers = _researchers(ctx, self.agent_names, {"fact_checker", "researcher"})
        if not verifiers:
            ctx.warn("claim verification skipped: no agent has a search tool")
            return
        ctx.state.claims.assess(ctx.state.evidence, ctx.state.challenges)
        candidates = sorted(
            (
                c
                for c in ctx.state.claims.active()
                if c.type in VERIFIABLE
                and c.status not in (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED)
            ),
            key=lambda c: -c.confidence,
        )[: self.max_claims]
        assignments: dict[str, list[str]] = {a.name: [] for a in verifiers}
        for i, claim in enumerate(candidates):
            eligible = [a for a in verifiers if a.name != claim.agent]
            if eligible:
                assignments[eligible[i % len(eligible)].name].append(claim.id)

        async def verify(agent: Agent) -> None:
            claims = [c for cid in assignments[agent.name] if (c := ctx.state.claims.get(cid))]
            if not claims:
                return
            found: list[Evidence] = []
            for claim in claims:
                for tool in search_tools(agent):
                    found.extend(await agent.call_tool(ctx, tool, {"query": claim.statement}))
            if not found:
                return
            links = await agent.link_evidence(ctx, claims, found)
            allowed_claims = {c.id for c in claims}
            allowed_evidence = {e.id for e in found}
            for link in links.links:
                if link.claim_id in allowed_claims and link.evidence_id in allowed_evidence:
                    if link.relation == "supports":
                        ctx.state.evidence.link(link.evidence_id, link.claim_id)
                    elif link.relation == "contradicts":
                        ctx.state.evidence.link(link.evidence_id, link.claim_id, contradicts=True)

        await for_each_agent(ctx, [a for a in verifiers if assignments[a.name]], verify)
        ctx.state.claims.assess(ctx.state.evidence, ctx.state.challenges)
