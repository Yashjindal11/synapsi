from typing import Any

from synapsi.agents import Agent
from synapsi.claims import Claim, ClaimStatus
from synapsi.core import Problem
from synapsi.evidence import SourceKind
from synapsi.providers import CompletionRequest, MockProvider
from synapsi.research.steps import EvidenceCollection, VerifyClaims
from synapsi.testing import scripted
from synapsi.tools import StaticSearch, web_search_tool
from synapsi.workflows import RunContext, RunSettings

CORPUS = {
    "https://stats.example/delays": "Average departure delays rose 12 percent in winter 2025.",
    "https://news.example/weather": "Snowstorms grounded flights across the northeast.",
}


async def test_evidence_collection_creates_web_evidence() -> None:
    researcher = Agent(
        "res",
        "researcher",
        scripted(lambda r: {"queries": ["winter delays", "snowstorms flights"]}),
        tools=[web_search_tool(StaticSearch(CORPUS))],
    )
    ctx = RunContext(Problem(question="Why were winter delays high?"), [researcher])
    await EvidenceCollection().run(ctx)
    urls = {e.provenance.source_id for e in ctx.state.evidence}
    assert urls == set(CORPUS)
    assert all(e.provenance.source_kind is SourceKind.WEB for e in ctx.state.evidence)
    assert all(e.provenance.query for e in ctx.state.evidence)


async def test_collection_without_search_tools_warns() -> None:
    ctx = RunContext(Problem(question="q"), [Agent("a", model=MockProvider())])
    await EvidenceCollection().run(ctx)
    assert "no agent has a search tool" in ctx.state.warnings[0]


async def test_verify_claims_links_evidence_and_skips_own_claims() -> None:
    def link(req: CompletionRequest) -> dict[str, Any] | None:
        if req.metadata["task"] != "link_evidence":
            return None
        return {
            "links": [
                {
                    "evidence_id": eid,
                    "claim_id": req.metadata["claim_ids"][0],
                    "relation": "supports",
                }
                for eid in req.metadata["evidence_ids"]
                if eid
            ][:1]
        }

    checker = Agent(
        "fc", "fact_checker", scripted(link), tools=[web_search_tool(StaticSearch(CORPUS))]
    )
    ctx = RunContext(
        Problem(question="q"), [checker, Agent("an", model=MockProvider())], settings=RunSettings()
    )
    mine = ctx.state.claims.add(Claim(statement="winter delays rose 12 percent", agent="fc"))
    theirs = ctx.state.claims.add(Claim(statement="winter delays rose 12 percent", agent="an"))
    await VerifyClaims().run(ctx)
    assert ctx.state.claims.get(theirs.id).status is ClaimStatus.SUPPORTED  # type: ignore[union-attr]
    assert ctx.state.claims.get(mine.id).status is ClaimStatus.UNSUPPORTED  # type: ignore[union-attr]
