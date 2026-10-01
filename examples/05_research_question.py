"""5. Research question over a local document collection (evidence-first)."""

import asyncio
from pathlib import Path

from _common import model, show

from synapsi import Agent, Council
from synapsi.tools import DocumentStore, document_search_tool

DOCS = Path(__file__).parent / "data" / "notes"


async def main() -> None:
    store = DocumentStore()
    store.add_directory(DOCS)
    search = document_search_tool(store)
    spec = model()
    agents = [
        Agent.from_role("researcher", spec, tools=[search]),
        Agent.from_role("fact_checker", spec, tools=[search]),
        Agent.from_role("statistician", spec),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="evidence_first", mode="deep", seed=2)
    result = await council.run(
        "Does remote check-in reduce passenger missed connections at hub airports?",
        options=["yes", "no", "insufficient evidence"],
    )
    show(result)
    print("\nDocument evidence and the claims it bears on:")
    for e in result.evidence:
        if e.provenance.source_kind.value == "document":
            print(
                f"  {e.id} {e.provenance.source_id} supports={e.supports} contradicts={e.contradicts}"
            )


if __name__ == "__main__":
    asyncio.run(main())
