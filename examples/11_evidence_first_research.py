"""11. Evidence-first research with web search.

Uses Tavily when TAVILY_API_KEY is set; otherwise an offline static corpus so
the example is reproducible. Either way, only tool results become `web`
evidence — an agent saying "studies show" stays `model_knowledge`.
"""

import asyncio
import os

from _common import model, show

from synapsi import Agent, Council
from synapsi.tools import StaticSearch, TavilySearch, fetch_url_tool, web_search_tool

OFFLINE = {
    "https://example.org/sleep-study": (
        "A 2024 randomized trial (n=212) found 30 minutes of afternoon napping improved "
        "reaction time by 6% in shift workers; no effect on memory tasks."
    ),
    "https://example.org/nap-review": (
        "A review of 18 studies reports benefits of short naps (10-30 min) on alertness, "
        "while naps over 45 minutes were associated with grogginess (sleep inertia)."
    ),
}


async def main() -> None:
    backend = TavilySearch() if os.environ.get("TAVILY_API_KEY") else StaticSearch(OFFLINE)
    search = web_search_tool(backend)
    spec = model()
    agents = [
        Agent.from_role("researcher", spec, tools=[search, fetch_url_tool()]),
        Agent.from_role("fact_checker", spec, tools=[search]),
        Agent.from_role("statistician", spec),
    ]
    council = Council(agents, strategy="evidence_first", mode="balanced", seed=21)
    result = await council.run(
        "Do short afternoon naps improve alertness for shift workers?",
        options=["yes", "no", "mixed evidence"],
    )
    show(result)
    print("\nProvenance:")
    for src in result.provenance:
        print(
            f"  {src.evidence_id} {src.source_kind} {src.source_id} (via {src.tool}, by {src.produced_by})"
        )
    dep = result.uncertainty.source_dependence
    print(
        f"distinct sources={dep.distinct_sources} shared-source ratio={dep.shared_source_ratio:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
