"""1. General decision analysis with a default role panel."""

import asyncio

from _common import model, show

from synapsi import Council


async def main() -> None:
    council = Council.preset("balanced", model(), strategy="independent_panel", seed=7)
    result = await council.run(
        "Should a 40-person startup move from a monolith to microservices this year?",
        options=["yes", "no", "partially"],
        context="Two backend teams, one product, deploys weekly, growing 3x per year.",
    )
    show(result)
    print("\nIndependent first-pass answers:", result.uncertainty.initial_answer_distribution)


if __name__ == "__main__":
    asyncio.run(main())
