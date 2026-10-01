"""6. Business analysis as a jury: agents argue, an isolated judge on its own model decides."""

import asyncio

from _common import model, show

from synapsi import Agent, Council, Judge


async def main() -> None:
    spec = model()
    agents = [
        Agent(
            "Finance", "domain_expert", spec, instructions="You are a corporate finance analyst."
        ),
        Agent("Market", "domain_expert", spec, instructions="You analyse customer demand."),
        Agent.from_role("risk_analyst", spec),
        Agent.from_role("skeptic", spec),
    ]
    judge = Judge(model(), blind=True, show_votes=False, temperature=0.0)
    council = Council(agents, strategy="jury", judge=judge, mode="fast", seed=9)
    result = await council.run(
        "Should a regional coffee chain launch a subscription plan?",
        options=["launch", "pilot first", "do not launch"],
        facts=[
            "40% of revenue comes from customers visiting 4+ times per week.",
            "Average ticket is $6.10; gross margin per drink is 68%.",
            "A competitor's $40/month unlimited plan was withdrawn after 9 months.",
        ],
    )
    show(result)
    if result.judgment and result.judgment.minority_positions:
        print("Minority positions kept visible:", result.judgment.minority_positions)


if __name__ == "__main__":
    asyncio.run(main())
