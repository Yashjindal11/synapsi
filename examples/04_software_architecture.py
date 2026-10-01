"""4. Software architecture review with two engineers holding different priorities."""

import asyncio

from _common import model, show

from synapsi import Agent, Council


async def main() -> None:
    spec = model()
    agents = [
        Agent(
            "Platform Engineer",
            "software_engineer",
            spec,
            instructions="You prioritise operability, on-call load, and failure isolation.",
        ),
        Agent(
            "Product Engineer",
            "software_engineer",
            spec,
            instructions="You prioritise delivery speed and developer experience.",
        ),
        Agent.from_role("risk_analyst", spec),
        Agent.from_role("devils_advocate", spec),
    ]
    council = Council(agents, strategy="adversarial", mode="fast", seed=5)
    result = await council.run(
        "Should we replace our Postgres job queue with Kafka?",
        options=["adopt Kafka", "keep Postgres queue", "adopt a managed queue"],
        context="Peak 400 jobs/s, at-least-once semantics needed, 3 engineers on the platform team.",
    )
    show(result)
    print("\nAdversarial perspectives:")
    for p in result.perspectives:
        if p.stance == "adversarial":
            print(f"  {p.agent}: argued '{p.answer}' (honest confidence {p.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
