"""3. Machine-learning model review: user-provided facts are the only external evidence."""

import asyncio

from _common import model, show

from synapsi import Agent, Council

FACTS = [
    "Validation AUC 0.94 on a random 80/20 split of 2023-2025 data.",
    "Test AUC on 2026 data (time-based holdout) is 0.81.",
    "The feature 'days_since_last_claim' is computed using the full dataset, including future rows.",
    "Positive class prevalence: 3.2% in training, 5.9% in 2026.",
]


async def main() -> None:
    spec = model()
    agents = [
        Agent.from_role("data_scientist", spec),
        Agent.from_role("statistician", spec),
        Agent.from_role("risk_analyst", spec),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="debate", mode="balanced", seed=11)
    result = await council.run(
        "Is this fraud model ready for production?",
        options=["ready", "not ready", "ready with conditions"],
        facts=FACTS,
    )
    show(result)
    print("\nChallenges raised:")
    for ch in result.challenges:
        print(
            f"  {ch.id} {ch.challenger} -> {ch.target_claim_id} [{ch.status.value}] {ch.problem[:90]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
