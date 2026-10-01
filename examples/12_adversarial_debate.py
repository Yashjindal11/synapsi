"""12. Adversarial debate: pressure-test the leading answer and watch for conformity."""

import asyncio

from _common import model, show

from synapsi import Agent, Council


async def main() -> None:
    spec = model()
    agents = [
        Agent.from_role("domain_expert", spec),
        Agent.from_role("statistician", spec),
        Agent.from_role("devils_advocate", spec),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="adversarial", mode="balanced", seed=31)
    result = await council.run(
        "Did the four-day work week pilot cause the productivity increase reported by the company?",
        options=["yes", "no", "cannot be determined"],
        facts=[
            "Output per employee rose 8% during the 6-month pilot.",
            "The pilot started the same quarter a new CRM system was rolled out.",
            "Only teams that volunteered joined the pilot.",
        ],
    )
    show(result)
    u = result.uncertainty
    print(f"\nagreement: initial {u.initial_agreement_rate} -> final {u.agreement_rate}")
    print(
        f"position changes: {u.position_changes} "
        f"({u.unsupported_position_changes} without new evidence or concession)"
    )
    for d in result.disagreements:
        if d.kind == "answer":
            for side in d.sides:
                print(
                    f"  side '{side.position}': {side.agents} "
                    f"({side.external_evidence_count} external evidence)"
                )


if __name__ == "__main__":
    asyncio.run(main())
