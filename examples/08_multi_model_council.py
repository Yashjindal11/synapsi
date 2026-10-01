"""8. Multi-model council: each agent on a different provider where keys are available.

Falls back to distinct mock models so the example always runs. Model diversity
is a hypothesis to test (see 13_experiment.py), not an assumed benefit.
"""

import asyncio
import os

from _common import show

from synapsi import Agent, Council, Judge

CANDIDATES = [
    ("OPENAI_API_KEY", "openai:gpt-4o-mini"),
    ("ANTHROPIC_API_KEY", "anthropic:claude-3-5-haiku-latest"),
    ("GEMINI_API_KEY", "gemini:gemini-2.0-flash"),
]


def pick_models() -> list[str]:
    models = [spec for env, spec in CANDIDATES if os.environ.get(env)]
    if os.environ.get("SYNAPSI_OLLAMA_MODEL"):
        models.append(f"ollama:{os.environ['SYNAPSI_OLLAMA_MODEL']}")
    return models or ["mock:model-a", "mock:model-b", "mock:model-c"]


async def main() -> None:
    models = pick_models()
    roles = ["researcher", "statistician", "skeptic", "risk_analyst"]
    agents = [Agent.from_role(role, models[i % len(models)]) for i, role in enumerate(roles)]
    for a in agents:
        print(f"{a.name:<14} -> {a.model_id}")
    judge = Judge(models[-1])
    council = Council(agents, strategy="peer_review", judge=judge, mode="fast", seed=1)
    result = await council.run(
        "Will a 10% fare increase on a monopoly route reduce total revenue?",
        options=["yes", "no", "depends on elasticity"],
    )
    show(result)
    print("\nUsage by model:")
    for model_id, usage in result.metadata.usage.by_model.items():
        print(f"  {model_id}: {usage.calls} calls, {usage.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
