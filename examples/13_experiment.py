"""13. Experiment: does deliberation beat a single model on base-rate questions?

Runs baselines and deliberative strategies on the same seeded problems with the
same agents and reports accuracy (with 95% CIs), coverage, calibration, flips
from a correct initial majority to a wrong final answer, cost, and an exact
McNemar test against the single-model baseline.

With the mock provider the numbers are meaningless; set SYNAPSI_MODEL.
"""

import asyncio

from _common import model

from synapsi import Agent
from synapsi.experiments import Experiment
from synapsi.experiments.suites import base_rates


async def main() -> None:
    spec = model()

    def agents() -> list[Agent]:
        return [Agent.from_role(r, spec) for r in ("analyst", "statistician", "skeptic")]

    experiment = Experiment(
        base_rates(n=12, seed=0),
        ["single_model", "majority_vote", "independent_panel", "debate", "adversarial"],
        agents=agents,
        mode="fast",
        repeats=1,
        seed=0,
        name="base-rates-vs-deliberation",
    )
    result = await experiment.run()
    print(result.to_markdown())
    result.save("runs/experiments/base-rates")


if __name__ == "__main__":
    asyncio.run(main())
