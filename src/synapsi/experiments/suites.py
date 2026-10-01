"""Synthetic, reproducible problem suites with computed gold answers.

These are small instruments for comparing strategies, not general-purpose
benchmarks. Every answer is computed from generated data, so nothing depends on
proprietary datasets or on any model's knowledge.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from synapsi.core.problem import Problem
from synapsi.experiments.problems import ProblemSet

NAMES = ["Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"]
HUBS = ["ORD", "DEN", "IAH", "EWR", "SFO", "IAD"]


def arithmetic(n: int = 20, seed: int = 0) -> ProblemSet:
    """Multi-step inventory arithmetic. Numeric answers."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        start, shipments, per, out, damaged = (
            rng.randint(100, 900),
            rng.randint(2, 9),
            rng.randint(12, 60),
            rng.randint(50, 300),
            rng.randint(1, 25),
        )
        answer = start + shipments * per - out - damaged
        problems.append(
            Problem(
                id=f"arith-{seed}-{i}",
                question=(
                    f"A warehouse starts with {start} boxes. It receives {shipments} shipments of "
                    f"{per} boxes each, ships out {out} boxes, and discards {damaged} damaged "
                    "boxes. How many boxes remain? Answer with a number."
                ),
                answer=str(answer),
                metadata={"suite": "arithmetic"},
            )
        )
    return ProblemSet(problems, name=f"arithmetic-{seed}")


def ordering(n: int = 20, seed: int = 0, size: int = 5) -> ProblemSet:
    """Transitive ordering puzzles with shuffled premises. Multiple choice."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        people = rng.sample(NAMES, size)
        premises = [f"{people[k]} arrived before {people[k + 1]}." for k in range(size - 1)]
        rng.shuffle(premises)
        ask_first = rng.random() < 0.5
        target = people[0] if ask_first else people[-1]
        options = sorted(people)
        problems.append(
            Problem(
                id=f"order-{seed}-{i}",
                question=" ".join(premises) + f" Who arrived {'first' if ask_first else 'last'}?",
                options=options,
                answer=target,
                metadata={"suite": "ordering"},
            )
        )
    return ProblemSet(problems, name=f"ordering-{seed}")


def base_rates(n: int = 20, seed: int = 0) -> ProblemSet:
    """Bayesian base-rate questions, a known source of intuitive errors. Yes/no."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        prevalence = rng.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
        sensitivity = rng.choice([0.8, 0.9, 0.95, 0.99])
        fpr = rng.choice([0.01, 0.02, 0.05, 0.1])
        posterior = sensitivity * prevalence / (sensitivity * prevalence + fpr * (1 - prevalence))
        threshold = rng.choice([0.25, 0.5, 0.75])
        problems.append(
            Problem(
                id=f"bayes-{seed}-{i}",
                question=(
                    f"A condition affects {prevalence:.1%} of a population. A test detects it "
                    f"{sensitivity:.0%} of the time when present and returns a false positive "
                    f"{fpr:.0%} of the time when absent. A random person tests positive. Is the "
                    f"probability they have the condition above {threshold:.0%}?"
                ),
                options=["yes", "no"],
                answer="yes" if posterior > threshold else "no",
                metadata={"suite": "base_rates", "posterior": round(posterior, 4)},
            )
        )
    return ProblemSet(problems, name=f"base_rates-{seed}")


def aviation_delays(n: int = 10, seed: int = 0, flights_per_hub: int = 12) -> ProblemSet:
    """Synthetic hub operations data supplied as user facts.

    Asks which hub had the highest share of departures delayed 15+ minutes
    (D15). Data is generated; it does not describe any real airline.
    """
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        hubs = rng.sample(HUBS, 4)
        facts: list[str] = []
        shares: dict[str, float] = {}
        for hub in hubs:
            base = rng.uniform(2, 22)
            delays = [max(0, round(rng.gauss(base, 14))) for _ in range(flights_per_hub)]
            shares[hub] = sum(d >= 15 for d in delays) / len(delays)
            facts.append(
                f"{hub} departure delays in minutes for {len(delays)} flights: "
                + ", ".join(map(str, delays))
            )
        best = max(shares.values())
        if sum(v == best for v in shares.values()) > 1:
            continue
        winner = max(shares, key=lambda h: shares[h])
        problems.append(
            Problem(
                id=f"avd15-{seed}-{i}",
                question=(
                    "Using only the provided data, which hub had the highest share of departures "
                    "delayed by 15 minutes or more?"
                ),
                options=sorted(hubs),
                facts=facts,
                answer=winner,
                metadata={"suite": "aviation_delays", "d15_share": shares},
            )
        )
    return ProblemSet(problems, name=f"aviation_delays-{seed}")


SUITES: dict[str, Callable[..., ProblemSet]] = {
    "arithmetic": arithmetic,
    "ordering": ordering,
    "base_rates": base_rates,
    "aviation_delays": aviation_delays,
}
