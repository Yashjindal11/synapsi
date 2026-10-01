"""7. Aviation operations on synthetic, reproducible data.

Generates per-flight departure delays for four hubs (no real airline data),
loads them into SQLite for agents to query, and asks which hub has the worst
D15 rate (share of departures delayed 15+ minutes). The gold answer is
computed from the data, so the run can be scored.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from _common import model, show

from synapsi import Agent, Council
from synapsi.experiments.scoring import auto_scorer
from synapsi.experiments.suites import aviation_delays
from synapsi.tools import calculator_tool, sql_tool


def to_sqlite(facts: list[str], path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE departures (hub TEXT, delay_min INTEGER)")
    for line in facts:
        hub, _, values = line.partition(" departure delays in minutes for ")
        delays = values.split(": ", 1)[1].split(", ")
        conn.executemany("INSERT INTO departures VALUES (?, ?)", [(hub, int(d)) for d in delays])
    conn.commit()
    conn.close()


async def main() -> None:
    problem = next(iter(aviation_delays(n=3, seed=2026)))
    db = Path(tempfile.mkdtemp()) / "ops.db"
    to_sqlite(problem.facts, db)
    tools = [sql_tool(db), calculator_tool()]
    spec = model()
    agents = [
        Agent("Ops Analyst", "data_scientist", spec, tools=tools),
        Agent("Statistician", "statistician", spec, tools=tools),
        Agent(
            "Network Planner",
            "domain_expert",
            spec,
            instructions="You know airline hub operations and on-time performance metrics.",
        ),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="debate", mode="fast", seed=4)
    result = await council.run(problem)
    show(result)
    print(
        f"\ngold={problem.answer} predicted={result.answer} "
        f"correct={auto_scorer(problem, result.answer)}"
    )
    print("D15 shares (ground truth):", problem.metadata["d15_share"])


if __name__ == "__main__":
    asyncio.run(main())
