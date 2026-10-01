"""2. Data science: agents query a (synthetic) A/B test database and calculate.

Tool results become `database` / `calculation` evidence with provenance, so
claims built on them can reach `supported`; unsupported recollections cannot.
"""

import asyncio
import random
import sqlite3
import tempfile
from pathlib import Path

from _common import model, show

from synapsi import Agent, Council
from synapsi.tools import calculator_tool, sql_tool


def build_db(path: Path, seed: int = 1) -> None:
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (variant TEXT, converted INTEGER, revenue REAL)")
    rows = []
    for variant, rate in (("A", 0.110), ("B", 0.118)):
        for _ in range(4000):
            converted = int(rng.random() < rate)
            rows.append((variant, converted, round(rng.uniform(20, 80), 2) if converted else 0.0))
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


async def main() -> None:
    db = Path(tempfile.mkdtemp()) / "ab_test.db"
    build_db(db)
    tools = [sql_tool(db), calculator_tool()]
    spec = model()
    agents = [
        Agent.from_role("data_scientist", spec, tools=tools),
        Agent.from_role("statistician", spec, tools=tools),
        Agent.from_role("skeptic", spec),
    ]
    council = Council(agents, strategy="peer_review", mode="fast", seed=3)
    result = await council.run(
        "Is variant B's conversion rate meaningfully better than A's, enough to ship B?",
        options=["ship B", "keep A", "run longer"],
    )
    show(result)
    print("\nEvidence produced by tools:")
    for e in result.evidence:
        if e.is_external:
            print(f"  {e.id} [{e.provenance.source_kind.value}] {e.content[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
