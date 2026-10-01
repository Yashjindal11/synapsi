"""Read-only SQLite queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from synapsi.evidence.models import SourceKind
from synapsi.tools.base import Tool, ToolResult

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def _authorizer(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def sql_tool(database: str | Path, *, max_rows: int = 50) -> Tool:
    """Query a SQLite database opened read-only; only SELECT statements run."""
    path = Path(database).resolve()

    def sql_query(query: str) -> ToolResult:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.set_authorizer(_authorizer)
            cursor = conn.execute(query)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
        finally:
            conn.close()
        truncated = len(rows) > max_rows
        lines = [" | ".join(columns)] + [" | ".join(map(str, r)) for r in rows[:max_rows]]
        if truncated:
            lines.append(f"... truncated to {max_rows} rows")
        return ToolResult(tool="sql_query", content=f"SQL: {query}\n" + "\n".join(lines))

    schema = _schema(path)
    return Tool(
        "sql_query",
        f"Run a read-only SQL SELECT against the database. Schema: {schema}",
        sql_query,
        parameters={"query": "str"},
        evidence_kind=SourceKind.DATABASE,
    )


def _schema(path: Path) -> str:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return "; ".join(r[0] for r in rows if r[0])[:1500]
