import sqlite3
from pathlib import Path

import httpx
import pytest

from synapsi.evidence import SourceKind
from synapsi.tools import (
    DocumentStore,
    StaticSearch,
    calculator_tool,
    document_search_tool,
    fetch_url_tool,
    sql_tool,
    tool,
    web_search_tool,
)
from synapsi.tools.calculator import CalculatorError, evaluate
from synapsi.tools.web import html_to_text


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("2 + 3 * 4", 14), ("sqrt(16)", 4.0), ("mean([1, 2, 3])", 2.0), ("-2 ** 2", -4), ("pi > 3", None)],
)
def test_calculator(expr: str, expected: float | None) -> None:
    if expected is None:
        with pytest.raises(CalculatorError):
            evaluate(expr)
    else:
        assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "open('x')", "2 ** 99999", "(1).__class__", "x", "a" * 600]
)
def test_calculator_rejects_unsafe(expr: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(expr)


async def test_tool_wraps_sync_functions_and_errors() -> None:
    calc = calculator_tool()
    ok = await calc(expression="1 + 1")
    assert ok.content == "1 + 1 = 2" and ok.error is None
    bad = await calc(expression="import os")
    assert bad.error and "invalid" in bad.error
    unknown = await calc(expr="1")
    assert unknown.error and "unknown arguments" in unknown.error
    [ev] = ok.to_evidence(SourceKind.CALCULATION, produced_by="A", call_id="T1")
    assert ev.provenance.source_kind is SourceKind.CALCULATION and ev.is_external


async def test_decorator_infers_parameters() -> None:
    @tool(description="Add numbers")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    assert add.parameters == {"a": "int", "b": "int (optional)"}
    assert (await add(a=1, b=2)).content == "3"


async def test_document_search_ranks_relevant_chunk() -> None:
    store = DocumentStore()
    store.add("Turnaround time at hub airports is 45 minutes.", source_id="ops.md")
    store.add("The cafeteria serves lunch at noon.", source_id="hr.md")
    result = await document_search_tool(store)(query="hub turnaround minutes")
    assert result.sources[0].source_id == "ops.md"
    [ev, *_] = result.to_evidence(SourceKind.DOCUMENT)
    assert ev.fingerprint == "document:ops.md"


async def test_sql_tool_is_read_only(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE flights (id INTEGER, delay REAL)")
    conn.executemany("INSERT INTO flights VALUES (?, ?)", [(1, 5.0), (2, 15.0)])
    conn.commit()
    conn.close()
    sql = sql_tool(db)
    assert "flights" in sql.description
    ok = await sql(query="SELECT avg(delay) FROM flights")
    assert "10.0" in ok.content
    denied = await sql(query="DELETE FROM flights")
    assert denied.error
    denied = await sql(query="ATTACH DATABASE ':memory:' AS other")
    assert denied.error


async def test_static_web_search() -> None:
    search = web_search_tool(StaticSearch({"https://a.org": "delays rose in winter"}))
    result = await search(query="winter delays")
    [ev] = result.to_evidence(search.evidence_kind, query="winter delays")
    assert ev.provenance.source_id == "https://a.org"
    assert ev.provenance.source_kind is SourceKind.WEB


async def test_fetch_url_blocks_private_addresses() -> None:
    fetch = fetch_url_tool()
    for url in ["http://127.0.0.1/admin", "http://169.254.169.254/latest", "file:///etc/passwd"]:
        assert (await fetch(url=url)).error


async def test_fetch_url_extracts_text_when_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        html = "<html><head><title>t</title></head><body><script>x()</script><p>Hello</p></body>"
        return httpx.Response(200, html=html)

    fetch = fetch_url_tool(allow_private=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await fetch(url="http://localhost/page")
    assert result.sources[0].content == "Hello"


def test_html_to_text_skips_scripts() -> None:
    assert html_to_text("<p>a</p><style>.x{}</style><p>b</p>") == "a b"
