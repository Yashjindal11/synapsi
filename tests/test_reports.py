from synapsi import Agent, Council
from synapsi.providers import MockProvider
from synapsi.testing import scripted


async def test_markdown_and_html_reports() -> None:
    agents = [Agent("A", model=MockProvider()), Agent("B", "skeptic", MockProvider())]
    result = await Council(agents, strategy="debate", mode="fast").run("Q?", options=["x", "y"])
    md = result.to_markdown()
    for heading in (
        "## Synthesis",
        "### Established",
        "### Disputed",
        "## Claims",
        "## Run metadata",
    ):
        assert heading in md
    assert "cost: $0.0000" in md
    page = result.to_html()
    assert page.startswith("<!doctype html>") and "Claims" in page


async def test_html_escapes_model_output() -> None:
    payload = "<script>alert(1)</script>"

    def script(req: object) -> dict[str, object] | None:
        return {"position": payload, "answer": "x", "claims": [{"statement": payload}]}

    result = await Council([Agent("A", model=scripted(script))], judge="structural").run(
        "Q?", options=["x", "y"]
    )
    page = result.to_html()
    assert payload not in page
    assert "&lt;script&gt;" in page
