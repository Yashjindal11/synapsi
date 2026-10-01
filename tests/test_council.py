import pytest

from synapsi import (
    Agent,
    Budget,
    Council,
    SynapSIResult,
    Verdict,
    build_strategy,
    list_strategies,
)
from synapsi.core.errors import ConfigError
from synapsi.observability import EventType, Recorder
from synapsi.providers import MockProvider
from synapsi.tools import StaticSearch, web_search_tool

STRATEGIES = sorted(list_strategies())


def panel() -> list[Agent]:
    search = web_search_tool(StaticSearch({"https://x.org": "delays rose sharply in winter"}))
    return [
        Agent("Researcher", "researcher", MockProvider(), tools=[search]),
        Agent("Statistician", "statistician", MockProvider()),
        Agent("Skeptic", "skeptic", MockProvider()),
        Agent("Reviewer", "judge", MockProvider(), level=2),
    ]


@pytest.mark.parametrize("strategy", STRATEGIES)
async def test_every_strategy_runs_end_to_end(strategy: str) -> None:
    recorder = Recorder()
    council = Council(panel(), strategy=strategy, mode="fast", seed=1, event_handlers=[recorder])
    result = await council.run("Did delays rise?", options=["yes", "no"], facts=["Winter 2025."])
    assert result.metadata.strategy == strategy
    assert not result.metadata.errors, result.metadata.errors
    assert result.judgment is not None and result.synthesis is not None
    assert result.perspectives
    assert any(e.provenance.source_kind.value == "user_provided" for e in result.evidence)
    assert recorder.of_type(EventType.SYNTHESIS_CREATED)
    restored = SynapSIResult.model_validate_json(result.to_json())
    assert restored.run_id == result.run_id
    assert result.claim_graph()["nodes"]


async def test_single_model_uses_only_first_agent() -> None:
    result = await Council(panel(), strategy="single_model").run("q", options=["a", "b"])
    assert {p.agent for p in result.perspectives} == {"Researcher"}
    assert result.judgment is not None and result.judgment.method == "single"


async def test_majority_vote_is_labelled_as_vote() -> None:
    result = await Council(panel(), strategy="majority_vote").run("q", options=["a", "b"])
    assert result.judgment is not None and result.judgment.method == "vote"


async def test_structural_judge_option() -> None:
    result = await Council(panel()[:2], judge="structural").run("q", options=["a", "b"])
    assert result.judgment is not None and result.judgment.method == "structural"
    # mock claims carry no external support, so nothing can be decided
    assert result.judgment.verdict is Verdict.INCONCLUSIVE


async def test_budget_stops_and_still_finalizes() -> None:
    council = Council(panel(), strategy="debate", budget=Budget(max_model_calls=3))
    result = await council.run("q", options=["a", "b"])
    assert "budget" in (result.metadata.stopped_reason or "")
    assert result.synthesis is not None
    assert result.metadata.usage.total.calls <= 3


async def test_runs_are_reproducible_with_seed() -> None:
    a = await Council(panel(), strategy="debate", seed=5).run("q", options=["a", "b"])
    b = await Council(panel(), strategy="debate", seed=5).run("q", options=["a", "b"])
    assert a.answer == b.answer
    assert [c.statement for c in a.claims] == [c.statement for c in b.claims]


def test_preset_sizes() -> None:
    assert len(Council.preset("fast").agents) == 3
    assert len(Council.preset("balanced").agents) == 5
    assert len(Council.preset("deep").agents) == 8
    with pytest.raises(ConfigError):
        build_strategy("nope")


async def test_trace_dir_writes_jsonl(tmp_path: object) -> None:
    council = Council(panel()[:2], trace_dir=str(tmp_path))
    result = await council.run("q")
    trace = (tmp_path / f"{result.run_id}.jsonl").read_text()  # type: ignore[operator]
    assert "workflow_completed" in trace
