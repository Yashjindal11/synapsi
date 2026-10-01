import pytest

from synapsi.agents import Agent, get_role, list_roles
from synapsi.agents.drafts import PerspectiveDraft
from synapsi.core import Problem
from synapsi.core.errors import ConfigError, StopWorkflow
from synapsi.evidence import SourceKind
from synapsi.observability import EventBus, EventType, Recorder
from synapsi.providers import MockProvider
from synapsi.tools import calculator_tool
from synapsi.workflows import (
    Budget,
    Conditional,
    FunctionStep,
    Loop,
    Parallel,
    RunContext,
    RunSettings,
    Sequential,
    Workflow,
)


def make_ctx(*agents: Agent, **settings: object) -> tuple[RunContext, Recorder]:
    recorder = Recorder()
    ctx = RunContext(
        Problem(question="Is it A or B?", options=["A", "B"], answer="A"),
        list(agents) or [Agent("a1")],
        settings=RunSettings(seed=1, **settings),  # type: ignore[arg-type]
        events=EventBus([recorder]),
    )
    return ctx, recorder


def test_roles_registry() -> None:
    assert {"skeptic", "statistician", "devils_advocate"} <= {r.key for r in list_roles()}
    assert get_role("Devil's Advocate").stance == "contrarian"
    custom = get_role("aviation_economist")
    assert custom.title == "Aviation Economist"


def test_context_hides_gold_answer_and_requires_unique_names() -> None:
    ctx, _ = make_ctx()
    assert ctx.problem.answer is None
    with pytest.raises(ConfigError):
        make_ctx(Agent("x"), Agent("x"))


def test_blind_labels_roundtrip() -> None:
    ctx, _ = make_ctx(Agent("alice"), Agent("bob"))
    label = ctx.label("alice")
    assert label.startswith("Analyst ") and ctx.unlabel(label) == "alice"
    ctx.settings.blind_review = False
    assert ctx.label("alice") == "alice"


async def test_workflow_composites_and_loop() -> None:
    ctx, recorder = make_ctx()
    trace: list[str] = []

    def mark(tag: str) -> FunctionStep:
        return FunctionStep(lambda c: trace.append(f"{tag}{c.state.round}"), name=tag)

    wf = Workflow(
        [
            Sequential([mark("s"), Parallel([mark("p"), mark("q")])]),
            Loop(mark("l"), max_iterations=5, until=lambda c: c.state.round >= 2),
            Conditional(lambda c: c.state.round == 2, mark("yes"), mark("no")),
        ]
    )
    await wf.execute(ctx)
    assert trace == ["s0", "p0", "q0", "l1", "l2", "yes2"]
    assert recorder.of_type(EventType.WORKFLOW_COMPLETED)


async def test_stop_workflow_still_runs_finalize() -> None:
    ctx, _ = make_ctx()
    ran: list[str] = []

    def stop(_: RunContext) -> None:
        raise StopWorkflow("enough")

    wf = Workflow(
        [FunctionStep(stop), FunctionStep(lambda c: ran.append("skipped"))],
        finalize=[FunctionStep(lambda c: ran.append("final"))],
    )
    await wf.execute(ctx)
    assert ran == ["final"] and ctx.state.stopped_reason == "enough"


async def test_agent_analyze_uses_structured_output_and_tracks_usage() -> None:
    agent = Agent("stat", "statistician", MockProvider())
    ctx, recorder = make_ctx(agent)
    draft = await agent.analyze(ctx)
    assert isinstance(draft, PerspectiveDraft)
    assert draft.answer in {"A", "B"}
    assert ctx.usage.total.calls == 1
    [call] = recorder.of_type(EventType.MODEL_CALL)
    assert call.data["task"] == "analyze" and call.agent == "stat"
    system = agent.provider.requests[0].messages[0].content  # type: ignore[attr-defined]
    assert "Statistician" in system and "not evidence" in system
    assert "A" not in agent.provider.requests[0].metadata.get("answer", "")  # type: ignore[attr-defined]


async def test_budget_stops_run_gracefully() -> None:
    agent = Agent("a", model=MockProvider())
    ctx, _ = make_ctx(agent, budget=Budget(max_model_calls=1))

    async def two_calls(c: RunContext) -> None:
        await agent.analyze(c)
        await agent.analyze(c)

    await Workflow([FunctionStep(two_calls)]).execute(ctx)
    assert ctx.usage.total.calls == 1
    assert "budget" in (ctx.state.stopped_reason or "")


async def test_agent_tool_loop_creates_external_evidence() -> None:
    replies = iter(
        [
            '{"tool_calls": [{"tool": "calculator", "arguments": {"expression": "6*7"}}]}',
            '{"tool_calls": []}',
        ]
    )
    agent = Agent(
        "calc", model=MockProvider(responder=lambda r: next(replies)), tools=[calculator_tool()]
    )
    ctx, recorder = make_ctx(agent)
    evidence = await agent.gather_with_tools(ctx)
    assert [e.content for e in evidence] == ["6*7 = 42"]
    assert evidence[0].provenance.source_kind is SourceKind.CALCULATION
    assert evidence[0].provenance.produced_by == "calc"
    assert recorder.of_type(EventType.TOOL_RESULT)
