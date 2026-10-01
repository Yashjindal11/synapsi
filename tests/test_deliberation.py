from typing import Any

from synapsi.agents import Agent
from synapsi.claims import ClaimStatus
from synapsi.core import Problem
from synapsi.deliberation import ChallengeStatus
from synapsi.deliberation.disagreement import DisagreementAnalysis
from synapsi.deliberation.registration import normalize_answer
from synapsi.deliberation.steps import (
    ClaimAssessment,
    CollectVotes,
    CrossExamination,
    IndependentAnalysis,
    PeerReview,
    RespondToChallenges,
    Revision,
    SequentialCritique,
    SteelmanAlternative,
    converged,
)
from synapsi.evidence import Evidence, Provenance, SourceKind
from synapsi.observability import EventBus, Recorder
from synapsi.providers import CompletionRequest, MockProvider
from synapsi.testing import scripted
from synapsi.workflows import RunContext, RunSettings, Workflow


def ctx_for(*agents: Agent, **settings: Any) -> RunContext:
    return RunContext(
        Problem(question="Did delays rise?", options=["yes", "no"]),
        list(agents),
        settings=RunSettings(seed=7, **settings),
        events=EventBus([Recorder()]),
    )


def test_normalize_answer() -> None:
    opts = ["yes", "no"]
    assert normalize_answer(" Yes. ", opts) == "yes"
    assert normalize_answer("b", opts) == "no"
    assert normalize_answer("I think no", opts) == "no"
    assert normalize_answer("unknown", opts) is None
    assert normalize_answer("maybe", opts) == "maybe"
    assert normalize_answer("42", None) == "42"


async def test_independent_analysis_isolates_agents() -> None:
    agents = [Agent(n, model=MockProvider()) for n in ("a", "b", "c")]
    ctx = ctx_for(*agents)
    await IndependentAnalysis().run(ctx)
    assert set(ctx.state.perspectives) == {"a", "b", "c"}
    assert all(p.independent for p in ctx.state.perspectives.values())
    for agent in agents:
        prompt = agent.provider.requests[0].messages[-1].content  # type: ignore[attr-defined]
        assert "EARLIER ANALYSES" not in prompt
        others = {"a", "b", "c"} - {agent.name}
        assert not any(f"({o})" in prompt for o in others)
    # mock "basis" text becomes model-knowledge evidence, never external
    assert all(not e.is_external for e in ctx.state.evidence)
    await ClaimAssessment().run(ctx)
    assert all(c.status is ClaimStatus.UNVERIFIED for c in ctx.state.claims)


async def test_failed_agent_is_recorded_not_fatal() -> None:
    def boom(_: CompletionRequest) -> str:
        raise RuntimeError("provider down")

    ctx = ctx_for(
        Agent("ok", model=MockProvider()), Agent("bad", model=MockProvider(responder=boom))
    )
    await IndependentAnalysis().run(ctx)
    assert set(ctx.state.perspectives) == {"ok"}
    assert any("bad failed" in e for e in ctx.state.errors)


def _two_agent_script(response_type: str, evidence: list[str]) -> tuple[Agent, Agent]:
    def a_script(req: CompletionRequest) -> dict[str, Any] | None:
        if req.metadata["task"] == "analyze":
            return {
                "position": "Delays rose",
                "answer": "yes",
                "claims": [{"statement": "Delays rose 10%", "type": "statistical"}],
                "confidence": 0.8,
            }
        if req.metadata["task"] == "respond":
            return {
                "rebuttals": [
                    {
                        "challenge_id": req.metadata["challenge_ids"][0],
                        "response_type": response_type,
                        "response": "reply",
                        "evidence_ids": evidence,
                        "revised_statement": "Delays rose 5%"
                        if response_type == "revise"
                        else None,
                    }
                ]
            }
        return None

    def b_script(req: CompletionRequest) -> dict[str, Any] | None:
        if req.metadata["task"] == "analyze":
            return {"position": "No change", "answer": "no", "claims": [], "confidence": 0.6}
        if req.metadata["task"] == "challenge":
            return {
                "challenges": [
                    {"target_claim_id": "C1", "problem": "No data cited", "question": "Source?"},
                    {"target_claim_id": "C999", "problem": "invented id"},
                ]
            }
        if req.metadata["task"] == "revise":
            return {
                "position": "Now yes",
                "answer": "yes",
                "confidence": 0.7,
                "reason_for_change": "persuaded",
            }
        return None

    return Agent("a", model=scripted(a_script)), Agent("b", model=scripted(b_script))


async def _debate(ctx: RunContext) -> None:
    await IndependentAnalysis().run(ctx)
    ctx.state.round = 1
    await CrossExamination(challengers=["b"]).run(ctx)
    await RespondToChallenges().run(ctx)


async def test_defence_without_evidence_leaves_challenge_open() -> None:
    a, b = _two_agent_script("defend", [])
    ctx = ctx_for(a, b)
    await _debate(ctx)
    [challenge] = ctx.state.challenges
    assert challenge.target_claim_id == "C1" and challenge.status is ChallengeStatus.OPEN
    assert ctx.state.invalid_references == {"b": 1}
    assert len(ctx.state.rebuttals) == 1
    await ClaimAssessment().run(ctx)
    assert ctx.state.claims.get("C1").status is ClaimStatus.UNCERTAIN  # type: ignore[union-attr]


async def test_defence_with_external_evidence_answers_challenge() -> None:
    a, b = _two_agent_script("defend", ["E1"])
    ctx = ctx_for(a, b)
    ctx.add_evidence(
        Evidence(content="BTS data", provenance=Provenance(source_kind=SourceKind.DATABASE))
    )
    await _debate(ctx)
    assert ctx.state.challenges[0].status is ChallengeStatus.ANSWERED
    await ClaimAssessment().run(ctx)
    assert ctx.state.claims.get("C1").status is ClaimStatus.SUPPORTED  # type: ignore[union-attr]


async def test_revision_withdraws_and_refines_claim() -> None:
    a, b = _two_agent_script("revise", [])
    ctx = ctx_for(a, b)
    await _debate(ctx)
    old, new = ctx.state.claims.get("C1"), ctx.state.claims.get("C2")
    assert old is not None and old.withdrawn
    assert new is not None and new.revised_from == "C1" and new.statement == "Delays rose 5%"
    assert ctx.state.perspectives["a"].claim_ids == ["C2"]


async def test_position_change_without_evidence_is_flagged() -> None:
    a, b = _two_agent_script("concede", [])
    ctx = ctx_for(a, b)
    await _debate(ctx)
    await Revision(agents=["b"]).run(ctx)
    [change] = ctx.state.position_changes
    assert (change.from_answer, change.to_answer) == ("no", "yes")
    assert not change.cited_new_evidence
    assert ctx.state.perspectives["b"].independent is False
    assert not converged(ctx)


async def test_disagreement_engine_reports_sides() -> None:
    a, b = _two_agent_script("defend", [])
    ctx = ctx_for(a, b)
    await _debate(ctx)
    await ClaimAssessment().run(ctx)
    await DisagreementAnalysis().run(ctx)
    kinds = [d.kind for d in ctx.state.disagreements]
    assert kinds == ["answer", "claim"]
    answer = ctx.state.disagreements[0]
    assert {s.position for s in answer.sides} == {"yes", "no"}
    claim = ctx.state.disagreements[1]
    assert claim.status == "unresolved" and claim.sides[1].agents == ["b"]


async def test_full_mock_pipeline_runs() -> None:
    agents = [
        Agent("r", "researcher", MockProvider()),
        Agent("s", "skeptic", MockProvider()),
        Agent("t", "statistician", MockProvider()),
    ]
    ctx = ctx_for(*agents)
    wf = Workflow(
        [
            IndependentAnalysis(),
            SteelmanAlternative(),
            PeerReview(),
            CrossExamination(adversarial=True),
            RespondToChallenges(),
            Revision(),
            CollectVotes(),
            ClaimAssessment(),
            DisagreementAnalysis(),
        ]
    )
    await wf.execute(ctx)
    assert not ctx.state.errors, ctx.state.errors
    assert any(p.stance == "adversarial" for p in ctx.state.history)
    assert ctx.state.reviews and ctx.state.votes
    assert all(r.blind for r in ctx.state.reviews)


async def test_sequential_critique_chains_agents() -> None:
    agents = [Agent(n, model=MockProvider()) for n in ("first", "second")]
    ctx = ctx_for(*agents)
    await SequentialCritique().run(ctx)
    assert ctx.state.perspectives["first"].independent
    assert not ctx.state.perspectives["second"].independent
    prompt = agents[1].provider.requests[-1].messages[-1].content  # type: ignore[attr-defined]
    assert "EARLIER ANALYSES" in prompt
