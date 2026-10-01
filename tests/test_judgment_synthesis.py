from typing import Any

from synapsi.agents import Agent
from synapsi.claims import Claim, ClaimStatus
from synapsi.core import Problem
from synapsi.deliberation.disagreement import DisagreementAnalysis
from synapsi.deliberation.models import Perspective, Vote
from synapsi.deliberation.steps import ClaimAssessment
from synapsi.evidence import Evidence, Provenance, SourceKind
from synapsi.judgment import EvidenceStrength, Verdict
from synapsi.judgment.judge import Judge, build_digest
from synapsi.judgment.steps import JudgeStep
from synapsi.judgment.structural import StructuralJudge, majority_judgment
from synapsi.providers import CompletionRequest, MockProvider
from synapsi.synthesis.synthesizer import SynthesisStep, Synthesizer
from synapsi.testing import scripted
from synapsi.workflows import RunContext, RunSettings


def _ctx(judge: Any = None) -> RunContext:
    agents = [Agent(n, model=MockProvider()) for n in ("a", "b", "c")]
    return RunContext(
        Problem(question="Q?", options=["x", "y"]),
        agents,
        judge=judge,
        settings=RunSettings(seed=3),
    )


def _web(ctx: RunContext, text: str, url: str) -> str:
    return ctx.add_evidence(
        Evidence(content=text, provenance=Provenance(source_kind=SourceKind.WEB, source_id=url))
    ).id


def _perspective(ctx: RunContext, agent: str, answer: str, claims: list[Claim]) -> None:
    ids = [ctx.state.claims.add(c).id for c in claims]
    p = Perspective(
        agent=agent,
        role="analyst",
        model="mock:mock",
        position=answer,
        answer=answer,
        claim_ids=ids,
    )
    ctx.state.perspectives[agent] = p
    ctx.state.history.append(p)


async def _minority_with_evidence() -> RunContext:
    """Two agents say x with no evidence; one says y with two external sources."""
    ctx = _ctx()
    e1, e2 = _web(ctx, "data 1", "https://one"), _web(ctx, "data 2", "https://two")
    _perspective(ctx, "a", "x", [Claim(statement="x because", agent="a")])
    _perspective(ctx, "b", "x", [Claim(statement="x since", agent="b")])
    _perspective(ctx, "c", "y", [Claim(statement="y per data", agent="c", evidence_ids=[e1, e2])])
    ctx.state.votes = [
        Vote(agent=p.agent, answer=p.answer or "") for p in ctx.state.perspectives.values()
    ]
    await ClaimAssessment().run(ctx)
    await DisagreementAnalysis().run(ctx)
    return ctx


async def test_structural_judge_follows_evidence_not_headcount() -> None:
    ctx = await _minority_with_evidence()
    judgment = await StructuralJudge().judge(ctx)
    assert judgment.answer == "y" and judgment.verdict is Verdict.DECIDED
    assert judgment.minority_positions == ["x"]
    assert majority_judgment(ctx).answer == "x"


async def test_structural_judge_is_inconclusive_without_evidence() -> None:
    ctx = _ctx()
    _perspective(ctx, "a", "x", [Claim(statement="s", agent="a")])
    _perspective(ctx, "b", "y", [Claim(statement="t", agent="b")])
    await ClaimAssessment().run(ctx)
    judgment = await StructuralJudge().judge(ctx)
    assert judgment.verdict is Verdict.INCONCLUSIVE and judgment.answer is None


def test_majority_tie_is_split() -> None:
    ctx = _ctx()
    ctx.state.votes = [Vote(agent="a", answer="x"), Vote(agent="b", answer="y")]
    j = majority_judgment(ctx)
    assert j.verdict is Verdict.SPLIT and j.answer is None


async def test_digest_hides_identities_and_headcount() -> None:
    ctx = await _minority_with_evidence()
    digest = build_digest(ctx)
    assert "(a," not in digest and "Analyst" in digest
    assert "analysts]" not in digest
    assert "analysts]" in build_digest(ctx, show_votes=True)


async def test_model_judge_cannot_overstate_evidence() -> None:
    def script(req: CompletionRequest) -> dict[str, Any] | None:
        return {
            "decision": "x it is",
            "answer": "x",
            "verdict": "decided",
            "supporting_claim_ids": ["C1", "C99"],
            "evidence_strength": "strong",
            "confidence": 0.9,
            "claim_assessments": [{"claim_id": "C1", "status": "supported", "note": "convincing"}],
        }

    ctx = _ctx(judge=Judge(scripted(script)))
    _perspective(ctx, "a", "x", [Claim(statement="s", agent="a")])
    await ClaimAssessment().run(ctx)
    await JudgeStep().run(ctx)
    j = ctx.state.judgment
    assert j is not None and j.answer == "x" and j.supporting_claim_ids == ["C1"]
    assert j.evidence_strength is EvidenceStrength.INSUFFICIENT
    assert j.claim_assessments[0].status is ClaimStatus.UNVERIFIED


async def test_judge_failure_falls_back_to_structural() -> None:
    ctx = _ctx(judge=Judge(MockProvider(responder=lambda r: "garbage")))
    _perspective(ctx, "a", "x", [Claim(statement="s", agent="a")])
    await JudgeStep().run(ctx)
    assert ctx.state.judgment is not None and ctx.state.judgment.method == "structural"
    assert any("judge failed" in e for e in ctx.state.errors)


async def test_synthesis_buckets() -> None:
    ctx = await _minority_with_evidence()
    await JudgeStep(method="structural").run(ctx)
    await SynthesisStep().run(ctx)
    s = ctx.state.synthesis
    assert s is not None and s.method == "structural" and s.answer == "y"
    assert [f.statement for f in s.established] == ["y per data"]
    assert any("disagree on the answer" in f.statement for f in s.disputed)
    assert {f.note for f in s.unknown} == {"no evidence offered"}


async def test_model_synthesis_uses_buckets() -> None:
    synth = Synthesizer(
        scripted(lambda r: {"summary": "Evidence favours y.", "recommendations": []})
    )
    ctx = await _minority_with_evidence()
    ctx.synthesizer = synth
    await JudgeStep(method="structural").run(ctx)
    await SynthesisStep().run(ctx)
    assert ctx.state.synthesis is not None
    assert ctx.state.synthesis.summary == "Evidence favours y."
    prompt = synth.provider.requests[0].messages[-1].content  # type: ignore[union-attr]
    assert "ESTABLISHED:\n- y per data" in prompt
