import pytest
from pydantic import ValidationError

from synapsi.claims import Claim, ClaimStatus
from synapsi.core import Problem, TokenUsage, Usage
from synapsi.evidence import Evidence, Provenance, SourceKind
from synapsi.judgment import Judgment, Verdict


def test_problem_gets_stable_id_and_public_copy_hides_answer() -> None:
    p1 = Problem(question="Is X true?", answer="yes")
    p2 = Problem(question="Is X true?")
    assert p1.id == p2.id and p1.id.startswith("p_")
    assert p1.public().answer is None
    assert p1.answer == "yes"


def test_problem_requires_question() -> None:
    with pytest.raises(ValidationError):
        Problem(question="")


def test_model_knowledge_is_not_external() -> None:
    mk = Evidence(
        content="LLM says so", provenance=Provenance(source_kind=SourceKind.MODEL_KNOWLEDGE)
    )
    web = Evidence(
        content="page text",
        provenance=Provenance(source_kind=SourceKind.WEB, source_id="https://example.org/a"),
    )
    assert not mk.is_external
    assert web.is_external
    assert web.fingerprint == "web:https://example.org/a"
    assert mk.fingerprint.startswith("model_knowledge:#")


def test_claim_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Claim(statement="x", confidence=1.5)
    assert Claim(statement="x").status is ClaimStatus.UNVERIFIED


def test_usage_cost_becomes_unknown_when_any_call_unpriced() -> None:
    u = Usage()
    u.add(TokenUsage(prompt_tokens=10, completion_tokens=5), cost_usd=0.01, latency_s=0.1)
    assert u.cost_usd == pytest.approx(0.01)
    u.add(TokenUsage(prompt_tokens=1, completion_tokens=1), cost_usd=None, latency_s=0.1)
    assert u.cost_usd is None
    assert u.total_tokens == 17 and u.calls == 2


def test_judgment_defaults() -> None:
    j = Judgment(judge="j", method="structural", decision="Inconclusive")
    assert j.verdict is Verdict.DECIDED
    assert j.minority_positions == []
