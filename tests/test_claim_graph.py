from synapsi.claims import Claim, ClaimGraph, ClaimStatus, RelationKind
from synapsi.deliberation import Challenge, ChallengeStatus
from synapsi.evidence import Evidence, EvidencePool, Provenance, SourceKind


def _ev(content: str, kind: SourceKind = SourceKind.WEB, source: str | None = None) -> Evidence:
    return Evidence(content=content, provenance=Provenance(source_kind=kind, source_id=source))


def _setup() -> tuple[ClaimGraph, EvidencePool]:
    return ClaimGraph(), EvidencePool()


def test_pool_assigns_ids_and_dedupes() -> None:
    pool = EvidencePool()
    a = pool.add(_ev("x", source="u1"))
    b = pool.add(_ev("x", source="u1"))
    c = pool.add(_ev("y", source="u1"))
    assert a.id == b.id == "E1" and c.id == "E2"
    assert pool.valid_ids(["E2", "E9", "E2"]) == ["E2"]


def test_status_rules() -> None:
    graph, pool = _setup()
    web = pool.add(_ev("page", source="https://a"))
    mk = pool.add(_ev("I recall", SourceKind.MODEL_KNOWLEDGE))
    contra = pool.add(_ev("counter", source="https://b"))

    supported = graph.add(Claim(statement="s", agent="A", evidence_ids=[web.id]))
    unverified = graph.add(Claim(statement="u", agent="A", evidence_ids=[mk.id]))
    unsupported = graph.add(Claim(statement="n", agent="B"))
    contradicted = graph.add(
        Claim(statement="c", agent="B", contradicting_evidence_ids=[contra.id])
    )
    disputed = graph.add(Claim(statement="d", agent="B", evidence_ids=[web.id]))
    challenged = graph.add(Claim(statement="ch", agent="B"))

    challenges = [
        Challenge(challenger="A", target_claim_id=disputed.id, target_agent="B", problem="p"),
        Challenge(challenger="A", target_claim_id=challenged.id, target_agent="B", problem="p"),
    ]
    graph.assess(pool, challenges)
    status = {c.id: c.status for c in graph}
    assert status[supported.id] is ClaimStatus.SUPPORTED
    assert status[unverified.id] is ClaimStatus.UNVERIFIED
    assert status[unsupported.id] is ClaimStatus.UNSUPPORTED
    assert status[contradicted.id] is ClaimStatus.CONTRADICTED
    assert status[disputed.id] is ClaimStatus.PARTIALLY_SUPPORTED
    assert status[challenged.id] is ClaimStatus.UNCERTAIN

    challenges[0].status = ChallengeStatus.ANSWERED
    graph.assess(pool, challenges)
    assert graph.get(disputed.id).status is ClaimStatus.SUPPORTED  # type: ignore[union-attr]


def test_agreement_does_not_create_support() -> None:
    graph, pool = _setup()
    for agent in "ABCDE":
        graph.add(Claim(statement="The sky is green", agent=agent, confidence=0.99))
    graph.assess(pool, [])
    assert all(c.status is ClaimStatus.UNSUPPORTED for c in graph)


def test_supported_claim_contradicts_unsupported_one() -> None:
    graph, pool = _setup()
    web = pool.add(_ev("data", source="https://a"))
    a = graph.add(Claim(statement="a", agent="A", evidence_ids=[web.id]))
    b = graph.add(Claim(statement="b", agent="B"))
    graph.relate(a.id, b.id, RelationKind.CONTRADICTS, agent="A")
    graph.relate(a.id, a.id, RelationKind.SUPPORTS)
    assert len(graph.relations) == 1
    graph.assess(pool, [])
    assert graph.get(b.id).status is ClaimStatus.CONTRADICTED  # type: ignore[union-attr]


def test_judge_cannot_support_without_external_evidence() -> None:
    graph, pool = _setup()
    claim = graph.add(Claim(statement="x", agent="A"))
    applied = graph.apply_assessment(claim.id, ClaimStatus.SUPPORTED, "convincing", pool)
    assert applied is ClaimStatus.UNVERIFIED
    graph.assess(pool, [])
    assert graph.get(claim.id).status is ClaimStatus.UNVERIFIED  # type: ignore[union-attr]


def test_source_dependence_detects_shared_sources() -> None:
    graph, pool = _setup()
    shared = pool.add(_ev("one page", source="https://same"))
    own = pool.add(_ev("other", source="https://own"))
    graph.add(Claim(statement="1", agent="A", evidence_ids=[shared.id]))
    graph.add(Claim(statement="2", agent="B", evidence_ids=[shared.id]))
    graph.add(Claim(statement="3", agent="C", evidence_ids=[own.id]))
    report = pool.dependence(graph.citations())
    assert report.distinct_sources == 2
    assert report.shared_sources[0].agents == ["A", "B"]
    assert report.shared_source_ratio == 0.5


def test_graph_export_includes_evidence_edges() -> None:
    graph, pool = _setup()
    ev = pool.add(_ev("e", source="s"))
    claim = graph.add(Claim(statement="c", agent="A", evidence_ids=[ev.id]))
    exported = graph.to_graph(pool)
    assert {n["id"] for n in exported["nodes"]} == {claim.id, ev.id}
    assert exported["edges"] == [{"source": ev.id, "target": claim.id, "kind": "supports"}]
