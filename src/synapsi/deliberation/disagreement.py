"""Disagreement engine: make disagreement explicit instead of averaging it away."""

from __future__ import annotations

from collections import defaultdict

from synapsi.claims.models import ClaimStatus, RelationKind
from synapsi.deliberation.models import (
    ChallengeStatus,
    Disagreement,
    DisagreementSide,
    Perspective,
)
from synapsi.observability.events import EventType
from synapsi.workflows.base import Step
from synapsi.workflows.context import RunContext


def _external(ctx: RunContext, claim_ids: list[str]) -> list[str]:
    ids: list[str] = []
    for cid in claim_ids:
        claim = ctx.state.claims.get(cid)
        if claim is not None and not claim.withdrawn:
            ids.extend(ctx.state.claims.external_support(claim, ctx.state.evidence))
    return list(dict.fromkeys(ids))


def answer_disagreement(ctx: RunContext, perspectives: list[Perspective]) -> Disagreement | None:
    groups: defaultdict[str, list[Perspective]] = defaultdict(list)
    for p in perspectives:
        if p.answer is not None:
            groups[p.answer].append(p)
    if len(groups) < 2:
        return None
    sides = []
    for answer, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        claim_ids = [c for p in members for c in p.claim_ids]
        external = _external(ctx, claim_ids)
        sides.append(
            DisagreementSide(
                position=answer,
                agents=sorted({p.agent for p in members}),
                claim_ids=claim_ids,
                evidence_ids=external,
                external_evidence_count=len(external),
            )
        )
    balance = "; ".join(f"{s.position}: {s.external_evidence_count} external" for s in sides)
    return Disagreement(
        topic=ctx.problem.question,
        kind="answer",
        sides=sides,
        claim_ids=[c for s in sides for c in s.claim_ids],
        reason=f"{len(sides)} distinct answers among agents",
        status="unresolved",
        evidence_balance=balance,
    )


def claim_disagreements(ctx: RunContext) -> list[Disagreement]:
    found: list[Disagreement] = []
    by_claim: defaultdict[str, list[str]] = defaultdict(list)
    for ch in ctx.state.challenges:
        by_claim[ch.target_claim_id].append(ch.id)
    contradicting: defaultdict[str, list[str]] = defaultdict(list)
    for rel in ctx.state.claims.relations:
        if rel.kind is RelationKind.CONTRADICTS:
            contradicting[rel.target].append(rel.source)

    challenges = {c.id: c for c in ctx.state.challenges}
    for claim in ctx.state.claims.all():
        challenge_ids = by_claim.get(claim.id, [])
        rivals = contradicting.get(claim.id, [])
        contradicted = claim.status is ClaimStatus.CONTRADICTED
        if not challenge_ids and not rivals and not contradicted:
            continue
        chs = [challenges[i] for i in challenge_ids]
        unresolved = any(c.status is ChallengeStatus.OPEN for c in chs) or (
            claim.status in (ClaimStatus.CONTRADICTED, ClaimStatus.PARTIALLY_SUPPORTED)
            and not claim.withdrawn
        )
        own_external = _external(ctx, [claim.id])
        opposing_evidence = list(
            dict.fromkeys(
                [e for c in chs for e in c.evidence_ids]
                + ctx.state.claims.external_contradiction(claim, ctx.state.evidence)
                + _external(ctx, rivals)
            )
        )
        rival_agents = {c.agent for r in rivals if (c := ctx.state.claims.get(r)) is not None}
        sides = [
            DisagreementSide(
                position=claim.statement,
                agents=[claim.agent],
                claim_ids=[claim.id],
                evidence_ids=own_external,
                external_evidence_count=len(own_external),
            ),
            DisagreementSide(
                position="; ".join(c.problem for c in chs)[:500] or "contradicting claims",
                agents=sorted({c.challenger for c in chs} | rival_agents),
                claim_ids=rivals,
                evidence_ids=opposing_evidence,
                external_evidence_count=sum(
                    1
                    for e in opposing_evidence
                    if (ev := ctx.state.evidence.get(e)) and ev.is_external
                ),
            ),
        ]
        statuses = ", ".join(f"{c.id}:{c.status.value}" for c in chs)
        found.append(
            Disagreement(
                topic=claim.statement,
                kind="claim",
                sides=sides,
                claim_ids=[claim.id, *rivals],
                reason=f"challenges [{statuses}]" if chs else "contradicted by other claims",
                status="unresolved" if unresolved else "resolved",
                evidence_balance=(
                    f"claim: {sides[0].external_evidence_count} external; "
                    f"opposition: {sides[1].external_evidence_count} external"
                ),
            )
        )
    return found


class DisagreementAnalysis(Step):
    """Detect answer-level and claim-level disagreements (no model call)."""

    name = "disagreement_analysis"

    async def run(self, ctx: RunContext) -> None:
        perspectives = [
            p
            for p in ctx.state.history
            if (p.stance == "own" and ctx.state.perspectives.get(p.agent) is p)
            or p.stance == "adversarial"
        ]
        found: list[Disagreement] = []
        answer = answer_disagreement(ctx, perspectives)
        if answer is not None:
            found.append(answer)
        found.extend(claim_disagreements(ctx))
        for i, d in enumerate(found, start=1):
            d.id = f"D{i}"
            ctx.emit(
                EventType.DISAGREEMENT_DETECTED,
                disagreement_id=d.id,
                kind=d.kind,
                status=d.status,
                sides=len(d.sides),
            )
        ctx.state.disagreements = found
