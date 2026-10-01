"""Convert model drafts into validated protocol objects.

This is where the framework enforces its rules: unknown ids are dropped (and
counted), agents cannot challenge their own claims, and anything an agent
asserts from memory is stored as ``model_knowledge``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from synapsi.agents.drafts import ChallengeDraft, ClaimDraft, PerspectiveDraft
from synapsi.claims.models import Claim
from synapsi.deliberation.models import Challenge, Perspective
from synapsi.evidence.models import Evidence, Provenance, SourceKind
from synapsi.observability.events import EventType

if TYPE_CHECKING:
    from synapsi.agents.base import Agent
    from synapsi.workflows.context import RunContext

UNDETERMINED = {"", "null", "none", "unknown", "undetermined", "inconclusive", "n/a", "na"}


def normalize_answer(answer: str | None, options: Sequence[str] | None) -> str | None:
    """Map a free-text answer onto one of ``options`` where unambiguous."""
    if answer is None:
        return None
    text = answer.strip().strip(".").strip()
    if text.lower() in UNDETERMINED:
        return None
    if not options:
        return text
    lowered = {o.lower(): o for o in options}
    if text.lower() in lowered:
        return lowered[text.lower()]
    letters = {chr(ord("a") + i): o for i, o in enumerate(options)}
    if len(text) <= 3 and text.lower().strip("()") in letters:
        return letters[text.lower().strip("()")]
    contained = [o for o in options if o.lower() in text.lower()]
    if len(contained) == 1:
        return contained[0]
    return text


def _valid_evidence(ctx: RunContext, agent: str, ids: Iterable[str]) -> list[str]:
    ids = list(ids)
    valid = ctx.state.evidence.valid_ids(ids)
    _count_invalid(ctx, agent, len(set(ids)) - len(valid))
    return valid


def _count_invalid(ctx: RunContext, agent: str, n: int) -> None:
    if n > 0:
        ctx.state.invalid_references[agent] = ctx.state.invalid_references.get(agent, 0) + n
        ctx.warn(f"{agent} referenced {n} unknown id(s); they were ignored")


def register_claims(ctx: RunContext, agent: Agent, drafts: Sequence[ClaimDraft]) -> list[str]:
    ids: list[str] = []
    for d in drafts:
        evidence_ids = _valid_evidence(ctx, agent.name, d.evidence_ids)
        if d.basis.strip():
            basis = ctx.add_evidence(
                Evidence(
                    content=d.basis.strip(),
                    provenance=Provenance(
                        source_kind=SourceKind.MODEL_KNOWLEDGE,
                        produced_by=agent.name,
                        claimed_source=d.basis.strip()[:200],
                    ),
                )
            )
            evidence_ids.append(basis.id)
        claim = ctx.state.claims.add(
            Claim(
                statement=d.statement,
                type=d.type,
                agent=agent.name,
                evidence_ids=evidence_ids,
                assumptions=d.assumptions,
                confidence=d.confidence,
                round=ctx.state.round,
            )
        )
        ctx.emit(EventType.CLAIM_CREATED, agent=agent.name, claim_id=claim.id, type=claim.type)
        ids.append(claim.id)
    return ids


def register_perspective(
    ctx: RunContext,
    agent: Agent,
    draft: PerspectiveDraft,
    *,
    independent: bool,
    stance: str = "own",
) -> Perspective:
    claim_ids = register_claims(ctx, agent, draft.claims)
    perspective = Perspective(
        agent=agent.name,
        role=agent.role.key,
        model=agent.model_id,
        position=draft.position,
        answer=normalize_answer(draft.answer, ctx.problem.options),
        claim_ids=claim_ids,
        assumptions=draft.assumptions,
        reasoning_summary=draft.reasoning_summary,
        uncertainty=draft.uncertainty,
        confidence=draft.confidence,
        counterarguments=draft.counterarguments,
        open_questions=draft.open_questions,
        round=ctx.state.round,
        independent=independent,
        stance="adversarial" if stance == "adversarial" else "own",
    )
    ctx.state.history.append(perspective)
    if perspective.stance == "own":
        ctx.state.perspectives[agent.name] = perspective
    ctx.emit(
        EventType.AGENT_COMPLETED,
        agent=agent.name,
        answer=perspective.answer,
        confidence=perspective.confidence,
        claims=len(claim_ids),
        independent=independent,
        stance=perspective.stance,
    )
    return perspective


def register_challenges(
    ctx: RunContext,
    challenger: Agent,
    drafts: Sequence[ChallengeDraft],
    *,
    allowed_claim_ids: set[str],
    limit: int,
) -> list[Challenge]:
    created: list[Challenge] = []
    invalid = 0
    for d in drafts:
        claim = ctx.state.claims.get(d.target_claim_id.strip())
        if claim is None or claim.id not in allowed_claim_ids or claim.agent == challenger.name:
            invalid += 1
            continue
        if len(created) >= limit:
            break
        challenge = ctx.state.add_challenge(
            Challenge(
                challenger=challenger.name,
                target_claim_id=claim.id,
                target_agent=claim.agent,
                kind=d.kind,
                problem=d.problem,
                question=d.question,
                evidence_ids=_valid_evidence(ctx, challenger.name, d.evidence_ids),
                round=ctx.state.round,
            )
        )
        for eid in challenge.evidence_ids:
            ev = ctx.state.evidence.get(eid)
            if ev is not None and ev.is_external:
                ctx.state.evidence.link(eid, claim.id, contradicts=True)
        ctx.emit(
            EventType.CHALLENGE_CREATED,
            agent=challenger.name,
            challenge_id=challenge.id,
            target_claim_id=claim.id,
            target_agent=claim.agent,
            kind=challenge.kind,
        )
        created.append(challenge)
    _count_invalid(ctx, challenger.name, invalid)
    return created
