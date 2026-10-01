"""Built-in deliberation steps. Strategies are compositions of these."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from synapsi.agents.drafts import ClaimDraft, RebuttalDraft, RevisionDraft
from synapsi.claims.models import Claim, RelationKind
from synapsi.core.errors import BudgetExceeded
from synapsi.deliberation.models import (
    Challenge,
    ChallengeStatus,
    Perspective,
    PositionChange,
    Rebuttal,
    ResponseType,
    Review,
    Vote,
)
from synapsi.deliberation.registration import (
    normalize_answer,
    register_challenges,
    register_claims,
    register_perspective,
)
from synapsi.observability.events import EventType
from synapsi.workflows.base import Step
from synapsi.workflows.context import RunContext

if TYPE_CHECKING:
    from synapsi.agents.base import Agent

R = TypeVar("R")


async def for_each_agent(
    ctx: RunContext,
    agents: Sequence[Agent],
    fn: Callable[[Agent], Awaitable[R]],
) -> list[tuple[Agent, R]]:
    """Run ``fn`` for every agent concurrently, isolating individual failures.

    A failed agent is recorded and skipped unless ``settings.fail_fast``.
    Budget exhaustion always propagates.
    """

    async def guarded(agent: Agent) -> R:
        ctx.emit(EventType.AGENT_STARTED, agent=agent.name)
        return await fn(agent)

    outcomes = await asyncio.gather(*(guarded(a) for a in agents), return_exceptions=True)
    results: list[tuple[Agent, R]] = []
    for agent, outcome in zip(agents, outcomes, strict=True):
        if isinstance(outcome, BudgetExceeded):
            raise outcome
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception) or ctx.settings.fail_fast:
                raise outcome
            message = (
                f"{agent.name} failed in {ctx.current_step}: {type(outcome).__name__}: {outcome}"
            )
            ctx.state.errors.append(message)
            ctx.emit(EventType.AGENT_FAILED, agent=agent.name, error=str(outcome)[:500])
            continue
        results.append((agent, outcome))
    return results


def leading_answer(perspectives: Sequence[Perspective]) -> str | None:
    counts = Counter(p.answer for p in perspectives if p.answer is not None)
    if not counts:
        return None
    (top, n), *rest = counts.most_common()
    return None if rest and rest[0][1] == n else top


class IndependentAnalysis(Step):
    """Every selected agent forms a position without seeing any other agent's output."""

    name = "independent_analysis"

    def __init__(
        self,
        agents: list[str] | None = None,
        *,
        level: int | None = None,
        first_n: int | None = None,
    ):
        super().__init__()
        self.agent_names = agents
        self.level = level
        self.first_n = first_n

    async def run(self, ctx: RunContext) -> None:
        agents = ctx.select(self.agent_names, level=self.level)
        if self.first_n is not None:
            agents = agents[: self.first_n]
        snapshot = {e.id for e in ctx.state.evidence}

        async def analyze(agent: Agent) -> Perspective:
            draft = await agent.analyze(ctx, evidence_ids=snapshot)
            return register_perspective(ctx, agent, draft, independent=True)

        results = await for_each_agent(ctx, agents, analyze)
        if not results:
            raise RuntimeError("every agent failed during independent analysis")

    def describe(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "agents": self.agent_names,
            "level": self.level,
            "first_n": self.first_n,
        }


class SequentialCritique(Step):
    """Agent k sees agent k-1's analysis, challenges it, then writes an improved one.

    Deliberately non-independent: this measures chained refinement.
    """

    name = "sequential_critique"

    def __init__(self, order: list[str] | None = None):
        super().__init__()
        self.order = order

    async def run(self, ctx: RunContext) -> None:
        agents = ctx.select(self.order)
        previous: Perspective | None = None
        for agent in agents:
            ctx.emit(EventType.AGENT_STARTED, agent=agent.name)
            if previous is not None:
                ctx.state.round += 1
                targets = ctx.state.claims.by_agent(previous.agent)
                result = await agent.challenge(
                    ctx, targets, max_challenges=ctx.settings.max_challenges_per_agent
                )
                register_challenges(
                    ctx,
                    agent,
                    result.challenges,
                    allowed_claim_ids={c.id for c in targets},
                    limit=ctx.settings.max_challenges_per_agent,
                )
            visible = [previous] if previous else []
            draft = await agent.analyze(ctx, visible=visible)
            previous = register_perspective(ctx, agent, draft, independent=previous is None)


class SteelmanAlternative(Step):
    """Contrarian agents build the strongest case against the leading position."""

    name = "steelman_alternative"

    def __init__(self, agents: list[str] | None = None):
        super().__init__()
        self.agent_names = agents

    async def run(self, ctx: RunContext) -> None:
        if self.agent_names is not None:
            agents = ctx.select(self.agent_names)
        else:
            agents = [a for a in ctx.agents if a.is_contrarian]
        if not agents:
            agents = [ctx.rng.choice(ctx.agents)]
            ctx.warn(f"no contrarian agent configured; {agents[0].name} argues the alternative")
        current = list(ctx.state.perspectives.values())
        leading = leading_answer(current)
        if leading is not None:
            target = f"the answer '{leading}'"
        else:
            target = "the position most analyses lean towards"
        instruction = (
            f"Construct the strongest evidence-based case AGAINST {target}. Argue for the best "
            "alternative. This is an assigned adversarial position; keep claims honest and set "
            "`confidence` to your genuine belief, not the strength of the advocacy."
        )

        async def steelman(agent: Agent) -> Perspective:
            draft = await agent.analyze(ctx, visible=current, stance_instruction=instruction)
            return register_perspective(ctx, agent, draft, independent=False, stance="adversarial")

        await for_each_agent(ctx, agents, steelman)


class CollectVotes(Step):
    """Record each agent's current answer. Votes are data, not a decision rule."""

    name = "collect_votes"

    async def run(self, ctx: RunContext) -> None:
        ctx.state.votes = [
            Vote(agent=p.agent, answer=p.answer, confidence=p.confidence)
            for p in ctx.state.perspectives.values()
            if p.answer is not None
        ]


class PeerReview(Step):
    """Each reviewer assesses the others' analyses, optionally blind and shuffled."""

    name = "peer_review"

    def __init__(
        self,
        reviewers: list[str] | None = None,
        *,
        reviewer_level: int | None = None,
        target_level: int | None = None,
        max_targets: int | None = None,
    ):
        super().__init__()
        self.reviewers = reviewers
        self.reviewer_level = reviewer_level
        self.target_level = target_level
        self.max_targets = max_targets

    async def run(self, ctx: RunContext) -> None:
        reviewers = ctx.select(self.reviewers, level=self.reviewer_level)
        perspectives = [
            p
            for p in ctx.state.perspectives.values()
            if self.target_level is None or ctx.agent(p.agent).level == self.target_level
        ]
        limit = ctx.settings.max_challenges_per_agent

        async def review(agent: Agent) -> list[Review]:
            targets = ctx.ordered([p for p in perspectives if p.agent != agent.name])
            if self.max_targets is not None:
                targets = targets[: self.max_targets]
            if not targets:
                return []
            labelled = [(ctx.label(p.agent), p) for p in targets]
            result = await agent.review(ctx, labelled, max_challenges=limit)
            reviews: list[Review] = []
            by_agent = {p.agent: p for p in targets}
            for item in result.reviews:
                target = ctx.unlabel(item.target)
                if target is None or target not in by_agent:
                    ctx.warn(f"{agent.name} reviewed unknown target {item.target!r}")
                    continue
                owned = set(by_agent[target].claim_ids)
                challenges = register_challenges(
                    ctx, agent, item.challenges, allowed_claim_ids=owned, limit=limit
                )
                record = Review(
                    reviewer=agent.name,
                    target_agent=target,
                    agreements=[
                        c for c in ctx.state.claims.valid_ids(item.agreements) if c in owned
                    ],
                    challenge_ids=[c.id for c in challenges],
                    assessment=item.assessment,
                    quality=item.quality,
                    blind=ctx.settings.blind_review,
                    round=ctx.state.round,
                )
                ctx.state.reviews.append(record)
                ctx.emit(EventType.REVIEW_CREATED, agent=agent.name, target_agent=target)
                reviews.append(record)
            return reviews

        await for_each_agent(ctx, reviewers, review)


class CrossExamination(Step):
    """Agents raise structured challenges against other agents' claims.

    In ``adversarial`` mode challengers target only claims from agents whose
    answer differs from their own (falling back to all others if none do).
    """

    name = "cross_examination"

    def __init__(self, challengers: list[str] | None = None, *, adversarial: bool = False):
        super().__init__()
        self.challengers = challengers
        self.adversarial = adversarial

    def _targets(self, ctx: RunContext, agent: Agent) -> list[Claim]:
        already = {c.target_claim_id for c in ctx.state.challenges if c.challenger == agent.name}
        current = [p for p in ctx.state.history if p.round <= ctx.state.round]
        latest: dict[tuple[str, str], Perspective] = {}
        for p in current:
            latest[(p.agent, p.stance)] = p
        mine = ctx.state.perspectives.get(agent.name)
        others = [p for (name, _), p in latest.items() if name != agent.name]
        if self.adversarial and mine is not None and mine.answer is not None:
            opposed = [p for p in others if p.answer != mine.answer]
            others = opposed or others
        claim_ids = {cid for p in others for cid in p.claim_ids}
        claims = [
            c
            for c in ctx.state.claims.active()
            if c.id in claim_ids and c.id not in already and c.agent != agent.name
        ]
        return ctx.ordered(claims)

    async def run(self, ctx: RunContext) -> None:
        limit = ctx.settings.max_challenges_per_agent

        async def examine(agent: Agent) -> list[Challenge]:
            targets = self._targets(ctx, agent)
            if not targets:
                return []
            instruction = ""
            if self.adversarial:
                instruction = (
                    f"These claims support positions opposed to yours. Raise at most {limit} "
                    "challenges against the claims that matter most to their conclusion. "
                    "Challenge substance, not wording."
                )
            result = await agent.challenge(
                ctx, targets, max_challenges=limit, instruction=instruction
            )
            return register_challenges(
                ctx,
                agent,
                result.challenges,
                allowed_claim_ids={c.id for c in targets},
                limit=limit,
            )

        await for_each_agent(ctx, ctx.select(self.challengers), examine)


class RespondToChallenges(Step):
    """Owners answer open challenges: defend, clarify, concede, or revise.

    A defence without evidence leaves the challenge open: assertion is not
    resolution. Clarifications close it; concessions and revisions are
    recorded and change claim status.
    """

    name = "respond_to_challenges"

    async def run(self, ctx: RunContext) -> None:
        pending = {
            agent.name: ctx.state.open_challenges(agent.name)
            for agent in ctx.agents
            if ctx.state.open_challenges(agent.name)
        }
        agents = [ctx.agent(name) for name in pending]

        async def respond(agent: Agent) -> None:
            challenges = pending[agent.name]
            result = await agent.respond(ctx, challenges)
            by_id = {c.id: c for c in challenges}
            for draft in result.rebuttals:
                challenge = by_id.pop(draft.challenge_id.strip(), None)
                if challenge is None:
                    continue
                self._apply(ctx, agent, challenge, draft)

        await for_each_agent(ctx, agents, respond)

    def _apply(
        self, ctx: RunContext, agent: Agent, challenge: Challenge, draft: RebuttalDraft
    ) -> None:
        evidence_ids = ctx.state.evidence.valid_ids(draft.evidence_ids)
        external = [e for e in evidence_ids if (ev := ctx.state.evidence.get(e)) and ev.is_external]
        claim = ctx.state.claims.get(challenge.target_claim_id)
        revised_id: str | None = None
        kind = draft.response_type
        if kind is ResponseType.CONCEDE:
            challenge.status = ChallengeStatus.CONCEDED
        elif kind is ResponseType.REVISE and claim is not None and draft.revised_statement:
            [revised_id] = register_claims(
                ctx,
                agent,
                [
                    ClaimDraft(
                        statement=draft.revised_statement,
                        type=claim.type,
                        evidence_ids=[*claim.evidence_ids, *evidence_ids],
                        confidence=claim.confidence,
                        assumptions=claim.assumptions,
                    )
                ],
            )
            revised = ctx.state.claims.get(revised_id)
            if revised is not None:
                revised.revised_from = claim.id
            ctx.state.claims.relate(revised_id, claim.id, RelationKind.REFINES, agent=agent.name)
            ctx.state.claims.withdraw(claim.id)
            perspective = ctx.state.perspectives.get(agent.name)
            if perspective is not None and claim.id in perspective.claim_ids:
                perspective.claim_ids = [
                    revised_id if c == claim.id else c for c in perspective.claim_ids
                ]
            challenge.status = ChallengeStatus.REVISED
        elif kind is ResponseType.CLARIFY or external:
            challenge.status = ChallengeStatus.ANSWERED
            if claim is not None:
                claim.evidence_ids = list(dict.fromkeys([*claim.evidence_ids, *external]))
        rebuttal = ctx.state.add_rebuttal(
            Rebuttal(
                challenge_id=challenge.id,
                agent=agent.name,
                response_type=kind,
                response=draft.response,
                evidence_ids=evidence_ids,
                revised_claim_id=revised_id,
                round=ctx.state.round,
            )
        )
        ctx.emit(
            EventType.REBUTTAL_CREATED,
            agent=agent.name,
            rebuttal_id=rebuttal.id,
            challenge_id=challenge.id,
            response_type=kind,
            challenge_status=challenge.status,
        )


class Revision(Step):
    """Agents update their positions after seeing challenges and responses.

    All agents revise concurrently from the same snapshot, so no agent sees
    another's revision from the same round. Position changes are recorded with
    whether they were backed by new external evidence or a concession, which
    makes conformity measurable.
    """

    name = "revision"

    def __init__(self, agents: list[str] | None = None, *, see_others: bool = True):
        super().__init__()
        self.agent_names = agents
        self.see_others = see_others

    async def run(self, ctx: RunContext) -> None:
        snapshot = dict(ctx.state.perspectives)
        agents = [a for a in ctx.select(self.agent_names) if a.name in snapshot]
        round_ = ctx.state.round

        async def revise(agent: Agent) -> Perspective:
            own = snapshot[agent.name]
            own_claims = set(own.claim_ids)
            challenges = [c for c in ctx.state.challenges if c.target_claim_id in own_claims]
            challenge_ids = {c.id for c in challenges}
            rebuttals = [r for r in ctx.state.rebuttals if r.challenge_id in challenge_ids]
            others = []
            if self.see_others:
                others = [
                    (ctx.label(p.agent), p)
                    for p in ctx.ordered(list(snapshot.values()))
                    if p.agent != agent.name
                ]
            draft = await agent.revise(
                ctx, own, challenges=challenges, rebuttals=rebuttals, others=others
            )
            return self._apply(ctx, agent, own, draft, challenges, round_)

        await for_each_agent(ctx, agents, revise)

    def _apply(
        self,
        ctx: RunContext,
        agent: Agent,
        own: Perspective,
        draft: RevisionDraft,
        challenges: list[Challenge],
        round_: int,
    ) -> Perspective:
        own_ids = set(own.claim_ids)
        for cid in ctx.state.claims.valid_ids(draft.withdraw_claim_ids):
            if cid in own_ids:
                ctx.state.claims.withdraw(cid)
        active = {c.id for c in ctx.state.claims.active()}
        keep = [
            cid
            for cid in ctx.state.claims.valid_ids(draft.keep_claim_ids)
            if cid in own_ids and cid in active
        ]
        if not draft.keep_claim_ids and not draft.withdraw_claim_ids:
            keep = [c for c in own.claim_ids if c in active]
        new_ids = register_claims(ctx, agent, draft.new_claims)
        prior_evidence = {
            e for cid in own.claim_ids if (c := ctx.state.claims.get(cid)) for e in c.evidence_ids
        }
        cited_new = any(
            e not in prior_evidence and (ev := ctx.state.evidence.get(e)) and ev.is_external
            for cid in new_ids
            if (c := ctx.state.claims.get(cid))
            for e in c.evidence_ids
        )
        updated = Perspective(
            agent=agent.name,
            role=agent.role.key,
            model=agent.model_id,
            position=draft.position,
            answer=normalize_answer(draft.answer, ctx.problem.options),
            claim_ids=keep + new_ids,
            assumptions=own.assumptions,
            reasoning_summary=draft.reasoning_summary or own.reasoning_summary,
            uncertainty=draft.uncertainty,
            confidence=draft.confidence,
            counterarguments=own.counterarguments,
            open_questions=draft.open_questions,
            round=round_,
            independent=False,
        )
        ctx.state.history.append(updated)
        ctx.state.perspectives[agent.name] = updated
        if updated.answer != own.answer:
            change = PositionChange(
                agent=agent.name,
                round=round_,
                from_answer=own.answer,
                to_answer=updated.answer,
                from_confidence=own.confidence,
                to_confidence=updated.confidence,
                reason=draft.reason_for_change,
                cited_new_evidence=bool(cited_new),
                after_concession=any(
                    c.status in (ChallengeStatus.CONCEDED, ChallengeStatus.REVISED)
                    for c in challenges
                ),
            )
            ctx.state.position_changes.append(change)
            ctx.emit(
                EventType.POSITION_CHANGED,
                agent=agent.name,
                from_answer=change.from_answer,
                to_answer=change.to_answer,
                cited_new_evidence=change.cited_new_evidence,
            )
        ctx.emit(EventType.AGENT_COMPLETED, agent=agent.name, answer=updated.answer, revised=True)
        return updated


class ClaimAssessment(Step):
    """Recompute claim status from evidence and challenges (no model call)."""

    name = "claim_assessment"

    async def run(self, ctx: RunContext) -> None:
        ctx.state.claims.assess(ctx.state.evidence, ctx.state.challenges)


def converged(ctx: RunContext) -> bool:
    """True when the latest round produced no position change and no new open challenge."""
    round_ = ctx.state.round
    changed = any(c.round == round_ for c in ctx.state.position_changes)
    open_new = any(c.round == round_ and c.status == "open" for c in ctx.state.challenges)
    return not changed and not open_new
