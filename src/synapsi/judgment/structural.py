"""Deterministic decision rules: structural judge, majority vote, single model."""

from __future__ import annotations

from collections import Counter, defaultdict

from synapsi.claims.models import ClaimStatus
from synapsi.deliberation.models import Perspective
from synapsi.judgment.models import EvidenceStrength, Judgment, Verdict
from synapsi.workflows.context import RunContext


def current_positions(ctx: RunContext) -> list[Perspective]:
    """Latest own perspective per agent plus any adversarial perspectives."""
    latest_adv: dict[str, Perspective] = {}
    for p in ctx.state.history:
        if p.stance == "adversarial":
            latest_adv[p.agent] = p
    return list(ctx.state.perspectives.values()) + list(latest_adv.values())


def evidence_strength(ctx: RunContext, claim_ids: list[str]) -> EvidenceStrength:
    claims = [c for cid in claim_ids if (c := ctx.state.claims.get(cid)) and not c.withdrawn]
    supported = [c for c in claims if c.status is ClaimStatus.SUPPORTED]
    partial = [c for c in claims if c.status is ClaimStatus.PARTIALLY_SUPPORTED]
    sources = {
        ev.fingerprint
        for c in supported
        for eid in ctx.state.claims.external_support(c, ctx.state.evidence)
        if (ev := ctx.state.evidence.get(eid))
    }
    contradicted = any(c.status is ClaimStatus.CONTRADICTED for c in claims)
    if len(sources) >= 3 and not contradicted:
        return EvidenceStrength.STRONG
    if supported:
        return EvidenceStrength.MODERATE
    if partial:
        return EvidenceStrength.WEAK
    return EvidenceStrength.INSUFFICIENT


def _score(ctx: RunContext, claim_ids: list[str]) -> float:
    score = 0.0
    for cid in set(claim_ids):
        claim = ctx.state.claims.get(cid)
        if claim is None or claim.withdrawn:
            continue
        if claim.status is ClaimStatus.SUPPORTED:
            score += 1.0
        elif claim.status is ClaimStatus.PARTIALLY_SUPPORTED:
            score += 0.5
        elif claim.status is ClaimStatus.CONTRADICTED:
            score -= 0.5
    return score


class StructuralJudge:
    """Prefer the answer whose claims carry the most external, unchallenged support.

    Makes no model call and never counts heads. If no answer is distinguished
    by evidence it returns ``inconclusive``; equal non-zero support yields
    ``split``. ``confidence`` is a heuristic (support share), not calibrated.
    """

    name = "structural"

    async def judge(self, ctx: RunContext) -> Judgment:
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for p in current_positions(ctx):
            if p.answer is not None:
                groups[p.answer].extend(p.claim_ids)
        if not groups:
            return Judgment(
                judge=self.name,
                method="structural",
                decision="No comparable answers were produced.",
                verdict=Verdict.INCONCLUSIVE,
                confidence=0.0,
                reasoning_summary="Agents gave no answers that can be compared structurally.",
            )
        scores = sorted(((_score(ctx, ids), a) for a, ids in groups.items()), reverse=True)
        (top_score, top), rest = scores[0], scores[1:]
        others = [a for _, a in rest]
        if top_score <= 0:
            verdict, answer = Verdict.INCONCLUSIVE, None
            decision = "Evidence does not distinguish between the answers."
        elif rest and rest[0][0] == top_score:
            verdict, answer = Verdict.SPLIT, None
            decision = "Answers are equally supported by external evidence."
        else:
            verdict, answer = Verdict.DECIDED, top
            decision = f"Best supported answer: {top}"
        total = sum(max(s, 0.0) for s, _ in scores)
        claim_ids = groups[top] if answer else []
        return Judgment(
            judge=self.name,
            method="structural",
            decision=decision,
            answer=answer,
            verdict=verdict,
            supporting_claim_ids=sorted(set(claim_ids)),
            contradicting_claim_ids=sorted({c for a in others for c in groups[a]})
            if answer
            else [],
            evidence_strength=evidence_strength(ctx, claim_ids),
            confidence=round(top_score / total, 3) if answer and total else 0.0,
            reasoning_summary="; ".join(f"{a}: support score {s:g}" for s, a in scores),
            minority_positions=others if answer else [],
            unresolved_issues=[
                d.topic
                for d in ctx.state.disagreements
                if d.status == "unresolved" and d.kind == "claim"
            ][:10],
        )


def majority_judgment(ctx: RunContext) -> Judgment:
    """Plurality of agent answers. Exists as a baseline, not as a truth criterion."""
    counts = Counter(v.answer for v in ctx.state.votes)
    total = sum(counts.values())
    if not counts:
        return Judgment(
            judge="majority_vote",
            method="vote",
            decision="No votes were cast.",
            verdict=Verdict.INCONCLUSIVE,
            confidence=0.0,
        )
    ranked = counts.most_common()
    (top, n), rest = ranked[0], ranked[1:]
    tie = bool(rest) and rest[0][1] == n
    supporting = [
        cid
        for p in ctx.state.perspectives.values()
        if p.answer == top and not tie
        for cid in p.claim_ids
    ]
    return Judgment(
        judge="majority_vote",
        method="vote",
        decision=("Tie between " + ", ".join(a for a, k in ranked if k == n))
        if tie
        else f"Majority answer: {top} ({n}/{total} votes)",
        answer=None if tie else top,
        verdict=Verdict.SPLIT if tie else Verdict.DECIDED,
        supporting_claim_ids=supporting,
        evidence_strength=evidence_strength(ctx, supporting),
        confidence=0.0 if tie else round(n / total, 3),
        reasoning_summary=", ".join(f"{a}: {k}" for a, k in ranked),
        minority_positions=[f"{a} ({k} votes)" for a, k in rest],
    )


def single_judgment(ctx: RunContext) -> Judgment:
    """Adopt the only agent's answer (single-model baseline)."""
    perspectives = list(ctx.state.perspectives.values())
    if not perspectives:
        return Judgment(
            judge="single",
            method="single",
            decision="The agent produced no answer.",
            verdict=Verdict.INCONCLUSIVE,
            confidence=0.0,
        )
    p = perspectives[0]
    return Judgment(
        judge=p.agent,
        method="single",
        decision=p.position,
        answer=p.answer,
        verdict=Verdict.DECIDED if p.answer is not None else Verdict.INCONCLUSIVE,
        supporting_claim_ids=p.claim_ids,
        evidence_strength=evidence_strength(ctx, p.claim_ids),
        confidence=p.confidence,
        reasoning_summary=p.reasoning_summary,
        unresolved_issues=p.open_questions,
    )
