from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from synapsi.agents.prompts import render_evidence
from synapsi.claims.models import ClaimStatus
from synapsi.deliberation.registration import normalize_answer
from synapsi.judgment.models import ClaimAssessment, EvidenceStrength, Judgment, Verdict
from synapsi.judgment.structural import current_positions
from synapsi.providers.base import Message, ModelProvider
from synapsi.providers.registry import create_provider
from synapsi.providers.wrappers import managed
from synapsi.workflows.context import RunContext

JUDGE_SYSTEM = """\
You are an impartial judge evaluating a structured deliberation. You see claims, evidence,
challenges, and responses, not conversations.

Rules:
- Decide on evidence and reasoning quality. The number of analysts holding a position is
  irrelevant and is deliberately hidden from you.
- Claims and arguments are not evidence. Only items under EVIDENCE are evidence, and
  `model_knowledge` items are unverified recollections.
- If the evidence does not distinguish the positions, return verdict "inconclusive". If
  positions are equally well supported, return "split". Do not force a consensus.
- Preserve strong minority positions in `minority_positions`.
- Give a concise reasoning summary, not step-by-step internal deliberation.
- Respond with one JSON object in the requested shape."""

JUDGE_SHAPE = """\
Return JSON:
{"decision": str, "answer": str|null, "verdict": "decided|inconclusive|split",
 "supporting_claim_ids": [str], "contradicting_claim_ids": [str],
 "evidence_strength": "strong|moderate|weak|insufficient", "unresolved_issues": [str],
 "confidence": float, "reasoning_summary": str,
 "claim_assessments": [{"claim_id": str, "note": str, "status":
   "supported|partially_supported|contradicted|unsupported|uncertain|unverified"}],
 "minority_positions": [str]}"""


class JudgmentDraft(BaseModel):
    decision: str = Field(min_length=1)
    answer: str | None = None
    verdict: Verdict = Verdict.DECIDED
    supporting_claim_ids: list[str] = Field(default_factory=list)
    contradicting_claim_ids: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    unresolved_issues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = ""
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    minority_positions: list[str] = Field(default_factory=list)


def build_digest(ctx: RunContext, *, blind: bool = True, show_votes: bool = False) -> str:
    """Compact, transcript-free view of the deliberation for the judge."""
    state = ctx.state

    def who(agent: str) -> str:
        return ctx.label(agent) if blind else agent

    parts = [f"PROBLEM:\n{ctx.problem.question}"]
    if ctx.problem.context:
        parts.append(f"CONTEXT:\n{ctx.problem.context}")
    if ctx.problem.options:
        parts.append(f"OPTIONS: {'; '.join(ctx.problem.options)}")
    parts.append(render_evidence(state.evidence))

    groups: defaultdict[str, list[str]] = defaultdict(list)
    supporters: defaultdict[str, set[str]] = defaultdict(set)
    texts: dict[str, str] = {}
    for p in current_positions(ctx):
        key = p.answer if p.answer is not None else p.position.strip()[:300]
        groups[key].extend(p.claim_ids)
        supporters[key].add(p.agent)
        texts.setdefault(key, p.position.strip()[:400])
    rows = []
    for i, (key, ids) in enumerate(ctx.ordered(list(groups.items())), start=1):
        count = f" [{len(supporters[key])} analysts]" if show_votes else ""
        label = f"answer={key}" if ctx.problem.options else "position"
        rows.append(
            f"P{i} {label}{count}: {texts[key]}\n   claims: {', '.join(dict.fromkeys(ids))}"
        )
    parts.append("POSITIONS:\n" + "\n".join(rows))

    claim_rows = []
    for c in ctx.state.claims.active():
        ev = ", ".join(c.evidence_ids) or "none"
        claim_rows.append(
            f"{c.id} ({who(c.agent)}, {c.type.value}) {c.statement} | evidence: {ev} | "
            f"status: {c.status.value}"
        )
    parts.append("CLAIMS:\n" + ("\n".join(claim_rows) or "(none)"))

    if state.challenges:
        rebuttals = {r.challenge_id: r for r in state.rebuttals}
        ch_rows = []
        for ch in state.challenges:
            line = (
                f"{ch.id} -> {ch.target_claim_id} [{ch.kind.value}, {ch.status.value}]: "
                f"{ch.problem}"
            )
            if ch.evidence_ids:
                line += f" (evidence {', '.join(ch.evidence_ids)})"
            if (r := rebuttals.get(ch.id)) is not None:
                line += f"\n   response [{r.response_type.value}]: {r.response}"
            ch_rows.append(line)
        parts.append("CHALLENGES:\n" + "\n".join(ch_rows))
    open_disputes = [d for d in state.disagreements if d.status == "unresolved"]
    if open_disputes:
        parts.append(
            "UNRESOLVED DISAGREEMENTS:\n"
            + "\n".join(
                f"{d.id} [{d.kind}] {d.topic} ({d.evidence_balance})" for d in open_disputes
            )
        )
    return "\n\n".join(parts)


class Judge:
    """Model-based judge working on a structured digest.

    ``blind`` hides agent identities. ``show_votes`` (off by default) reveals
    how many agents hold each position; keep it off to test whether judges
    follow evidence or headcount.
    """

    def __init__(
        self,
        model: str | ModelProvider = "mock",
        *,
        name: str = "judge",
        blind: bool = True,
        show_votes: bool = False,
        instructions: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ):
        self.provider = (
            model if isinstance(model, ModelProvider) else managed(create_provider(model))
        )
        self.name = name
        self.blind = blind
        self.show_votes = show_votes
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self.provider.id

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model_id,
            "blind": self.blind,
            "show_votes": self.show_votes,
        }

    async def judge(self, ctx: RunContext) -> Judgment:
        system = JUDGE_SYSTEM + (f"\n\n{self.instructions}" if self.instructions else "")
        digest = build_digest(ctx, blind=self.blind, show_votes=self.show_votes)
        claim_ids = [c.id for c in ctx.state.claims.active()]
        draft = await ctx.generate(
            provider=self.provider,
            schema=JudgmentDraft,
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=f"{digest}\n\n{JUDGE_SHAPE}"),
            ],
            agent=self.name,
            task="judge",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            hints={"claim_ids": claim_ids, "mock_list_len": 1},
        )
        claims = ctx.state.claims
        answer = normalize_answer(draft.answer, ctx.problem.options)
        verdict = draft.verdict
        if verdict is not Verdict.DECIDED:
            answer = None
        assessments = []
        for a in draft.claim_assessments:
            applied = claims.apply_assessment(a.claim_id, a.status, a.note, ctx.state.evidence)
            if applied is not None:
                assessments.append(
                    ClaimAssessment(claim_id=a.claim_id, status=applied, note=a.note)
                )
        return Judgment(
            judge=self.name,
            method="model",
            decision=draft.decision,
            answer=answer,
            verdict=verdict,
            supporting_claim_ids=claims.valid_ids(draft.supporting_claim_ids),
            contradicting_claim_ids=claims.valid_ids(draft.contradicting_claim_ids),
            evidence_strength=_cap_strength(ctx, draft.evidence_strength),
            unresolved_issues=draft.unresolved_issues,
            confidence=draft.confidence,
            reasoning_summary=draft.reasoning_summary,
            claim_assessments=assessments,
            minority_positions=draft.minority_positions,
        )


def _cap_strength(ctx: RunContext, claimed: EvidenceStrength) -> EvidenceStrength:
    """A judge cannot report strong evidence when no external evidence exists."""
    has_external = any(e.is_external for e in ctx.state.evidence)
    supported = any(c.status is ClaimStatus.SUPPORTED for c in ctx.state.claims.active())
    if not has_external:
        return EvidenceStrength.INSUFFICIENT
    if claimed is EvidenceStrength.STRONG and not supported:
        return EvidenceStrength.WEAK
    return claimed
