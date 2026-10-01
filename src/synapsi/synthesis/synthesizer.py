from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from synapsi.claims.models import Claim, ClaimStatus
from synapsi.observability.events import EventType
from synapsi.providers.base import Message, ModelProvider
from synapsi.providers.registry import create_provider
from synapsi.providers.wrappers import managed
from synapsi.synthesis.models import Finding, Synthesis
from synapsi.workflows.base import Step
from synapsi.workflows.context import RunContext

_WORD = re.compile(r"[a-z0-9]+")

SYNTH_SYSTEM = """\
You write the final summary of a structured deliberation for a decision-maker.
Use only the findings provided. Do not add facts, sources, or certainty that are not there.
Keep the distinction between established, probable, disputed, and unknown explicit.
If the verdict is inconclusive or split, say so plainly. Respond with one JSON object."""


class SynthesisDraft(BaseModel):
    summary: str = Field(min_length=1)
    recommendations: list[str] = Field(default_factory=list, max_length=8)


def _similar(a: str, b: str, threshold: float) -> bool:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _merge(findings: list[Finding], threshold: float) -> list[Finding]:
    merged: list[Finding] = []
    for f in findings:
        for m in merged:
            if _similar(f.statement, m.statement, threshold):
                m.claim_ids = list(dict.fromkeys(m.claim_ids + f.claim_ids))
                m.evidence_ids = list(dict.fromkeys(m.evidence_ids + f.evidence_ids))
                m.agents = sorted(set(m.agents) | set(f.agents))
                break
        else:
            merged.append(f.model_copy(deep=True))
    return merged


class Synthesizer:
    """Group claims by epistemic status; optionally have a model write the summary.

    The grouping is deterministic and driven by claim status, so the summary
    model cannot promote a disputed claim to established.

    - established: ``supported`` and not under unresolved dispute
    - probable: ``partially_supported`` without unresolved dispute
    - disputed: ``contradicted``, or any claim in an unresolved disagreement,
      plus answer-level disagreement among agents
    - unknown: ``uncertain``/``unverified``/``unsupported`` claims and open issues
    """

    def __init__(
        self,
        model: str | ModelProvider | None = None,
        *,
        similarity: float = 0.8,
        temperature: float = 0.2,
        max_findings: int = 12,
    ):
        if model is None or isinstance(model, ModelProvider):
            self.provider = model
        else:
            self.provider = managed(create_provider(model))
        self.similarity = similarity
        self.temperature = temperature
        self.max_findings = max_findings

    def describe(self) -> dict[str, Any]:
        return {"model": self.provider.id if self.provider else None, "similarity": self.similarity}

    def _finding(self, ctx: RunContext, claim: Claim, note: str = "") -> Finding:
        external = ctx.state.claims.external_support(claim, ctx.state.evidence)
        return Finding(
            statement=claim.statement,
            claim_ids=[claim.id],
            evidence_ids=external,
            agents=[claim.agent],
            confidence=claim.confidence,
            note=note or claim.status_reason,
        )

    def buckets(self, ctx: RunContext) -> dict[str, list[Finding]]:
        disputed_ids = {
            cid
            for d in ctx.state.disagreements
            if d.kind == "claim" and d.status == "unresolved"
            for cid in d.claim_ids
        }
        out: dict[str, list[Finding]] = {
            k: [] for k in ("established", "probable", "disputed", "unknown")
        }
        for d in ctx.state.disagreements:
            if d.kind == "answer" and d.status == "unresolved":
                sides = " vs ".join(f"{s.position} ({len(s.agents)})" for s in d.sides)
                out["disputed"].append(
                    Finding(
                        statement=f"Agents disagree on the answer: {sides}",
                        agents=sorted({a for s in d.sides for a in s.agents}),
                        note=d.evidence_balance,
                    )
                )
        for claim in ctx.state.claims.active():
            status = claim.status
            if claim.id in disputed_ids or status is ClaimStatus.CONTRADICTED:
                out["disputed"].append(self._finding(ctx, claim))
            elif status is ClaimStatus.SUPPORTED:
                out["established"].append(self._finding(ctx, claim))
            elif status is ClaimStatus.PARTIALLY_SUPPORTED:
                out["probable"].append(self._finding(ctx, claim))
            else:
                note = {
                    ClaimStatus.UNVERIFIED: "not verified against external evidence",
                    ClaimStatus.UNSUPPORTED: "no evidence offered",
                    ClaimStatus.UNCERTAIN: claim.status_reason,
                }.get(status, "")
                out["unknown"].append(self._finding(ctx, claim, note))
        if ctx.state.judgment:
            seen = [f.statement for f in out["disputed"] + out["unknown"]]
            for issue in ctx.state.judgment.unresolved_issues:
                if not any(_similar(issue, s, self.similarity) for s in seen):
                    out["unknown"].append(Finding(statement=issue, note="unresolved issue"))
        return {
            k: sorted(_merge(v, self.similarity), key=lambda f: -len(f.agents))[: self.max_findings]
            for k, v in out.items()
        }

    def recommendations(self, ctx: RunContext) -> list[str]:
        items: list[str] = []
        for ch in ctx.state.challenges:
            if ch.status == "open" and ch.question:
                items.append(ch.question)
        for p in ctx.state.perspectives.values():
            items.extend(p.open_questions)
        unique: list[str] = []
        for item in items:
            if item.strip() and not any(_similar(item, u, self.similarity) for u in unique):
                unique.append(item.strip())
        return unique[:8]

    async def synthesize(self, ctx: RunContext) -> Synthesis:
        buckets = self.buckets(ctx)
        judgment = ctx.state.judgment
        recs = self.recommendations(ctx)
        verdict = judgment.verdict.value if judgment else "none"
        decision = judgment.decision if judgment else "No judgment was produced."
        summary = (
            f"{decision} (verdict: {verdict}). "
            f"Established: {len(buckets['established'])}, probable: {len(buckets['probable'])}, "
            f"disputed: {len(buckets['disputed'])}, unknown: {len(buckets['unknown'])}."
        )
        method: str = "structural"
        if self.provider is not None:
            rendered = "\n".join(
                f"{k.upper()}:\n" + "\n".join(f"- {f.statement} ({f.note})" for f in v)
                for k, v in buckets.items()
            )
            prompt = (
                f"QUESTION: {ctx.problem.question}\nDECISION: {decision}\nVERDICT: {verdict}\n"
                f"CONFIDENCE: {judgment.confidence if judgment else 'n/a'}\n\n{rendered}\n\n"
                f"OPEN QUESTIONS:\n" + "\n".join(f"- {r}" for r in recs) + "\n\n"
                'Return JSON: {"summary": str (3-6 sentences), "recommendations": [str]}'
            )
            draft = await ctx.generate(
                provider=self.provider,
                schema=SynthesisDraft,
                messages=[
                    Message(role="system", content=SYNTH_SYSTEM),
                    Message(role="user", content=prompt),
                ],
                agent="synthesizer",
                task="synthesize",
                temperature=self.temperature,
            )
            summary = draft.summary
            recs = draft.recommendations or recs
            method = "model"
        return Synthesis(
            summary=summary,
            answer=judgment.answer if judgment else None,
            established=buckets["established"],
            probable=buckets["probable"],
            disputed=buckets["disputed"],
            unknown=buckets["unknown"],
            recommendations=recs,
            confidence=judgment.confidence if judgment else None,
            method="model" if method == "model" else "structural",
        )


class SynthesisStep(Step):
    name = "synthesis"

    async def run(self, ctx: RunContext) -> None:
        synthesizer = ctx.synthesizer or Synthesizer()
        try:
            ctx.state.synthesis = await synthesizer.synthesize(ctx)
        except Exception as exc:  # budget, provider, or schema failure
            ctx.state.errors.append(f"synthesis failed: {type(exc).__name__}: {exc}")
            ctx.warn("model synthesis failed; falling back to structural synthesis")
            ctx.state.synthesis = await Synthesizer().synthesize(ctx)
        ctx.emit(
            EventType.SYNTHESIS_CREATED,
            method=ctx.state.synthesis.method,
            established=len(ctx.state.synthesis.established),
            disputed=len(ctx.state.synthesis.disputed),
        )
