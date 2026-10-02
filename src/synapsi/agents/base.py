from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from synapsi.agents import prompts as P
from synapsi.agents.drafts import (
    ChallengeSet,
    EvidenceLinkSet,
    PerspectiveDraft,
    QueryPlan,
    RebuttalSet,
    ReviewSet,
    RevisionDraft,
    ToolPlan,
)
from synapsi.agents.roles import RoleSpec, get_role
from synapsi.claims.models import Claim
from synapsi.deliberation.models import Challenge, Perspective, Rebuttal
from synapsi.evidence.models import Evidence
from synapsi.observability.events import EventType
from synapsi.providers.base import Message, ModelProvider
from synapsi.providers.registry import create_provider
from synapsi.providers.wrappers import managed
from synapsi.tools.base import Tool

if TYPE_CHECKING:
    from synapsi.workflows.context import RunContext

T = TypeVar("T", bound=BaseModel)


class Agent:
    """A participant in a deliberation.

    Each protocol task is a separate async method so subclasses can override
    one behaviour (or replace the LLM entirely) without touching the rest.

    ``level`` groups agents for hierarchical workflows (1 = analysts,
    2 = reviewers).
    """

    def __init__(
        self,
        name: str,
        role: str | RoleSpec = "analyst",
        model: str | ModelProvider = "mock",
        *,
        instructions: str = "",
        tools: Sequence[Tool] = (),
        capabilities: Sequence[str] = (),
        temperature: float = 0.7,
        max_tokens: int | None = None,
        level: int = 1,
        config: dict[str, Any] | None = None,
    ):
        self.name = name
        self.role = role if isinstance(role, RoleSpec) else get_role(role)
        self.provider = (
            model if isinstance(model, ModelProvider) else managed(create_provider(model))
        )
        self.instructions = instructions
        self.tools = {t.name: t for t in tools}
        self.capabilities = tuple(capabilities)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.level = level
        self.config = dict(config or {})

    # -- identity -------------------------------------------------------------
    @classmethod
    def from_role(cls, role: str, model: str | ModelProvider = "mock", **kw: Any) -> Agent:
        spec = get_role(role)
        return cls(kw.pop("name", None) or spec.title, spec, model, **kw)

    @property
    def model_id(self) -> str:
        return self.provider.id

    @property
    def is_contrarian(self) -> bool:
        return self.role.stance == "contrarian"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.key,
            "model": self.model_id,
            "tools": sorted(self.tools),
            "level": self.level,
            "temperature": self.temperature,
        }

    def system_prompt(self) -> str:
        return P.system_prompt(self)

    def __repr__(self) -> str:
        return f"<Agent {self.name} ({self.role.key}) {self.model_id}>"

    # -- model access -------------------------------------------------------
    async def _generate(
        self,
        ctx: RunContext,
        schema: type[T],
        user: str,
        *,
        task: str,
        hints: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> T:
        return await ctx.generate(
            provider=self.provider,
            schema=schema,
            messages=[
                Message(role="system", content=self.system_prompt()),
                Message(role="user", content=user),
            ],
            agent=self.name,
            task=task,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            hints=hints,
        )

    # -- tools --------------------------------------------------------------
    async def gather_with_tools(self, ctx: RunContext, focus: str = "") -> list[Evidence]:
        """Let the agent call its tools; every result is stored as evidence."""
        gathered: list[Evidence] = []
        if not self.tools:
            return gathered
        specs = json.dumps([t.spec() for t in self.tools.values()])
        log: list[str] = []
        for _ in range(ctx.settings.max_tool_rounds):
            prompt = "\n\n".join(
                [
                    P.render_problem(ctx),
                    f"FOCUS: {focus}" if focus else "",
                    f"AVAILABLE TOOLS: {specs}",
                    "RESULTS SO FAR:\n" + ("\n".join(log) if log else "(none)"),
                    "Decide which tool calls (if any) would produce evidence for this problem.",
                    P.TOOL_SHAPE,
                ]
            )
            plan = await self._generate(
                ctx, ToolPlan, prompt, task="tool_plan", hints={"tools": list(self.tools)}
            )
            if not plan.tool_calls:
                break
            for call in plan.tool_calls:
                evidence = await self.call_tool(ctx, call.tool, call.arguments)
                gathered.extend(evidence)
                ids = ", ".join(e.id for e in evidence) or "no evidence"
                log.append(f"{call.tool}({json.dumps(call.arguments)[:200]}) -> {ids}")
        return gathered

    async def call_tool(
        self, ctx: RunContext, name: str, arguments: dict[str, Any]
    ) -> list[Evidence]:
        tool = self.tools.get(name)
        call_id = f"T{len(ctx.state.evidence) + 1}-{self.name}"
        ctx.emit(EventType.TOOL_CALL, agent=self.name, tool=name, arguments=arguments)
        if tool is None:
            ctx.warn(f"{self.name} requested unknown tool {name!r}")
            return []
        result = await tool(**arguments)
        ctx.emit(
            EventType.TOOL_RESULT,
            agent=self.name,
            tool=name,
            error=result.error,
            sources=len(result.sources),
        )
        if result.error:
            ctx.warn(f"{self.name}: tool {name} failed: {result.error}")
            return []
        query = next((str(v) for v in arguments.values()), None)
        return [
            ctx.add_evidence(ev)
            for ev in result.to_evidence(
                tool.evidence_kind, produced_by=self.name, query=query, call_id=call_id
            )
        ]

    # -- protocol tasks -------------------------------------------------------
    async def analyze(
        self,
        ctx: RunContext,
        *,
        visible: Sequence[Perspective] = (),
        stance_instruction: str = "",
        evidence_ids: set[str] | None = None,
        use_tools: bool = True,
    ) -> PerspectiveDraft:
        """Form a perspective. With empty ``visible`` this is independent reasoning.

        ``evidence_ids`` restricts shared evidence to a snapshot so concurrently
        running agents do not see each other's tool results or recollections.
        """
        if use_tools and self.tools and ctx.settings.max_tool_rounds:
            await self.gather_with_tools(ctx)
        evidence = ctx.visible_evidence(self.name, evidence_ids)
        sections = [P.render_problem(ctx), P.render_evidence(evidence)]
        if visible:
            others = "\n\n".join(P.render_perspective(ctx, ctx.label(p.agent), p) for p in visible)
            sections.append(f"EARLIER ANALYSES (claims, not evidence):\n{others}")
        if stance_instruction:
            sections.append(f"ASSIGNMENT: {stance_instruction}")
        sections.append(P.PERSPECTIVE_SHAPE)
        return await self._generate(
            ctx,
            PerspectiveDraft,
            "\n\n".join(sections),
            task="analyze",
            hints={"evidence_ids": [e.id for e in evidence]},
        )

    async def challenge(
        self,
        ctx: RunContext,
        targets: Sequence[Claim],
        *,
        max_challenges: int,
        instruction: str = "",
    ) -> ChallengeSet:
        evidence = ctx.visible_evidence(self.name)
        sections = [
            P.render_problem(ctx),
            P.render_evidence(evidence),
            f"CLAIMS TO EXAMINE:\n{P.render_claims(ctx, targets)}",
            instruction
            or (
                f"Raise at most {max_challenges} specific challenges against the weakest "
                "load-bearing claims. Each must name the problem and the question that would "
                "resolve it. Raise none if the claims are sound."
            ),
            P.CHALLENGE_SHAPE,
        ]
        return await self._generate(
            ctx,
            ChallengeSet,
            "\n\n".join(sections),
            task="challenge",
            hints={
                "claim_ids": [c.id for c in targets],
                "evidence_ids": [e.id for e in evidence],
                "mock_list_len": min(max_challenges, 2),
            },
        )

    async def review(
        self, ctx: RunContext, targets: Sequence[tuple[str, Perspective]], *, max_challenges: int
    ) -> ReviewSet:
        rendered = "\n\n".join(P.render_perspective(ctx, label, p) for label, p in targets)
        evidence = ctx.visible_evidence(self.name)
        sections = [
            P.render_problem(ctx),
            P.render_evidence(evidence),
            f"ANALYSES TO REVIEW:\n{rendered}",
            "Review each analysis on evidence and reasoning, not on whether it matches your "
            f"view. List claims you find well supported and at most {max_challenges} "
            "challenges per analysis.",
            P.REVIEW_SHAPE,
        ]
        claim_ids = [cid for _, p in targets for cid in p.claim_ids]
        return await self._generate(
            ctx,
            ReviewSet,
            "\n\n".join(sections),
            task="review",
            hints={
                "targets": [label for label, _ in targets],
                "claim_ids": claim_ids,
                "evidence_ids": [e.id for e in evidence],
                "mock_list_len": len(targets),
            },
        )

    async def respond(self, ctx: RunContext, challenges: Sequence[Challenge]) -> RebuttalSet:
        own = ctx.state.claims.by_agent(self.name)
        evidence = ctx.visible_evidence(self.name)
        sections = [
            P.render_problem(ctx),
            P.render_evidence(evidence),
            f"YOUR CLAIMS:\n{P.render_claims(ctx, own, show_owner=False)}",
            f"CHALLENGES TO YOUR CLAIMS:\n{P.render_challenges(challenges, ctx)}",
            "Respond to each challenge. Concede or revise when the challenge is right; defend "
            "only with evidence ids or a clear argument.",
            P.REBUTTAL_SHAPE,
        ]
        return await self._generate(
            ctx,
            RebuttalSet,
            "\n\n".join(sections),
            task="respond",
            hints={
                "challenge_ids": [c.id for c in challenges],
                "evidence_ids": [e.id for e in evidence],
                "mock_list_len": len(challenges),
            },
        )

    async def revise(
        self,
        ctx: RunContext,
        own: Perspective,
        *,
        challenges: Sequence[Challenge],
        rebuttals: Sequence[Rebuttal],
        others: Sequence[tuple[str, Perspective]],
    ) -> RevisionDraft:
        other_text = "\n\n".join(P.render_perspective(ctx, label, p) for label, p in others)
        evidence = ctx.visible_evidence(self.name)
        sections = [
            P.render_problem(ctx),
            P.render_evidence(evidence),
            f"YOUR CURRENT POSITION:\n{P.render_perspective(ctx, 'You', own)}",
            f"CHALLENGES RAISED:\n{P.render_challenges(challenges, ctx) or '(none)'}",
            f"RESPONSES:\n{P.render_rebuttals(rebuttals) or '(none)'}",
            f"OTHER ANALYSES (claims, not evidence):\n{other_text or '(none)'}",
            "Update your position only where evidence or arguments warrant it. Agreement by "
            "others is not a reason to change. State the reason for any change.",
            P.REVISION_SHAPE,
        ]
        return await self._generate(
            ctx,
            RevisionDraft,
            "\n\n".join(sections),
            task="revise",
            hints={
                "own_claim_ids": own.claim_ids,
                "evidence_ids": [e.id for e in evidence],
                "mock_list_len": 1,
            },
        )

    async def plan_queries(
        self, ctx: RunContext, *, max_queries: int, focus: str = ""
    ) -> QueryPlan:
        prompt = "\n\n".join(
            [
                P.render_problem(ctx),
                f"FOCUS: {focus}" if focus else "",
                f"Write at most {max_queries} short search queries whose results would most "
                "reduce uncertainty about this problem. Include at least one query that could "
                "find evidence against the most likely answer.",
                P.QUERY_SHAPE,
            ]
        )
        return await self._generate(ctx, QueryPlan, prompt, task="plan_queries")

    async def link_evidence(
        self, ctx: RunContext, claims: Sequence[Claim], evidence: Sequence[Evidence]
    ) -> EvidenceLinkSet:
        prompt = "\n\n".join(
            [
                P.render_evidence(evidence),
                f"CLAIMS:\n{P.render_claims(ctx, claims, show_owner=False)}",
                "For each evidence item and claim it bears on, decide whether it supports or "
                "contradicts the claim. Omit irrelevant pairs. Judge only what the text says.",
                P.LINK_SHAPE,
            ]
        )
        return await self._generate(
            ctx,
            EvidenceLinkSet,
            prompt,
            task="link_evidence",
            temperature=0.0,
            hints={
                "claim_ids": [c.id for c in claims],
                "evidence_ids": [e.id for e in evidence],
                "mock_list_len": 1,
            },
        )
