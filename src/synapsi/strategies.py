"""Built-in deliberation strategies, each an ordinary :class:`Workflow`.

Steps read rounds, challenge limits, and verification from ``RunSettings`` at
run time, so one strategy definition serves every mode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from synapsi.core.errors import ConfigError
from synapsi.deliberation.disagreement import DisagreementAnalysis
from synapsi.deliberation.steps import (
    ClaimAssessment,
    CollectVotes,
    CrossExamination,
    IndependentAnalysis,
    PeerReview,
    RespondToChallenges,
    Revision,
    SequentialCritique,
    SteelmanAlternative,
    converged,
)
from synapsi.judgment.steps import JudgeStep
from synapsi.research.steps import EvidenceCollection, VerifyClaims
from synapsi.synthesis.synthesizer import SynthesisStep
from synapsi.workflows.base import Conditional, Loop, Step, Workflow
from synapsi.workflows.context import RunContext

StrategyBuilder = Callable[[], Workflow]

_STRATEGIES: dict[str, tuple[StrategyBuilder, str]] = {}


def register_strategy(name: str, builder: StrategyBuilder, description: str = "") -> None:
    _STRATEGIES[name] = (builder, description)


def list_strategies() -> dict[str, str]:
    return {name: desc for name, (_, desc) in _STRATEGIES.items()}


def build_strategy(name: str) -> Workflow:
    if name not in _STRATEGIES:
        raise ConfigError(f"unknown strategy {name!r}; known: {', '.join(_STRATEGIES)}")
    workflow = _STRATEGIES[name][0]()
    workflow.name = name
    return workflow


def finalize(method: Literal["auto", "structural", "vote", "single"] = "auto") -> list[Step]:
    """Model-free bookkeeping plus judgment and synthesis.

    Runs even when the workflow stops early (budget, stop condition).
    """
    return [ClaimAssessment(), DisagreementAnalysis(), JudgeStep(method), SynthesisStep()]


def _verify() -> Step:
    return Conditional(lambda c: c.settings.verify_evidence, VerifyClaims(), name="maybe_verify")


def _rounds(ctx: RunContext) -> int:
    return ctx.settings.rounds


def _stop(ctx: RunContext) -> bool:
    return ctx.settings.early_stop and converged(ctx)


def single_model() -> Workflow:
    return Workflow([IndependentAnalysis(first_n=1), _verify()], finalize=finalize("single"))


def independent_panel() -> Workflow:
    return Workflow([IndependentAnalysis(), _verify()], finalize=finalize())


def majority_vote() -> Workflow:
    return Workflow([IndependentAnalysis(), CollectVotes()], finalize=finalize("vote"))


def peer_review() -> Workflow:
    return Workflow(
        [
            IndependentAnalysis(),
            _verify(),
            PeerReview(),
            RespondToChallenges(),
            Revision(see_others=False),
        ],
        finalize=finalize(),
    )


def debate() -> Workflow:
    return Workflow(
        [
            IndependentAnalysis(),
            _verify(),
            Loop(
                [CrossExamination(), RespondToChallenges(), Revision(see_others=True)],
                max_iterations=_rounds,
                until=_stop,
                name="debate_round",
            ),
        ],
        finalize=finalize(),
    )


def adversarial() -> Workflow:
    return Workflow(
        [
            IndependentAnalysis(),
            SteelmanAlternative(),
            _verify(),
            Loop(
                [CrossExamination(adversarial=True), RespondToChallenges(), Revision()],
                max_iterations=_rounds,
                until=_stop,
                name="adversarial_round",
            ),
        ],
        finalize=finalize(),
    )


def jury() -> Workflow:
    return Workflow(
        [
            EvidenceCollection(),
            IndependentAnalysis(),
            _verify(),
            CrossExamination(),
            RespondToChallenges(),
        ],
        finalize=finalize(),
    )


def evidence_first() -> Workflow:
    return Workflow(
        [
            EvidenceCollection(),
            IndependentAnalysis(),
            VerifyClaims(),
            PeerReview(),
            RespondToChallenges(),
        ],
        finalize=finalize(),
    )


def hierarchical() -> Workflow:
    def has_reviewers(ctx: RunContext) -> bool:
        if any(a.level >= 2 for a in ctx.agents):
            return True
        ctx.warn("hierarchical strategy has no level-2 agents; analysts review each other")
        return False

    return Workflow(
        [
            IndependentAnalysis(level=1),
            _verify(),
            Conditional(
                has_reviewers,
                PeerReview(reviewer_level=2, target_level=1),
                PeerReview(target_level=1),
                name="review_level",
            ),
            RespondToChallenges(),
            Revision(see_others=False),
        ],
        finalize=finalize(),
    )


def sequential_critique() -> Workflow:
    return Workflow([SequentialCritique(), _verify()], finalize=finalize())


for _name, _builder, _desc in [
    ("single_model", single_model, "First agent alone (baseline)."),
    ("independent_panel", independent_panel, "Independent analyses, then judge."),
    ("majority_vote", majority_vote, "Independent analyses, plurality answer (baseline)."),
    ("peer_review", peer_review, "Blind peer review, responses, private revision."),
    ("debate", debate, "Rounds of cross-examination, responses, and revision."),
    ("adversarial", adversarial, "Steelman the alternative, then adversarial rounds."),
    ("jury", jury, "Evidence collection and cross-examination; judge decides."),
    ("evidence_first", evidence_first, "Research and claim verification before review."),
    ("hierarchical", hierarchical, "Level-1 analysts reviewed by level-2 reviewers."),
    ("sequential_critique", sequential_critique, "Each agent critiques and improves the last."),
]:
    register_strategy(_name, _builder, _desc)
