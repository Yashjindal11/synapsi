from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RoleSpec(BaseModel):
    """Reusable description of an expertise or stance."""

    key: str
    title: str
    description: str
    instructions: str = ""
    stance: Literal["neutral", "contrarian"] = "neutral"
    suggested_tools: tuple[str, ...] = ()


_ROLES: dict[str, RoleSpec] = {}


def register_role(spec: RoleSpec) -> RoleSpec:
    _ROLES[spec.key] = spec
    return spec


def get_role(key: str) -> RoleSpec:
    normalized = key.strip().lower().replace(" ", "_").replace("-", "_").replace("'", "")
    if normalized in _ROLES:
        return _ROLES[normalized]
    # Unknown roles are allowed: the key becomes a free-form expertise.
    title = key.strip().replace("_", " ").title()
    return RoleSpec(key=normalized, title=title, description=f"Expert in {key.strip()}.")


def list_roles() -> list[RoleSpec]:
    return list(_ROLES.values())


for _spec in [
    RoleSpec(
        key="analyst",
        title="Analyst",
        description="Generalist who reasons carefully about the problem as posed.",
    ),
    RoleSpec(
        key="researcher",
        title="Researcher",
        description="Finds and evaluates information relevant to the problem.",
        instructions=(
            "Prefer claims you can tie to evidence ids. Distinguish what sources say from "
            "what you infer. Note gaps where information is missing."
        ),
        suggested_tools=("web_search", "document_search"),
    ),
    RoleSpec(
        key="data_scientist",
        title="Data Scientist",
        description="Analyses quantitative evidence, data quality, and modelling choices.",
        instructions=(
            "Check sample sizes, leakage, confounders, baselines, and whether metrics fit "
            "the decision. Use the calculator for any arithmetic."
        ),
        suggested_tools=("calculator", "sql_query"),
    ),
    RoleSpec(
        key="statistician",
        title="Statistician",
        description="Evaluates statistical claims, uncertainty, and inference validity.",
        instructions=(
            "Separate correlation from causation, quantify uncertainty where possible, and "
            "flag multiple-comparison, selection, and base-rate problems."
        ),
        suggested_tools=("calculator",),
    ),
    RoleSpec(
        key="software_engineer",
        title="Software Engineer",
        description="Reviews technical feasibility, architecture, and implementation risk.",
        instructions=(
            "Consider maintainability, failure modes, operational cost, security, and "
            "migration paths. Be concrete about trade-offs."
        ),
    ),
    RoleSpec(
        key="domain_expert",
        title="Domain Expert",
        description="Provides specialised reasoning about the problem's domain.",
        instructions="Explain domain constraints a generalist would miss.",
    ),
    RoleSpec(
        key="skeptic",
        title="Skeptic",
        description="Looks for weaknesses, unsupported leaps, and missing evidence.",
        instructions=(
            "Do not invent objections. Target the weakest load-bearing claims and say what "
            "evidence would resolve each concern."
        ),
        stance="contrarian",
    ),
    RoleSpec(
        key="devils_advocate",
        title="Devil's Advocate",
        description="Constructs the strongest case for an alternative position.",
        instructions=(
            "Steelman the best competing answer even if you doubt it. Be explicit that this "
            "is an argued position, and report your honest confidence separately."
        ),
        stance="contrarian",
    ),
    RoleSpec(
        key="fact_checker",
        title="Fact Checker",
        description="Verifies factual claims against available evidence.",
        instructions=(
            "For each factual claim, decide whether the evidence supports it, contradicts it, "
            "or is silent. Never treat another agent's assertion as evidence."
        ),
        suggested_tools=("web_search", "document_search"),
    ),
    RoleSpec(
        key="risk_analyst",
        title="Risk Analyst",
        description="Identifies failure modes, tail risks, and their likelihood and impact.",
        instructions="Consider second-order effects and what happens if key assumptions fail.",
    ),
    RoleSpec(
        key="judge",
        title="Judge",
        description="Evaluates competing arguments on evidence and reasoning quality.",
    ),
    RoleSpec(
        key="synthesizer",
        title="Synthesizer",
        description="Summarises what is established, probable, disputed, and unknown.",
    ),
]:
    register_role(_spec)
