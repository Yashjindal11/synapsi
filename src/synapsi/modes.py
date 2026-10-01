"""Cost/depth presets. Modes only set run parameters; they never add agents."""

from __future__ import annotations

from synapsi.core.errors import ConfigError
from synapsi.workflows.context import RunSettings

MODES: dict[str, dict[str, object]] = {
    "fast": {
        "rounds": 1,
        "max_challenges_per_agent": 2,
        "verify_evidence": False,
        "max_tool_rounds": 1,
    },
    "balanced": {
        "rounds": 2,
        "max_challenges_per_agent": 3,
        "verify_evidence": False,
        "max_tool_rounds": 2,
    },
    "deep": {
        "rounds": 3,
        "max_challenges_per_agent": 4,
        "verify_evidence": True,
        "max_tool_rounds": 3,
    },
}

PRESET_ROLES: dict[str, list[str]] = {
    "fast": ["analyst", "domain_expert", "skeptic"],
    "balanced": ["researcher", "statistician", "domain_expert", "risk_analyst", "skeptic"],
    "deep": [
        "researcher",
        "data_scientist",
        "statistician",
        "domain_expert",
        "risk_analyst",
        "fact_checker",
        "skeptic",
        "devils_advocate",
    ],
}


def mode_settings(mode: str, **overrides: object) -> RunSettings:
    if mode == "custom":
        return RunSettings.model_validate(overrides)
    if mode not in MODES:
        raise ConfigError(f"unknown mode {mode!r}; use fast, balanced, deep, or custom")
    return RunSettings.model_validate({**MODES[mode], **overrides})
