"""SynapSI: many minds, one intelligence.

An open-source framework for structured multi-agent deliberation and for
measuring whether it actually helps.
"""

from synapsi._version import __version__
from synapsi.agents import Agent, RoleSpec, get_role, list_roles, register_role
from synapsi.claims import Claim, ClaimGraph, ClaimStatus, ClaimType
from synapsi.core import Problem
from synapsi.council import Council
from synapsi.evidence import Evidence, EvidencePool, Provenance, SourceKind
from synapsi.judgment.judge import Judge
from synapsi.judgment.models import Judgment, Verdict
from synapsi.judgment.structural import StructuralJudge
from synapsi.modes import mode_settings
from synapsi.providers import ModelProvider, create_provider, register_provider
from synapsi.result import SynapSIResult
from synapsi.strategies import build_strategy, list_strategies, register_strategy
from synapsi.synthesis.models import Synthesis
from synapsi.synthesis.synthesizer import Synthesizer
from synapsi.tools import Tool, tool
from synapsi.workflows import Budget, RunContext, RunSettings, Step, Workflow

__all__ = [
    "Agent",
    "Budget",
    "Claim",
    "ClaimGraph",
    "ClaimStatus",
    "ClaimType",
    "Council",
    "Evidence",
    "EvidencePool",
    "Judge",
    "Judgment",
    "ModelProvider",
    "Problem",
    "Provenance",
    "RoleSpec",
    "RunContext",
    "RunSettings",
    "SourceKind",
    "Step",
    "StructuralJudge",
    "SynapSIResult",
    "Synthesis",
    "Synthesizer",
    "Tool",
    "Verdict",
    "Workflow",
    "__version__",
    "build_strategy",
    "create_provider",
    "get_role",
    "list_roles",
    "list_strategies",
    "mode_settings",
    "register_provider",
    "register_role",
    "register_strategy",
    "tool",
]
