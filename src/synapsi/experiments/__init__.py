"""Experiment engine: compare strategies, models, and settings on the same problems."""

from synapsi.experiments.engine import (
    Comparison,
    Experiment,
    ExperimentResult,
    StrategySummary,
    TrialRecord,
    summarise,
)
from synapsi.experiments.problems import ProblemSet
from synapsi.experiments.scoring import auto_scorer, exact_scorer, numeric_scorer
from synapsi.experiments.suites import SUITES

__all__ = [
    "SUITES",
    "Comparison",
    "Experiment",
    "ExperimentResult",
    "ProblemSet",
    "StrategySummary",
    "TrialRecord",
    "auto_scorer",
    "exact_scorer",
    "numeric_scorer",
    "summarise",
]
