from __future__ import annotations

import math
import re
from typing import Protocol

from synapsi.core.problem import Problem

_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


class Scorer(Protocol):
    def __call__(self, problem: Problem, answer: str | None) -> bool: ...


def _norm(text: str) -> str:
    return " ".join(text.strip().strip(".").lower().split())


def exact_scorer(problem: Problem, answer: str | None) -> bool:
    return (
        answer is not None and problem.answer is not None and _norm(answer) == _norm(problem.answer)
    )


def parse_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    return float(match.group().replace(",", "")) if match else None


def numeric_scorer(problem: Problem, answer: str | None, *, rel_tol: float = 1e-6) -> bool:
    if answer is None or problem.answer is None:
        return False
    got, want = parse_number(answer), parse_number(problem.answer)
    return (
        got is not None
        and want is not None
        and math.isclose(got, want, rel_tol=rel_tol, abs_tol=1e-9)
    )


def auto_scorer(problem: Problem, answer: str | None) -> bool:
    """Exact match for multiple choice; numeric comparison when the gold answer is a number."""
    if problem.options or problem.answer is None:
        return exact_scorer(problem, answer)
    if parse_number(problem.answer) is not None and _NUMBER.fullmatch(problem.answer.strip()):
        return numeric_scorer(problem, answer)
    return exact_scorer(problem, answer)
