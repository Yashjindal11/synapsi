"""Small, dependency-free statistics for comparing strategies."""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def brier(confidences: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    if not confidences:
        return None
    return sum((c - float(o)) ** 2 for c, o in zip(confidences, outcomes, strict=True)) / len(
        confidences
    )


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[bool], bins: int = 10
) -> float | None:
    if not confidences:
        return None
    total = len(confidences)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)
        ]
        if not idx:
            continue
        acc = sum(outcomes[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(acc - conf)
    return ece


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pair counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability a random positive scores higher than a random negative (ties = 0.5)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
