"""Shared helpers for the examples.

Every example runs offline with the mock provider (placeholder text, useful to
see the pipeline and data structures). Set ``SYNAPSI_MODEL`` for real output:

    SYNAPSI_MODEL=openai:gpt-4o-mini python examples/01_decision_analysis.py
    SYNAPSI_MODEL=ollama:llama3.1 python examples/01_decision_analysis.py
"""

from __future__ import annotations

import os
import sys

from synapsi import SynapSIResult


def model(default: str = "mock") -> str:
    spec = os.environ.get("SYNAPSI_MODEL", default)
    if spec.startswith("mock"):
        print(
            "[mock provider: output is placeholder text. Set SYNAPSI_MODEL for real reasoning.]",
            file=sys.stderr,
        )
    return spec


def show(result: SynapSIResult) -> None:
    j, s = result.judgment, result.synthesis
    print(f"\nQ: {result.problem.question}")
    if j:
        print(
            f"verdict={j.verdict.value} answer={j.answer} confidence={j.confidence:.2f} "
            f"evidence={j.evidence_strength.value} (judge: {j.method})"
        )
    if s:
        print(f"summary: {s.summary}")
        for name in ("established", "probable", "disputed", "unknown"):
            items = getattr(s, name)
            print(f"  {name}: {len(items)}")
            for f in items[:3]:
                print(f"    - {f.statement[:110]}")
    u = result.metadata.usage.total
    print(
        f"calls={u.calls} tokens={u.total_tokens} cost={u.cost_usd} "
        f"time={result.metadata.latency_s:.2f}s strategy={result.metadata.strategy}"
    )
