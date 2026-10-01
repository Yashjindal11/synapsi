from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from synapsi.core.pricing import PricingTable
from synapsi.core.usage import Usage
from synapsi.providers.base import Completion


class UsageReport(BaseModel):
    total: Usage
    by_agent: dict[str, Usage] = Field(default_factory=dict)
    by_model: dict[str, Usage] = Field(default_factory=dict)
    by_task: dict[str, Usage] = Field(default_factory=dict)


class UsageTracker:
    def __init__(self, pricing: PricingTable | None = None):
        self.pricing = pricing or PricingTable()
        self.total = Usage()
        self._by_agent: defaultdict[str, Usage] = defaultdict(Usage)
        self._by_model: defaultdict[str, Usage] = defaultdict(Usage)
        self._by_task: defaultdict[str, Usage] = defaultdict(Usage)

    def record(self, completion: Completion, *, agent: str, task: str) -> float | None:
        cost = 0.0 if completion.cached else self.pricing.cost(completion.model, completion.usage)
        for bucket in (
            self.total,
            self._by_agent[agent],
            self._by_model[completion.model],
            self._by_task[task],
        ):
            bucket.add(
                completion.usage,
                cost_usd=cost,
                latency_s=completion.latency_s,
                cached=completion.cached,
            )
        return cost

    def report(self) -> UsageReport:
        return UsageReport(
            total=self.total.model_copy(),
            by_agent={k: v.model_copy() for k, v in self._by_agent.items()},
            by_model={k: v.model_copy() for k, v in self._by_model.items()},
            by_task={k: v.model_copy() for k, v in self._by_task.items()},
        )
