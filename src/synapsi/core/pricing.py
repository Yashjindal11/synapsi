"""User-supplied model pricing.

SynapSI ships no price list: prices change and stale numbers would silently
corrupt cost comparisons. Supply your own, in USD per million tokens.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from synapsi.core.usage import TokenUsage


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class PricingTable:
    def __init__(self, prices: Mapping[str, ModelPrice | tuple[float, float]] | None = None):
        self._prices: dict[str, ModelPrice] = {}
        for model_id, price in (prices or {}).items():
            self.set(model_id, price)

    def set(self, model_id: str, price: ModelPrice | tuple[float, float]) -> None:
        if isinstance(price, tuple):
            price = ModelPrice(input_per_mtok=price[0], output_per_mtok=price[1])
        self._prices[model_id] = price

    def cost(self, model_id: str, usage: TokenUsage) -> float | None:
        if model_id.startswith("mock:"):
            return 0.0
        price = self._prices.get(model_id)
        if price is None:
            return None
        return (
            usage.prompt_tokens * price.input_per_mtok
            + usage.completion_tokens * price.output_per_mtok
        ) / 1_000_000

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._prices
