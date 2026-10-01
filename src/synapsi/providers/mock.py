from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Awaitable, Callable
from typing import Any

from synapsi.core.usage import TokenUsage
from synapsi.providers.base import Completion, CompletionRequest, ModelProvider

Responder = Callable[[CompletionRequest], "str | Awaitable[str]"]


class MockProvider(ModelProvider):
    """Deterministic offline provider for tests, demos, and pipeline debugging.

    Without a ``responder`` it fabricates schema-valid placeholder output from
    ``request.response_schema``, using ids passed in ``request.metadata`` so
    references between artifacts stay consistent. Its content is meaningless:
    never draw conclusions from runs that use it.
    """

    provider = "mock"

    def __init__(self, model: str = "mock", responder: Responder | None = None):
        super().__init__(model)
        self.responder = responder
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        if self.responder is not None:
            out = self.responder(request)
            text = out if isinstance(out, str) else await out
        elif request.response_schema is not None:
            text = json.dumps(schema_placeholder(request))
        else:
            text = f"[mock:{self.model}] {request.messages[-1].content[:80]}"
        prompt_chars = sum(len(m.content) for m in request.messages)
        return Completion(
            text=text,
            model=self.id,
            usage=TokenUsage(prompt_tokens=prompt_chars // 4, completion_tokens=len(text) // 4),
            finish_reason="stop",
        )


def _seed(request: CompletionRequest) -> int:
    key = json.dumps(
        [request.metadata.get(k) for k in ("agent", "task", "round")]
        + [request.messages[-1].content if request.messages else ""],
        default=str,
    )
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def schema_placeholder(request: CompletionRequest) -> Any:
    schema = request.response_schema or {}
    rng = random.Random(_seed(request))
    return _Gen(schema.get("$defs", {}), request.metadata, rng).value(schema, "root")


class _Gen:
    def __init__(self, defs: dict[str, Any], hints: dict[str, Any], rng: random.Random):
        self.defs = defs
        self.hints = hints
        self.rng = rng
        self.agent = str(hints.get("agent", "agent"))

    def _pick(self, key: str) -> list[str]:
        values = self.hints.get(key) or []
        return [str(v) for v in values]

    def value(self, schema: dict[str, Any], name: str) -> Any:
        if "$ref" in schema:
            return self.value(self.defs[schema["$ref"].split("/")[-1]], name)
        if "anyOf" in schema:
            options = [s for s in schema["anyOf"] if s.get("type") != "null"]
            return self.value(options[0], name) if options else None
        if "enum" in schema:
            return self.rng.choice(schema["enum"])
        kind = schema.get("type")
        if kind == "object":
            props = schema.get("properties", {})
            return {k: self.value(v, k) for k, v in props.items()}
        if kind == "array":
            return self._array(schema, name)
        if kind == "string":
            return self._string(name)
        if kind == "number":
            lo, hi = schema.get("minimum", 0.0), schema.get("maximum", 1.0)
            return round(self.rng.uniform(max(lo, 0.4), min(hi, 0.9)), 2)
        if kind == "integer":
            lo, hi = schema.get("minimum", 1), schema.get("maximum", 5)
            return self.rng.randint(lo, hi)
        if kind == "boolean":
            return False
        return None

    def _array(self, schema: dict[str, Any], name: str) -> list[Any]:
        id_lists = {
            "claim_ids": "claim_ids",
            "agreements": "claim_ids",
            "supporting_claim_ids": "claim_ids",
            "contradicting_claim_ids": "claim_ids",
            "evidence_ids": "evidence_ids",
            "keep_claim_ids": "own_claim_ids",
        }
        if name in id_lists:
            pool = self._pick(id_lists[name])
            return self.rng.sample(pool, k=min(len(pool), 1))
        if name == "withdraw_claim_ids":
            return []
        items = schema.get("items", {})
        count = int(self.hints.get("mock_list_len", 2))
        maximum = schema.get("maxItems")
        if maximum is not None:
            count = min(count, maximum)
        if self._needs_targets(items) and not self._pick("claim_ids"):
            return []
        return [self.value(items, name.rstrip("s")) for _ in range(count)]

    def _needs_targets(self, items: dict[str, Any]) -> bool:
        if "$ref" in items:
            items = self.defs[items["$ref"].split("/")[-1]]
        props = items.get("properties", {})
        return "target_claim_id" in props or "claim_id" in props

    def _string(self, name: str) -> str:
        if name == "answer":
            options = self._pick("options")
            return self.rng.choice(options) if options else "undetermined"
        if name in ("target_claim_id", "claim_id"):
            ids = self._pick("claim_ids")
            return self.rng.choice(ids) if ids else "C0"
        if name == "challenge_id":
            ids = self._pick("challenge_ids")
            return self.rng.choice(ids) if ids else "X0"
        if name in ("target", "target_agent"):
            targets = self._pick("targets")
            return self.rng.choice(targets) if targets else "unknown"
        if name in ("tool", "tool_name"):
            tools = self._pick("tools")
            return tools[0] if tools else "none"
        if name == "query":
            return str(self.hints.get("question", "query"))[:80]
        return f"[mock {self.agent}] {name.replace('_', ' ')}"
