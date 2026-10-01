from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from synapsi.evidence.models import Evidence, Provenance, SourceKind


class ToolSource(BaseModel):
    """One retrievable unit returned by a tool (a search hit, a document chunk...)."""

    content: str
    source_id: str | None = None
    title: str | None = None
    locator: str | None = None


class ToolResult(BaseModel):
    tool: str
    content: str = ""
    sources: list[ToolSource] = Field(default_factory=list)
    error: str | None = None

    def to_evidence(
        self,
        kind: SourceKind,
        *,
        produced_by: str | None = None,
        query: str | None = None,
        call_id: str | None = None,
    ) -> list[Evidence]:
        if self.error:
            return []
        items = self.sources or [ToolSource(content=self.content)]
        return [
            Evidence(
                content=s.content,
                provenance=Provenance(
                    source_kind=kind,
                    source_id=s.source_id,
                    title=s.title,
                    locator=s.locator,
                    tool=self.tool,
                    tool_call_id=call_id,
                    query=query,
                    produced_by=produced_by,
                ),
            )
            for s in items
            if s.content.strip()
        ]


class Tool:
    """A callable capability an agent may invoke.

    ``function`` may be sync or async and may return ``str``, ``ToolResult``,
    or a list of ``ToolSource``. Results become evidence of ``evidence_kind``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        *,
        parameters: dict[str, str] | None = None,
        evidence_kind: SourceKind = SourceKind.TOOL,
        timeout: float = 30.0,
        capabilities: tuple[str, ...] = (),
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters if parameters is not None else _infer_parameters(function)
        self.evidence_kind = evidence_kind
        self.timeout = timeout
        self.capabilities = capabilities

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    async def __call__(self, **arguments: Any) -> ToolResult:
        unknown = set(arguments) - set(self.parameters)
        if unknown:
            return ToolResult(tool=self.name, error=f"unknown arguments: {sorted(unknown)}")
        try:
            if inspect.iscoroutinefunction(self.function):
                coro = self.function(**arguments)
            else:
                coro = asyncio.to_thread(self.function, **arguments)
            output = await asyncio.wait_for(coro, timeout=self.timeout)
        except TimeoutError:
            return ToolResult(tool=self.name, error=f"timed out after {self.timeout}s")
        except Exception as exc:
            return ToolResult(tool=self.name, error=f"{type(exc).__name__}: {exc}")
        return _coerce(self.name, output)

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


def _coerce(name: str, output: Any) -> ToolResult:
    if isinstance(output, ToolResult):
        return output
    if isinstance(output, list) and all(isinstance(o, ToolSource) for o in output):
        return ToolResult(tool=name, sources=output, content=f"{len(output)} results")
    return ToolResult(tool=name, content=str(output))


def _infer_parameters(function: Callable[..., Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    for p in inspect.signature(function).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        annotation = p.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        if annotation is inspect.Parameter.empty:
            type_name = "any"
        optional = "" if p.default is inspect.Parameter.empty else " (optional)"
        params[p.name] = f"{type_name}{optional}"
    return params


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    evidence_kind: SourceKind = SourceKind.TOOL,
    timeout: float = 30.0,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator turning a function into a :class:`Tool`."""

    def wrap(function: Callable[..., Any]) -> Tool:
        return Tool(
            name or function.__name__,
            description or (inspect.getdoc(function) or "").split("\n")[0],
            function,
            evidence_kind=evidence_kind,
            timeout=timeout,
        )

    return wrap
