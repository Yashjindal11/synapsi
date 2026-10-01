"""Helpers for testing custom agents, steps, and workflows offline."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from synapsi.providers.base import CompletionRequest
from synapsi.providers.mock import MockProvider, schema_placeholder

Script = Callable[[CompletionRequest], dict[str, Any] | None]


def scripted(script: Script, model: str = "scripted") -> MockProvider:
    """Mock provider whose replies are chosen per request.

    ``script`` receives the request (``request.metadata["task"]`` names the
    protocol task) and returns a dict to send as JSON, or ``None`` to fall back
    to schema-valid placeholder output.
    """

    def responder(request: CompletionRequest) -> str:
        out = script(request)
        return json.dumps(out if out is not None else schema_placeholder(request))

    return MockProvider(model, responder=responder)
