from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synapsi.core.errors import ConfigError
from synapsi.providers.base import ModelProvider
from synapsi.providers.mock import MockProvider

ProviderFactory = Callable[..., ModelProvider]

_FACTORIES: dict[str, ProviderFactory] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register_provider(name: str, factory: ProviderFactory, description: str = "") -> None:
    """Make ``name:model`` specs resolve through ``factory(model, **options)``."""
    _FACTORIES[name] = factory
    _DESCRIPTIONS[name] = description


def available_providers() -> dict[str, str]:
    return dict(_DESCRIPTIONS)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:model``. Model names may themselves contain colons."""
    name, sep, model = spec.partition(":")
    if not sep:
        return name, name if name == "mock" else ""
    return name, model


def create_provider(spec: str | ModelProvider, **options: Any) -> ModelProvider:
    """Resolve a spec such as ``openai:gpt-4o-mini`` or ``ollama:llama3.1:8b``."""
    if isinstance(spec, ModelProvider):
        return spec
    name, model = parse_spec(spec.strip())
    factory = _FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES))
        raise ConfigError(f"unknown provider {name!r} in {spec!r}; known: {known}")
    if not model:
        raise ConfigError(f"model spec {spec!r} must look like 'provider:model'")
    return factory(model, **options)


register_provider(
    "mock",
    lambda model, **kw: MockProvider(model, **kw),
    "Deterministic offline placeholder output (tests and demos only).",
)
