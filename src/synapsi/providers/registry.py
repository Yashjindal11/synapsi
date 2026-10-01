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


def _register_builtins() -> None:
    from synapsi.providers.anthropic import AnthropicProvider
    from synapsi.providers.gemini import GeminiProvider
    from synapsi.providers.openai_compat import (
        compatible_provider,
        huggingface_provider,
        ollama_provider,
        openai_provider,
    )

    register_provider(
        "mock",
        lambda model, **kw: MockProvider(model, **kw),
        "Deterministic offline placeholder output (tests and demos only).",
    )
    register_provider("openai", openai_provider, "OpenAI Chat Completions (OPENAI_API_KEY).")
    register_provider(
        "anthropic",
        lambda model, **kw: AnthropicProvider(model, **kw),
        "Anthropic Messages API (ANTHROPIC_API_KEY).",
    )
    register_provider(
        "gemini",
        lambda model, **kw: GeminiProvider(model, **kw),
        "Google Gemini API (GEMINI_API_KEY).",
    )
    register_provider("ollama", ollama_provider, "Local Ollama server (http://localhost:11434).")
    register_provider(
        "huggingface", huggingface_provider, "Hugging Face inference router (HF_TOKEN)."
    )
    register_provider(
        "openai-compatible",
        compatible_provider,
        "Any OpenAI-compatible server; requires base_url.",
    )


PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
}

_register_builtins()
