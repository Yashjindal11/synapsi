"""Model providers. The core depends only on :class:`ModelProvider`."""

from synapsi.providers.base import Completion, CompletionRequest, Message, ModelProvider
from synapsi.providers.mock import MockProvider
from synapsi.providers.registry import (
    available_providers,
    create_provider,
    parse_spec,
    register_provider,
)
from synapsi.providers.structured import extract_json, generate_structured

__all__ = [
    "Completion",
    "CompletionRequest",
    "Message",
    "MockProvider",
    "ModelProvider",
    "available_providers",
    "create_provider",
    "extract_json",
    "generate_structured",
    "parse_spec",
    "register_provider",
]
