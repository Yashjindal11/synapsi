import pytest
from pydantic import BaseModel, Field

from synapsi.core.errors import ConfigError, StructuredOutputError
from synapsi.providers import (
    CompletionRequest,
    Message,
    MockProvider,
    create_provider,
    extract_json,
    generate_structured,
    parse_spec,
)


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    claim_ids: list[str] = []


def _req(**metadata: object) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content="q")], metadata=metadata)


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('prefix {"a": 3} suffix') == {"a": 3}
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def test_parse_spec_keeps_colons_in_model_names() -> None:
    assert parse_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert parse_spec("mock") == ("mock", "mock")


def test_create_provider_errors() -> None:
    with pytest.raises(ConfigError):
        create_provider("nope:model")
    assert create_provider("mock").id == "mock:mock"


async def test_mock_generates_schema_valid_output_using_hints() -> None:
    provider = MockProvider()
    req = _req(agent="A", options=["yes", "no"], claim_ids=["C1", "C2"])
    result, completions = await generate_structured(provider.complete, req, Answer)
    assert result.answer in {"yes", "no"}
    assert set(result.claim_ids) <= {"C1", "C2"}
    assert len(completions) == 1


async def test_mock_is_deterministic() -> None:
    a, _ = await generate_structured(MockProvider().complete, _req(agent="A"), Answer)
    b, _ = await generate_structured(MockProvider().complete, _req(agent="A"), Answer)
    assert a == b


async def test_structured_output_repairs_once() -> None:
    replies = iter(["not json", '{"answer": "x", "confidence": 0.3}'])
    provider = MockProvider(responder=lambda _req: next(replies))
    result, completions = await generate_structured(provider.complete, _req(), Answer)
    assert result.answer == "x"
    assert len(completions) == 2
    assert "not valid" in provider.requests[1].messages[-1].content


async def test_structured_output_gives_up() -> None:
    provider = MockProvider(responder=lambda _req: '{"answer": "x", "confidence": 7}')
    with pytest.raises(StructuredOutputError):
        await generate_structured(provider.complete, _req(), Answer, repair_attempts=1)
    assert len(provider.requests) == 2
