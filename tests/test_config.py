from pathlib import Path

import pytest

from synapsi.config import DEFAULT_CONFIG, build_council, load_config
from synapsi.core.errors import ConfigError

TOML = """
[models.default]
spec = "mock"
price_per_mtok = [1.0, 2.0]

[[agents]]
role = "statistician"
tools = ["calculator", "document_search"]

[[agents]]
role = "skeptic"
name = "Doubter"

[judge]
kind = "structural"

[workflow]
strategy = "debate"
mode = "fast"
rounds = 2

[tools.document_search]
paths = ["notes.md"]
"""


def test_default_yaml_template_builds(tmp_path: Path) -> None:
    path = tmp_path / "synapsi.yaml"
    path.write_text(DEFAULT_CONFIG)
    council = build_council(load_config(path), base_dir=tmp_path)
    assert [a.role.key for a in council.agents] == [
        "researcher",
        "statistician",
        "domain_expert",
        "skeptic",
    ]
    assert council.settings.seed == 42 and council.settings.budget.max_model_calls == 200
    assert len({id(a.provider) for a in council.agents}) == 1


async def test_toml_config_with_tools_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Hub turnaround is 45 minutes.")
    path = tmp_path / "c.toml"
    path.write_text(TOML)
    council = build_council(load_config(path), base_dir=tmp_path)
    assert council.settings.rounds == 2 and council.settings.max_challenges_per_agent == 2
    assert council.judge is None
    assert set(council.agents[0].tools) == {"calculator", "document_search"}
    assert council.agents[1].name == "Doubter"
    assert "mock:mock" in council.pricing
    result = await council.run("How long is turnaround?")
    assert result.metadata.strategy == "debate"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"agents": []}', "agents"),
        ('{"agents": [{"role": "x", "api_key": "sk-123"}]}', "api_key"),
        (
            '{"agents": [{"role": "x", "tools": ["web_search"]}], '
            '"models": {"default": {"spec": "mock"}}}',
            "backend",
        ),
        ('{"agents": [{"role": "x"}]}', "default"),
    ],
)
def test_invalid_configs(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "c.json"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        build_council(load_config(path), base_dir=tmp_path)


def test_model_override_replaces_all_models(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    (tmp_path / "notes.md").write_text("x")
    path.write_text(TOML.replace('spec = "mock"', 'spec = "openai:gpt-x"'))
    council = build_council(load_config(path), base_dir=tmp_path, model_override="mock:other")
    assert {a.model_id for a in council.agents} == {"mock:other"}
