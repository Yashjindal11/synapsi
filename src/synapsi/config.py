"""YAML/TOML/JSON configuration.

API keys are never read from configuration files: models name the environment
variable that holds the key (``api_key_env``). Unknown fields are rejected so
typos fail loudly.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from synapsi.agents.base import Agent
from synapsi.core.errors import ConfigError
from synapsi.core.pricing import PricingTable
from synapsi.council import Council
from synapsi.judgment.judge import Judge
from synapsi.memory import JSONLMemoryStore
from synapsi.modes import mode_settings
from synapsi.providers.base import ModelProvider
from synapsi.providers.registry import create_provider
from synapsi.providers.wrappers import managed
from synapsi.synthesis.synthesizer import Synthesizer
from synapsi.tools.base import Tool
from synapsi.tools.calculator import calculator_tool
from synapsi.tools.documents import DocumentStore, document_search_tool
from synapsi.tools.sql import sql_tool
from synapsi.tools.web import fetch_url_tool, search_backend, web_search_tool
from synapsi.workflows.context import Budget


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    spec: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float | None = None
    retries: int = 2
    max_concurrency: int = 8
    requests_per_minute: float | None = None
    cache: bool | str = False
    price_per_mtok: tuple[float, float] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(_Strict):
    role: str
    name: str | None = None
    model: str = "default"
    instructions: str = ""
    tools: list[str] = Field(default_factory=list)
    level: int = 1
    temperature: float = 0.7
    max_tokens: int | None = None


class JudgeConfig(_Strict):
    kind: Literal["model", "structural"] = "model"
    model: str | None = None
    blind: bool = True
    show_votes: bool = False
    instructions: str = ""


class SynthesizerConfig(_Strict):
    model: str | None = None


class WorkflowConfig(_Strict):
    strategy: str = "independent_panel"
    mode: str = "balanced"
    seed: int | None = None
    rounds: int | None = None
    max_challenges_per_agent: int | None = None
    verify_evidence: bool | None = None
    blind_review: bool | None = None
    randomize_order: bool | None = None
    early_stop: bool | None = None
    max_tool_rounds: int | None = None


class MemoryConfig(_Strict):
    enabled: bool = False
    path: str = ".synapsi/memory.jsonl"


class SynapSIConfig(_Strict):
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    agents: list[AgentConfig] = Field(min_length=1)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    trace_dir: str | None = None


def load_config(path: str | Path) -> SynapSIConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("YAML config requires: pip install 'synapsi[yaml]'") from exc
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        data = tomllib.loads(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"unsupported config format {suffix!r}; use .yaml, .toml, or .json")
    try:
        return SynapSIConfig.model_validate(data or {})
    except Exception as exc:
        raise ConfigError(f"invalid config {p}: {exc}") from exc


class _Resolver:
    def __init__(self, cfg: SynapSIConfig, base_dir: Path, model_override: str | None):
        self.cfg = cfg
        self.base_dir = base_dir
        self.override = model_override
        self._providers: dict[str, ModelProvider] = {}
        self._tools: dict[str, Tool] = {}
        self.pricing = PricingTable()

    def provider(self, ref: str) -> ModelProvider:
        if self.override:
            ref = self.override
        if ref in self._providers:
            return self._providers[ref]
        mc = self.cfg.models.get(ref)
        if mc is None:
            if ref == "default":
                raise ConfigError("agent uses model 'default' but no 'default' model is defined")
            mc = ModelConfig(spec=ref)
        kwargs = dict(mc.options)
        for key in ("base_url", "api_key_env", "timeout"):
            if (value := getattr(mc, key)) is not None:
                kwargs[key] = value
        raw = create_provider(mc.spec, **kwargs)
        cache = mc.cache
        if isinstance(cache, str):
            cache = str(self.base_dir / cache)
        provider = managed(
            raw,
            retries=mc.retries,
            max_concurrency=mc.max_concurrency,
            requests_per_minute=mc.requests_per_minute,
            cache=cache,
        )
        if mc.price_per_mtok is not None:
            self.pricing.set(provider.id, mc.price_per_mtok)
        self._providers[ref] = provider
        return provider

    def tool(self, name: str) -> Tool:
        if name in self._tools:
            return self._tools[name]
        opts = self.cfg.tools.get(name, {})
        tool: Tool
        if name == "calculator":
            tool = calculator_tool()
        elif name == "fetch_url":
            tool = fetch_url_tool(allow_private=bool(opts.get("allow_private", False)))
        elif name == "web_search":
            if "backend" not in opts:
                raise ConfigError("tools.web_search.backend must be 'tavily' or 'brave'")
            tool = web_search_tool(search_backend(opts["backend"]), k=int(opts.get("k", 4)))
        elif name == "document_search":
            store = DocumentStore()
            for path in opts.get("paths", []):
                full = self.base_dir / path
                store.add_directory(full) if full.is_dir() else store.add_file(full)
            if not store.chunks:
                raise ConfigError("tools.document_search.paths produced no documents")
            tool = document_search_tool(store, k=int(opts.get("k", 3)))
        elif name == "sql_query":
            if "database" not in opts:
                raise ConfigError("tools.sql_query.database is required")
            tool = sql_tool(self.base_dir / opts["database"])
        else:
            raise ConfigError(f"unknown tool {name!r}")
        self._tools[name] = tool
        return tool


def build_council(
    cfg: SynapSIConfig,
    *,
    base_dir: str | Path = ".",
    model_override: str | None = None,
    strategy_override: str | None = None,
    mode_override: str | None = None,
) -> Council:
    """Construct a :class:`~synapsi.council.Council` from configuration."""
    resolver = _Resolver(cfg, Path(base_dir), model_override)
    agents: list[Agent] = []
    for i, a in enumerate(cfg.agents):
        provider = resolver.provider(a.model)
        agent = Agent.from_role(
            a.role,
            provider,
            name=a.name or None,
            instructions=a.instructions,
            tools=[resolver.tool(t) for t in a.tools],
            level=a.level,
            temperature=a.temperature,
            max_tokens=a.max_tokens,
        )
        if a.name is None and any(x.name == agent.name for x in agents):
            agent.name = f"{agent.name} {i + 1}"
        agents.append(agent)

    wf = cfg.workflow
    mode = mode_override or wf.mode
    overrides = {
        k: v for k, v in wf.model_dump(exclude={"strategy", "mode"}).items() if v is not None
    }
    settings = mode_settings(mode, **overrides)
    settings.budget = cfg.budget

    judge: Judge | str
    if cfg.judge.kind == "structural":
        judge = "structural"
    else:
        provider = resolver.provider(cfg.judge.model) if cfg.judge.model else agents[0].provider
        judge = Judge(
            provider,
            blind=cfg.judge.blind,
            show_votes=cfg.judge.show_votes,
            instructions=cfg.judge.instructions,
        )
    synth_model = cfg.synthesizer.model
    synthesizer = Synthesizer(resolver.provider(synth_model) if synth_model else None)
    memory = None
    if cfg.memory.enabled:
        memory = JSONLMemoryStore(Path(base_dir) / cfg.memory.path)
    return Council(
        agents,
        strategy=strategy_override or wf.strategy,
        judge=judge,
        synthesizer=synthesizer,
        mode=mode,
        settings=settings,
        pricing=resolver.pricing,
        memory=memory,
        trace_dir=Path(base_dir) / cfg.trace_dir if cfg.trace_dir else None,
    )


DEFAULT_CONFIG = """\
# SynapSI configuration. API keys come from environment variables, never this file.
models:
  default:
    spec: mock            # e.g. openai:gpt-4o-mini, anthropic:<model>, ollama:llama3.1
    # api_key_env: OPENAI_API_KEY
    # price_per_mtok: [0.15, 0.60]   # your own prices (USD per 1M input/output tokens)
  # local:
  #   spec: ollama:llama3.1

agents:
  - role: researcher
    model: default
  - role: statistician
    model: default
    tools: [calculator]
  - role: domain_expert
    model: default
  - role: skeptic
    model: default

judge:
  kind: model             # or: structural (deterministic, no model call)
  model: default
  blind: true
  show_votes: false

synthesizer:
  model: null             # null = structural summary, or a model alias

workflow:
  strategy: debate        # see: synapsi workflows
  mode: balanced          # fast | balanced | deep | custom
  seed: 42

tools: {}
  # web_search: {backend: tavily}     # needs TAVILY_API_KEY
  # document_search: {paths: [docs/]}
  # sql_query: {database: data.db}

budget:
  max_model_calls: 200
  # max_cost_usd: 1.0

memory:
  enabled: false

trace_dir: runs/traces
"""
