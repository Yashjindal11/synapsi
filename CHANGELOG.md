# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-01

First public milestone: the full deliberation pipeline, baselines, and the
experiment engine. No benchmark results are claimed in this release.

### Added
- Deliberation protocol: perspectives, claims, evidence with provenance,
  challenges, rebuttals, revisions, disagreements, judgments, synthesis.
- Claim graph with deterministic, evidence-based status rules; agreement never
  creates support, and judges cannot mark claims supported without external
  evidence.
- Providers: OpenAI, Anthropic, Gemini, Ollama, Hugging Face, any
  OpenAI-compatible server, and a deterministic mock; retry, rate limiting,
  caching, opt-in pricing.
- Agents with 12 built-in roles, custom roles, and overridable protocol tasks.
- Workflow engine (sequential, parallel, loop, conditional, function steps,
  stop conditions, budgets) and ten built-in strategies.
- Model judge (blind, vote counts hidden), structural judge, majority-vote and
  single-model baselines; status-driven synthesis.
- Tools: calculator, document search, read-only SQL, web search (Tavily,
  Brave, static), guarded URL fetching; evidence collection and claim
  verification steps.
- Optional long-term memory, 

### Security
- API keys only from environment variables; config files reject unknown keys.
- Safe AST calculator, read-only SQL, `fetch_url` refuses private addresses,
  HTML output escaped, optional bearer token for the API.

### Known limitations
- Evidence links (which evidence supports which claim) are produced by models
  and can be wrong; claim verification reduces but does not remove this.
- The structural judge does not check that claims entail the answer they are
  attached to.
- Tool use is JSON-protocol based, not native provider tool calling.
- No published experiment results yet; the synthetic suites are small
  instruments, not general benchmarks.
- The web API is designed for local use (no user accounts).

### Migration
- First release; nothing to migrate. Prompt protocol version: 2.

[Unreleased]: https://github.com/Yashjindal11/synapsi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Yashjindal11/synapsi/releases/tag/v0.1.0events and JSONL traces, usage and cost tracking.
- YAML/TOML/JSON configuration, CLI, REST/WebSocket API, React dashboard with
  claim-graph visualisation.
- Experiment engine with synthetic suites, accuracy CIs, coverage,
  calibration, flip analysis, and exact McNemar comparisons.
- Thirteen runnable examples.
