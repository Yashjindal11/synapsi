# Contributing to SynapSI

Thanks for helping. SynapSI is research infrastructure, so two things matter as
much as code quality: claims must be backed by experiments, and results must
be reproducible.

## Setup

```bash
git clone https://github.com/Yashjindal11/synapsi && cd synapsi
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11+
pip install -e ".[dev]"
pre-commit install
scripts/check.sh            # format, lint, mypy --strict, tests
```

Web dashboard:

```bash
cd web/frontend && npm install && npm run dev   # proxies /api to :8000
synapsi serve                                   # in another terminal
```

No API keys are needed for development: tests and examples use the mock
provider, and `synapsi.testing.scripted` lets you script model replies per task.

## Where things live

| Path | Purpose |
| --- | --- |
| `src/synapsi/providers/` | Model adapters and wrappers (retry, rate limit, cache) |
| `src/synapsi/agents/` | `Agent`, roles, prompt construction, output drafts |
| `src/synapsi/deliberation/` | Protocol objects, steps, disagreement engine |
| `src/synapsi/claims/`, `evidence/` | Claim graph and evidence pool |
| `src/synapsi/judgment/`, `synthesis/` | Judges and synthesizer |
| `src/synapsi/workflows/` | Workflow engine and run context |
| `src/synapsi/strategies.py` | Built-in strategies (compositions of steps) |
| `src/synapsi/experiments/` | Experiment engine, suites, metrics |
| `src/synapsi/server/`, `web/frontend/` | API and dashboard |

## Extending without forking

- **Provider**: subclass `ModelProvider`, then `register_provider("name", factory)`.
- **Role**: `register_role(RoleSpec(...))`.
- **Agent behaviour**: subclass `Agent` and override `analyze`, `challenge`,
  `review`, `respond`, or `revise` (see `examples/09_custom_agent.py`).
- **Tool**: `Tool(name, description, function)` or the `@tool` decorator.
- **Step / strategy**: subclass `Step`; compose a `Workflow`; optionally
  `register_strategy("name", builder)`.
- **Judge / synthesizer**: any object with `async judge(ctx)` /
  `async synthesize(ctx)`.
- **Scorer**: any callable `(problem, answer) -> bool`.

If you need to change core code to add one of these, that is a bug in the
extension points — please open an issue.

## Rules of the road

- Conventional Commits (`feat(scope): ...`, `fix: ...`, `docs: ...`, `test: ...`).
- Keep commits focused; each should leave tests passing.
- Changing a prompt changes results. Bump `PROMPT_VERSION` in
  `src/synapsi/agents/prompts.py` and say so in the changelog.
- Never commit secrets, `.env`, run outputs with private data, or real
  customer datasets. Example data must be synthetic or openly licensed.
- Do not add benchmark numbers to docs unless they come from a saved
  `synapsi evaluate`/`benchmark` run whose config is included.

## Releasing (maintainers)

1. Update `src/synapsi/_version.py` and move `Unreleased` notes in
   `CHANGELOG.md` under the new version heading.
2. `scripts/check.sh`, `python -m build`, and a clean-venv install of the wheel.
3. Commit `chore: release vX.Y.Z`, tag `vX.Y.Z`, push the tag. The release
   workflow builds artifacts and creates the GitHub release from the changelog.
