## What and why

<!-- What does this change and which problem does it solve? Link issues. -->

## How it was tested

- [ ] `scripts/check.sh` passes (format, lint, mypy, tests)
- [ ] New behaviour has tests (the mock provider or `synapsi.testing.scripted` keeps them offline)
- [ ] Web changes: `npm run build` in `web/frontend`

## Checklist

- [ ] Public behaviour changes are documented (README/docs) and noted in `CHANGELOG.md`
- [ ] No API keys, tokens, `.env` files, or user data are included
- [ ] Prompt changes bump `PROMPT_VERSION` in `src/synapsi/agents/prompts.py`
- [ ] No claim that a strategy "improves" results without experiment output to back it
