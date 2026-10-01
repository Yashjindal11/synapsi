# Custom Workflows

A workflow is a list of steps run over a shared `RunContext`. Strategies are
just workflows, so anything a built-in strategy does, you can rearrange.

## The pieces

| Building block | Purpose |
| --- | --- |
| `Step` | Subclass and implement `async run(ctx)`. |
| `FunctionStep(fn)` | Wrap a plain (async) function `fn(ctx)`. |
| `Sequential([...])` | Run steps in order. |
| `Parallel([...])` | Run independent steps concurrently. |
| `Loop(body, max_iterations=n, until=pred)` | Repeat with a limit; `ctx.state.round` increments each iteration. |
| `Conditional(pred, then, otherwise)` | Branch on state. |
| `Workflow(steps, stop_when=pred, finalize=[...])` | Top level. `finalize` steps run even after early stops or budget exhaustion. |
| `StopWorkflow(reason)` | Raise from any step to stop early (not an error). |

`synapsi.strategies.finalize(method)` returns the standard model-free tail:
`ClaimAssessment`, `DisagreementAnalysis`, `JudgeStep(method)`,
`SynthesisStep`.

## What a step can use

```python
class MyStep(Step):
    name = "my_step"

    async def run(self, ctx: RunContext) -> None:
        ctx.problem                     # question, options, context (no gold answer)
        ctx.agents, ctx.select(names, level=2)
        ctx.state.perspectives          # latest per agent
        ctx.state.history               # every perspective, incl. revisions/adversarial
        ctx.state.claims                # ClaimGraph
        ctx.state.evidence              # EvidencePool
        ctx.state.challenges, ctx.state.rebuttals, ctx.state.disagreements
        ctx.settings                    # rounds, limits, seed, budget, blind_review...
        ctx.rng                         # seeded random.Random
        ctx.label(agent_name)           # "Analyst 3" when blind review is on
        ctx.ordered(items)              # seeded shuffle to avoid positional bias
        ctx.emit(EventType.WARNING, message="...")
        await ctx.generate(provider=..., schema=MyModel, messages=[...], agent="x", task="y")
```

`ctx.generate` is the only way steps should call models: it enforces the
budget, validates structured output, and records usage and events.

## Example: debate only when needed

```python
class StopIfUnanimous(Step):
    name = "stop_if_unanimous"
    async def run(self, ctx):
        answers = {p.answer for p in ctx.state.perspectives.values()}
        if len(answers) == 1 and None not in answers:
            raise StopWorkflow("independent agents agreed")

workflow = Workflow(
    [
        IndependentAnalysis(),
        StopIfUnanimous(),
        Loop([CrossExamination(), RespondToChallenges(), Revision(see_others=False)],
             max_iterations=3, until=converged),
    ],
    name="debate_if_needed",
    finalize=finalize(),
)
```

Note the trade-off this encodes: unanimous independent answers can still be
wrong (correlated errors). Whether skipping debate in that case loses accuracy
is something to measure with `Experiment`, comparing this workflow against
plain `debate`.

## Registering a strategy

```python
from synapsi import register_strategy
register_strategy("debate_if_needed", lambda: build(), "Debate only on disagreement")
```

It is then available to `Council(strategy="debate_if_needed")`, config files,
the CLI, and the API.

## Independence checklist for new steps

- Does the step show agents other agents' positions? If so, register their
  output with `independent=False`.
- Are reviewers shown identities? Use `ctx.label()` so blind review works.
- Is presentation order fixed? Use `ctx.ordered()`.
- Does the step treat agent statements as evidence? It must not; only tools,
  documents, and the user create external evidence.
