# Running Experiments

The experiment engine answers one question per run: *on these problems, with
these agents and models, does strategy B do better than strategy A — and at
what cost?*

## Design for a fair comparison

1. **Same problems, same seeds.** Each `(problem, repeat)` gets a seed derived
   from the experiment seed, shared across strategies.
2. **Same agents.** Named strategies are built from one `agents` factory. Use a
   mapping of councils only when the agents themselves are the variable.
3. **Always include baselines.** At least `single_model` and `majority_vote`.
   The first strategy listed is the default baseline for paired tests.
4. **Repeat.** Sampling noise is large; use `repeats >= 3` with non-zero
   temperature, and report confidence intervals.
5. **Count abstentions.** Inconclusive verdicts are wrong for accuracy but are
   reported separately as coverage; compare selective accuracy too.
6. **Price your models** (`price_per_mtok`) so cost comparisons are real.

## Metrics

| Metric | Meaning |
| --- | --- |
| accuracy, 95% CI | Wilson interval over trials |
| coverage | share of trials with a committed answer |
| selective accuracy | accuracy over answered trials |
| Brier, ECE | calibration of the final confidence |
| initial majority accuracy | accuracy of the independent first-pass plurality |
| R→W, W→R | trials where deliberation broke / fixed the initial majority |
| disagreement→error AUROC | does initial disagreement predict a wrong answer? |
| calls, tokens, cost, latency | means per trial |
| McNemar p | exact paired test vs. the baseline |

## Outputs

`result.save(dir)` writes `records.jsonl` (one row per trial),
`summary.json` (config, summaries, comparisons, environment, SynapSI and
prompt versions), and `report.md`. With `save_runs=...` every full run result
is saved too, so any single trial can be opened in the dashboard.

## Problem files

```jsonl
{"id": "q1", "question": "...", "options": ["A", "B"], "answer": "A"}
{"id": "q2", "question": "...", "answer": "1250", "facts": ["..."]}
```

Scoring defaults to exact match for multiple choice and numeric comparison for
numeric gold answers; pass `scorer=` for anything else.

## Suites

`synapsi.experiments.suites` generates problems with computed answers:

| Suite | Tests |
| --- | --- |
| `arithmetic` | multi-step arithmetic |
| `ordering` | transitive reasoning with shuffled premises |
| `base_rates` | Bayesian base-rate reasoning (a known intuitive trap) |
| `aviation_delays` | reading synthetic operational data supplied as user facts |

They are instruments for comparing strategies, not general benchmarks, and
they are cheap enough to run many repeats on local models.
