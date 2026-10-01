# Deliberation Strategies

All strategies are ordinary `Workflow` objects built from public steps. They are
starting points for experiments, not recommendations. Which one helps — if any —
is an empirical question for your models and problems.

| Strategy | Steps | Model calls (n agents, r rounds) | What it tests |
| --- | --- | --- | --- |
| `single_model` | analyze (first agent only) | 1 | Baseline. |
| `independent_panel` | analyze ×n → judge → synth | n (+1 judge) | Value of independent samples without interaction. |
| `majority_vote` | analyze ×n → vote | n | Classic self-consistency style aggregation. |
| `peer_review` | analyze → blind review → respond → revise → judge | ≈4n (+1) | Whether critique improves positions. |
| `debate` | analyze → r × (challenge → respond → revise) | ≈n + 3nr (+1) | Iterative argument exchange. |
| `adversarial` | analyze → steelman alternative → r × (challenge opposing side → respond → revise) | ≈n + k + 3nr (+1) | Deliberate pressure on the leading answer. |
| `jury` | collect evidence → analyze → challenge → respond → judge | ≈3n (+1) | Judge-centred decision with no revision. |
| `evidence_first` | collect evidence → analyze → verify claims → review → respond → judge | varies | Whether grounding before reasoning helps. |
| `hierarchical` | level-1 analyze → level-2 review → respond → revise → judge | varies | Separate analysts from reviewers. |
| `sequential_critique` | agent k critiques and improves agent k-1 | ≈2n (+1) | Chain refinement (non-independent by design). |

Judgment uses the configured `Judge` when present, otherwise a deterministic
`StructuralJudge` that prefers the position whose claims carry the most
external, unchallenged evidence — and declares `inconclusive` when nothing
separates them. Vote counts are not shown to the model judge by default.

Debate-style loops stop early when no agent changed position and no challenge
remains open.

## Modes

Modes only set strategy parameters; they never add agents behind your back.

| Mode | Rounds | Challenges per agent | Claim verification |
| --- | --- | --- | --- |
| `fast` | 1 | 2 | off |
| `balanced` | 2 | 3 | off |
| `deep` | 3 | 4 | on |
