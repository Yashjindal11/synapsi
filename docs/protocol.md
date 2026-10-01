# Deliberation Protocol

Agents never exchange free-form transcripts. Every message is a validated
Pydantic object. Identifiers are short and sequential (`C3`, `E2`, `X1`, `R1`)
so models can reference them reliably.

## Perspective

An agent's position on the problem.

| Field | Meaning |
| --- | --- |
| `position` | One-paragraph stance. |
| `answer` | Normalised answer label when the problem has options. |
| `claim_ids` | Claims the agent asserts (stored in the claim graph). |
| `assumptions` | Explicit assumptions. |
| `reasoning_summary` | Short, inspectable justification (not chain-of-thought). |
| `uncertainty` | What the agent is unsure about. |
| `confidence` | 0–1 self-reported confidence. |
| `counterarguments` | Strongest arguments against its own position. |
| `open_questions` | Information that would change its mind. |
| `independent` | `True` if formed without seeing other agents' output. |

## Claim

`statement`, `type` (factual, causal, statistical, predictive, normative,
definitional, methodological, other), `agent`, `evidence_ids`,
`contradicting_evidence_ids`, `assumptions`, `confidence`, `status`, `withdrawn`.

Relations between claims (`supports`, `contradicts`, `refines`) are stored as
edges in the `ClaimGraph`.

### Status rules

Status is derived deterministically, then may be adjusted by the judge.

| Condition | Status |
| --- | --- |
| Contradicting evidence, no external support | `contradicted` |
| External support and contradicting evidence, or external support with an open challenge | `partially_supported` |
| External support, no open challenges | `supported` |
| Only model-knowledge support, unchallenged | `unverified` |
| Challenged without external support | `uncertain` |
| No evidence offered | `unsupported` |

The judge may not mark a claim `supported` unless it has at least one piece of
external evidence; such overrides are capped at `partially_supported`.

## Evidence

`content` + `Provenance` (`source_kind`, `source_id`, `title`, `tool`, `query`,
`produced_by`, `retrieved_at`, `locator`, `claimed_source`).

`source_kind` is one of: `user_provided`, `document`, `web`, `database`, `api`,
`calculation`, `experiment`, `tool`, `model_knowledge`.

Only tools, document stores, and the user create non-`model_knowledge`
evidence. If an agent writes "according to source X", that is stored as
`model_knowledge` with `claimed_source="X"` — the framework cannot verify it.

## Challenge and rebuttal

```text
Challenge  X4
  challenger   : Skeptic
  target_claim : C2 (owned by Statistician)
  kind         : evidence
  problem      : The cited data only shows correlation.
  question     : What supports a causal interpretation?
  evidence_ids : [E3]

Rebuttal  R2 -> X4
  response_type : revise        # defend | concede | revise | clarify
  response      : Agreed; the claim is narrowed to association.
  revised_statement : ...
```

Challenge status: `open`, `answered` (defended with evidence or clarified),
`conceded`, `revised`.

## Disagreement

`topic`, `kind` (`answer` or `claim`), `sides` (each with position, agents,
claim ids, evidence ids), `reason`, `status` (`resolved`/`unresolved`), and an
`evidence_balance` summary. Disagreements are reported even when a judge reaches
a decision.

## Judgment

`decision`, `answer`, `verdict` (`decided`, `inconclusive`, `split`),
`supporting_claim_ids`, `contradicting_claim_ids`, `evidence_strength`
(`strong`, `moderate`, `weak`, `insufficient`), `unresolved_issues`,
`confidence`, `reasoning_summary`, `claim_assessments`, `minority_positions`,
`method` (`model`, `structural`, `vote`, `single`).

## Synthesis

`summary`, `answer`, and findings grouped into `established`, `probable`,
`disputed`, `unknown`, plus `recommendations` for further investigation.
