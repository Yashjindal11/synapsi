// Mirrors the subset of synapsi.result.SynapSIResult the dashboard renders.

export type ClaimStatus =
  | "supported"
  | "partially_supported"
  | "contradicted"
  | "unsupported"
  | "uncertain"
  | "unverified";

export interface Problem {
  question: string;
  context?: string | null;
  options?: string[] | null;
  facts: string[];
}

export interface Perspective {
  agent: string;
  role: string;
  model: string;
  position: string;
  answer: string | null;
  claim_ids: string[];
  assumptions: string[];
  reasoning_summary: string;
  uncertainty: string;
  confidence: number;
  counterarguments: string[];
  open_questions: string[];
  round: number;
  independent: boolean;
  stance: "own" | "adversarial";
}

export interface Claim {
  id: string;
  statement: string;
  type: string;
  agent: string;
  evidence_ids: string[];
  contradicting_evidence_ids: string[];
  confidence: number;
  status: ClaimStatus;
  status_reason: string;
  round: number;
  withdrawn: boolean;
  revised_from: string | null;
}

export interface ClaimRelation {
  source: string;
  target: string;
  kind: "supports" | "contradicts" | "refines";
  agent: string;
}

export interface Evidence {
  id: string;
  content: string;
  provenance: {
    source_kind: string;
    source_id: string | null;
    title: string | null;
    tool: string | null;
    query: string | null;
    produced_by: string | null;
    claimed_source: string | null;
  };
  supports: string[];
  contradicts: string[];
}

export interface Challenge {
  id: string;
  challenger: string;
  target_claim_id: string;
  target_agent: string;
  kind: string;
  problem: string;
  question: string;
  evidence_ids: string[];
  round: number;
  status: "open" | "answered" | "conceded" | "revised";
}

export interface Rebuttal {
  id: string;
  challenge_id: string;
  agent: string;
  response_type: string;
  response: string;
  evidence_ids: string[];
  revised_claim_id: string | null;
}

export interface Disagreement {
  id: string;
  topic: string;
  kind: "answer" | "claim";
  sides: {
    position: string;
    agents: string[];
    claim_ids: string[];
    evidence_ids: string[];
    external_evidence_count: number;
  }[];
  reason: string;
  status: "resolved" | "unresolved";
  evidence_balance: string;
}

export interface Judgment {
  judge: string;
  method: string;
  decision: string;
  answer: string | null;
  verdict: "decided" | "inconclusive" | "split";
  supporting_claim_ids: string[];
  contradicting_claim_ids: string[];
  evidence_strength: string;
  unresolved_issues: string[];
  confidence: number;
  reasoning_summary: string;
  minority_positions: string[];
}

export interface Finding {
  statement: string;
  claim_ids: string[];
  evidence_ids: string[];
  agents: string[];
  note: string;
}

export interface Synthesis {
  summary: string;
  answer: string | null;
  established: Finding[];
  probable: Finding[];
  disputed: Finding[];
  unknown: Finding[];
  recommendations: string[];
  method: string;
}

export interface Usage {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number | null;
  latency_s: number;
  cached_calls: number;
}

export interface PositionChange {
  agent: string;
  round: number;
  from_answer: string | null;
  to_answer: string | null;
  reason: string;
  cited_new_evidence: boolean;
  after_concession: boolean;
}

export interface Result {
  run_id: string;
  problem: Problem;
  perspectives: Perspective[];
  final_perspectives: Record<string, Perspective>;
  claims: Claim[];
  claim_relations: ClaimRelation[];
  evidence: Evidence[];
  challenges: Challenge[];
  rebuttals: Rebuttal[];
  position_changes: PositionChange[];
  disagreements: Disagreement[];
  judgment: Judgment | null;
  synthesis: Synthesis | null;
  uncertainty: Record<string, unknown>;
  metadata: {
    synapsi_version: string;
    prompt_version: string;
    strategy: string;
    mode: string;
    settings: Record<string, unknown>;
    agents: { name: string; role: string; model: string; tools: string[]; level: number }[];
    judge: Record<string, unknown> | null;
    latency_s: number;
    usage: {
      total: Usage;
      by_agent: Record<string, Usage>;
      by_model: Record<string, Usage>;
      by_task: Record<string, Usage>;
    };
    stopped_reason: string | null;
    errors: string[];
    warnings: string[];
  };
}

export interface RunSummary {
  run_id: string;
  status: "running" | "completed" | "failed";
  question: string;
  strategy: string | null;
  verdict: string | null;
  answer: string | null;
  error: string | null;
}

export interface RunEvent {
  type: string;
  timestamp: string;
  step: string | null;
  agent: string | null;
  data: Record<string, unknown>;
}

export interface Meta {
  strategies: Record<string, string>;
  modes: string[];
  roles: string[];
  council: { agents: { name: string; role: string; model: string }[]; strategy: string };
  auth_required: boolean;
}
