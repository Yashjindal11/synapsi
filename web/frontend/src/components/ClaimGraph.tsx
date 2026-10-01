import { useMemo, useState } from "react";
import type { Claim, Evidence, Result } from "../types";

const COL_W = 230;
const GAP_X = 70;
const NODE_H = 58;
const GAP_Y = 16;
const HEADER = 44;

const STATUS_FILL: Record<string, string> = {
  supported: "#dff3e7",
  partially_supported: "#eef6d8",
  contradicted: "#fde3e0",
  uncertain: "#fff1d6",
  unverified: "#eef0f4",
  unsupported: "#f6f6f8",
};

const EDGE_COLOR: Record<string, string> = {
  supports: "#1f8a4c",
  contradicts: "#c0392b",
  refines: "#8792a2",
  challenge: "#d68910",
};

interface Box {
  id: string;
  x: number;
  y: number;
  kind: "claim" | "evidence" | "agent";
}

interface Edge {
  from: string;
  to: string;
  kind: keyof typeof EDGE_COLOR;
}

function clip(text: string, n: number) {
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}

export function ClaimGraph({ result }: { result: Result }) {
  const [showModelKnowledge, setShowModelKnowledge] = useState(false);
  const [showWithdrawn, setShowWithdrawn] = useState(false);
  const [showChallenges, setShowChallenges] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);

  const layout = useMemo(() => {
    const claims = result.claims.filter((c) => showWithdrawn || !c.withdrawn);
    const claimIds = new Set(claims.map((c) => c.id));
    const evidence = result.evidence.filter(
      (e) => showModelKnowledge || e.provenance.source_kind !== "model_knowledge",
    );
    const agents: string[] = [];
    for (const c of claims) if (!agents.includes(c.agent)) agents.push(c.agent);

    const boxes = new Map<string, Box>();
    const evX = 20;
    evidence.forEach((e, i) => boxes.set(e.id, { id: e.id, x: evX, y: HEADER + i * (NODE_H + GAP_Y), kind: "evidence" }));
    const offset = evidence.length ? COL_W + GAP_X : 0;
    agents.forEach((agent, col) => {
      const x = 20 + offset + col * (COL_W + GAP_X);
      boxes.set(`agent:${agent}`, { id: `agent:${agent}`, x, y: 4, kind: "agent" });
      claims
        .filter((c) => c.agent === agent)
        .forEach((c, row) => boxes.set(c.id, { id: c.id, x, y: HEADER + row * (NODE_H + GAP_Y), kind: "claim" }));
    });

    const edges: Edge[] = [];
    const seen = new Set<string>();
    const add = (from: string, to: string, kind: Edge["kind"]) => {
      const key = `${from}|${to}|${kind}`;
      if (boxes.has(from) && boxes.has(to) && !seen.has(key)) {
        seen.add(key);
        edges.push({ from, to, kind });
      }
    };
    for (const c of claims) {
      c.evidence_ids.forEach((e) => add(e, c.id, "supports"));
      c.contradicting_evidence_ids.forEach((e) => add(e, c.id, "contradicts"));
    }
    for (const e of evidence) {
      e.supports.forEach((c) => add(e.id, c, "supports"));
      e.contradicts.forEach((c) => add(e.id, c, "contradicts"));
    }
    for (const r of result.claim_relations) if (claimIds.has(r.source) && claimIds.has(r.target)) add(r.source, r.target, r.kind);
    if (showChallenges) for (const ch of result.challenges) add(`agent:${ch.challenger}`, ch.target_claim_id, "challenge");

    const width = 40 + offset + Math.max(agents.length, 1) * (COL_W + GAP_X);
    const rows = Math.max(evidence.length, ...agents.map((a) => claims.filter((c) => c.agent === a).length), 1);
    const height = HEADER + rows * (NODE_H + GAP_Y) + 20;
    return { boxes, edges, claims, evidence, width, height };
  }, [result, showModelKnowledge, showWithdrawn, showChallenges]);

  const claimById = useMemo(() => new Map(result.claims.map((c) => [c.id, c])), [result]);
  const evidenceById = useMemo(() => new Map(result.evidence.map((e) => [e.id, e])), [result]);

  const anchor = (box: Box, side: "left" | "right") => ({
    x: box.x + (side === "right" ? COL_W : 0),
    y: box.y + (box.kind === "agent" ? 16 : NODE_H / 2),
  });

  const path = (edge: Edge) => {
    const a = layout.boxes.get(edge.from)!;
    const b = layout.boxes.get(edge.to)!;
    const forward = a.x < b.x;
    const sameCol = a.x === b.x;
    const p1 = anchor(a, forward || sameCol ? "right" : "left");
    const p2 = anchor(b, sameCol ? "right" : forward ? "left" : "right");
    const bend = sameCol ? 60 : Math.max(40, Math.abs(p2.x - p1.x) / 2);
    const c1x = p1.x + (forward || sameCol ? bend : -bend);
    const c2x = p2.x + (sameCol ? bend : forward ? -bend : bend);
    return `M${p1.x},${p1.y} C${c1x},${p1.y} ${c2x},${p2.y} ${p2.x},${p2.y}`;
  };

  const related = (id: string) =>
    layout.edges.some((e) => (e.from === selected && e.to === id) || (e.to === selected && e.from === id));

  return (
    <section>
      <div className="graph-controls small">
        <label>
          <input type="checkbox" checked={showChallenges} onChange={(e) => setShowChallenges(e.target.checked)} /> challenges
        </label>
        <label>
          <input type="checkbox" checked={showModelKnowledge} onChange={(e) => setShowModelKnowledge(e.target.checked)} /> model-knowledge items
        </label>
        <label>
          <input type="checkbox" checked={showWithdrawn} onChange={(e) => setShowWithdrawn(e.target.checked)} /> withdrawn claims
        </label>
        <span className="legend">
          {Object.entries(EDGE_COLOR).map(([k, c]) => (
            <span key={k}>
              <i style={{ background: c }} /> {k}
            </span>
          ))}
        </span>
      </div>
      <div className="graph-scroll">
        <svg width={layout.width} height={layout.height} role="img" aria-label="Claim graph">
          <defs>
            {Object.entries(EDGE_COLOR).map(([k, c]) => (
              <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill={c} />
              </marker>
            ))}
          </defs>
          {layout.edges.map((e, i) => {
            const active = !selected || e.from === selected || e.to === selected;
            return (
              <path
                key={i}
                d={path(e)}
                fill="none"
                stroke={EDGE_COLOR[e.kind]}
                strokeWidth={active && selected ? 2.2 : 1.4}
                strokeDasharray={e.kind === "challenge" ? "3 4" : e.kind === "refines" ? "6 4" : undefined}
                opacity={active ? 0.9 : 0.12}
                markerEnd={`url(#arrow-${e.kind})`}
              />
            );
          })}
          {[...layout.boxes.values()].map((box) => {
            const dim = selected && selected !== box.id && !related(box.id) ? 0.3 : 1;
            if (box.kind === "agent") {
              const name = box.id.slice(6);
              return (
                <g key={box.id} opacity={dim}>
                  <text x={box.x} y={box.y + 20} className="graph-agent">
                    {clip(name, 28)}
                  </text>
                </g>
              );
            }
            if (box.kind === "evidence") {
              const ev = evidenceById.get(box.id)!;
              return (
                <g key={box.id} opacity={dim} onClick={() => setSelected(selected === box.id ? null : box.id)} className="node">
                  <rect x={box.x} y={box.y} width={COL_W} height={NODE_H} rx={8} fill="#eef3ff" stroke="#9fb3e8" />
                  <text x={box.x + 8} y={box.y + 17} className="graph-id">
                    {ev.id} · {ev.provenance.source_kind}
                  </text>
                  <text x={box.x + 8} y={box.y + 36} className="graph-text">
                    {clip(ev.content, 34)}
                  </text>
                  <text x={box.x + 8} y={box.y + 51} className="graph-sub">
                    {clip(ev.provenance.source_id ?? ev.provenance.tool ?? "", 36)}
                  </text>
                </g>
              );
            }
            const c = claimById.get(box.id)!;
            return (
              <g key={box.id} opacity={dim} onClick={() => setSelected(selected === box.id ? null : box.id)} className="node">
                <rect
                  x={box.x}
                  y={box.y}
                  width={COL_W}
                  height={NODE_H}
                  rx={8}
                  fill={STATUS_FILL[c.status] ?? "#fff"}
                  stroke={selected === c.id ? "#4b4bd8" : "#c7cdd8"}
                  strokeWidth={selected === c.id ? 2 : 1}
                  strokeDasharray={c.withdrawn ? "4 3" : undefined}
                />
                <text x={box.x + 8} y={box.y + 17} className="graph-id">
                  {c.id} · {c.status.replace("_", " ")} · {c.confidence.toFixed(2)}
                </text>
                <text x={box.x + 8} y={box.y + 36} className="graph-text">
                  {clip(c.statement, 34)}
                </text>
                <text x={box.x + 8} y={box.y + 51} className="graph-sub">
                  {c.type}
                  {c.withdrawn ? " · withdrawn" : ""}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {selected && <Detail id={selected} claim={claimById.get(selected)} evidence={evidenceById.get(selected)} result={result} />}
    </section>
  );
}

function Detail({ id, claim, evidence, result }: { id: string; claim?: Claim; evidence?: Evidence; result: Result }) {
  if (evidence) {
    const p = evidence.provenance;
    return (
      <div className="card detail">
        <b>{id}</b> <span className="badge">{p.source_kind}</span>
        <p>{evidence.content}</p>
        <p className="small muted">
          {[p.source_id, p.title, p.tool && `tool ${p.tool}`, p.query && `query “${p.query}”`, p.produced_by && `by ${p.produced_by}`, p.claimed_source && `claimed source: ${p.claimed_source}`]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
    );
  }
  if (!claim) return null;
  const challenges = result.challenges.filter((c) => c.target_claim_id === id);
  return (
    <div className="card detail">
      <b>{id}</b> by {claim.agent} <span className={`badge ${claim.status}`}>{claim.status}</span>
      <p>{claim.statement}</p>
      <p className="small muted">
        {claim.type} · confidence {claim.confidence.toFixed(2)} · {claim.status_reason}
        {claim.revised_from && ` · revises ${claim.revised_from}`}
      </p>
      {challenges.map((ch) => (
        <div key={ch.id} className="challenge small">
          <b>{ch.id}</b> {ch.challenger} [{ch.kind}, {ch.status}]: {ch.problem}
        </div>
      ))}
    </div>
  );
}
