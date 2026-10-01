import { useMemo, useState } from "react";
import type { Claim, Finding, Result } from "../types";
import { ClaimGraph } from "./ClaimGraph";

const TABS = ["Overview", "Agents", "Claim graph", "Claims", "Evidence", "Debate", "Disagreement", "Judge", "Metadata"] as const;
type Tab = (typeof TABS)[number];

function Status({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{status.replace("_", " ")}</span>;
}

function Refs({ ids }: { ids: string[] }) {
  if (!ids.length) return null;
  return <span className="refs">{ids.join(", ")}</span>;
}

function Findings({ title, items, tone }: { title: string; items: Finding[]; tone: string }) {
  return (
    <div className={`bucket ${tone}`}>
      <h3>
        {title} <span className="count">{items.length}</span>
      </h3>
      <ul>
        {items.map((f, i) => (
          <li key={i}>
            {f.statement} <Refs ids={[...f.claim_ids, ...f.evidence_ids]} />
            {f.note && <div className="muted small">{f.note}</div>}
          </li>
        ))}
        {items.length === 0 && <li className="muted">none</li>}
      </ul>
    </div>
  );
}

export function RunView({ result }: { result: Result }) {
  const [tab, setTab] = useState<Tab>("Overview");
  const claims = useMemo(() => new Map(result.claims.map((c) => [c.id, c])), [result]);
  const { judgment: j, synthesis: s, metadata: m } = result;

  return (
    <article>
      <header className="run-header">
        <h2>{result.problem.question}</h2>
        <div className="chips">
          <span className="chip">{m.strategy}</span>
          <span className="chip">{m.mode}</span>
          {j && <span className={`chip verdict ${j.verdict}`}>{j.verdict}</span>}
          {j?.answer && <span className="chip answer">answer: {j.answer}</span>}
          {j && <span className="chip">confidence {j.confidence.toFixed(2)}</span>}
          {j && <span className="chip">evidence {j.evidence_strength}</span>}
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>

      {tab === "Overview" && (
        <section>
          {result.problem.options && <p className="muted">Options: {result.problem.options.join(" · ")}</p>}
          {s && (
            <>
              <p className="summary">{s.summary}</p>
              <div className="buckets">
                <Findings title="Established" items={s.established} tone="good" />
                <Findings title="Probable" items={s.probable} tone="ok" />
                <Findings title="Disputed" items={s.disputed} tone="bad" />
                <Findings title="Unknown" items={s.unknown} tone="neutral" />
              </div>
              {s.recommendations.length > 0 && (
                <>
                  <h3>To resolve remaining uncertainty</h3>
                  <ul>
                    {s.recommendations.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </section>
      )}

      {tab === "Agents" && (
        <section className="grid">
          {result.perspectives.map((p, i) => (
            <div key={i} className={`card ${p.stance}`}>
              <div className="card-head">
                <b>{p.agent}</b>
                <span className="muted small">
                  {p.role} · {p.model}
                </span>
              </div>
              <div className="small">
                {p.independent ? "independent" : `round ${p.round}`}
                {p.stance === "adversarial" && " · assigned adversarial position"}
              </div>
              <p>
                answer: <b>{p.answer ?? "—"}</b> · confidence {p.confidence.toFixed(2)}
              </p>
              <p>{p.position}</p>
              {p.reasoning_summary && <p className="muted">{p.reasoning_summary}</p>}
              <ul className="small">
                {p.claim_ids.map((id) => {
                  const c = claims.get(id);
                  return c ? (
                    <li key={id}>
                      <b>{id}</b> <Status status={c.status} /> {c.statement}
                    </li>
                  ) : null;
                })}
              </ul>
              {p.open_questions.length > 0 && (
                <p className="small muted">Open questions: {p.open_questions.join("; ")}</p>
              )}
            </div>
          ))}
        </section>
      )}

      {tab === "Claim graph" && <ClaimGraph result={result} />}

      {tab === "Claims" && <ClaimTable claims={result.claims} />}

      {tab === "Evidence" && (
        <section>
          <p className="muted small">
            Only tool, document, database and user-provided items are external evidence. Model
            recollections are labelled <code>model_knowledge</code> and never count as support.
          </p>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Kind</th>
                <th>Content</th>
                <th>Source</th>
                <th>Produced by</th>
              </tr>
            </thead>
            <tbody>
              {result.evidence.map((e) => {
                const src = e.provenance.source_id ?? e.provenance.claimed_source ?? "";
                const safe = /^https?:\/\//.test(src);
                return (
                  <tr key={e.id} className={e.provenance.source_kind === "model_knowledge" ? "dim" : ""}>
                    <td>{e.id}</td>
                    <td>
                      <span className="badge">{e.provenance.source_kind}</span>
                    </td>
                    <td>{e.content}</td>
                    <td className="small">
                      {safe ? (
                        <a href={src} target="_blank" rel="noopener noreferrer nofollow">
                          {src}
                        </a>
                      ) : (
                        src
                      )}
                    </td>
                    <td className="small">{e.provenance.produced_by ?? ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {tab === "Debate" && (
        <section>
          {result.challenges.length === 0 && <p className="muted">No challenges were raised.</p>}
          {result.challenges.map((ch) => {
            const target = claims.get(ch.target_claim_id);
            const replies = result.rebuttals.filter((r) => r.challenge_id === ch.id);
            return (
              <div key={ch.id} className="thread">
                <div className="claim-ref">
                  <b>{ch.target_claim_id}</b> ({ch.target_agent}) {target?.statement}
                </div>
                <div className="challenge">
                  <b>{ch.id}</b> {ch.challenger} · {ch.kind} · <span className={`badge ${ch.status}`}>{ch.status}</span>
                  <p>{ch.problem}</p>
                  {ch.question && <p className="muted">Question: {ch.question}</p>}
                  <Refs ids={ch.evidence_ids} />
                </div>
                {replies.map((r) => (
                  <div key={r.id} className="rebuttal">
                    <b>{r.id}</b> {r.agent} · {r.response_type}
                    <p>{r.response}</p>
                    <Refs ids={r.evidence_ids} />
                  </div>
                ))}
              </div>
            );
          })}
          {result.position_changes.length > 0 && (
            <>
              <h3>Position changes</h3>
              <ul>
                {result.position_changes.map((pc, i) => (
                  <li key={i}>
                    {pc.agent} (round {pc.round}): {pc.from_answer ?? "—"} → {pc.to_answer ?? "—"}{" "}
                    {!pc.cited_new_evidence && !pc.after_concession && (
                      <span className="badge uncertain">no new evidence</span>
                    )}
                    {pc.reason && <div className="muted small">{pc.reason}</div>}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}

      {tab === "Disagreement" && (
        <section>
          {result.disagreements.length === 0 && <p className="muted">No disagreements detected.</p>}
          {result.disagreements.map((d) => (
            <div key={d.id} className={`card ${d.status}`}>
              <div className="card-head">
                <b>
                  {d.id} · {d.kind}
                </b>
                <span className={`badge ${d.status === "unresolved" ? "contradicted" : "supported"}`}>{d.status}</span>
              </div>
              <p>{d.topic}</p>
              <div className="sides">
                {d.sides.map((side, i) => (
                  <div key={i} className="side">
                    <b>{side.position}</b>
                    <div className="small">agents: {side.agents.join(", ") || "—"}</div>
                    <div className="small">external evidence: {side.external_evidence_count}</div>
                    <Refs ids={[...side.claim_ids, ...side.evidence_ids]} />
                  </div>
                ))}
              </div>
              <p className="muted small">{d.reason}</p>
            </div>
          ))}
        </section>
      )}

      {tab === "Judge" && j && (
        <section>
          <p>
            <b>{j.decision}</b>
          </p>
          <p>{j.reasoning_summary}</p>
          <p className="muted small">
            judge {j.judge} · method {j.method} · verdict {j.verdict} · evidence {j.evidence_strength}
          </p>
          <h3>Supporting claims</h3>
          <ClaimTable claims={j.supporting_claim_ids.map((id) => claims.get(id)).filter(Boolean) as Claim[]} />
          {j.contradicting_claim_ids.length > 0 && (
            <>
              <h3>Contradicting claims</h3>
              <ClaimTable claims={j.contradicting_claim_ids.map((id) => claims.get(id)).filter(Boolean) as Claim[]} />
            </>
          )}
          {j.minority_positions.length > 0 && (
            <>
              <h3>Minority positions</h3>
              <ul>{j.minority_positions.map((p, i) => <li key={i}>{p}</li>)}</ul>
            </>
          )}
          {j.unresolved_issues.length > 0 && (
            <>
              <h3>Unresolved</h3>
              <ul>{j.unresolved_issues.map((p, i) => <li key={i}>{p}</li>)}</ul>
            </>
          )}
        </section>
      )}

      {tab === "Metadata" && (
        <section className="meta">
          <p>
            run <code>{result.run_id}</code> · SynapSI {m.synapsi_version} · prompts v{m.prompt_version} ·{" "}
            {m.latency_s.toFixed(2)}s
          </p>
          {m.stopped_reason && <p className="warn">Stopped early: {m.stopped_reason}</p>}
          <h3>Usage by agent</h3>
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Calls</th>
                <th>Tokens</th>
                <th>Cost</th>
                <th>Model time</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(m.usage.by_agent).map(([name, u]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{u.calls}</td>
                  <td>{u.prompt_tokens + u.completion_tokens}</td>
                  <td>{u.cost_usd == null ? "unknown" : `$${u.cost_usd.toFixed(4)}`}</td>
                  <td>{u.latency_s.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Agents</h3>
          <ul>
            {m.agents.map((a) => (
              <li key={a.name}>
                {a.name} · {a.role} · {a.model} · level {a.level}
                {a.tools.length > 0 && ` · tools: ${a.tools.join(", ")}`}
              </li>
            ))}
          </ul>
          <h3>Uncertainty signals</h3>
          <pre>{JSON.stringify(result.uncertainty, null, 2)}</pre>
          <h3>Settings</h3>
          <pre>{JSON.stringify(m.settings, null, 2)}</pre>
          {m.errors.length > 0 && (
            <>
              <h3>Errors</h3>
              <ul>{m.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </>
          )}
          {m.warnings.length > 0 && (
            <>
              <h3>Warnings</h3>
              <ul>{m.warnings.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </>
          )}
        </section>
      )}
    </article>
  );
}

function ClaimTable({ claims }: { claims: Claim[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Agent</th>
          <th>Type</th>
          <th>Status</th>
          <th>Statement</th>
          <th>Evidence</th>
        </tr>
      </thead>
      <tbody>
        {claims.map((c) => (
          <tr key={c.id} className={c.withdrawn ? "dim" : ""}>
            <td>{c.id}</td>
            <td>{c.agent}</td>
            <td className="small">{c.type}</td>
            <td>
              <Status status={c.status} />
            </td>
            <td>
              {c.statement}
              {c.withdrawn && <span className="muted small"> (withdrawn)</span>}
              {c.status_reason && <div className="muted small">{c.status_reason}</div>}
            </td>
            <td className="small">{c.evidence_ids.join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
