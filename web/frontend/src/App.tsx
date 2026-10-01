import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { RunForm } from "./components/RunForm";
import { RunView } from "./components/RunView";
import type { Meta, Result, RunEvent, RunSummary } from "./types";

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [offline, setOffline] = useState(false);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [live, setLive] = useState<{ id: string; events: RunEvent[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.runs().then(setRuns).catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setOffline(true));
    refresh();
  }, [refresh]);

  const open = async (id: string) => {
    setError(null);
    try {
      const body = await api.run(id);
      if (body.result) setResult(body.result);
      else if (body.status === "running") follow(id);
      else setError(body.error ?? `run ${body.status}`);
    } catch (e) {
      setError(String(e));
    }
  };

  const follow = (id: string) => {
    setResult(null);
    setLive({ id, events: [] });
    api.stream(
      id,
      (event) => setLive((prev) => (prev && prev.id === id ? { id, events: [...prev.events, event] } : prev)),
      async (status) => {
        refresh();
        if (status === "completed") {
          const body = await api.run(id);
          if (body.result) setResult(body.result);
          setLive(null);
        } else {
          setError(`run ended: ${status}`);
        }
      },
    );
  };

  const loadFile = async (file: File) => {
    try {
      setResult(JSON.parse(await file.text()) as Result);
      setLive(null);
      setError(null);
    } catch {
      setError("not a valid SynapSI result JSON file");
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">
          Synap<span>SI</span>
        </h1>
        <p className="tagline">Many minds. One intelligence.</p>
        {offline ? (
          <p className="muted">
            API not reachable. Start <code>synapsi serve</code>, or open a saved result file below.
          </p>
        ) : (
          <RunForm
            meta={meta}
            onStarted={(id) => {
              refresh();
              follow(id);
            }}
            onError={setError}
          />
        )}
        <label className="file">
          Open result JSON
          <input type="file" accept="application/json" onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])} />
        </label>
        <h2>Runs</h2>
        <ul className="runs">
          {runs.map((r) => (
            <li key={r.run_id}>
              <button onClick={() => open(r.run_id)} title={r.run_id}>
                <span className="q">{r.question}</span>
                <span className="muted">
                  {r.strategy ?? "default"} · {r.status === "completed" ? `${r.verdict ?? ""} ${r.answer ?? ""}` : r.status}
                </span>
              </button>
            </li>
          ))}
          {runs.length === 0 && <li className="muted">No runs yet.</li>}
        </ul>
      </aside>
      <main className="main">
        {error && <div className="error">{error}</div>}
        {live && (
          <section className="card">
            <h2>Running…</h2>
            <ol className="events">
              {live.events.map((e, i) => (
                <li key={i}>
                  <code>{e.type}</code> {e.step && <span className="muted">{e.step}</span>}{" "}
                  {e.agent && <b>{e.agent}</b>}{" "}
                  {"answer" in e.data && e.data.answer != null && <span>→ {String(e.data.answer)}</span>}
                </li>
              ))}
            </ol>
          </section>
        )}
        {result ? (
          <RunView result={result} />
        ) : (
          !live && (
            <section className="empty">
              <h2>Structured deliberation, inspectable end to end</h2>
              <p>
                Start a run or open a saved result to see independent perspectives, the claim graph,
                evidence with provenance, challenges, disagreements, and the judge&apos;s assessment.
              </p>
            </section>
          )
        )}
      </main>
    </div>
  );
}
