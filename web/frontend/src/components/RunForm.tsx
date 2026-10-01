import { useState } from "react";
import { api } from "../api";
import type { Meta } from "../types";

interface Props {
  meta: Meta | null;
  onStarted: (id: string) => void;
  onError: (message: string) => void;
}

export function RunForm({ meta, onStarted, onError }: Props) {
  const [question, setQuestion] = useState("");
  const [options, setOptions] = useState("");
  const [facts, setFacts] = useState("");
  const [strategy, setStrategy] = useState("");
  const [mode, setMode] = useState("fast");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const opts = options.split(",").map((s) => s.trim()).filter(Boolean);
      const run = await api.start(
        {
          question,
          options: opts.length ? opts : null,
          facts: facts.split("\n").map((s) => s.trim()).filter(Boolean),
          strategy: strategy || null,
          mode,
        },
        token || undefined,
      );
      onStarted(run.run_id);
    } catch (err) {
      onError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const models = meta ? Array.from(new Set(meta.council.agents.map((a) => a.model))).join(", ") : "";

  return (
    <form className="run-form" onSubmit={submit}>
      <textarea
        required
        placeholder="Question to deliberate…"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={3}
      />
      <input placeholder="Options, comma separated (optional)" value={options} onChange={(e) => setOptions(e.target.value)} />
      <textarea
        placeholder="Facts you provide, one per line (become user evidence)"
        value={facts}
        onChange={(e) => setFacts(e.target.value)}
        rows={2}
      />
      <div className="row">
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)} title="strategy">
          <option value="">default strategy</option>
          {meta &&
            Object.entries(meta.strategies).map(([name, desc]) => (
              <option key={name} value={name} title={desc}>
                {name}
              </option>
            ))}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value)} title="mode">
          {(meta?.modes ?? ["fast", "balanced", "deep"]).map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      </div>
      {meta?.auth_required && (
        <input type="password" placeholder="API token" value={token} onChange={(e) => setToken(e.target.value)} />
      )}
      <button disabled={busy || !question.trim()}>{busy ? "Starting…" : "Deliberate"}</button>
      {models && <p className="muted small">Models (server config): {models}</p>}
      {models.includes("mock:") && (
        <p className="warn small">Mock provider: output is placeholder text, not reasoning.</p>
      )}
    </form>
  );
}
