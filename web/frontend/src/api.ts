import type { Meta, Result, RunEvent, RunSummary } from "./types";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status}: ${body.slice(0, 200)}`);
  }
  return (await response.json()) as T;
}

export const api = {
  meta: () => fetch("/api/meta").then((r) => json<Meta>(r)),
  runs: () => fetch("/api/runs").then((r) => json<RunSummary[]>(r)),
  run: (id: string) =>
    fetch(`/api/runs/${encodeURIComponent(id)}`).then((r) =>
      json<{ status: string; result?: Result; events?: RunEvent[]; error?: string }>(r),
    ),
  start: (body: Record<string, unknown>, token?: string) =>
    fetch("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
    }).then((r) => json<RunSummary>(r)),
  stream(id: string, onEvent: (e: RunEvent) => void, onEnd: (status: string) => void): () => void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/runs/${encodeURIComponent(id)}/stream`);
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data as string) as { type: string; event?: RunEvent; status?: string };
      if (data.type === "event" && data.event) onEvent(data.event);
      if (data.type === "end") onEnd(data.status ?? "unknown");
    };
    ws.onerror = () => onEnd("error");
    return () => ws.close();
  },
};
