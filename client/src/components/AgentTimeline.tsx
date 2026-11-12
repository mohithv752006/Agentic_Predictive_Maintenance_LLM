import type { AgentFinding, TraceEvent } from "../types";
import { StatusBadge } from "./Badges";
import { horizonsOf, label, pct } from "./format";

export interface TimelineRow {
  agent: string;
  status: string; // pending | running | ok | degraded | failed
  finding?: AgentFinding;
  durationMs?: number;
}

/** Build per-agent rows from the planned agents and the streamed trace events. */
export function timelineRows(agents: string[], trace: TraceEvent[], findings: AgentFinding[]): TimelineRow[] {
  const rows = new Map<string, TimelineRow>(agents.map((a) => [a, { agent: a, status: "pending" }]));
  for (const ev of trace) {
    if (!ev.agent) continue;
    const row = rows.get(ev.agent) ?? { agent: ev.agent, status: "pending" };
    row.status = ev.status === "started" ? "running" : ev.status;
    if (typeof ev.data?.duration_ms === "number") row.durationMs = ev.data.duration_ms as number;
    rows.set(ev.agent, row);
  }
  for (const f of findings) {
    const row = rows.get(f.agent) ?? { agent: f.agent, status: f.status };
    row.finding = f;
    row.status = f.status;
    rows.set(f.agent, row);
  }
  return [...rows.values()];
}

export function AgentTimeline({
  rows,
  highlight,
}: {
  rows: TimelineRow[];
  highlight?: string | null;
}) {
  if (!rows.length) return <div className="text-sm text-slate-500">No agents planned yet.</div>;
  return (
    <ol className="relative space-y-3 border-l border-slate-200 pl-5">
      {rows.map((r) => {
        const f = r.finding;
        const active = highlight && f && highlight === f.id;
        return (
          <li
            key={r.agent}
            id={f ? `cite-${f.id}` : undefined}
            className={`rounded-lg p-3 transition ${active ? "bg-sky-50 ring-2 ring-sky-300" : "bg-slate-50"}`}
          >
            <span className="absolute -left-1.5 mt-1.5 h-3 w-3 rounded-full border-2 border-white bg-slate-300" />
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold capitalize text-slate-800">{label(r.agent)}</span>
              <StatusBadge status={r.status} />
              {f && <span className="font-mono text-xs text-slate-400">{f.id}</span>}
              {r.durationMs !== undefined && (
                <span className="text-xs text-slate-400">{(r.durationMs / 1000).toFixed(1)} s</span>
              )}
              {f && Object.keys(f.risk).length > 0 && (
                <span className="ml-auto flex gap-2 text-xs tabular-nums text-slate-600">
                  {horizonsOf(f.risk).map((h) => (
                    <span key={h}>
                      {h}d <strong>{pct(f.risk[String(h)])}</strong>
                    </span>
                  ))}
                </span>
              )}
            </div>
            {f && <p className="mt-1.5 text-sm text-slate-700">{f.summary}</p>}
            {f?.error && f.status === "failed" && <p className="mt-1 text-xs text-rose-600">{f.error}</p>}
            {f && f.evidence.length > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-slate-500">
                  {f.evidence.length} evidence items · confidence {f.confidence.toFixed(2)}
                </summary>
                <ul className="mt-1 space-y-1">
                  {f.evidence.map((e) => (
                    <li key={e.id} className="text-xs text-slate-600">
                      <span className="font-mono text-slate-400">{e.id}</span> {e.label}
                      {e.value !== null && e.value !== "" ? `: ${e.value}` : ""}{" "}
                      <span className="text-slate-500">{e.detail}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </li>
        );
      })}
    </ol>
  );
}
