import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { API_BASE, api } from "../api/client";
import { Card, ErrorNote, Loading } from "../components/Card";

function fmt(v: unknown): string {
  if (typeof v === "number") return Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (v === null || v === undefined) return "–";
  return String(v);
}

function DataTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <div className="text-sm text-slate-500">Empty table.</div>;
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            {cols.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1.5 font-medium">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-slate-100">
              {cols.map((c) => <td key={c} className="whitespace-nowrap px-2 py-1.5 tabular-nums">{fmt(r[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Experiments() {
  const list = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const [selected, setSelected] = useState<string | null>(null);
  const id = selected ?? list.data?.[0]?.id ?? null;
  const detail = useQuery({ queryKey: ["experiment", id], queryFn: () => api.experiment(id!), enabled: !!id });
  const d = detail.data;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Experiments</h1>
        <p className="text-sm text-slate-500">
          Reports written by <code>python -m scripts.evaluate</code>. All numbers are computed on the
          held-out test periods; LLM-dependent systems run on the stratified sample stated in each report.
        </p>
      </div>
      <ErrorNote error={list.error ?? detail.error} />
      {list.data && list.data.length === 0 && (
        <Card><div className="text-sm text-slate-500">No experiment reports yet.</div></Card>
      )}
      <div className="flex flex-wrap gap-2">
        {list.data?.map((e) => (
          <button
            key={e.id}
            onClick={() => setSelected(e.id)}
            className={`rounded-md border px-3 py-1.5 text-sm ${e.id === id ? "border-slate-900 bg-slate-900 text-white" : "border-slate-300 bg-white"}`}
          >
            {e.name}
            <span className="ml-2 text-xs opacity-70">{e.created_at.slice(0, 16).replace("T", " ")}</span>
          </button>
        ))}
      </div>
      {id && !d && <Loading />}
      {d && (
        <>
          {typeof d.metrics.notes === "string" && (
            <Card title="Notes"><p className="whitespace-pre-line text-sm text-slate-700">{d.metrics.notes}</p></Card>
          )}
          {Object.entries(d.tables).map(([name, rows]) => (
            <Card key={name} title={name.replace(/_/g, " ")}>
              <DataTable rows={rows} />
            </Card>
          ))}
          {d.plots.length > 0 && (
            <div className="grid gap-4 lg:grid-cols-2">
              {d.plots.map((p) => (
                <Card key={p} title={p.split("/").pop()?.replace(".png", "").replace(/_/g, " ")}>
                  <img src={`${API_BASE}${p}`} alt={p} className="w-full" />
                </Card>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
