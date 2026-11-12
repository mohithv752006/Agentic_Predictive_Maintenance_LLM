import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type DriveFilters } from "../api/client";
import { ActionBadge } from "../components/Badges";
import { Card, ErrorNote, KpiCard, Loading } from "../components/Card";
import { label, pct, tb } from "../components/format";

const SORTABLE: { key: NonNullable<DriveFilters["sort"]>; label: string }[] = [
  { key: "serial", label: "Serial" },
  { key: "model", label: "Model" },
  { key: "last_date", label: "Last snapshot" },
  { key: "risk", label: "Risk" },
];

export default function FleetOverview() {
  const [filters, setFilters] = useState<DriveFilters>({ page: 1, page_size: 25, sort: "risk", order: "desc" });
  const stats = useQuery({ queryKey: ["fleet-stats"], queryFn: api.fleetStats });
  const drives = useQuery({
    queryKey: ["drives", filters],
    queryFn: () => api.drives(filters),
    placeholderData: keepPreviousData,
  });
  const set = (patch: Partial<DriveFilters>) => setFilters((f) => ({ ...f, page: 1, ...patch }));
  const toggleSort = (key: NonNullable<DriveFilters["sort"]>) =>
    setFilters((f) => ({ ...f, sort: key, order: f.sort === key && f.order === "desc" ? "asc" : "desc" }));
  const s = stats.data;
  const d = drives.data;
  const pages = d ? Math.max(1, Math.ceil(d.total / d.page_size)) : 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Fleet overview</h1>
        <p className="text-sm text-slate-500">
          Latest snapshot of every drive, scored by the calibrated numeric pipeline (no LLM). Run a full
          multi-agent analysis from a drive's page.
        </p>
      </div>
      <ErrorNote error={stats.error ?? drives.error} />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Drives" value={s ? s.total.toLocaleString() : "–"} hint={s ? `${Object.keys(s.by_model).length} models` : undefined} />
        <KpiCard label="High risk" value={s ? s.high_risk_count.toLocaleString() : "–"} hint="replacement recommended" />
        <KpiCard label="Escalations" value={s ? s.escalated_count.toLocaleString() : "–"} hint="snapshot + analysed runs" />
        <KpiCard label="Failed (in data)" value={s ? s.failed.toLocaleString() : "–"} hint="drives with a failure record" />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title={`Risk distribution: P(fail ≤ ${s?.risk_horizon ?? 30} days)`} className="lg:col-span-2">
          {s ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={s.risk_histogram.map((b) => ({ bin: `${pct(b.start, b.start < 0.01 ? 3 : 1)}`, count: b.count }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} scale="sqrt" width={48} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Bar dataKey="count" fill="#334155" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Loading />
          )}
          <p className="mt-2 text-xs text-slate-500">Log-spaced bins; bar height on a square-root scale.</p>
        </Card>
        <Card title="Recommended actions">
          {s ? (
            <ul className="space-y-2">
              {Object.entries(s.by_action).map(([a, n]) => (
                <li key={a} className="flex items-center justify-between text-sm">
                  <ActionBadge action={a} />
                  <span className="tabular-nums">{n.toLocaleString()}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Loading />
          )}
        </Card>
      </div>

      <Card
        title="Drives"
        actions={
          <div className="flex flex-wrap gap-2">
            <input
              placeholder="Search serial"
              className="rounded-md border border-slate-300 px-2 py-1 text-sm"
              onChange={(e) => set({ search: e.target.value || undefined })}
            />
            <select className="rounded-md border border-slate-300 px-2 py-1 text-sm" onChange={(e) => set({ model: e.target.value || undefined })}>
              <option value="">All models</option>
              {d?.models.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
            <select className="rounded-md border border-slate-300 px-2 py-1 text-sm" onChange={(e) => set({ action: e.target.value || undefined })}>
              <option value="">All actions</option>
              {s && Object.keys(s.by_action).map((a) => <option key={a} value={a}>{label(a)}</option>)}
            </select>
          </div>
        }
      >
        {!d ? (
          <Loading />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                    {SORTABLE.map((c) => (
                      <th key={c.key} className="cursor-pointer px-2 py-2 select-none" onClick={() => toggleSort(c.key)}>
                        {c.label} {filters.sort === c.key ? (filters.order === "desc" ? "▼" : "▲") : ""}
                      </th>
                    ))}
                    <th className="px-2 py-2">Capacity</th>
                    <th className="px-2 py-2">Action</th>
                    <th className="px-2 py-2">Split</th>
                  </tr>
                </thead>
                <tbody>
                  {d.items.map((r) => (
                    <tr key={r.serial_number} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="px-2 py-2 font-mono">
                        <Link to={`/drives/${encodeURIComponent(r.serial_number)}`} className="text-sky-700 hover:underline">
                          {r.serial_number}
                        </Link>
                        {r.failed && <span className="ml-2 rounded bg-rose-100 px-1 text-xs text-rose-700">failed</span>}
                      </td>
                      <td className="px-2 py-2">{r.model}</td>
                      <td className="px-2 py-2 tabular-nums">{r.last_date}</td>
                      <td className="px-2 py-2 tabular-nums">{pct(r.latest_risk)}</td>
                      <td className="px-2 py-2">{tb(r.capacity_bytes)}</td>
                      <td className="px-2 py-2"><ActionBadge action={r.action} /></td>
                      <td className="px-2 py-2 text-slate-500">{r.split ?? "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-slate-600">
              <span>{d.total.toLocaleString()} drives</span>
              <div className="flex items-center gap-2">
                <button disabled={d.page <= 1} className="rounded border px-2 py-1 disabled:opacity-40" onClick={() => setFilters((f) => ({ ...f, page: (f.page ?? 1) - 1 }))}>
                  Prev
                </button>
                <span>
                  page {d.page} / {pages}
                </span>
                <button disabled={d.page >= pages} className="rounded border px-2 py-1 disabled:opacity-40" onClick={() => setFilters((f) => ({ ...f, page: (f.page ?? 1) + 1 }))}>
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
