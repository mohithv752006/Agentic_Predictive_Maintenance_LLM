import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { api } from "../api/client";
import { streamPrediction } from "../api/sse";
import { AgentTimeline, timelineRows } from "../components/AgentTimeline";
import { StatusBadge } from "../components/Badges";
import { Card, ErrorNote, Loading } from "../components/Card";
import { DecisionCard, FusedRiskBars } from "../components/DecisionCard";
import { CasesList, ExplanationView, KnowledgeList } from "../components/ExplanationView";
import { SmartCharts } from "../components/SmartCharts";
import { tb } from "../components/format";
import type { AgentFinding, PredictResponse, TraceEvent } from "../types";

const NODE_ORDER = ["load_context", "planner", "retrieve", "fuse", "decide", "explain"];

function SerialPicker() {
  const navigate = useNavigate();
  const [serial, setSerial] = useState("");
  return (
    <Card title="Analyse a drive">
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (serial.trim()) navigate(`/drives/${encodeURIComponent(serial.trim())}`);
        }}
      >
        <input
          value={serial}
          onChange={(e) => setSerial(e.target.value)}
          placeholder="Serial number (pick one from the fleet overview)"
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <button className="rounded-md bg-slate-900 px-4 py-2 text-sm text-white">Open</button>
      </form>
    </Card>
  );
}

export default function DriveAnalysis() {
  const { serial } = useParams();
  if (!serial) return <SerialPicker />;
  return <DriveView serial={serial} />;
}

function DriveView({ serial }: { serial: string }) {
  const queryClient = useQueryClient();
  const history = useQuery({ queryKey: ["drive", serial], queryFn: () => api.drive(serial) });
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [asOf, setAsOf] = useState<string>("");
  const [costRatio, setCostRatio] = useState<number>(50);
  const [offline, setOffline] = useState(false);
  const [running, setRunning] = useState(false);
  const [agents, setAgents] = useState<string[]>([]);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [findings, setFindings] = useState<AgentFinding[]>([]);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const closeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (settings.data) setCostRatio(settings.data.default_cost_ratio);
  }, [settings.data]);
  useEffect(() => {
    const h = history.data;
    if (!h || asOf) return;
    // default as-of: latest snapshot, or the day before the failure record
    const last = h.dates[h.dates.length - 1];
    setAsOf(h.failure_date && h.dates.length > 1 ? h.dates[h.dates.length - 2] : last);
  }, [history.data, asOf]);
  useEffect(() => () => closeRef.current?.(), []);

  const run = () => {
    closeRef.current?.();
    setRunning(true);
    setTrace([]);
    setFindings([]);
    setResult(null);
    setError(null);
    closeRef.current = streamPrediction(
      { serial_number: serial, as_of_date: asOf || undefined, cost_ratio: costRatio, offline },
      {
        onStart: (info) => setAgents(info.agents),
        onUpdate: (u) => {
          setTrace((t) => [...t, ...u.trace]);
          if (u.findings.length) setFindings((f) => [...f, ...u.findings]);
        },
        onResult: (r) => {
          setResult(r);
          setTrace(r.trace);
          setFindings(r.findings);
          setRunning(false);
          queryClient.invalidateQueries({ queryKey: ["fleet-stats"] });
        },
        onError: (m) => {
          setError(m);
          setRunning(false);
        },
      },
    );
  };

  const cite = (id: string) => {
    setHighlight(id);
    document.getElementById(`cite-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const h = history.data;
  const rows = timelineRows(agents, trace, findings);
  const nodeStatus = (node: string) => {
    const evs = trace.filter((t) => t.node === node);
    return evs.length ? evs[evs.length - 1].status : running ? "pending" : "–";
  };
  const ctx = result?.context_summary as Record<string, unknown> | undefined;

  return (
    <div className="space-y-6">
      <ErrorNote error={history.error} />
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-mono text-xl font-semibold">{serial}</h1>
          {h && (
            <p className="text-sm text-slate-500">
              {h.model} · {tb(h.capacity_bytes)} · {h.dates.length} daily snapshots ({h.dates[0]} – {h.dates[h.dates.length - 1]})
              {h.failure_date && <span className="ml-2 rounded bg-rose-100 px-1.5 text-rose-700">failure recorded {h.failure_date}</span>}
            </p>
          )}
        </div>
      </div>

      <Card title="Run analysis">
        <div className="flex flex-wrap items-end gap-6">
          <label className="text-sm">
            <span className="block text-xs text-slate-500">As-of date</span>
            <input
              type="date"
              value={asOf}
              min={h?.dates[0]}
              max={h?.dates[h.dates.length - 1]}
              onChange={(e) => setAsOf(e.target.value)}
              className="mt-1 rounded-md border border-slate-300 px-2 py-1.5"
            />
          </label>
          <label className="min-w-64 flex-1 text-sm">
            <span className="block text-xs text-slate-500">
              Cost ratio (missed failure / planned replacement): <strong>{costRatio}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={300}
              step={1}
              value={costRatio}
              onChange={(e) => setCostRatio(Number(e.target.value))}
              className="mt-2 w-full"
            />
            <div className="flex justify-between text-[10px] text-slate-400">
              {(settings.data?.cost_ratios ?? [10, 50, 200]).map((c) => (
                <button key={c} type="button" onClick={() => setCostRatio(c)} className="hover:text-slate-700">
                  {c}
                </button>
              ))}
            </div>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={offline} onChange={(e) => setOffline(e.target.checked)} />
            Offline (no LLM)
          </label>
          <button
            onClick={run}
            disabled={running || !h}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {running ? "Running…" : "Run analysis"}
          </button>
        </div>
        {error && <div className="mt-3"><ErrorNote error={error} /></div>}
      </Card>

      {(running || result) && (
        <div className="grid gap-4 lg:grid-cols-5">
          <div className="space-y-4 lg:col-span-3">
            <Card
              title="Agent timeline"
              actions={
                <div className="flex flex-wrap gap-1.5">
                  {NODE_ORDER.map((n) => (
                    <span key={n} className="flex items-center gap-1 text-xs text-slate-500">
                      {n.replace("_", " ")} <StatusBadge status={nodeStatus(n)} />
                    </span>
                  ))}
                </div>
              }
            >
              <AgentTimeline rows={rows} highlight={highlight} />
            </Card>
            {result?.explanation && <ExplanationView explanation={result.explanation} onCite={cite} />}
          </div>
          <div className="space-y-4 lg:col-span-2">
            <Card title="Fused failure probability">
              {result?.fused ? <FusedRiskBars fused={result.fused} /> : <Loading text={running ? "Waiting for agents…" : "No fused risk"} />}
            </Card>
            {result?.decision && <DecisionCard decision={result.decision} />}
            {result && (
              <Card title="Run details">
                <dl className="grid grid-cols-2 gap-y-1 text-sm">
                  <dt className="text-slate-500">LLM</dt>
                  <dd>{result.llm_available ? `available (${result.llm_calls} calls)` : "unavailable (degraded mode)"}</dd>
                  <dt className="text-slate-500">Latency</dt>
                  <dd>{(result.latency_ms / 1000).toFixed(1)} s</dd>
                  <dt className="text-slate-500">Plan</dt>
                  <dd>{result.plan.join(", ") || "–"}</dd>
                  <dt className="text-slate-500">Drift</dt>
                  <dd>
                    {result.drift ? (
                      <>
                        <StatusBadge
                          status={result.drift.level === "ok" ? "ok" : result.drift.level === "warn" ? "degraded" : "failed"}
                          text={result.drift.level}
                        />{" "}
                        <span className="text-xs text-slate-500">
                          {Object.entries(result.drift.psi_by_family).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")}
                        </span>
                      </>
                    ) : (
                      "–"
                    )}
                  </dd>
                  <dt className="text-slate-500">Window</dt>
                  <dd>
                    {String(ctx?.window_days_observed ?? "–")}/{String(ctx?.window_days ?? "–")} days to {String(ctx?.as_of_date ?? "–")}
                  </dd>
                  <dt className="text-slate-500">Run id</dt>
                  <dd className="font-mono text-xs">{result.run_id}</dd>
                </dl>
                {result.errors.length > 0 && (
                  <ul className="mt-2 space-y-1 text-xs text-rose-600">
                    {result.errors.map((e) => <li key={e}>{e}</li>)}
                  </ul>
                )}
              </Card>
            )}
          </div>
        </div>
      )}

      {result && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Similar past drives (training period)">
            <CasesList cases={result.retrieved_cases} highlight={highlight} />
          </Card>
          <Card title="Knowledge snippets">
            <KnowledgeList items={result.knowledge} highlight={highlight} />
          </Card>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">SMART history by family</h2>
        {h ? <SmartCharts history={h} asOf={asOf} /> : <Loading />}
      </div>
    </div>
  );
}
