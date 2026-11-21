/**
 * Public demo mode (VITE_DEMO=1): the dashboard reads recorded JSON from public/demo/
 * (written by server/scripts/export_demo.py) instead of calling the FastAPI backend, and
 * "Run analysis" replays a recorded run of the real system step by step.
 */
import type { AgentFinding, DriveListResponse, DriveSummary, PredictResponse } from "../types";
import type { StreamHandlers, StreamParams } from "./sse";

export const DEMO = import.meta.env.VITE_DEMO === "1";
export const DEMO_BASE = `${import.meta.env.BASE_URL}demo`;
const REPLAY_MS = 12_000; // a recorded run (about 60-100 s) is replayed in about this time

export class DemoError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const cache = new Map<string, Promise<unknown>>();

async function load<T>(file: string): Promise<T> {
  if (!cache.has(file)) {
    cache.set(file, fetch(`${DEMO_BASE}/${file}`).then(async (res) => {
      if (!res.ok) throw new DemoError(404, "This item is not part of the public demo.");
      return res.json();
    }));
  }
  try {
    return (await cache.get(file)) as T;
  } catch (e) {
    cache.delete(file);
    throw e;
  }
}

interface DriveIndex {
  items: DriveSummary[];
  models: string[];
  risk_horizon: number;
}

export interface RecordedRun {
  as_of: string;
  run_id: string;
  cost_ratio: number;
}

export const demoAnalyses = () => load<Record<string, RecordedRun[]>>("analyses/index.json");

function listDrives(index: DriveIndex, q: URLSearchParams): DriveListResponse {
  let items = [...index.items];
  const model = q.get("model");
  const minRisk = q.get("min_risk");
  const action = q.get("action");
  const failed = q.get("failed");
  const search = q.get("search");
  if (model) items = items.filter((d) => d.model === model);
  if (minRisk) items = items.filter((d) => (d.latest_risk ?? -1) >= Number(minRisk));
  if (action) items = items.filter((d) => d.action === action);
  if (failed) items = items.filter((d) => d.failed === (failed === "true"));
  if (search) items = items.filter((d) => d.serial_number.toLowerCase().includes(search.toLowerCase()));
  const sort = q.get("sort") ?? "risk";
  const dir = q.get("order") === "asc" ? 1 : -1;
  const key = (d: DriveSummary): string | number =>
    sort === "serial" ? d.serial_number : sort === "model" ? d.model
      : sort === "last_date" ? d.last_date : (d.latest_risk ?? -1);
  items.sort((a, b) => (key(a) < key(b) ? -dir : key(a) > key(b) ? dir : 0));
  const page = Number(q.get("page") ?? 1);
  const size = Number(q.get("page_size") ?? 50);
  return { items: items.slice((page - 1) * size, page * size), total: items.length, page,
           page_size: size, models: index.models, risk_horizon: index.risk_horizon };
}

/** Answer an API request from the recorded files. */
export async function demoRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const [route, qs = ""] = path.split("?");
  const q = new URLSearchParams(qs);
  const method = (init?.method ?? "GET").toUpperCase();
  const parts = route.split("/").filter(Boolean);
  if (method !== "GET") {
    const what = parts[0] === "feedback" ? "Recording feedback" : parts[0] === "predict" ? "Live analysis"
      : "Changing settings";
    throw new DemoError(403, `${what} is disabled in the public demo (no backend is running).`);
  }
  switch (parts[0]) {
    case "health":
      return load<T>("health.json");
    case "settings":
      return load<T>("settings.json");
    case "escalations":
      return load<T>("escalations.json");
    case "runs":
      return load<T>(`runs/${parts[1]}.json`);
    case "experiments":
      return parts[1] ? load<T>(`experiments/${parts[1]}.json`) : load<T>("experiments.json");
    case "drives":
      if (!parts[1]) return listDrives(await load<DriveIndex>("drives.json"), q) as T;
      if (parts[1] === "stats") return load<T>("fleet_stats.json");
      return load<T>(`drives/${decodeURIComponent(parts[1])}.json`);
    default:
      throw new DemoError(404, `${route} is not available in the public demo.`);
  }
}

function closest(runs: RecordedRun[], asOf?: string): RecordedRun {
  if (!asOf) return runs[0];
  const t = Date.parse(asOf);
  return runs.reduce((best, r) =>
    Math.abs(Date.parse(r.as_of) - t) < Math.abs(Date.parse(best.as_of) - t) ? r : best);
}

/** Replay a recorded run through the same handlers the live SSE stream uses. */
export function replayPrediction(params: StreamParams, handlers: StreamHandlers): () => void {
  const timers: number[] = [];
  let cancelled = false;
  (async () => {
    try {
      const recorded = (await demoAnalyses())[params.serial_number];
      if (!recorded?.length) throw new DemoError(404, "This drive has no recorded analysis in the public demo.");
      const run = await load<PredictResponse>(`runs/${closest(recorded, params.as_of_date).run_id}.json`);
      if (cancelled) return;
      const agents = [...new Set([...run.plan, ...run.findings.map((f) => f.agent)])];
      handlers.onStart?.({ agents, llm_configured: run.llm_available });
      const events = run.trace;
      const t0 = Date.parse(events[0]?.ts ?? "");
      const span = Math.max(1, Date.parse(events[events.length - 1]?.ts ?? "") - t0);
      const scale = Math.min(1, REPLAY_MS / span);
      const byAgent = new Map<string, AgentFinding[]>();
      for (const f of run.findings) byAgent.set(f.agent, [...(byAgent.get(f.agent) ?? []), f]);
      let last = 0;
      events.forEach((ev) => {
        const at = Math.max(last + 120, Number.isNaN(t0) ? last + 400 : (Date.parse(ev.ts) - t0) * scale);
        last = at;
        const done = ev.agent && ev.status !== "started";
        timers.push(window.setTimeout(() => handlers.onUpdate?.({
          type: "event", node: ev.node, run_id: run.run_id, trace: [ev],
          findings: done ? byAgent.get(ev.agent!) ?? [] : [],
        }), at));
      });
      timers.push(window.setTimeout(() => handlers.onResult(run), last + 500));
    } catch (e) {
      if (!cancelled) handlers.onError(e instanceof Error ? e.message : String(e));
    }
  })();
  return () => {
    cancelled = true;
    timers.forEach((t) => window.clearTimeout(t));
  };
}
