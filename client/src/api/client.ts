import type {
  DriveHistory,
  DriveListResponse,
  EscalationItem,
  ExperimentDetail,
  ExperimentSummary,
  FeedbackRecord,
  FeedbackRequest,
  FleetStats,
  HealthResponse,
  LLMHealth,
  PredictRequest,
  PredictResponse,
  RuntimeSettingsUpdate,
  RuntimeSettingsView,
} from "../types";

export const API_BASE: string = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export interface DriveFilters {
  page?: number;
  page_size?: number;
  model?: string;
  min_risk?: number;
  action?: string;
  search?: string;
  failed?: boolean;
  sort?: "risk" | "serial" | "last_date" | "model";
  order?: "asc" | "desc";
}

export const api = {
  health: (refresh = false) => request<HealthResponse>(`/health${query({ refresh })}`),
  settings: () => request<RuntimeSettingsView>("/settings"),
  updateSettings: (body: RuntimeSettingsUpdate) =>
    request<RuntimeSettingsView>("/settings", { method: "PUT", body: JSON.stringify(body) }),
  testConnection: (body?: RuntimeSettingsUpdate) =>
    request<LLMHealth>("/settings/test", {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
    }),
  drives: (f: DriveFilters) =>
    request<DriveListResponse>(`/drives${query({ ...f })}`),
  fleetStats: () => request<FleetStats>("/drives/stats"),
  drive: (serial: string) => request<DriveHistory>(`/drives/${encodeURIComponent(serial)}`),
  predict: (body: PredictRequest) =>
    request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify(body) }),
  run: (runId: string) => request<PredictResponse>(`/runs/${runId}`),
  experiments: () => request<ExperimentSummary[]>("/experiments"),
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${id}`),
  feedback: (body: FeedbackRequest) =>
    request<FeedbackRecord>("/feedback", { method: "POST", body: JSON.stringify(body) }),
  escalations: () => request<EscalationItem[]>("/escalations"),
};
