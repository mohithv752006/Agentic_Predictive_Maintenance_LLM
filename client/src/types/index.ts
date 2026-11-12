// TypeScript mirrors of server/src/pdm_agents/schemas.py.
// Horizon-keyed maps arrive as JSON objects with string keys ("7", "14", "30").

export type HorizonMap = Record<string, number>;
export type AgentStatus = "ok" | "degraded" | "failed";
export type DriftLevel = "ok" | "warn" | "alert";
export type TraceStatus = "started" | "ok" | "degraded" | "failed" | "skipped" | "info";
export type EvidenceKind = "metric" | "trend" | "model" | "case" | "knowledge" | "context" | "note";

export interface PredictRequest {
  serial_number: string;
  as_of_date?: string | null;
  horizons?: number[];
  cost_ratio?: number | null;
  enabled_agents?: string[] | null;
  offline?: boolean;
}

export interface EvidenceItem {
  id: string;
  kind: EvidenceKind;
  label: string;
  value: number | string | null;
  detail: string;
}

export interface AgentFinding {
  id: string;
  agent: string;
  serial_number: string;
  window_start: string | null;
  window_end: string | null;
  summary: string;
  risk: HorizonMap;
  confidence: number;
  evidence: EvidenceItem[];
  status: AgentStatus;
  error: string | null;
}

export interface RetrievedCase {
  id: string;
  serial_number: string;
  model: string;
  similarity: number;
  outcome_failed: boolean;
  days_to_failure: number | null;
  summary: string;
}

export interface KnowledgeSnippet {
  id: string;
  source_file: string;
  text: string;
  score: number | null;
}

export interface FusedRisk {
  probabilities: HorizonMap;
  uncertainty: number;
  disagreement: number;
  method: string;
  contributing_agents: string[];
}

export interface DriftReport {
  psi_by_family: Record<string, number>;
  level: DriftLevel;
  reference: string;
}

export interface Decision {
  action: string;
  expected_costs: Record<string, number>;
  cost_ratio: number;
  escalated: boolean;
  reasons: string[];
}

export interface Explanation {
  text: string;
  cited_finding_ids: string[];
  cited_case_ids: string[];
  cited_knowledge_ids: string[];
  source: "llm" | "template";
}

export interface TraceEvent {
  ts: string;
  node: string;
  agent: string | null;
  status: TraceStatus;
  message: string;
  data: Record<string, unknown>;
}

export interface PredictResponse {
  run_id: string;
  request: PredictRequest;
  context_summary: Record<string, unknown>;
  plan: string[];
  fused: FusedRisk | null;
  decision: Decision | null;
  explanation: Explanation | null;
  findings: AgentFinding[];
  retrieved_cases: RetrievedCase[];
  knowledge: KnowledgeSnippet[];
  drift: DriftReport | null;
  llm_available: boolean;
  latency_ms: number;
  llm_calls: number;
  errors: string[];
  trace: TraceEvent[];
}

export interface LLMHealth {
  configured: boolean;
  reachable: boolean;
  base_url: string;
  model_id: string | null;
  latency_ms: number | null;
  error: string | null;
  guided_json: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  env: string;
  data_ready: boolean;
  models_ready: boolean;
  indexes_ready: boolean;
  llm: LLMHealth;
}

export interface RuntimeSettingsView {
  llm_base_url: string;
  llm_api_key_set: boolean;
  llm_api_key_hint: string;
  llm_model: string;
  default_cost_ratio: number;
  enabled_agents: string[];
  available_agents: string[];
  cost_ratios: number[];
}

export interface RuntimeSettingsUpdate {
  llm_base_url?: string;
  llm_api_key?: string;
  llm_model?: string;
  default_cost_ratio?: number;
  enabled_agents?: string[];
}

export interface DriveSummary {
  serial_number: string;
  model: string;
  capacity_bytes: number | null;
  last_date: string;
  failed: boolean;
  latest_risk: number | null;
  action: string | null;
  split: string | null;
}

export interface DriveListResponse {
  items: DriveSummary[];
  total: number;
  page: number;
  page_size: number;
  models: string[];
  risk_horizon: number;
}

export interface HistogramBin {
  start: number;
  end: number;
  count: number;
}

export interface FleetStats {
  total: number;
  failed: number;
  by_action: Record<string, number>;
  by_model: Record<string, number>;
  risk_horizon: number;
  risk_histogram: HistogramBin[];
  high_risk_count: number;
  escalated_count: number;
  snapshot_available: boolean;
}

export interface SmartSeries {
  column: string;
  family: string;
  values: (number | null)[];
}

export interface DriveHistory {
  serial_number: string;
  model: string;
  capacity_bytes: number | null;
  dates: string[];
  series: SmartSeries[];
  failure_date: string | null;
}

export interface FeedbackRequest {
  run_id: string;
  serial_number: string;
  verdict: "approve" | "override";
  final_action: string;
  note?: string;
}

export interface FeedbackRecord extends FeedbackRequest {
  id: string;
  created_at: string;
}

export interface EscalationItem {
  run_id: string;
  serial_number: string;
  model: string;
  as_of_date: string;
  action: string;
  reasons: string[];
  probabilities: HorizonMap;
  created_at: string;
  feedback: FeedbackRecord | null;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  created_at: string;
  systems: string[];
  headline: Record<string, unknown>;
}

export interface ExperimentDetail {
  id: string;
  name: string;
  created_at: string;
  metrics: Record<string, unknown>;
  tables: Record<string, Record<string, unknown>[]>;
  plots: string[];
  config: Record<string, unknown>;
}

/** Server-sent event payload for one graph update. */
export interface StreamUpdate {
  type: "event";
  node: string;
  run_id: string;
  trace: TraceEvent[];
  findings: AgentFinding[];
}
