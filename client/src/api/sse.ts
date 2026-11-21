import type { PredictResponse, StreamUpdate } from "../types";
import { API_BASE } from "./client";
import { DEMO, replayPrediction } from "./demo";

export interface StreamParams {
  serial_number: string;
  as_of_date?: string;
  cost_ratio?: number;
  offline?: boolean;
  agents?: string[];
}

export interface StreamHandlers {
  onStart?: (info: { agents: string[]; llm_configured: boolean }) => void;
  onUpdate?: (update: StreamUpdate) => void;
  onResult: (result: PredictResponse) => void;
  onError: (message: string) => void;
}

/** Open the /predict/stream SSE connection; returns a function that closes it. */
export function streamPrediction(params: StreamParams, handlers: StreamHandlers): () => void {
  if (DEMO) return replayPrediction(params, handlers);
  const q = new URLSearchParams({ serial_number: params.serial_number });
  if (params.as_of_date) q.set("as_of_date", params.as_of_date);
  if (params.cost_ratio !== undefined) q.set("cost_ratio", String(params.cost_ratio));
  if (params.offline) q.set("offline", "true");
  if (params.agents?.length) q.set("agents", params.agents.join(","));
  const source = new EventSource(`${API_BASE}/predict/stream?${q.toString()}`);
  let finished = false;

  source.addEventListener("start", (e) => handlers.onStart?.(JSON.parse((e as MessageEvent).data)));
  source.addEventListener("update", (e) => handlers.onUpdate?.(JSON.parse((e as MessageEvent).data)));
  source.addEventListener("result", (e) => {
    finished = true;
    handlers.onResult(JSON.parse((e as MessageEvent).data));
    source.close();
  });
  source.addEventListener("error", (e) => {
    if (finished) return;
    const data = (e as MessageEvent).data;
    finished = true;
    source.close();
    handlers.onError(data ? JSON.parse(data).message : "stream connection failed");
  });
  return () => {
    finished = true;
    source.close();
  };
}
