import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { api } from "../api/client";
import { DEMO } from "../api/demo";

const REPO_URL = "https://github.com/mohithv752006/Agentic_Predictive_Maintenance_LLM";

function DemoBanner() {
  return (
    <div className="border-b border-sky-200 bg-sky-50 px-6 py-2 text-sm text-sky-900">
      <strong>Public demo.</strong> No backend runs here: drive data, agent analyses (recorded from the real
      system with the live Qwen3.5-4B LLM) and evaluation results are saved copies, and "Run analysis"
      replays a recorded run.{" "}
      <a href={REPO_URL} target="_blank" rel="noreferrer" className="font-medium underline">
        Source code on GitHub
      </a>
    </div>
  );
}

/** Shown whenever the API or the Colab LLM is unreachable (degraded mode). */
export function OfflineBanner() {
  if (DEMO) return <DemoBanner />;
  return <LiveBanner />;
}

function LiveBanner() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.health(), refetchInterval: 30_000 });
  if (health.isError) {
    return (
      <div className="border-b border-rose-200 bg-rose-50 px-6 py-2 text-sm text-rose-800">
        API unreachable: start the server with <code>python -m scripts.serve</code>.
      </div>
    );
  }
  const h = health.data;
  if (!h) return null;
  if (!h.data_ready || !h.models_ready) {
    return (
      <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-sm text-amber-900">
        Data or models are not ready. Run <code>scripts.prepare</code>, <code>scripts.train</code> and{" "}
        <code>scripts.index</code>.
      </div>
    );
  }
  if (!h.llm.reachable) {
    return (
      <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-sm text-amber-900">
        <strong>LLM offline: degraded mode.</strong> Predictions and decisions still come from the
        calibrated models; explanations use templates.{" "}
        {h.llm.configured ? `(${h.llm.error ?? "unreachable"}) ` : "No endpoint configured. "}
        <Link to="/settings" className="font-medium underline">
          Open settings
        </Link>
      </div>
    );
  }
  return null;
}
