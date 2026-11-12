import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { api } from "../api/client";

/** Shown whenever the API or the Colab LLM is unreachable (degraded mode). */
export function OfflineBanner() {
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
