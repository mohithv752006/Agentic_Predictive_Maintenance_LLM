import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Card, ErrorNote, Loading } from "../components/Card";
import { label } from "../components/format";
import type { LLMHealth, RuntimeSettingsUpdate } from "../types";

export default function Settings() {
  const qc = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const [url, setUrl] = useState("");
  const [key, setKey] = useState("");
  const [model, setModel] = useState("");
  const [ratio, setRatio] = useState(50);
  const [agents, setAgents] = useState<string[]>([]);
  const [test, setTest] = useState<LLMHealth | null>(null);

  useEffect(() => {
    const s = settings.data;
    if (!s) return;
    setUrl(s.llm_base_url);
    setModel(s.llm_model);
    setRatio(s.default_cost_ratio);
    setAgents(s.enabled_agents);
  }, [settings.data]);

  const body = (): RuntimeSettingsUpdate => ({
    llm_base_url: url,
    llm_model: model,
    default_cost_ratio: ratio,
    enabled_agents: agents,
    ...(key ? { llm_api_key: key } : {}),
  });
  const save = useMutation({
    mutationFn: () => api.updateSettings(body()),
    onSuccess: () => {
      setKey("");
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["health"] });
    },
  });
  const tester = useMutation({ mutationFn: () => api.testConnection(body()), onSuccess: setTest });

  if (!settings.data) return <Loading />;
  const s = settings.data;
  return (
    <div className="max-w-3xl space-y-6">
      <h1 className="text-xl font-semibold">Settings</h1>
      <ErrorNote error={settings.error ?? save.error ?? tester.error} />
      <Card title="LLM endpoint (Colab)">
        <p className="mb-3 text-xs text-slate-500">
          Paste the block printed by <code>qwen35_colab_server.ipynb</code>. The tunnel URL changes every Colab session.
        </p>
        <div className="space-y-3">
          <label className="block text-sm">
            <span className="text-xs text-slate-500">LLM_BASE_URL</span>
            <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://xxxx.trycloudflare.com/v1" className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm" />
          </label>
          <label className="block text-sm">
            <span className="text-xs text-slate-500">LLM_API_KEY {s.llm_api_key_set ? `(saved: ${s.llm_api_key_hint}; leave blank to keep)` : "(not set)"}</span>
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm" />
          </label>
          <label className="block text-sm">
            <span className="text-xs text-slate-500">LLM_MODEL</span>
            <input value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm" />
          </label>
          <div className="flex items-center gap-3">
            <button onClick={() => tester.mutate()} className="rounded-md border border-slate-300 px-3 py-1.5 text-sm">
              {tester.isPending ? "Testing…" : "Test connection"}
            </button>
            {test && (
              <span className={`text-sm ${test.reachable ? "text-emerald-700" : "text-rose-700"}`}>
                {test.reachable
                  ? `Reachable: ${test.model_id} (${test.latency_ms} ms, guided JSON ${test.guided_json ? "yes" : "no"})`
                  : `Not reachable: ${test.error}`}
              </span>
            )}
          </div>
        </div>
      </Card>
      <Card title="Agents">
        <div className="grid grid-cols-2 gap-2">
          {s.available_agents.map((a) => (
            <label key={a} className="flex items-center gap-2 text-sm capitalize">
              <input
                type="checkbox"
                checked={agents.includes(a)}
                onChange={(e) => setAgents((cur) => (e.target.checked ? [...cur, a] : cur.filter((x) => x !== a)))}
              />
              {label(a)}
            </label>
          ))}
        </div>
      </Card>
      <Card title="Decision">
        <label className="block text-sm">
          <span className="text-xs text-slate-500">Default cost ratio (missed failure / planned replacement): <strong>{ratio}</strong></span>
          <input type="range" min={1} max={300} value={ratio} onChange={(e) => setRatio(Number(e.target.value))} className="mt-2 w-full" />
        </label>
      </Card>
      <div className="flex items-center gap-3">
        <button onClick={() => save.mutate()} className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white">
          {save.isPending ? "Saving…" : "Save settings"}
        </button>
        {save.isSuccess && <span className="text-sm text-emerald-700">Saved.</span>}
      </div>
    </div>
  );
}
