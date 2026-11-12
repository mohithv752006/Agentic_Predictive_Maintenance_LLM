import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router";
import { api } from "../api/client";
import { PAGES } from "../routes";
import { OfflineBanner } from "./OfflineBanner";

export function Layout() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.health(), refetchInterval: 30_000 });
  const llmOk = health.data?.llm.reachable;
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-56 shrink-0 border-r border-slate-200 bg-white md:block">
        <div className="px-5 py-5">
          <div className="text-sm font-semibold text-slate-900">Drive Failure Agents</div>
          <div className="text-xs text-slate-500">cost-aware multi-agent RAG</div>
        </div>
        <nav className="space-y-0.5 px-3">
          {PAGES.filter((p) => p.nav).map((p) => (
            <NavLink
              key={p.path}
              to={p.navPath ?? p.path}
              end={p.path === "/"}
              className={({ isActive }) =>
                `block rounded-md px-3 py-2 text-sm ${
                  isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
                }`
              }
            >
              {p.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-6 px-5 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${llmOk ? "bg-emerald-500" : "bg-amber-500"}`} />
            LLM {llmOk ? "online" : "offline"}
          </div>
          {health.data?.llm.model_id && <div className="mt-1 truncate">{health.data.llm.model_id}</div>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <nav className="flex gap-1 overflow-x-auto border-b border-slate-200 bg-white px-3 py-2 md:hidden">
          {PAGES.filter((p) => p.nav).map((p) => (
            <NavLink
              key={p.path}
              to={p.navPath ?? p.path}
              end={p.path === "/"}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-3 py-1.5 text-sm ${
                  isActive ? "bg-slate-900 text-white" : "text-slate-600"
                }`
              }
            >
              {p.label}
            </NavLink>
          ))}
        </nav>
        <OfflineBanner />
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
          <Outlet />
        </main>
        <footer className="px-6 py-4 text-xs text-slate-400">
          Data: Backblaze Drive Stats (backblaze.com). Probabilities come from calibrated models; the
          LLM only plans and explains.
        </footer>
      </div>
    </div>
  );
}
