export const pct = (p: number | null | undefined, digits = 2): string =>
  p === null || p === undefined || Number.isNaN(p) ? "–" : `${(p * 100).toFixed(digits)}%`;

export const tb = (bytes: number | null | undefined): string =>
  bytes ? `${(bytes / 1e12).toFixed(1)} TB` : "–";

export const label = (s: string): string => s.replace(/_/g, " ");

export const horizonsOf = (map: Record<string, number>): number[] =>
  Object.keys(map)
    .map(Number)
    .sort((a, b) => a - b);

export const ACTION_STYLES: Record<string, string> = {
  monitor: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  schedule_replacement: "bg-amber-50 text-amber-800 ring-amber-600/20",
  replace_now: "bg-rose-50 text-rose-700 ring-rose-600/20",
  escalate_to_human: "bg-violet-50 text-violet-700 ring-violet-600/20",
};

export const STATUS_STYLES: Record<string, string> = {
  pending: "bg-slate-100 text-slate-500",
  started: "bg-sky-100 text-sky-700",
  running: "bg-sky-100 text-sky-700",
  ok: "bg-emerald-100 text-emerald-700",
  degraded: "bg-amber-100 text-amber-800",
  failed: "bg-rose-100 text-rose-700",
  skipped: "bg-slate-100 text-slate-500",
  info: "bg-slate-100 text-slate-600",
};

export const FAMILY_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"];
