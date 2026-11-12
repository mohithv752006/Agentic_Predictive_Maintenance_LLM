import { ACTION_STYLES, STATUS_STYLES, label } from "./format";

export function ActionBadge({ action }: { action: string | null | undefined }) {
  if (!action) return <span className="text-slate-400">–</span>;
  return (
    <span
      className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
        ACTION_STYLES[action] ?? "bg-slate-50 text-slate-700 ring-slate-600/20"
      }`}
    >
      {label(action)}
    </span>
  );
}

/** Coloured status pill; `text` overrides the shown word (e.g. drift "alert" styled as failed). */
export function StatusBadge({ status, text }: { status: string; text?: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
        STATUS_STYLES[status] ?? STATUS_STYLES.info
      }`}
    >
      {status === "running" || status === "started" ? (
        <span className="mr-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-sky-500" />
      ) : null}
      {text ?? status}
    </span>
  );
}
