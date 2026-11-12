import { Fragment } from "react";
import type { Explanation, KnowledgeSnippet, RetrievedCase } from "../types";
import { Card } from "./Card";

const CITE = /\[((?:F|C|K)-[0-9A-Za-z_-]+)\]/g;

/** Explanation text with clickable [F-..], [C-..], [K-..] citations. */
export function ExplanationView({
  explanation,
  onCite,
}: {
  explanation: Explanation;
  onCite: (id: string) => void;
}) {
  const parts: (string | { id: string })[] = [];
  let last = 0;
  for (const m of explanation.text.matchAll(CITE)) {
    parts.push(explanation.text.slice(last, m.index));
    parts.push({ id: m[1] });
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(explanation.text.slice(last));
  const cited = [
    ...explanation.cited_finding_ids,
    ...explanation.cited_case_ids,
    ...explanation.cited_knowledge_ids,
  ];
  return (
    <Card
      title="Explanation"
      actions={
        <span className="text-xs text-slate-500">
          {explanation.source === "llm" ? "written by the LLM from cited evidence" : "template (LLM unavailable)"}
        </span>
      }
    >
      <p className="text-sm leading-relaxed text-slate-800">
        {parts.map((p, i) =>
          typeof p === "string" ? (
            <Fragment key={i}>{p}</Fragment>
          ) : (
            <button
              key={i}
              onClick={() => onCite(p.id)}
              className="mx-0.5 rounded bg-sky-100 px-1 font-mono text-xs text-sky-800 hover:bg-sky-200"
            >
              {p.id}
            </button>
          ),
        )}
      </p>
      {cited.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-slate-500">
          Sources:
          {cited.map((id) => (
            <button key={id} onClick={() => onCite(id)} className="font-mono text-sky-700 hover:underline">
              {id}
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}

export function CasesList({ cases, highlight }: { cases: RetrievedCase[]; highlight?: string | null }) {
  if (!cases.length) return <div className="text-sm text-slate-500">No similar drives retrieved.</div>;
  return (
    <ul className="space-y-2">
      {cases.map((c) => (
        <li
          key={c.id}
          id={`cite-${c.id}`}
          className={`rounded-lg p-3 text-sm ${highlight === c.id ? "bg-sky-50 ring-2 ring-sky-300" : "bg-slate-50"}`}
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs text-slate-400">{c.id}</span>
            <span className="font-medium">{c.serial_number}</span>
            <span
              className={`rounded px-1.5 text-xs ${c.outcome_failed ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"}`}
            >
              {c.outcome_failed ? `failed after ${c.days_to_failure ?? "?"} d` : "survived"}
            </span>
            <span className="ml-auto text-xs text-slate-500">similarity {c.similarity.toFixed(3)}</span>
          </div>
          <p className="mt-1 text-xs text-slate-600">{c.summary}</p>
        </li>
      ))}
    </ul>
  );
}

export function KnowledgeList({ items, highlight }: { items: KnowledgeSnippet[]; highlight?: string | null }) {
  if (!items.length) return <div className="text-sm text-slate-500">No knowledge snippets.</div>;
  return (
    <ul className="space-y-2">
      {items.map((k) => (
        <li
          key={k.id}
          id={`cite-${k.id}`}
          className={`rounded-lg p-3 text-sm ${highlight === k.id ? "bg-sky-50 ring-2 ring-sky-300" : "bg-slate-50"}`}
        >
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-slate-400">{k.id}</span>
            <span className="text-xs text-slate-500">{k.source_file}</span>
          </div>
          <p className="mt-1 text-xs text-slate-700">{k.text}</p>
        </li>
      ))}
    </ul>
  );
}
