import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import { api } from "../api/client";
import { ActionBadge } from "../components/Badges";
import { Card, ErrorNote, Loading } from "../components/Card";
import { horizonsOf, label, pct } from "../components/format";
import type { EscalationItem } from "../types";

const ACTIONS = ["monitor", "schedule_replacement", "replace_now"];

function suggested(item: EscalationItem): string {
  const r = item.reasons.find((x) => x.startsWith("suggested action for the reviewer:"));
  return r ? r.split(":")[1].trim() : "monitor";
}

function Row({ item }: { item: EscalationItem }) {
  const qc = useQueryClient();
  const [action, setAction] = useState(suggested(item));
  const [note, setNote] = useState("");
  const submit = useMutation({
    mutationFn: (verdict: "approve" | "override") =>
      api.feedback({ run_id: item.run_id, serial_number: item.serial_number, verdict, final_action: verdict === "approve" ? suggested(item) : action, note }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["escalations"] }),
  });
  return (
    <Card
      title={
        <span>
          <Link to={`/drives/${encodeURIComponent(item.serial_number)}`} className="font-mono text-sky-700 hover:underline">
            {item.serial_number}
          </Link>{" "}
          <span className="font-normal text-slate-500">· {item.model} · as of {item.as_of_date}</span>
        </span>
      }
      actions={item.feedback ? <ActionBadge action={item.feedback.final_action} /> : <ActionBadge action={item.action} />}
    >
      <div className="flex flex-wrap gap-4 text-sm tabular-nums">
        {horizonsOf(item.probabilities).map((h) => (
          <span key={h}>≤{h}d <strong>{pct(item.probabilities[String(h)])}</strong></span>
        ))}
      </div>
      <ul className="mt-2 list-disc pl-5 text-sm text-slate-700">
        {item.reasons.map((r) => <li key={r}>{r}</li>)}
      </ul>
      {item.feedback ? (
        <p className="mt-3 text-sm text-slate-600">
          Reviewed ({item.feedback.verdict}): <strong>{label(item.feedback.final_action)}</strong>
          {item.feedback.note && ` – ${item.feedback.note}`}
        </p>
      ) : (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button onClick={() => submit.mutate("approve")} className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm text-white">
            Approve suggestion ({label(suggested(item))})
          </button>
          <select value={action} onChange={(e) => setAction(e.target.value)} className="rounded-md border border-slate-300 px-2 py-1.5 text-sm">
            {ACTIONS.map((a) => <option key={a} value={a}>{label(a)}</option>)}
          </select>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="note (optional)" className="rounded-md border border-slate-300 px-2 py-1.5 text-sm" />
          <button onClick={() => submit.mutate("override")} className="rounded-md border border-slate-400 px-3 py-1.5 text-sm">
            Override
          </button>
          <ErrorNote error={submit.error} />
        </div>
      )}
    </Card>
  );
}

export default function Escalations() {
  const items = useQuery({ queryKey: ["escalations"], queryFn: api.escalations });
  const open = items.data?.filter((i) => !i.feedback) ?? [];
  const done = items.data?.filter((i) => i.feedback) ?? [];
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Escalations</h1>
        <p className="text-sm text-slate-500">Analysed runs the policy sent to a human. Decisions are stored for later analysis.</p>
      </div>
      <ErrorNote error={items.error} />
      {!items.data && <Loading />}
      {items.data && items.data.length === 0 && <Card><div className="text-sm text-slate-500">No escalated runs yet.</div></Card>}
      {open.map((i) => <Row key={i.run_id} item={i} />)}
      {done.length > 0 && <h2 className="pt-2 text-sm font-semibold text-slate-600">Reviewed</h2>}
      {done.map((i) => <Row key={i.run_id} item={i} />)}
    </div>
  );
}
