import type { Decision, FusedRisk } from "../types";
import { ActionBadge } from "./Badges";
import { Card } from "./Card";
import { horizonsOf, label, pct } from "./format";

export function FusedRiskBars({ fused }: { fused: FusedRisk }) {
  const hs = horizonsOf(fused.probabilities);
  const max = Math.max(...hs.map((h) => fused.probabilities[String(h)]), 0.01);
  return (
    <div className="space-y-2">
      {hs.map((h) => {
        const p = fused.probabilities[String(h)];
        return (
          <div key={h} className="flex items-center gap-3 text-sm">
            <span className="w-16 text-slate-500">≤ {h} days</span>
            <div className="h-2.5 flex-1 rounded-full bg-slate-100">
              <div
                className="h-2.5 rounded-full bg-rose-500"
                style={{ width: `${Math.max(2, (p / max) * 100)}%` }}
              />
            </div>
            <span className="w-16 text-right font-semibold tabular-nums">{pct(p)}</span>
          </div>
        );
      })}
      <div className="flex gap-4 pt-1 text-xs text-slate-500">
        <span>uncertainty {fused.uncertainty.toFixed(2)}</span>
        <span>agent disagreement {fused.disagreement.toFixed(2)}</span>
        <span>fusion: {fused.method}</span>
      </div>
    </div>
  );
}

export function DecisionCard({ decision }: { decision: Decision }) {
  const costs = Object.entries(decision.expected_costs).sort((a, b) => a[1] - b[1]);
  const max = Math.max(...costs.map(([, c]) => c), 1e-9);
  return (
    <Card title="Decision" actions={<ActionBadge action={decision.action} />}>
      <div className="text-xs text-slate-500">
        Expected cost per action (1 unit = one planned replacement; missed failure ={" "}
        {decision.cost_ratio} units)
      </div>
      <div className="mt-2 space-y-1.5">
        {costs.map(([action, cost]) => (
          <div key={action} className="flex items-center gap-3 text-sm">
            <span className={`w-44 capitalize ${action === decision.action ? "font-semibold" : "text-slate-600"}`}>
              {label(action)}
            </span>
            <div className="h-2 flex-1 rounded-full bg-slate-100">
              <div className="h-2 rounded-full bg-slate-500" style={{ width: `${(cost / max) * 100}%` }} />
            </div>
            <span className="w-16 text-right tabular-nums">{cost.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-700">
        {decision.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
    </Card>
  );
}
