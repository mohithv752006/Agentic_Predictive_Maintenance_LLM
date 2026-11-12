import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DriveHistory } from "../types";
import { Card } from "./Card";
import { FAMILY_COLORS, label } from "./format";

/** One line chart per SMART family (agent), with the as-of date marked. */
export function SmartCharts({ history, asOf }: { history: DriveHistory; asOf?: string }) {
  const families = useMemo(() => {
    const groups = new Map<string, typeof history.series>();
    for (const s of history.series) groups.set(s.family, [...(groups.get(s.family) ?? []), s]);
    return [...groups.entries()];
  }, [history]);
  const rows = useMemo(
    () =>
      history.dates.map((d, i) => {
        const row: Record<string, string | number | null> = { date: d };
        for (const s of history.series) row[s.column] = s.values[i];
        return row;
      }),
    [history],
  );
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {families.map(([family, series]) => (
        <Card key={family} title={<span className="capitalize">{label(family)}</span>}>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows} margin={{ top: 5, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis tick={{ fontSize: 10 }} width={56} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {asOf && <ReferenceLine x={asOf} stroke="#0f172a" strokeDasharray="4 2" label={{ value: "as of", fontSize: 10 }} />}
                {history.failure_date && (
                  <ReferenceLine x={history.failure_date} stroke="#e11d48" label={{ value: "failure", fontSize: 10 }} />
                )}
                {series.map((s, i) => (
                  <Line
                    key={s.column}
                    type="monotone"
                    dataKey={s.column}
                    stroke={FAMILY_COLORS[i % FAMILY_COLORS.length]}
                    dot={false}
                    strokeWidth={1.5}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      ))}
    </div>
  );
}
