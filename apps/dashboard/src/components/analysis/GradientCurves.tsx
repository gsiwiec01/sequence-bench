import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type { Experiment } from "@/api/experiments";
import type { GradientParamTrends } from "@/api/gradients";
import type { EpochMetric } from "@/api/results";
import { cn } from "@/lib/utils";
import ParamHeatmap from "./ParamHeatmap";
import type { GradientHealth } from "./constants";

export default function GradientCurves({
  selected,
  epochData,
  loading,
  gradientHealth,
  paramTrendData,
}: {
  selected: { exp: Experiment; color: string }[];
  epochData: Record<string, EpochMetric[]>;
  loading: boolean;
  gradientHealth: Record<string, GradientHealth>;
  paramTrendData: Record<string, GradientParamTrends>;
}) {
  const [sharedScale, setSharedScale] = useState(false);

  const chartData = useMemo(() => {
    const allEpochs = [
      ...new Set(selected.flatMap(({ exp }) => (epochData[exp.id] ?? []).map((m) => m.epoch))),
    ].sort((a, b) => a - b);

    return allEpochs.map((epoch) => {
      const row: Record<string, number | null | undefined> = { epoch };
      for (const { exp } of selected) {
        const pt = (epochData[exp.id] ?? []).find((m) => m.epoch === epoch);
        const mean = pt?.grad_norm_mean;
        const max = pt?.grad_norm_max;

        row[`mean_${exp.id}`] = mean != null && mean > 0 ? mean : null;
        row[`max_${exp.id}`] = max != null && max > 0 ? max : null;
      }

      return row;
    });
  }, [selected, epochData]);

  const globalLogRange = useMemo(() => {
    if (!sharedScale) return null;

    const allVals = selected.flatMap(({ exp }) => {
      const d = paramTrendData[exp.id];
      if (!d) return [];

      return Object.values(d.params).flatMap((series) =>
        series.filter((v): v is number => v != null && v > 0),
      );
    });

    if (allVals.length === 0) return null;

    return { logMin: Math.log10(Math.min(...allVals)), logMax: Math.log10(Math.max(...allVals)) };
  }, [sharedScale, selected, paramTrendData]);

  if (selected.length === 0) {
    return <p className="py-12 text-center text-sm text-muted-foreground">Zaznacz eksperymenty w panelu po lewej.</p>;
  }

  const hasAnyData = chartData.some((row) => selected.some(({ exp }) => row[`mean_${exp.id}`] != null));
  const hasParamData = selected.some(({ exp }) => {
    const d = paramTrendData[exp.id];
    return d && Object.keys(d.params).length > 0 && d.epochs.length > 0;
  });

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Skala log -<span className="font-semibold">━━</span> mean,{" "}
            <span className="font-semibold">-</span> max
          </p>

          {loading && <span className="text-xs text-muted-foreground animate-pulse">Wczytuje dane...</span>}
        </div>

        {hasAnyData ? (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 36, right: 20, bottom: 28, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: "epoka", position: "insideBottom", offset: -12, fontSize: 11 }} />
              <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 10 }} tickFormatter={(v: number) => (v != null && isFinite(v) ? v.toExponential(0) : "")} width={56} label={{ value: "norma gradientu", angle: -90, position: "insideLeft", offset: 15, fontSize: 10 }} />

              <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v: number, key: string) => {
                const isMean = key.startsWith("mean_");
                const id = key.replace(/^(mean|max)_/, "");
                const item = selected.find((s) => s.exp.id === id);
                const label = item ? `${isMean ? "mean" : "max"} -${item.exp.architecture.toUpperCase()} k₁=${item.exp.k1} k₂=${item.exp.k2}` : key;

                return [v != null && isFinite(v) ? v.toExponential(3) : "-", label];
              }} />

              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} formatter={(key: string) => {
                const id = key.replace(/^mean_/, "");
                const item = selected.find((s) => s.exp.id === id);

                return <span style={{ marginLeft: 4 }}>{item ? `${item.exp.architecture.toUpperCase()} k₁=${item.exp.k1} k₂=${item.exp.k2} seed=${item.exp.seed}` : key}</span>;
              }} />

              {selected.flatMap(({ exp, color }) => [
                <Line key={`mean_${exp.id}`} type="monotone" dataKey={`mean_${exp.id}`} stroke={color} strokeWidth={2} dot={false} connectNulls />,
                <Line key={`max_${exp.id}`} type="monotone" dataKey={`max_${exp.id}`} stroke={color} strokeWidth={1.5} strokeDasharray="4 2" dot={false} connectNulls legendType="none" />,
              ])}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="py-8 text-center text-sm text-muted-foreground">Brak danych gradientów dla wybranych eksperymentów.</p>
        )}
      </div>

      <div className="flex flex-wrap gap-3">
        {selected.map(({ exp, color }) => {
          const h = gradientHealth[exp.id];
          if (!h?.status) return null;

          return (
            <div key={exp.id} className="flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs">
              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: color }} />
              <span className="font-medium uppercase">{exp.architecture}</span>
              <span className="text-muted-foreground">k₂={exp.k2}</span>
              <span className={cn("font-semibold",
                h.status === "stabilne" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive",
              )}>
                {h.status}
              </span>
            </div>
          );
        })}
      </div>

      {hasParamData && (
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Heatmapy parametrów</p>

            {selected.length > 1 && (
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
                <input type="checkbox" checked={sharedScale} onChange={(e) => setSharedScale(e.target.checked)} className="accent-primary" />
                Wspólna skala
              </label>
            )}
          </div>

          <div className={cn("grid gap-8", selected.length > 1 ? "xl:grid-cols-2" : "grid-cols-1")}>
            {selected.map(({ exp }) => (
              <ParamHeatmap key={exp.id} exp={exp} paramTrends={paramTrendData[exp.id] ?? null} sharedLogRange={globalLogRange} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
