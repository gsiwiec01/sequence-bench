import {ReactNode, useMemo, useState} from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {Experiment, AdditionalMetricSeries} from "@/api/experiments";
import type {EpochMetric} from "@/api/results";

export default function LearningCurves({
                                         selected,
                                         epochData,
                                         additionalData,
                                         loading,
                                         additionalMetrics,
                                       }: {
  selected: { exp: Experiment; color: string }[];
  epochData: Record<string, EpochMetric[]>;
  additionalData: Record<string, AdditionalMetricSeries>;
  loading: boolean;
  additionalMetrics: string[];
}) {
  const [showTrain, setShowTrain] = useState(false);

  const allEpochs = useMemo(
    () =>
      [...new Set(selected.flatMap(({exp}) => (epochData[exp.id] ?? []).map((m) => m.epoch)))].sort(
        (a, b) => a - b,
      ),
    [selected, epochData],
  );

  const chartData = useMemo(() => {
    if (selected.length === 0) return [];

    return allEpochs.map((epoch) => {
      const row: Record<string, number | null | undefined> = {epoch};

      for (const {exp} of selected) {
        const pt = (epochData[exp.id] ?? []).find((m) => m.epoch === epoch);
        row[`val_${exp.id}`] = pt?.val_loss ?? null;
        if (showTrain) row[`train_${exp.id}`] = pt?.train_loss ?? null;
      }

      return row;
    });
  }, [selected, epochData, showTrain, allEpochs]);

  const additionalChartData = useMemo(() => {
    const charts = {} as Record<string, Record<string, number | null | undefined>[]>;
    for (const name of additionalMetrics) {
      const allMetricEpochs = [
        ...new Set(selected.flatMap(({exp}) =>
          (additionalData[exp.id]?.[name] ?? [])
            .map((p) => p.epoch))
        ),
      ].sort((a, b) => a - b);

      charts[name] = allMetricEpochs.map((epoch) => {
        const row: Record<string, number | null | undefined> = {epoch};

        for (const {exp} of selected) {
          const pt = (additionalData[exp.id]?.[name] ?? []).find((p) => p.epoch === epoch);
          row[exp.id] = pt?.value ?? null;
        }

        return row;
      });
    }

    return charts;
  }, [selected, additionalData, additionalMetrics]);

  if (selected.length === 0) {
    return <p className="py-12 text-center text-sm text-muted-foreground">Zaznacz eksperymenty w panelu po lewej</p>;
  }

  function expLabel(exp: Experiment, short = false) {
    return short
      ? `${exp.architecture.toUpperCase()} k₁=${exp.k1} k₂=${exp.k2}`
      : `${exp.architecture.toUpperCase()} k₁=${exp.k1} k₂=${exp.k2} seed=${exp.seed}`;
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 flex-wrap">
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
          <input type="checkbox" checked={showTrain} onChange={(e) => setShowTrain(e.target.checked)}
                 className="accent-primary"/>
          Pokaż train loss
        </label>

        {loading && <span className="text-xs text-muted-foreground animate-pulse">Wczytuje krzywe...</span>}
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{top: 36, right: 20, bottom: 28, left: 10}}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>
          <XAxis dataKey="epoch" tick={{fontSize: 11}}
                 label={{value: "epoka", position: "insideBottom", offset: -12, fontSize: 11}}/>
          <YAxis yAxisId="loss" tick={{fontSize: 11}} width={52}
                 label={{value: "strata", angle: -90, position: "insideLeft", offset: 15, fontSize: 11}}/>

          <Tooltip
            contentStyle={{fontSize: 11}}
            formatter={(v: number, key: string) => {
              const isTrain = key.startsWith("train_");
              const id = key.replace(/^(val|train)_/, "");
              const item = selected.find((s) => s.exp.id === id);
              const type = isTrain ? "train" : "val";

              return [v.toFixed(4), item ? `${type} · ${expLabel(item.exp, true)}` : key];
            }}
          />

          <Legend
            verticalAlign="top"
            wrapperStyle={{fontSize: 11}}
            formatter={(key: string) => {
              const isTrain = key.startsWith("train_");
              const id = key.replace(/^(val|train)_/, "");
              const item = selected.find((s) => s.exp.id === id);
              const prefix = isTrain ? "train · " : "";

              return <span style={{marginLeft: 4}}>{item ? `${prefix}${expLabel(item.exp)}` : key}</span>;
            }}
          />

          {selected.flatMap(({exp, color}) => {
            const valNonNull = chartData.filter((r) => r[`val_${exp.id}`] != null).length;
            const lines: ReactNode[] = [
              <Line key={`val_${exp.id}`} yAxisId="loss" type="monotone" dataKey={`val_${exp.id}`} stroke={color}
                    strokeWidth={2} dot={valNonNull <= 1 ? {r: 4, fill: color, strokeWidth: 0} : false} connectNulls/>,
            ];

            if (showTrain) {
              const trainNonNull = chartData.filter((r) => r[`train_${exp.id}`] != null).length;
              lines.push(
                <Line key={`train_${exp.id}`} yAxisId="loss" type="monotone" dataKey={`train_${exp.id}`} stroke={color}
                      strokeWidth={1.5} strokeDasharray="4 2"
                      dot={trainNonNull <= 1 ? {r: 3, fill: color, strokeWidth: 0} : false} connectNulls/>,
              );
            }

            return lines;
          })}
        </LineChart>
      </ResponsiveContainer>

      {additionalMetrics.map((name) => (
        <div key={name} className="space-y-1 pt-2">
          <p className="text-sm font-medium font-mono">{name}</p>

          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={additionalChartData[name] ?? []} margin={{top: 36, right: 20, bottom: 28, left: 10}}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>
              <XAxis dataKey="epoch" tick={{fontSize: 11}}
                     label={{value: "epoka", position: "insideBottom", offset: -12, fontSize: 11}}/>
              <YAxis tick={{fontSize: 11}} width={52}
                     label={{value: name, angle: -90, position: "insideLeft", offset: 15, fontSize: 10}}/>

              <Tooltip contentStyle={{fontSize: 11}} formatter={(v: number, key: string) => {
                const item = selected.find((s) => s.exp.id === key);
                return [v.toFixed(4), item ? expLabel(item.exp, true) : key];
              }}/>

              <Legend verticalAlign="top" wrapperStyle={{fontSize: 11}} formatter={(key: string) => {
                const item = selected.find((s) => s.exp.id === key);
                return <span style={{marginLeft: 4}}>{item ? expLabel(item.exp) : key}</span>;
              }}/>

              {selected.map(({exp, color}) => {
                const nonNull = (additionalChartData[name] ?? []).filter((r) => r[exp.id] != null).length;
                return (
                  <Line key={exp.id} type="monotone" dataKey={exp.id} stroke={color} strokeWidth={2}
                        dot={nonNull <= 1 ? {r: 4, fill: color, strokeWidth: 0} : false} connectNulls/>
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
