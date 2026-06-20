import {useState} from "react";
import {
  CartesianGrid, ErrorBar, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {useConvergence, type ConvergencePoint} from "@/api/results";
import {downloadFile} from "@/api/client";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

interface Props {
  datasetId: string;
  groupBy?: string;
  groupId?: string | null;
  availableMetrics?: string[];
}

const GROUP_COLORS: Record<string, string> = {
  lstm: "#2563eb",
  gru: "#dc2626",
  rnn: "#16a34a",
};
const FALLBACK_COLORS = ["#ca8a04", "#9333ea", "#0891b2", "#be185d"];

function CustomDot(props: {
  cx?: number; cy?: number; payload?: Record<string, unknown>;
  groupKey: string; color: string; r?: number;
}) {
  const {cx, cy, payload, groupKey, color, r = 4} = props;
  if (cx == null || cy == null || payload == null) return null;
  const isPartial = (payload[`${groupKey}_partial`] as number) === 1;
  return isPartial
    ? <circle cx={cx} cy={cy} r={r} stroke={color} strokeWidth={2} fill="white"/>
    : <circle cx={cx} cy={cy} r={r} fill={color}/>;
}

function ConvergenceTooltip({
                              active, payload, label, pointByGroupRatio, allRatios,
                            }: {
  active?: boolean;
  payload?: Array<{ dataKey?: string | number; value?: number }>;
  label?: string;
  pointByGroupRatio: Map<string, ConvergencePoint>;
  allRatios: number[];
}) {
  if (!active || !payload?.length) return null;
  const visible = payload.filter(p => {
    const k = String(p.dataKey ?? "");
    return !k.endsWith("_gap") && !k.endsWith("_std") && !k.endsWith("_partial");
  });
  if (!visible.length) return null;
  const rawRatio = allRatios.find(r => `${(r * 100).toFixed(0)}%` === label) ?? 0;
  return (
    <div style={{background: "white", border: "1px solid #e2e8f0", borderRadius: 6, padding: "8px 12px", fontSize: 12}}>
      <p style={{margin: "0 0 4px", fontWeight: 600}}>{label}</p>
      {visible.map(entry => {
        const gname = String(entry.dataKey ?? "");
        const pt = pointByGroupRatio.get(`${gname}_${rawRatio}`);
        const val = entry.value;
        if (val == null) return null;
        const std = pt?.convergence_epoch_std;
        const stdStr = std != null && (pt?.n_converged ?? 0) > 1 ? ` ±${std.toFixed(0)}` : "";
        const conv = pt ? `${pt.n_converged}/${pt.n_seeds} seedów` : "";
        return (
          <p key={gname} style={{margin: "2px 0"}}>
            {gname.toUpperCase()}: {val.toFixed(0)}{stdStr} epok ({conv})
          </p>
        );
      })}
    </div>
  );
}

export default function ConvergenceChart({
                                           datasetId, groupBy = "architecture", groupId, availableMetrics = [],
                                         }: Props) {
  // Metryka, po której oceniamy zbieżność (pusty = domyślna dla zadania).
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  const [thresholdInput, setThresholdInput] = useState("");

  const metric = selectedMetric || availableMetrics[0] || undefined;
  const parsedThreshold = thresholdInput.trim() === "" ? undefined : Number(thresholdInput);
  const threshold = parsedThreshold != null && Number.isFinite(parsedThreshold) ? parsedThreshold : undefined;

  const {data, isLoading, isError} = useConvergence(datasetId, groupBy, groupId, metric, threshold);

  const downloadPng = () => {
    const params = new URLSearchParams({dataset_id: datasetId, group_by: groupBy});
    if (groupId) params.set("group_id", groupId);
    if (metric) params.set("metric", metric);
    if (threshold != null) params.set("metric_threshold", String(threshold));
    void downloadFile(`/api/results/convergence.png?${params}`, `convergence_${datasetId}.png`);
  };

  const effectiveThreshold = data?.threshold ?? null;
  const thresholdMode = data?.threshold_mode ?? null;
  const effectiveMetric = data?.metric ?? null;

  const controls = (
    <div className="flex flex-wrap items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 text-xs">
      <span className="text-muted-foreground">Metryka zbieżności:</span>
      <Select value={metric ?? ""} onValueChange={setSelectedMetric}>
        <SelectTrigger className="h-6 w-40 text-xs">
          <SelectValue placeholder={effectiveMetric ?? "domyślna"}/>
        </SelectTrigger>
        <SelectContent>
          {availableMetrics.map((m) => (
            <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <label className="flex items-center gap-1.5">
        <span className="text-muted-foreground">
          Próg{thresholdMode ? ` (${effectiveMetric ?? "metryka"} ${thresholdMode === "min" ? "<" : ">"})` : ""}:
        </span>
        <input
          type="number"
          step="any"
          value={thresholdInput}
          onChange={(e) => setThresholdInput(e.target.value)}
          placeholder={effectiveThreshold != null ? String(effectiveThreshold) : "auto"}
          className="h-6 w-24 rounded border bg-background px-1.5 font-mono"
        />
        {thresholdInput.trim() === "" && (
          <span className="text-muted-foreground">(domyślny)</span>
        )}
      </label>
    </div>
  );

  if (isLoading) {
    return (
      <div className="space-y-4">
        {controls}
        <p className="text-sm text-muted-foreground animate-pulse">Ładowanie...</p>
      </div>
    );
  }
  if (isError) {
    return (
      <div className="space-y-4">
        {controls}
        <p className="text-sm text-destructive">Błąd pobierania danych zbieżności.</p>
      </div>
    );
  }
  if (!data || !data.points.length) {
    return (
      <div className="space-y-4">
        {controls}
        <p className="text-sm text-muted-foreground text-center py-6">
          Brak danych zbieżności dla wybranego datasetu.
        </p>
      </div>
    );
  }

  // Group points by group_by value
  const groups = new Map<string, ConvergencePoint[]>();
  for (const pt of data.points) {
    const key = String(pt[groupBy] ?? "");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(pt);
  }

  const groupNames = [...groups.keys()].sort();
  const allRatios = [...new Set(data.points.map((p) => p.k2_ratio))].sort((a, b) => a - b);
  const pointByGroupRatio = new Map<string, ConvergencePoint>();
  for (const pt of data.points) {
    pointByGroupRatio.set(`${String(pt[groupBy])}_${pt.k2_ratio}`, pt);
  }

  const k2ForRatio = new Map<number, number>();
  for (const pt of data.points) {
    if (!k2ForRatio.has(pt.k2_ratio)) k2ForRatio.set(pt.k2_ratio, pt.k2);
  }

  const chartData = allRatios.map((ratio) => {
    const row: Record<string, number | string | null> = {
      k2_ratio: `${(ratio * 100).toFixed(0)}%`,
    };
    for (const gname of groupNames) {
      const pt = pointByGroupRatio.get(`${gname}_${ratio}`);
      if (pt && pt.n_converged > 0) {
        row[gname] = pt.convergence_epoch_mean;
        row[`${gname}_gap`] = pt.convergence_epoch_mean;
        row[`${gname}_std`] = pt.convergence_epoch_std;
        row[`${gname}_partial`] = pt.n_converged < pt.n_seeds ? 1 : 0;
      } else {
        row[gname] = null;
        row[`${gname}_gap`] = null;
      }
    }
    return row;
  });

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <p className="text-xs text-muted-foreground max-w-xl">
          Niżej = szybsza zbieżność. Pusty okrąg = nie wszystkie seedy zbiegły.
          Brak punktu = żaden seed nie zbiegł.
        </p>
        <button
          onClick={downloadPng}
          className="rounded border px-2 py-0.5 text-xs hover:bg-muted shrink-0"
          title="Pobierz PNG"
        >
          PNG
        </button>
      </div>

      {controls}

      <div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{top: 36, right: 30, bottom: 36, left: 20}}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>
            <XAxis
              dataKey="k2_ratio"
              tick={{fontSize: 13}}
              label={{value: "k₂ / T", position: "insideBottom", offset: -20, fontSize: 14}}
            />
            <YAxis
              tick={{fontSize: 13}}
              label={{
                value: "epoka zbieżności",
                angle: -90,
                position: "insideLeft",
                fontSize: 14,
                offset: 15,
              }}
            />
            <Tooltip
              content={(props) => (
                <ConvergenceTooltip
                  {...props}
                  pointByGroupRatio={pointByGroupRatio}
                  allRatios={allRatios}
                />
              )}
            />
            <Legend verticalAlign="top" wrapperStyle={{fontSize: 13}}/>

            {groupNames.map((g, idx) => (
              <Line
                key={`${g}_gap`}
                type="monotone"
                dataKey={`${g}_gap`}
                stroke={GROUP_COLORS[g] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length]}
                strokeWidth={2}
                strokeDasharray="5 4"
                connectNulls={true}
                dot={false}
                activeDot={false}
                legendType="none"
                isAnimationActive={false}
              />
            ))}

            {groupNames.map((g, idx) => {
              const color = GROUP_COLORS[g] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
              return (
                <Line
                  key={g}
                  type="monotone"
                  dataKey={g}
                  stroke={color}
                  strokeWidth={2.5}
                  name={g.toUpperCase()}
                  connectNulls={false}
                  dot={(dotProps) => (
                    <CustomDot
                      key={`dot-${g}-${dotProps.index}`}
                      cx={dotProps.cx}
                      cy={dotProps.cy}
                      payload={dotProps.payload as Record<string, unknown>}
                      groupKey={g}
                      color={color}
                    />
                  )}
                >
                  <ErrorBar
                    dataKey={`${g}_std`}
                    width={5}
                    strokeWidth={2}
                    stroke={color}
                    direction="y"
                  />
                </Line>
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-auto rounded-md border text-sm">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="whitespace-nowrap">k₂</TableHead>
              <TableHead className="whitespace-nowrap">k₂/T</TableHead>
              {groupNames.map((g) => (
                <TableHead key={`h-${g}-e`} className="whitespace-nowrap">
                  {g.toUpperCase()} epoka ± std
                </TableHead>
              ))}
              {groupNames.map((g) => (
                <TableHead key={`h-${g}-n`} className="whitespace-nowrap text-center">
                  n {g.toUpperCase()}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {allRatios.map((ratio) => (
              <TableRow key={ratio}>
                <TableCell className="font-mono">{k2ForRatio.get(ratio) ?? "?"}</TableCell>
                <TableCell className="font-mono">{(ratio * 100).toFixed(1)}%</TableCell>

                {groupNames.map((g) => {
                  const pt = pointByGroupRatio.get(`${g}_${ratio}`);
                  const val =
                    pt && pt.n_converged > 0
                      ? pt.n_converged > 1 && pt.convergence_epoch_std != null
                        ? `${pt.convergence_epoch_mean!.toFixed(1)} ± ${pt.convergence_epoch_std.toFixed(1)}`
                        : pt.convergence_epoch_mean!.toFixed(1)
                      : "-";
                  return <TableCell key={`${ratio}-${g}-e`} className="font-mono">{val}</TableCell>;
                })}

                {groupNames.map((g) => {
                  const pt = pointByGroupRatio.get(`${g}_${ratio}`);
                  const nStr = pt
                    ? pt.n_converged === 0 ? `0/${pt.n_seeds}` : String(pt.n_converged)
                    : "-";
                  return (
                    <TableCell key={`${ratio}-${g}-n`} className="font-mono text-center">
                      {nStr}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
