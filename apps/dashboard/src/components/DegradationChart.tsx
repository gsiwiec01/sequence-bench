import {useState} from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ReferenceLine, ErrorBar, ResponsiveContainer,
} from "recharts";
import {useDegradationCurves} from "@/api/results";
import {useGroupDegradationCurves} from "@/api/groups";
import {downloadFile} from "@/api/client";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

interface Props {
  datasetId?: string;
  groupId?: string;
  baselineK2?: number;
  baselineK1?: number;
  availableMetrics?: string[];
}

const GROUP_COLORS: Record<string, string> = {
  lstm: "#2563eb",
  gru: "#dc2626",
};
const FALLBACK_COLORS = ["#16a34a", "#ca8a04", "#9333ea", "#0891b2"];

const findIdx = (ratios: number[], target: number) =>
  ratios.findIndex((r) => Math.abs(r - target) < 1e-6);

type GroupPoint = { delta_mean: number; delta_std: number; n: number };

export default function DegradationChart({datasetId, groupId, baselineK2, baselineK1, availableMetrics = []}: Props) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [onlyConverged, setOnlyConverged] = useState(false);

  const effectiveMetric = selectedMetric ?? availableMetrics[0];

  const downloadPng = () => {
    const params = new URLSearchParams({group_by: "architecture"});
    if (baselineK2 !== undefined) params.set("baseline_k2", String(baselineK2));
    if (baselineK1 !== undefined) params.set("baseline_k1", String(baselineK1));
    if (effectiveMetric) params.set("metric", effectiveMetric);
    if (onlyConverged) params.set("only_converged", "true");

    let path: string;
    if (groupId) {
      path = `/api/groups/${groupId}/degradation.png?${params}`;
    } else {
      params.set("dataset_id", datasetId ?? "");
      path = `/api/results/degradation.png?${params}`;
    }

    void downloadFile(path, "degradation.png");
  };

  const dsQuery = useDegradationCurves(datasetId ?? "", "architecture", baselineK2, effectiveMetric, baselineK1, onlyConverged);
  const grpQuery = useGroupDegradationCurves(groupId ?? "", "architecture", baselineK2, effectiveMetric, baselineK1, onlyConverged);
  const {data, isLoading, isError, error} = groupId ? grpQuery : dsQuery;

  if (isLoading) return <p className="text-sm text-muted-foreground animate-pulse">Ładowanie...</p>;
  if (isError) {
    return (
      <p className="text-sm text-destructive py-6 text-center">
        Błąd pobierania danych: {error instanceof Error ? error.message : "nieznany błąd"}
      </p>
    );
  }
  if (!data) return null;

  const {groups: groupsData, meta} = data;

  const controls = (
    <div className="flex flex-wrap items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 text-xs">
      {availableMetrics.length > 0 && (
        <>
          <span className="text-muted-foreground">Metryka:</span>
          <Select value={effectiveMetric} onValueChange={setSelectedMetric}>
            <SelectTrigger className="h-6 w-44 text-xs">
              <SelectValue/>
            </SelectTrigger>
            <SelectContent>
              {availableMetrics.map((m) => (
                <SelectItem key={m} value={m} className="text-xs font-mono">{m}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </>
      )}

      <label className="flex items-center gap-1.5 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={onlyConverged}
          onChange={(e) => setOnlyConverged(e.target.checked)}
          className="accent-primary"
        />
        <span className="text-muted-foreground">Tylko zbieżne</span>
      </label>
    </div>
  );

  if (!groupsData || !Object.keys(groupsData).length) {
    return (
      <div className="space-y-3 py-2">
        {controls}
        <p className="text-sm text-muted-foreground text-center">
          Brak ukończonych eksperymentów z ustawioną metryką.
        </p>
      </div>
    );
  }

  const groupNames = Object.keys(groupsData).sort();

  const allRatiosSet = new Set<number>();
  for (const g of groupNames) groupsData[g].k2_ratios.forEach((r) => allRatiosSet.add(r));
  const allRatios = [...allRatiosSet].sort((a, b) => a - b);

  const k2ForRatio = new Map<number, number>();
  for (const g of groupNames) {
    groupsData[g].k2_ratios.forEach((r, i) => {
      if (!k2ForRatio.has(r)) k2ForRatio.set(r, groupsData[g].k2_values[i]);
    });
  }

  const chartData = allRatios.map((ratio) => {
    const entry: Record<string, number | string | undefined> = {
      k2_ratio: `${(ratio * 100).toFixed(0)}%`,
    };

    for (const g of groupNames) {
      const gd = groupsData[g];
      const idx = findIdx(gd.k2_ratios, ratio);
      if (idx !== -1) {
        entry[g] = gd.delta_mean[idx];
        entry[`${g}_std`] = gd.delta_std[idx];
      }
    }

    return entry;
  });

  const tableRows = allRatios.map((ratio) => ({
    ratio,
    k2: k2ForRatio.get(ratio),
    pts: Object.fromEntries(
      groupNames.map((g) => {
        const gd = groupsData[g];
        const idx = findIdx(gd.k2_ratios, ratio);
        return [
          g,
          idx !== -1
            ? ({delta_mean: gd.delta_mean[idx], delta_std: gd.delta_std[idx], n: gd.n_per_ratio[idx]} as GroupPoint)
            : undefined,
        ];
      })
    ) as Record<string, GroupPoint | undefined>,
  }));

  return (
    <div className="space-y-4">
      {controls}

      <div className="flex items-center justify-end">
        <button
          onClick={downloadPng}
          className="rounded border px-2 py-0.5 text-xs hover:bg-muted shrink-0 ml-2"
        >
          PNG
        </button>
      </div>

      <div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{top: 36, right: 30, bottom: 36, left: 10}}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>
            <XAxis
              dataKey="k2_ratio"
              tick={{fontSize: 13}}
              label={{value: "k₂ / T", position: "insideBottom", offset: -20, fontSize: 14}}
            />
            <YAxis
              tick={{fontSize: 13}}
              label={{value: "δ(k₂)", angle: -90, position: "insideLeft", offset: 15, fontSize: 14}}
            />

            <Tooltip contentStyle={{fontSize: 13}} formatter={(v: number) => v.toFixed(4)}/>
            <Legend verticalAlign="top" wrapperStyle={{fontSize: 13}}/>

            <ReferenceLine
              y={1}
              stroke="#9ca3af"
              strokeDasharray="4 2"
              label={{value: "Baseline", fontSize: 10, fill: "#9ca3af"}}
            />
            {groupNames.map((g, idx) => {
              const color = GROUP_COLORS[g] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
              return (
                <Line
                  key={g}
                  type="monotone"
                  dataKey={g}
                  stroke={color}
                  strokeWidth={2.5}
                  dot={{r: 5, fill: color}}
                  name={g.toUpperCase()}
                  connectNulls={false}
                >
                  <ErrorBar dataKey={`${g}_std`} width={5} strokeWidth={2} stroke={color} direction="y"/>
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
                <TableHead key={`h-${g}-ds`} className="whitespace-nowrap">{g.toUpperCase()} śr. δ ± SD</TableHead>
              ))}

              {groupNames.map((g) => (
                <TableHead key={`h-${g}-n`} className="whitespace-nowrap text-center">
                  Liczba przebiegów {g.toUpperCase()}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>

          <TableBody>
            {tableRows.map(({ratio, k2, pts}) => (
              <TableRow key={ratio}>
                <TableCell className="font-mono">{k2 ?? "?"}</TableCell>
                <TableCell className="font-mono">{(ratio * 100).toFixed(1)}%</TableCell>

                {groupNames.map((g) => {
                  const pt = pts[g];
                  const val = pt
                    ? pt.n > 1
                      ? `${pt.delta_mean.toFixed(4)} ± ${pt.delta_std.toFixed(4)}`
                      : pt.delta_mean.toFixed(4)
                    : "-";
                  return <TableCell key={`${ratio}-${g}-ds`} className="font-mono">{val}</TableCell>;
                })}

                {groupNames.map((g) => (
                  <TableCell key={`${ratio}-${g}-n`} className="font-mono text-center">
                    {pts[g] ? pts[g]!.n : "-"}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
