import {useEffect, useMemo, useState, type ReactNode} from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import {useQueryClient} from "@tanstack/react-query";
import {useEpochMetrics, resultKeys} from "@/api/results";
import {useAdditionalMetrics, experimentKeys} from "@/api/experiments";
import {downloadFile} from "@/api/client";
import {useSSE} from "@/hooks/useSSE";
import {Badge} from "@/components/ui/badge";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

interface Props {
  experimentId: string;
  status: string;
  additionalMetricNames?: string[];
}

const EXTRA_COLORS = [
  "#7c3aed", "#059669", "#d97706", "#0284c7",
  "#dc2626", "#9333ea", "#16a34a", "#ca8a04",
];

function MetricChart({name, series, color, experimentId, dropdown}: {
  name: string;
  series: { epoch: number; value: number }[];
  color: string;
  experimentId: string;
  dropdown?: ReactNode;
}) {
  const downloadPng = () => {
    const params = new URLSearchParams({name, color});
    void downloadFile(
      `/api/experiments/${experimentId}/metric.png?${params}`,
      `${experimentId}_${name}.png`,
    );
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="font-mono text-base">{name}</CardTitle>
          <div className="flex items-center gap-2">
            {dropdown}
            <button
              onClick={downloadPng}
              className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
              title="Pobierz PNG"
            >
              PNG
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {series.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">Brak danych.</p>
        ) : (
          <div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={series} margin={{top: 5, right: 20, bottom: 32, left: 10}}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>

                <XAxis dataKey="epoch" tick={{fontSize: 13}}
                       label={{value: "epoka", position: "insideBottom", offset: -14, fontSize: 13}}/>

                <YAxis tick={{fontSize: 13}} width={56}
                       label={{value: name, angle: -90, position: "insideLeft", offset: 15, fontSize: 12}}/>

                <Tooltip contentStyle={{fontSize: 13}} formatter={(v: number) => v.toFixed(4)}/>
                <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2.5} dot={false} name={name}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function TrainingMonitor({
                                          experimentId,
                                          status,
                                          additionalMetricNames = [],
                                        }: Props) {
  const qc = useQueryClient();
  const isLive = status === "running" || status === "pending";

  const {metrics: liveMetrics, status: sseStatus} = useSSE(isLive ? experimentId : null);
  const {data: historicMetrics = []} = useEpochMetrics(isLive ? "" : experimentId);

  useEffect(() => {
    if (sseStatus !== "completed" && sseStatus !== "failed") return;
    qc.invalidateQueries({queryKey: experimentKeys.all});
    qc.invalidateQueries({queryKey: resultKeys.epochs(experimentId)});
  }, [sseStatus, experimentId, qc]);

  const [selectedMetric, setSelectedMetric] = useState<string>(() => additionalMetricNames[0] ?? "");

  const {data: historicAdditional = {}} = useAdditionalMetrics(
    experimentId,
    additionalMetricNames.length > 0 ? additionalMetricNames : undefined,
    0,
    undefined,
    !isLive && additionalMetricNames.length > 0,
  );

  const mainData = isLive ? liveMetrics : historicMetrics;
  const last = mainData.at(-1);

  const liveAdditional = useMemo(() => {
    if (!isLive || additionalMetricNames.length === 0) return {};
    const byMetric = {} as Record<string, { epoch: number; value: number }[]>;
    for (const name of additionalMetricNames) {
      byMetric[name] = liveMetrics
        .filter((m) => m[name] != null)
        .map((m) => ({epoch: m.epoch, value: m[name] as number}));
    }
    return byMetric;
  }, [isLive, liveMetrics, additionalMetricNames]);

  const additionalData = isLive ? liveAdditional : historicAdditional;

  const activeSeries = useMemo(
    () => (additionalData[selectedMetric] ?? []).map((p) => ({epoch: p.epoch, value: p.value})),
    [selectedMetric, additionalData],
  );

  const selectedMetricColor = useMemo(() => {
    const idx = additionalMetricNames.indexOf(selectedMetric);
    return EXTRA_COLORS[idx >= 0 ? idx % EXTRA_COLORS.length : 0];
  }, [selectedMetric, additionalMetricNames]);

  const downloadLossPng = () =>
    void downloadFile(`/api/results/${experimentId}/loss.png`, `${experimentId}_loss.png`);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Monitoring treningu</CardTitle>
            <div className="flex items-center gap-2">
              {isLive && (
                <span className="flex items-center gap-1 text-xs text-green-600 font-medium">
                  <span className="inline-block h-2 w-2 rounded-full bg-green-500 animate-pulse"/>
                  live
                </span>
              )}

              {sseStatus === "completed" && (
                <Badge variant="default">Ukończono</Badge>
              )}

              {sseStatus === "failed" && (
                <Badge variant="destructive">Bład</Badge>
              )}

              {sseStatus === "disconnected" && (
                <Badge variant="outline">Połączenie zerwane</Badge>
              )}

              {mainData.length > 0 && (
                <button
                  onClick={downloadLossPng}
                  className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
                  title="Pobierz PNG"
                >
                  PNG
                </button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {mainData.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {isLive ? "Czekam na start treningu..." : "Brak danych."}
            </p>
          ) : (
            <div className="space-y-3">
              {last && (
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div className="rounded-md border p-2.5">
                    <p className="text-xs text-muted-foreground">Epoka</p>
                    <p className="font-mono font-semibold">{last.epoch}</p>
                  </div>

                  <div className="rounded-md border p-2.5">
                    <p className="text-xs text-muted-foreground">Val loss</p>
                    <p className="font-mono font-semibold">{last.val_loss?.toFixed(4) ?? "-"}</p>
                  </div>

                  <div className="rounded-md border p-2.5">
                    <p className="text-xs text-muted-foreground">GPU</p>
                    <p className="font-mono font-semibold">{last.gpu_memory_mb ? `${last.gpu_memory_mb.toFixed(0)} MB` : "-"}</p>
                  </div>
                </div>
              )}
              <div>
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart data={mainData} margin={{top: 36, right: 20, bottom: 32, left: 10}}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>

                    <XAxis dataKey="epoch" tick={{fontSize: 13}}
                           label={{value: "epoka", position: "insideBottom", offset: -14, fontSize: 13}}/>
                    <YAxis tick={{fontSize: 13}} width={56}
                           label={{value: "strata", angle: -90, position: "insideLeft", offset: 15, fontSize: 13}}/>

                    <Tooltip contentStyle={{fontSize: 13}} formatter={(v: number) => v.toFixed(4)}/>
                    <Legend verticalAlign="top" wrapperStyle={{fontSize: 13}}/>

                    <Line type="monotone" dataKey="train_loss" stroke="#2563eb" strokeWidth={2.5} dot={false}
                          name="Train loss"/>
                    <Line type="monotone" dataKey="val_loss" stroke="#dc2626" strokeWidth={2.5} dot={false}
                          name="Val loss"/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {additionalMetricNames.length > 0 && (
        <MetricChart
          name={selectedMetric}
          series={activeSeries}
          color={selectedMetricColor}
          experimentId={experimentId}
          dropdown={
            additionalMetricNames.length > 1 ? (
              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger className="h-7 w-48 text-xs">
                  <SelectValue/>
                </SelectTrigger>
                <SelectContent>
                  {additionalMetricNames.map((m) => (
                    <SelectItem key={m} value={m} className="text-xs font-mono">
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : undefined
          }
        />
      )}
    </div>
  );
}
