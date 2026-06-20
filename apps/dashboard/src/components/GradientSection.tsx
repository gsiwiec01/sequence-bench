import {useMemo, useState} from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {useEpochMetrics} from "@/api/results";
import {useGradientEpochs, useGradientNorms, useGradientParamTrends} from "@/api/gradients";
import {useExperiment, type Experiment} from "@/api/experiments";
import {useDatasets, type Dataset} from "@/api/datasets";
import GradientHeatmap from "@/components/GradientHeatmap";
import {Badge} from "@/components/ui/badge";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {Label} from "@/components/ui/label";
import {Slider} from "@/components/ui/slider";
import {Button} from "@/components/ui/button";
import {ChevronLeft, ChevronRight} from "lucide-react";
import {downloadFile} from "@/api/client";

interface Props {
  experimentId: string;
  status: string;
}

function paramColor(name: string): string {
  if (name.includes("decoder")) return "#000000";
  if (name.includes("_hh")) return "#0072B2";
  if (name.includes("_ih")) return "#D55E00";
  return "#6b7280";
}

function paramStroke(name: string): string | undefined {
  return name.includes("bias") ? "4 2" : undefined;
}

function buildLabel(
  exp: Pick<Experiment, "architecture" | "k1" | "k2" | "seed" | "dataset_id"> | undefined,
  datasetMap: Map<string, Pick<Dataset, "name">>,
): string {
  if (!exp) return "-";

  const dsName = datasetMap.get(exp.dataset_id)?.name ?? "?";
  return `${dsName} · ${exp.architecture.toUpperCase()} k₁=${exp.k1} k₂=${exp.k2} seed=${exp.seed}`;
}

export default function GradientSection({experimentId, status}: Props) {
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);

  const downloadTrendPng = () =>
    void downloadFile(
      `/api/gradients/${experimentId}/param-trends.png`,
      `${experimentId}_gradient_trend.png`,
    );

  const isLive = status === "running" || status === "pending";

  const {data: epochMetrics = []} = useEpochMetrics(experimentId, isLive ? 5_000 : false);
  const {data: mainExp} = useExperiment(experimentId);
  const {data: epochsData} = useGradientEpochs(experimentId);
  const {data: trends} = useGradientParamTrends(experimentId);
  const {data: datasets = []} = useDatasets();

  const datasetMap = useMemo(
    () => new Map(datasets.map((d) => [d.id, d])),
    [datasets],
  );

  const heatmapEpochs = useMemo(
    () => (epochsData?.epochs ?? []).slice().sort((a, b) => a - b),
    [epochsData],
  );

  const effectiveEpoch = selectedEpoch ?? heatmapEpochs.at(-1) ?? null;
  const heatmapIdx = effectiveEpoch != null ? heatmapEpochs.indexOf(effectiveEpoch) : -1;

  useGradientNorms(experimentId, effectiveEpoch);

  const trendChartData = useMemo(() => {
    if (!trends) return [];

    return trends.epochs.map((epoch, i) => {
      const row: Record<string, number | null | undefined> = {epoch};
      for (const [pname, vals] of Object.entries(trends.params)) {
        const v = vals[i];
        row[pname] = v != null && v > 0 ? v : null;
      }

      return row;
    });
  }, [trends]);

  const trendParamNames = Object.keys(trends?.params ?? {});

  const hasAnyData = trendChartData.length > 0 || heatmapEpochs.length > 0;
  if (!hasAnyData) return null;

  const handleChartClick = (payload: any) => {
    if (heatmapEpochs.length === 0) return;

    const clicked = Number(payload?.activeLabel);
    if (isNaN(clicked)) return;

    const nearest = heatmapEpochs.reduce((p, c) =>
      Math.abs(c - clicked) < Math.abs(p - clicked) ? c : p,
    );
    setSelectedEpoch(nearest);
  };

  const mainLabel = buildLabel(mainExp, datasetMap);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Gradienty</CardTitle>
      </CardHeader>

      <CardContent className="space-y-8">
        {trendChartData.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium">Trend norm gradientów</p>
              <button
                onClick={downloadTrendPng}
                className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
                title="Pobierz PNG"
              >
                PNG
              </button>
            </div>

            {heatmapEpochs.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Kliknięcie na wykresie wybranej epoki synchronizacje heatmapy.
              </p>
            )}

            <div>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart
                  data={trendChartData}
                  onClick={handleChartClick}
                  className={heatmapEpochs.length > 0 ? "cursor-pointer" : undefined}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted"/>

                  <XAxis
                    dataKey="epoch"
                    tick={{fontSize: 13}}
                    label={{value: "epoka", position: "insideBottom", offset: -4, fontSize: 13}}
                  />

                  <YAxis
                    scale="log"
                    domain={["auto", "auto"]}
                    tick={{fontSize: 13}}
                    tickFormatter={(v: number) => (v != null && isFinite(v) ? v.toExponential(0) : "")}
                    width={60}
                    label={{value: "norma gradientu", angle: -90, position: "insideLeft", offset: 10, fontSize: 13}}
                  />

                  <Tooltip
                    contentStyle={{fontSize: 13}}
                    formatter={(v: number, name: string) => [v != null && isFinite(v) ? v.toExponential(3) : "-", name]}
                  />

                  <Legend
                    verticalAlign="bottom"
                    wrapperStyle={{fontSize: 13, top: 305}}
                  />

                  {effectiveEpoch != null && (
                    <ReferenceLine
                      x={effectiveEpoch}
                      stroke="hsl(var(--muted-foreground))"
                      strokeDasharray="4 2"
                    />
                  )}

                  {trendParamNames.map((name) => (
                    <Line
                      key={name}
                      type="monotone"
                      dataKey={name}
                      legendType="plainline"
                      stroke={paramColor(name)}
                      strokeWidth={2}
                      strokeDasharray={paramStroke(name)}
                      dot={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {heatmapEpochs.length > 0 && (
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">
                Wybrana epoka: {effectiveEpoch ?? "-"}
              </Label>

              <div className="flex items-center gap-2">
                <Slider
                  min={0}
                  max={heatmapEpochs.length - 1}
                  step={1}
                  value={[heatmapIdx >= 0 ? heatmapIdx : heatmapEpochs.length - 1]}
                  onValueChange={([v]) => setSelectedEpoch(heatmapEpochs[v])}
                  className="w-40"
                />

                <Button
                  variant="outline"
                  size="icon"
                  disabled={heatmapIdx <= 0}
                  onClick={() => setSelectedEpoch(heatmapEpochs[heatmapIdx - 1])}
                >
                  <ChevronLeft/>
                </Button>

                <Button
                  variant="outline"
                  size="icon"
                  disabled={heatmapIdx >= heatmapEpochs.length - 1}
                  onClick={() => setSelectedEpoch(heatmapEpochs[heatmapIdx + 1])}
                >
                  <ChevronRight/>
                </Button>
              </div>
            </div>
          </div>
        )}

        {heatmapEpochs.length > 0 && effectiveEpoch != null && (
          <GradientHeatmap
            experimentId={experimentId}
            epoch={effectiveEpoch}
          />
        )}
      </CardContent>
    </Card>
  );
}
