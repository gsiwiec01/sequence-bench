import type { Experiment, AdditionalMetricSeries } from "@/api/experiments";
import type { GradientParamTrends } from "@/api/gradients";
import type { EpochMetric } from "@/api/results";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import GradientCurves from "./GradientCurves";
import LearningCurves from "./LearningCurves";
import MetricsTable from "./MetricsTable";
import WeightTrajectoryTab from "./WeightTrajectoryTab";
import type { GradientHealth } from "./constants";

export default function CompareMode({
  selected,
  epochData,
  paramTrendData,
  gradientHealth,
  additionalMetrics,
  additionalBests,
  additionalData,
  curvesLoading,
  activeTab,
  onTabChange,
}: {
  selected: { exp: Experiment; color: string }[];
  epochData: Record<string, EpochMetric[]>;
  paramTrendData: Record<string, GradientParamTrends>;
  gradientHealth: Record<string, GradientHealth>;
  additionalMetrics: string[];
  additionalBests: Record<string, Record<string, number | null>>;
  additionalData: Record<string, AdditionalMetricSeries>;
  curvesLoading: boolean;
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Porównanie przebiegów</h2>

        {selected.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {selected.map(({ exp, color }) => (
              <span key={exp.id} className="flex items-center gap-1.5 rounded-full border bg-card pl-2.5 pr-2 py-1 text-xs">
                <span className="h-2 w-2 rounded-full shrink-0" style={{ background: color }} />
                <span className="font-medium uppercase">{exp.architecture}</span>
                <span className="text-muted-foreground">k₁={exp.k1} k₂={exp.k2}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      <Card>
        <CardContent className="pt-4">
          <Tabs value={activeTab} onValueChange={onTabChange}>
            <TabsList>
              <TabsTrigger value="curves">Krzywe uczenia</TabsTrigger>
              <TabsTrigger value="gradients">Gradienty</TabsTrigger>
              <TabsTrigger value="metrics">Metryki</TabsTrigger>
              <TabsTrigger value="trajectories">Trajektorie wag</TabsTrigger>
            </TabsList>

            <TabsContent value="curves" className="pt-4">
              <LearningCurves
                selected={selected}
                epochData={epochData}
                additionalData={additionalData}
                loading={curvesLoading}
                additionalMetrics={additionalMetrics}
              />
            </TabsContent>

            <TabsContent value="gradients" className="pt-4">
              <GradientCurves
                selected={selected}
                epochData={epochData}
                loading={curvesLoading}
                gradientHealth={gradientHealth}
                paramTrendData={paramTrendData}
              />
            </TabsContent>

            <TabsContent value="metrics" className="pt-4">
              <MetricsTable
                selected={selected}
                additionalMetrics={additionalMetrics}
                additionalBests={additionalBests}
                gradientHealth={gradientHealth}
              />
            </TabsContent>

            <TabsContent value="trajectories" className="pt-4">
              <WeightTrajectoryTab selected={selected} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
