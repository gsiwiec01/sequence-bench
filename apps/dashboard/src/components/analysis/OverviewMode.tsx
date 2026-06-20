import { useState } from "react";
import type { Experiment } from "@/api/experiments";
import type { Dataset } from "@/api/datasets";
import type { ExperimentGroup } from "@/api/groups";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ConvergenceChart from "@/components/ConvergenceChart";
import DegradationChart from "@/components/DegradationChart";
import { ChevronRight } from "lucide-react";
import OverviewTable from "./OverviewTable";

export default function OverviewMode({
  experiments,
  datasets,
  groups,
  selectedGroupId,
  selectedDatasetId,
  selectedExperimentIds,
  onToggle,
  onSwitchToCompare,
  activeTab,
  onTabChange,
}: {
  experiments: Experiment[];
  datasets: Dataset[];
  groups: ExperimentGroup[];
  selectedGroupId: string | null;
  selectedDatasetId: string | null;
  selectedExperimentIds: string[];
  onToggle: (id: string) => void;
  onSwitchToCompare: () => void;
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  const [baselineK2, setBaselineK2] = useState<number | undefined>(undefined);
  const [baselineK1, setBaselineK1] = useState<number | undefined>(undefined);

  const completedExps = experiments.filter((e) => e.status === "completed");
  const availableK2 = [...new Set(completedExps.map((e) => e.k2))].sort((a, b) => a - b);
  const availableK1 = [...new Set(completedExps.map((e) => e.k1))].sort((a, b) => a - b);
  const degradationActive = !!(selectedGroupId || selectedDatasetId);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Przegląd grupy</h2>
        <Button
          variant="outline"
          size="sm"
          disabled={selectedExperimentIds.length < 2}
          onClick={onSwitchToCompare}
        >
          Porównaj zaznaczone ({selectedExperimentIds.length})
          <ChevronRight className="ml-1 h-3.5 w-3.5" />
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={onTabChange}>
        <TabsList>
          <TabsTrigger value="table">Tabela wyników</TabsTrigger>
          <TabsTrigger value="degradation">Krzywa degradacji</TabsTrigger>
          <TabsTrigger value="convergence">Szybkość zbieżności</TabsTrigger>
        </TabsList>

        <TabsContent value="table" className="pt-4">
          <Card>
            <CardContent className="pt-4">
              <OverviewTable
                experiments={experiments}
                datasets={datasets}
                selectedIds={selectedExperimentIds}
                onToggle={onToggle}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="degradation" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Krzywa degradacji</CardTitle>
            </CardHeader>

            <CardContent className="space-y-4">
              {degradationActive ? (
                <>
                  {selectedDatasetId && (availableK2.length > 0 || availableK1.length > 0) && (
                    <div className="flex items-end gap-4 flex-wrap">
                      {availableK1.length > 1 && (
                        <div className="w-48 space-y-1.5">
                          <Label>Baseline k₁</Label>

                          <Select
                            value={baselineK1 !== undefined ? String(baselineK1) : "_all"}
                            onValueChange={(v) => setBaselineK1(v === "_all" ? undefined : +v)}
                          >
                            <SelectTrigger><SelectValue /></SelectTrigger>

                            <SelectContent>
                              <SelectItem value="_all">Wszystkie k₁</SelectItem>
                              {availableK1.map((k) => (
                                <SelectItem key={k} value={String(k)}>k₁ = {k}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}

                      {availableK2.length > 0 && (
                        <div className="w-48 space-y-1.5">
                          <Label>Baseline k₂</Label>

                          <Select
                            value={baselineK2 !== undefined ? String(baselineK2) : "_max"}
                            onValueChange={(v) => setBaselineK2(v === "_max" ? undefined : +v)}
                          >
                            <SelectTrigger><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="_max">Max k₂ (BPTT)</SelectItem>
                              {availableK2.map((k) => (
                                <SelectItem key={k} value={String(k)}>k₂ = {k}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </div>
                  )}

                  <DegradationChart
                    datasetId={selectedDatasetId ?? undefined}
                    groupId={selectedGroupId ?? undefined}
                    baselineK2={baselineK2}
                    baselineK1={baselineK1}
                    availableMetrics={completedExps[0]?.additional_metrics}
                  />
                </>
              ) : (
                <p className="py-6 text-center text-sm text-muted-foreground">
                  Wybierz grupę lub dataset w panelu po lewej.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="convergence" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Szybkość zbieżności</CardTitle>
            </CardHeader>

            <CardContent>
              {selectedDatasetId ? (
                <ConvergenceChart
                  datasetId={selectedDatasetId}
                  groupBy="architecture"
                  groupId={selectedGroupId}
                  availableMetrics={completedExps[0]?.additional_metrics}
                />
              ) : (
                <p className="py-6 text-center text-sm text-muted-foreground">
                  Wybierz dataset w panelu po lewej.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
