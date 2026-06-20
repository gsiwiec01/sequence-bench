import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQueries } from "@tanstack/react-query";
import { useExperiments, experimentKeys } from "@/api/experiments";
import type { Experiment, AdditionalMetricSeries } from "@/api/experiments";
import { useDatasets } from "@/api/datasets";
import { useGroups } from "@/api/groups";
import { request } from "@/api/client";
import { gradientKeys } from "@/api/gradients";
import type { GradientParamTrends } from "@/api/gradients";
import { resultKeys } from "@/api/results";
import type { EpochMetric } from "@/api/results";
import { cn } from "@/lib/utils";
import { ChevronRight, Menu } from "lucide-react";
import CompareMode from "@/components/analysis/CompareMode";
import LeftPanel from "@/components/analysis/LeftPanel";
import OverviewMode from "@/components/analysis/OverviewMode";
import { COLORS, MAX_SELECTED, MIN_METRICS } from "@/components/analysis/constants";
import type { ActiveMode, GradientHealth } from "@/components/analysis/constants";

export const Route = createFileRoute("/analysis")({
  component: RouteComponent,
});

function RouteComponent() {
  const [activeMode, setActiveMode] = useState<ActiveMode>("overview");
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedExperimentIds, setSelectedExperimentIds] = useState<string[]>([]);
  const [overviewTab, setOverviewTab] = useState<string>("table");
  const [compareTab, setCompareTab] = useState<string>("curves");
  const [panelOpen, setPanelOpen] = useState(true);

  const { data: groups = [] } = useGroups();
  const { data: datasets = [] } = useDatasets();
  const { data: allExperiments = [] } = useExperiments();

  const experimentParams = selectedGroupId
    ? { group_id: selectedGroupId }
    : selectedDatasetId
      ? { dataset_id: selectedDatasetId }
      : undefined;

  const { data: filteredExperiments = [] } = useExperiments(experimentParams);

  function pickGroup(groupId: string | null) {
    setSelectedGroupId(groupId);
    if (groupId) {
      const group = groups.find((g) => g.id === groupId);
      if (group?.dataset) {
        const ds = datasets.find((d) => d.name === group.dataset);
        if (ds) setSelectedDatasetId(ds.id);
      }
    }
    setSelectedExperimentIds([]);
  }

  function pickDataset(datasetId: string | null) {
    setSelectedDatasetId(datasetId);
    setSelectedExperimentIds([]);
  }

  function switchMode(mode: ActiveMode) {
    setActiveMode(mode);
    if (mode === "overview") setOverviewTab("table");
    else if (mode === "compare") setCompareTab("curves");
  }

  function toggle(id: string) {
    setSelectedExperimentIds((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < MAX_SELECTED
          ? [...prev, id]
          : prev,
    );
  }

  function selectAll() {
    const toAdd = filteredExperiments
      .filter((e) => !selectedExperimentIds.includes(e.id))
      .slice(0, MAX_SELECTED - selectedExperimentIds.length)
      .map((e) => e.id);
    setSelectedExperimentIds((prev) => [...prev, ...toAdd].slice(0, MAX_SELECTED));
  }

  const selected = useMemo(
    () =>
      selectedExperimentIds
        .map((id, i) => ({ exp: allExperiments.find((e) => e.id === id)!, color: COLORS[i] }))
        .filter((x) => !!x.exp),
    [selectedExperimentIds, allExperiments],
  );

  const epochQueries = useQueries({
    queries: selectedExperimentIds.map((id) => ({
      queryKey: resultKeys.epochs(id),
      queryFn: () => request<EpochMetric[]>(`/api/results/${id}/epochs`),
    })),
  });

  const epochData = useMemo(() => {
    const byId = {} as Record<string, EpochMetric[]>;
    selectedExperimentIds.forEach((id, i) => {
      const d = epochQueries[i]?.data;
      if (d) byId[id] = d;
    });
    return byId;
  }, [selectedExperimentIds, epochQueries]);

  const curvesLoading = epochQueries.some((q) => q.isLoading);

  const paramTrendQueries = useQueries({
    queries: selectedExperimentIds.map((id) => ({
      queryKey: gradientKeys.paramTrends(id),
      queryFn: () => request<GradientParamTrends>(`/api/gradients/${id}/param-trends`),
    })),
  });

  const paramTrendData = useMemo(() => {
    const byId = {} as Record<string, GradientParamTrends>;
    selectedExperimentIds.forEach((id, i) => {
      const d = paramTrendQueries[i]?.data;
      if (d) byId[id] = d;
    });
    return byId;
  }, [selectedExperimentIds, paramTrendQueries]);

  const allAdditionalMetrics = useMemo(
    () => [...new Set(selected.flatMap(({ exp }) => exp.additional_metrics ?? []))].sort(),
    [selected],
  );

  const additionalQueries = useQueries({
    queries: selected.map(({ exp }) => {
      const names = exp.additional_metrics ?? [];
      return {
        queryKey: experimentKeys.metrics(exp.id, names.join(","), 0, undefined),
        queryFn: () =>
          request<AdditionalMetricSeries>(`/api/experiments/${exp.id}/metrics?names=${names.join(",")}`),
        enabled: names.length > 0,
      };
    }),
  });

  const additionalData = useMemo(() => {
    const byId = {} as Record<string, AdditionalMetricSeries>;
    selected.forEach(({ exp }, i) => {
      const d = additionalQueries[i]?.data;
      if (d) byId[exp.id] = d;
    });
    return byId;
  }, [selected, additionalQueries]);

  const additionalBests = useMemo(() => {
    const bests = {} as Record<string, Record<string, number | null>>;
    for (const { exp } of selected) {
      bests[exp.id] = {};
      for (const name of allAdditionalMetrics) {
        const series = additionalData[exp.id]?.[name] ?? [];
        const vals = series.map((p) => p.value);
        bests[exp.id][name] = vals.length > 0
          ? (MIN_METRICS.has(name) ? Math.min(...vals) : Math.max(...vals))
          : null;
      }
    }
    return bests;
  }, [selected, additionalData, allAdditionalMetrics]);

  const gradientHealth = useMemo(() => {
    const health = {} as Record<string, GradientHealth>;
    for (const { exp } of selected) {
      const series = epochData[exp.id] ?? [];
      const lastWithMean = [...series].reverse().find((m) => m.grad_norm_mean != null);
      const lastWithMax = [...series].reverse().find((m) => m.grad_norm_max != null);
      const lastNorm = lastWithMean?.grad_norm_mean ?? null;
      const lastMax = lastWithMax?.grad_norm_max ?? null;
      let status: GradientHealth["status"] = null;
      if (lastNorm != null) {
        if (lastNorm < 1e-5) status = "vanishing";
        else if (lastMax != null && lastMax > 100) status = "exploding";
        else status = "stabilne";
      }
      health[exp.id] = { lastNorm, status };
    }
    return health;
  }, [selected, epochData]);

  const modeLabels: Record<ActiveMode, string> = {
    overview: "Przegląd",
    compare: "Porównanie",
  };

  return (
    <>
      <h1 className="text-2xl font-bold mb-6">Analiza</h1>
      <div className="-mx-6 -mb-8 flex flex-col" style={{ minHeight: "calc(100vh - 52px - 64px)" }}>
        <div className="border-b bg-card/50 px-6 py-2.5 flex items-center gap-1 shrink-0">
          <button
            onClick={() => setPanelOpen((v) => !v)}
            className="mr-2 rounded-md p-1.5 text-muted-foreground hover:bg-muted md:hidden"
            aria-label="Toggle panel"
          >
            <Menu className="h-4 w-4" />
          </button>
          {(["overview", "compare"] as ActiveMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => switchMode(mode)}
              className={cn(
                "px-4 py-1.5 text-sm rounded-md transition-colors",
                activeMode === mode
                  ? "bg-primary text-primary-foreground font-medium"
                  : "text-muted-foreground hover:bg-muted",
              )}
            >
              {modeLabels[mode]}
            </button>
          ))}
        </div>

        <div className="flex flex-1 overflow-hidden">
          <aside
            className={cn(
              "shrink-0 border-r bg-card overflow-y-auto transition-all",
              panelOpen ? "w-[280px]" : "w-0 overflow-hidden",
            )}
          >
            <LeftPanel
              groups={groups}
              datasets={datasets}
              experiments={filteredExperiments}
              selectedGroupId={selectedGroupId}
              selectedDatasetId={selectedDatasetId}
              selectedExperimentIds={selectedExperimentIds}
              onGroupChange={pickGroup}
              onDatasetChange={pickDataset}
              onToggle={toggle}
              onSelectAll={selectAll}
              onDeselectAll={() => setSelectedExperimentIds([])}
            />
          </aside>

          <button
            onClick={() => setPanelOpen((v) => !v)}
            className="hidden md:flex items-center justify-center w-4 shrink-0 border-r bg-muted/30 hover:bg-muted text-muted-foreground transition-colors"
            title={panelOpen ? "Zwiń panel" : "Rozwiń panel"}
          >
            <ChevronRight className={cn("h-3 w-3 transition-transform", panelOpen && "rotate-180")} />
          </button>

          <main className="flex-1 overflow-y-auto p-6 min-w-0">
            {activeMode === "overview" && (
              <OverviewMode
                experiments={filteredExperiments}
                datasets={datasets}
                groups={groups}
                selectedGroupId={selectedGroupId}
                selectedDatasetId={selectedDatasetId}
                selectedExperimentIds={selectedExperimentIds}
                onToggle={toggle}
                onSwitchToCompare={() => switchMode("compare")}
                activeTab={overviewTab}
                onTabChange={setOverviewTab}
              />
            )}
            {activeMode === "compare" && (
              <CompareMode
                selected={selected}
                epochData={epochData}
                paramTrendData={paramTrendData}
                gradientHealth={gradientHealth}
                additionalMetrics={allAdditionalMetrics}
                additionalBests={additionalBests}
                additionalData={additionalData}
                curvesLoading={curvesLoading}
                activeTab={compareTab}
                onTabChange={setCompareTab}
              />
            )}
          </main>
        </div>
      </div>
    </>
  );
}
