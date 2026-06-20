import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { request } from "@/api/client";
import { experimentKeys } from "@/api/experiments";
import type { Experiment, AdditionalMetricSeries } from "@/api/experiments";
import { useDatasets } from "@/api/datasets";

interface Props {
  experiments: Experiment[];
  onDelete?: (id: string) => void;
  onCancel?: (id: string) => void;
}

type SortKey = keyof Experiment;

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  completed: "default", running: "secondary", pending: "outline",
  failed: "destructive", cancelled: "outline",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "oczekuje",
  running: "trwa",
  completed: "ukończone",
  failed: "błąd",
  cancelled: "anulowane",
};

const MIN_METRICS = new Set(["mse", "mae", "mape", "perplexity", "cross_entropy", "loss"]);

function bestVal(series: { epoch: number; value: number }[], name: string): number | null {
  const vals = series.map((p) => p.value);
  if (!vals.length) return null;
  return MIN_METRICS.has(name) ? Math.min(...vals) : Math.max(...vals);
}

export default function ExperimentTable({ experiments, onDelete, onCancel }: Props) {
  const { data: datasets } = useDatasets();
  const datasetNames = useMemo(
    () => Object.fromEntries((datasets ?? []).map((d) => [d.id, d.name])),
    [datasets],
  );

  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const [filterArch, setFilterArch] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const [showExtraMetrics, setShowExtraMetrics] = useState(false);

  const uniqueArchitectures = useMemo(
    () => [...new Set(experiments.map((e) => e.architecture))].sort(),
    [experiments],
  );
  const [search, setSearch] = useState("");

  const toggle = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
    else { setSortKey(key); setSortDir(1); }
  };

  const filtered = experiments
    .filter((e) => filterArch === "all" || e.architecture === filterArch)
    .filter((e) => filterStatus === "all" || e.status === filterStatus)
    .filter((e) => !search || e.id.startsWith(search))
    .sort((a, b) => {
      const av = a[sortKey] ?? "";
      const bv = b[sortKey] ?? "";
      return sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
    });

  const extraMetricNames = filtered[0]?.additional_metrics ?? [];
  const extraMetricQueries = useQueries({
    queries: filtered.map((exp) => ({
      queryKey: experimentKeys.metrics(exp.id, extraMetricNames.join(","), 0, undefined),
      queryFn: () =>
        request<AdditionalMetricSeries>(
          `/api/experiments/${exp.id}/metrics?names=${extraMetricNames.join(",")}`,
        ),
      enabled: showExtraMetrics && extraMetricNames.length > 0,
      staleTime: 300_000,
    })),
  });

  const extraBests = useMemo(() => {
    if (!showExtraMetrics) return {};

    const bests = {} as Record<string, Record<string, number | null>>;
    filtered.forEach((exp, i) => {
      bests[exp.id] = {};
      const data = extraMetricQueries[i]?.data ?? {};
      for (const name of extraMetricNames) {
        bests[exp.id][name] = bestVal(data[name] ?? [], name);
      }
    });

    return bests;
  }, [showExtraMetrics, filtered, extraMetricQueries, extraMetricNames]);

  const SortableHead = ({ k, label }: { k: SortKey; label: string }) => (
    <TableHead onClick={() => toggle(k)}
               className="cursor-pointer select-none hover:bg-muted/50 whitespace-nowrap">
      {label}
      {sortKey === k && (
        <span className={cn("ml-1 text-muted-foreground")}>{sortDir === 1 ? "↑" : "↓"}</span>
      )}
    </TableHead>
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input placeholder="Szukaj po ID..." value={search}
               onChange={(e) => setSearch(e.target.value)} className="h-8 w-48" />

        <Select value={filterArch} onValueChange={setFilterArch}>
          <SelectTrigger className="h-8 w-40"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Wszystkie arch.</SelectItem>
            {uniqueArchitectures.map((arch) => (
              <SelectItem key={arch} value={arch}>{arch.toUpperCase()}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="h-8 w-44"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Wszystkie statusy</SelectItem>
            {["pending", "running", "completed", "failed", "cancelled"].map((s) => (
              <SelectItem key={s} value={s}>{STATUS_LABEL[s] ?? s}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <span className="text-xs text-muted-foreground ml-auto">{filtered.length} wyników</span>

        {extraMetricNames.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs"
            onClick={() => setShowExtraMetrics((v) => !v)}
          >
            {showExtraMetrics ? "Ukryj metryki" : "Pokaż więcej metryk"}
          </Button>
        )}
      </div>

      <div className="rounded-md border overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <SortableHead k="id" label="ID" />
              <SortableHead k="dataset_id" label="Dataset" />
              <SortableHead k="architecture" label="Arch" />
              <SortableHead k="k1" label="k₁" />
              <SortableHead k="k2" label="k₂" />
              <SortableHead k="seed" label="Seed" />
              <SortableHead k="status" label="Status" />
              <SortableHead k="best_metric" label="Metryka" />

              {showExtraMetrics && extraMetricNames.map((name) => (
                <TableHead key={name} className="whitespace-nowrap font-mono text-xs">{name}</TableHead>
              ))}

              <TableHead></TableHead>
            </TableRow>
          </TableHeader>

          <TableBody>
            {filtered.map((e) => (
              <TableRow key={e.id}>
                <TableCell className="font-mono text-xs">
                  <Link to="/experiments/$id" params={{ id: e.id }}
                        className="hover:underline">{e.id.slice(0, 8)}</Link>
                </TableCell>
                <TableCell className="text-xs max-w-[10rem] truncate" title={datasetNames[e.dataset_id] ?? e.dataset_id}>
                  {datasetNames[e.dataset_id] ?? e.dataset_id.slice(0, 8)}
                </TableCell>
                <TableCell className="uppercase font-medium">{e.architecture}</TableCell>
                <TableCell>{e.k1}</TableCell>
                <TableCell>{e.k2}</TableCell>
                <TableCell>{e.seed}</TableCell>

                <TableCell>
                  <Badge variant={STATUS_VARIANT[e.status] ?? "outline"}>{STATUS_LABEL[e.status] ?? e.status}</Badge>
                </TableCell>

                <TableCell>
                  {e.best_metric != null ? (
                    <span className="font-mono">{e.best_metric.toFixed(4)}</span>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                  <span className="ml-1 text-[10px] text-muted-foreground">{e.early_stopping_metric}</span>
                </TableCell>

                {showExtraMetrics && extraMetricNames.map((name) => {
                  const val = extraBests[e.id]?.[name];
                  const loading = extraMetricQueries[filtered.indexOf(e)]?.isLoading;
                  return (
                    <TableCell key={name} className="font-mono text-xs">
                      {loading ? "…" : val != null ? val.toFixed(4) : "-"}
                    </TableCell>
                  );
                })}

                <TableCell className="text-right space-x-1">
                  {e.status === "running" && onCancel && (
                    <Button size="sm" variant="outline" onClick={() => onCancel(e.id)}>Zatrzymaj</Button>
                  )}
                  {onDelete && (
                    <Button size="sm" variant="ghost" onClick={() => onDelete(e.id)}>Usuń</Button>
                  )}
                </TableCell>
              </TableRow>
            ))}

            {filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={9 + (showExtraMetrics ? extraMetricNames.length : 0)}
                           className="text-center text-muted-foreground py-8">
                  Brak wyników
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
