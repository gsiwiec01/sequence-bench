import { useMemo, useState } from "react";
import { Link } from "@tanstack/react-router";
import type { Experiment } from "@/api/experiments";
import type { Dataset } from "@/api/datasets";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { STATUS_LABEL, STATUS_VARIANT } from "@/lib/status";
import { COLORS, MAX_SELECTED, getSortValue } from "./constants";
import type { ResultSortKey } from "./constants";

export default function OverviewTable({
  experiments,
  datasets,
  selectedIds,
  onToggle,
}: {
  experiments: Experiment[];
  datasets: Dataset[];
  selectedIds: string[];
  onToggle: (id: string) => void;
}) {
  const [sortKey, setSortKey] = useState<ResultSortKey>("created_at");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const [filterArch, setFilterArch] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");

  const uniqueArchitectures = useMemo(
    () => [...new Set(experiments.map((e) => e.architecture))].sort(),
    [experiments],
  );

  const datasetMap = useMemo(
    () => new Map(datasets.map((d) => [d.id, d.name])),
    [datasets],
  );

  const toggle = (key: ResultSortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
    else { setSortKey(key); setSortDir(1); }
  };

  const filtered = useMemo(
    () =>
      experiments
        .filter((e) => filterArch === "all" || e.architecture === filterArch)
        .filter((e) => filterStatus === "all" || e.status === filterStatus)
        .sort((a, b) => {
          const av = getSortValue(a, sortKey);
          const bv = getSortValue(b, sortKey);
          return sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
        }),
    [experiments, filterArch, filterStatus, sortKey, sortDir],
  );

  const SH = ({ k, label }: { k: ResultSortKey; label: string }) => (
    <TableHead
      onClick={() => toggle(k)}
      className="cursor-pointer select-none whitespace-nowrap hover:bg-muted/50"
    >
      {label}
      {sortKey === k && (
        <span className="ml-1 text-muted-foreground">{sortDir === 1 ? "↑" : "↓"}</span>
      )}
    </TableHead>
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={filterArch} onValueChange={setFilterArch}>
          <SelectTrigger className="h-8 w-36">
            <SelectValue />
          </SelectTrigger>

          <SelectContent>
            <SelectItem value="all">Wszystkie arch.</SelectItem>
            {uniqueArchitectures.map((a) => (
              <SelectItem key={a} value={a}>{a.toUpperCase()}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="h-8 w-44">
            <SelectValue />
          </SelectTrigger>

          <SelectContent>
            <SelectItem value="all">Wszystkie statusy</SelectItem>
            {["pending", "running", "completed", "failed", "cancelled"].map((s) => (
              <SelectItem key={s} value={s}>{STATUS_LABEL[s] ?? s}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <span className="ml-auto text-xs text-muted-foreground">{filtered.length} wyników</span>
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8" />
              <TableHead className="w-24">ID</TableHead>
              <TableHead>Dataset</TableHead>

              <SH k="architecture" label="Arch" />
              <SH k="k1" label="k₁" />
              <SH k="k2" label="k₂" />
              <SH k="seed" label="Seed" />
              <SH k="best_metric" label="Najlepsza metryka" />
              <SH k="convergence_epoch" label="Konw." />
              <SH k="total_training_time_s" label="Czas (s)" />
              <SH k="status" label="Status" />
            </TableRow>
          </TableHeader>

          <TableBody>
            {filtered.map((e) => {
              const isSelected = selectedIds.includes(e.id);
              const selIdx = selectedIds.indexOf(e.id);
              const color = selIdx >= 0 ? COLORS[selIdx] : undefined;
              const disabled = !isSelected && selectedIds.length >= MAX_SELECTED;

              return (
                <TableRow
                  key={e.id}
                  className={cn(
                    "cursor-pointer transition-colors",
                    isSelected ? "bg-muted/60" : "hover:bg-muted/30",
                    disabled && "opacity-50",
                  )}
                  onClick={() => !disabled && onToggle(e.id)}
                >
                  <TableCell className="py-2 pl-3">
                    <div className="relative">
                      <Checkbox checked={isSelected} disabled={disabled} />
                      {color && (
                        <span
                          className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full border-2 border-background"
                          style={{ background: color }}
                        />
                      )}
                    </div>
                  </TableCell>

                  <TableCell className="font-mono text-xs">
                    <Link
                      to="/experiments/$id"
                      params={{ id: e.id }}
                      className="hover:underline"
                      onClick={(ev) => ev.stopPropagation()}
                    >
                      {e.id.slice(0, 8)}
                    </Link>
                  </TableCell>

                  <TableCell className="text-xs max-w-[8rem] truncate" title={datasetMap.get(e.dataset_id)}>
                    {datasetMap.get(e.dataset_id) ?? <span className="text-muted-foreground">-</span>}
                  </TableCell>

                  <TableCell className="font-medium uppercase">{e.architecture}</TableCell>
                  <TableCell>{e.k1}</TableCell>
                  <TableCell>{e.k2}</TableCell>
                  <TableCell>{e.seed}</TableCell>

                  <TableCell>
                    {e.best_metric != null ? (
                      <span className="font-mono">{e.best_metric.toFixed(4)}</span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                    <span className="ml-1 text-[10px] text-muted-foreground">{e.early_stopping_metric}</span>
                  </TableCell>

                  <TableCell>{e.convergence_epoch ?? "-"}</TableCell>

                  <TableCell className="font-mono">
                    {e.total_training_time_s != null ? e.total_training_time_s.toFixed(1) : "-"}
                  </TableCell>

                  <TableCell>
                    <Badge variant={STATUS_VARIANT[e.status] ?? "outline"}>
                      {STATUS_LABEL[e.status] ?? e.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              );
            })}

            {filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={11} className="py-8 text-center text-muted-foreground">
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
