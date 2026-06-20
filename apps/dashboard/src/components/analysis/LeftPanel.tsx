import { useMemo } from "react";
import type { Experiment } from "@/api/experiments";
import type { Dataset } from "@/api/datasets";
import type { ExperimentGroup } from "@/api/groups";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { STATUS_LABEL, STATUS_VARIANT } from "@/lib/status";
import { COLORS, MAX_SELECTED } from "./constants";

interface LeftPanelProps {
  groups: ExperimentGroup[];
  datasets: Dataset[];
  experiments: Experiment[];
  selectedGroupId: string | null;
  selectedDatasetId: string | null;
  selectedExperimentIds: string[];
  onGroupChange: (id: string | null) => void;
  onDatasetChange: (id: string | null) => void;
  onToggle: (id: string) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export default function LeftPanel({
  groups,
  datasets,
  experiments,
  selectedGroupId,
  selectedDatasetId,
  selectedExperimentIds,
  onGroupChange,
  onDatasetChange,
  onToggle,
  onSelectAll,
  onDeselectAll,
}: LeftPanelProps) {
  const datasetMap = useMemo(() => new Map(datasets.map((d) => [d.id, d])), [datasets]);

  return (
    <div className="flex flex-col h-full gap-3 p-3">
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Grupa</Label>

        <Select
          value={selectedGroupId ?? "_all"}
          onValueChange={(v) => onGroupChange(v === "_all" ? null : v)}
        >
          <SelectTrigger className="h-8 text-xs w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="_all">-wszystkie grupy -</SelectItem>
            {groups.map((g) => (
              <SelectItem key={g.id} value={g.id}>
                {g.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">Dataset</Label>
        <Select
          value={selectedDatasetId ?? "_all"}
          onValueChange={(v) => onDatasetChange(v === "_all" ? null : v)}
        >
          <SelectTrigger className="h-8 text-xs w-full">
            <SelectValue />
          </SelectTrigger>

          <SelectContent>
            <SelectItem value="_all">-wszystkie -</SelectItem>
            {datasets.map((d) => (
              <SelectItem key={d.id} value={d.id}>
                {d.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-1.5">
        <button
          onClick={onSelectAll}
          disabled={experiments.length === 0 || selectedExperimentIds.length >= MAX_SELECTED}
          className="flex-1 rounded border bg-muted/40 px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Zaznacz wszystkie
        </button>

        <button
          onClick={onDeselectAll}
          disabled={selectedExperimentIds.length === 0}
          className="rounded border bg-muted/40 px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Odznacz
        </button>
      </div>

      <p className="text-[11px] text-muted-foreground px-0.5">
        {selectedExperimentIds.length} / {MAX_SELECTED} zaznaczonych
      </p>

      <ScrollArea className="flex-1 rounded-md border">
        <div className="p-1.5 space-y-0.5">
          {experiments.length === 0 && (
            <p className="py-6 text-center text-xs text-muted-foreground">
              Brak eksperymentów
            </p>
          )}
          {experiments.map((e) => {
            const isSelected = selectedExperimentIds.includes(e.id);
            const selIdx = selectedExperimentIds.indexOf(e.id);
            const color = selIdx >= 0 ? COLORS[selIdx] : undefined;
            const disabled = !isSelected && selectedExperimentIds.length >= MAX_SELECTED;

            return (
              <label
                key={e.id}
                className={cn(
                  "flex items-start gap-2 rounded px-2 py-1.5 cursor-pointer transition-colors",
                  isSelected ? "bg-muted" : "hover:bg-muted/50",
                  disabled && "opacity-40 cursor-not-allowed",
                )}
              >
                <div className="relative mt-0.5 shrink-0">
                  <Checkbox
                    checked={isSelected}
                    onCheckedChange={() => !disabled && onToggle(e.id)}
                    disabled={disabled}
                  />
                  {color && (
                    <span
                      className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full border-2 border-background"
                      style={{ background: color }}
                    />
                  )}
                </div>

                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-mono text-[11px] text-muted-foreground">
                      {e.id.slice(0, 8)}
                    </span>
                    <span className="text-xs font-medium uppercase">{e.architecture}</span>
                    <Badge
                      variant={STATUS_VARIANT[e.status] ?? "outline"}
                      className="h-4 px-1 text-[10px]"
                    >
                      {STATUS_LABEL[e.status] ?? e.status}
                    </Badge>
                  </div>

                  <div className="text-xs text-muted-foreground">
                    k₁={e.k1} k₂={e.k2} seed={e.seed}
                    {e.best_metric != null && (
                      <span className="ml-1.5 font-mono text-foreground">
                        {e.best_metric.toFixed(4)}
                      </span>
                    )}
                  </div>

                  {e.dataset_id && (
                    <div className="text-[11px] text-muted-foreground/60 truncate">
                      {datasetMap.get(e.dataset_id)?.name ?? "-"}
                    </div>
                  )}
                </div>
              </label>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
