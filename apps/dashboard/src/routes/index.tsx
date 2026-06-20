import { useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  DndContext, DragOverlay, PointerSensor,
  useSensor, useSensors,
  type DragStartEvent, type DragEndEvent,
} from "@dnd-kit/core";
import { useExperiments, useDeleteExperiment, useCancelExperiment } from "@/api/experiments";
import { useDatasets } from "@/api/datasets";
import {
  useGroups, useAddExperimentsToGroup, useRemoveExperimentFromGroup,
} from "@/api/groups";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Experiment } from "@/api/experiments";
import type { ExperimentGroup } from "@/api/groups";
import { STATUS_LABEL, STATUS_VARIANT } from "@/lib/status";
import { UNGROUPED_ID } from "@/components/experiments/constants";
import BulkActionBar from "@/components/experiments/BulkActionBar";
import CreateGroupDialog from "@/components/experiments/CreateGroupDialog";
import GroupSection from "@/components/experiments/GroupSection";
import UngroupedSection from "@/components/experiments/UngroupedSection";

const STATUSES = ["pending", "running", "completed", "failed", "cancelled"] as const;

export const Route = createFileRoute("/")({
  component: RouteComponent,
});

function DragPreview({ experiment }: { experiment: Experiment }) {
  return (
    <div className="flex items-center gap-3 rounded-md border bg-card shadow-lg px-3 py-2 text-xs font-mono opacity-95 pointer-events-none">
      <span className="text-muted-foreground">{experiment.id.slice(0, 8)}</span>
      <span className="uppercase font-medium">{experiment.architecture}</span>
      <span className="text-muted-foreground">k₂={experiment.k2}</span>
      <Badge variant={STATUS_VARIANT[experiment.status] ?? "outline"} className="text-[10px] px-1.5 py-0">
        {STATUS_LABEL[experiment.status] ?? experiment.status}
      </Badge>
    </div>
  );
}

function RouteComponent() {
  const { data: experiments = [], isLoading: expsLoading } = useExperiments();
  const { data: groups = [], isLoading: groupsLoading } = useGroups();
  const { data: datasets } = useDatasets();
  const datasetMap = useMemo(
    () => new Map((datasets ?? []).map((d) => [d.id, d.name])),
    [datasets],
  );

  const deleteMutation = useDeleteExperiment();
  const cancelMutation = useCancelExperiment();
  const assignMutation = useAddExperimentsToGroup();
  const detachMutation = useRemoveExperimentFromGroup();

  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);

      return next;
    });
  }

  async function deleteSelected() {
    await Promise.all([...selectedIds].map((id) => deleteMutation.mutateAsync(id)));
    setSelectedIds(new Set());
  }

  function moveToGroup(groupId: string) {
    assignMutation.mutate(
      { groupId, experimentIds: [...selectedIds] },
      { onSuccess: () => setSelectedIds(new Set()) },
    );
  }

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  );

  const statusCounts = useMemo(
    () =>
      experiments.reduce<Record<string, number>>((acc, e) => {
        acc[e.status] = (acc[e.status] ?? 0) + 1;
        return acc;
      }, {}),
    [experiments],
  );

  const { grouped, ungrouped, expById } = useMemo(() => {
    const grouped = {} as Record<string, Experiment[]>;
    const ungrouped: Experiment[] = [];
    const expById = {} as Record<string, Experiment>;

    for (const exp of experiments) {
      expById[exp.id] = exp;
      if (exp.group_id) {
        (grouped[exp.group_id] ??= []).push(exp);
      } else {
        ungrouped.push(exp);
      }
    }

    return { grouped, ungrouped, expById };
  }, [experiments]);

  const canExport = [...selectedIds].some((id) => expById[id]?.status === "completed");

  async function exportZip() {
    const res = await fetch("/api/experiments/export-bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_ids: [...selectedIds] }),
    });

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "experiments_export.zip";
    a.click();
    URL.revokeObjectURL(url);
  }

  function dropExp({ active, over }: DragEndEvent) {
    setActiveId(null);
    if (!over) return;

    const expId = String(active.id);
    const exp = expById[expId];
    if (!exp) return;

    const targetId = String(over.id);

    if (targetId === UNGROUPED_ID) {
      if (exp.group_id) {
        detachMutation.mutate({ groupId: exp.group_id, expId });
      }
    } else {
      if (exp.group_id !== targetId) {
        assignMutation.mutate({ groupId: targetId, experimentIds: [expId] });
      }
    }
  }

  const activeExp = activeId ? expById[activeId] : null;
  const isLoading = expsLoading || groupsLoading;

  const sortedGroups = [...groups].sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <DndContext sensors={sensors} onDragStart={({ active }) => setActiveId(String(active.id))} onDragEnd={dropExp}>
      <div className="space-y-8">
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <div className="flex items-center gap-2">
            <CreateGroupDialog />
            <Button asChild>
              <Link to="/experiments/new">Nowy eksperyment</Link>
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 sm:grid-cols-5">
          {STATUSES.map((s) => (
            <Card key={s}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {STATUS_LABEL[s]}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{statusCounts[s] ?? 0}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {isLoading ? (
          <p className="text-sm text-muted-foreground">Ładowanie...</p>
        ) : experiments.length === 0 && groups.length === 0 ? (
          <div className="rounded-lg border border-dashed p-12 text-center">
            <p className="text-sm text-muted-foreground">
              Brak eksperymentów.
            </p>
          </div>
        ) : (
          <div className="space-y-8">
            {sortedGroups.map((g) => (
              <GroupSection
                key={g.id}
                group={g}
                experiments={grouped[g.id] ?? []}
                activeId={activeId}
                selectedIds={selectedIds}
                onToggleSelect={toggle}
                onDelete={(id) => deleteMutation.mutate(id)}
                onCancel={(id) => cancelMutation.mutate(id)}
                onDetach={(expId) => detachMutation.mutate({ groupId: g.id, expId })}
                datasetMap={datasetMap}
              />
            ))}

            {ungrouped.length > 0 && (
              <UngroupedSection
                experiments={ungrouped}
                activeId={activeId}
                selectedIds={selectedIds}
                onToggleSelect={toggle}
                onDelete={(id) => deleteMutation.mutate(id)}
                onCancel={(id) => cancelMutation.mutate(id)}
                datasetMap={datasetMap}
              />
            )}
          </div>
        )}
      </div>

      <DragOverlay dropAnimation={null}>
        {activeExp && <DragPreview experiment={activeExp} />}
      </DragOverlay>

      <BulkActionBar
        count={selectedIds.size}
        groups={sortedGroups}
        canExport={canExport}
        onDelete={deleteSelected}
        onMove={moveToGroup}
        onExport={exportZip}
        onClear={() => setSelectedIds(new Set())}
      />
    </DndContext>
  );
}
