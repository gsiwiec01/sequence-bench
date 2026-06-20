import { useState } from "react";
import { useCancelAllRunning } from "@/api/groups";
import type { ExperimentGroup } from "@/api/groups";
import type { Experiment } from "@/api/experiments";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { useDroppable } from "@dnd-kit/core";
import DeleteGroupDialog from "./DeleteGroupDialog";
import DraggableRow from "./DraggableRow";
import ExpTableHead from "./ExpTableHead";
import RenameGroupDialog from "./RenameGroupDialog";

export default function GroupSection({
  group,
  experiments,
  activeId,
  selectedIds,
  onToggleSelect,
  onDelete,
  onCancel,
  onDetach,
  datasetMap,
}: {
  group: ExperimentGroup;
  experiments: Experiment[];
  activeId: string | null;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
  onDetach: (expId: string) => void;
  datasetMap: Map<string, string>;
}) {
  const { isOver, setNodeRef } = useDroppable({ id: group.id });
  const cancelMutation = useCancelAllRunning(group.id);
  const hasActive = group.n_running > 0 || group.n_pending > 0;
  const [renameOpen, setRenameOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const ids = experiments.map((e) => e.id);
  const allSelected = ids.length > 0 && ids.every((id) => selectedIds.has(id));
  const someSelected = !allSelected && ids.some((id) => selectedIds.has(id));

  function toggleAll() {
    if (allSelected) ids.forEach((id) => onToggleSelect(id));
    else ids.filter((id) => !selectedIds.has(id)).forEach((id) => onToggleSelect(id));
  }

  return (
    <div ref={setNodeRef}>
      <RenameGroupDialog group={group} open={renameOpen} onOpenChange={setRenameOpen} />

      <div className={cn(
        "flex items-center justify-between gap-3 py-2 border-b transition-colors",
        isOver ? "border-primary/60 bg-primary/5" : "border-border/60",
      )}>
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={() => setCollapsed((v) => !v)}
            className="text-muted-foreground/60 hover:text-foreground transition-colors shrink-0"
            aria-label={collapsed ? "Rozwiń" : "Zwiń"}
          >
            <svg
              className={cn("h-3.5 w-3.5 transition-transform", collapsed && "-rotate-90")}
              viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8"
            >
              <path d="M2 4l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          <button
            onClick={() => setRenameOpen(true)}
            className="text-sm font-semibold truncate rounded px-1 -mx-1 hover:bg-muted transition-colors"
          >
            {group.name}
          </button>

          <span className="text-xs text-muted-foreground shrink-0">({experiments.length})</span>
          {group.n_running > 0 && <span className="text-xs shrink-0">{group.n_running} trwa</span>}
          {group.n_pending > 0 && <span className="text-xs shrink-0">{group.n_pending} oczekuje</span>}
          {group.n_failed > 0 && <span className="text-xs text-destructive shrink-0">{group.n_failed} błąd</span>}
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {hasActive && (
            <button
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              className="text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
            >
              Anuluj wszystkie
            </button>
          )}
          <DeleteGroupDialog group={group} experimentCount={experiments.length} />
        </div>
      </div>

      {!collapsed && (
        <Table>
          <ExpTableHead allSelected={allSelected} someSelected={someSelected} onToggleAll={toggleAll} />
          <TableBody>
            {experiments.length === 0 && !isOver ? (
              <TableRow>
                <TableCell colSpan={10} className="py-6 text-center text-xs text-muted-foreground">
                  Przeciągnij eksperymenty
                </TableCell>
              </TableRow>
            ) : (
              experiments.map((e) => (
                <DraggableRow
                  key={e.id}
                  experiment={e}
                  groupId={group.id}
                  isDragging={activeId === e.id}
                  isSelected={selectedIds.has(e.id)}
                  onToggleSelect={() => onToggleSelect(e.id)}
                  onDelete={() => onDelete(e.id)}
                  onCancel={() => onCancel(e.id)}
                  onDetach={() => onDetach(e.id)}
                  datasetName={datasetMap.get(e.dataset_id)}
                />
              ))
            )}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
