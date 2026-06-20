import { useDroppable } from "@dnd-kit/core";
import type { Experiment } from "@/api/experiments";
import { Table, TableBody } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { UNGROUPED_ID } from "./constants";
import DraggableRow from "./DraggableRow";
import ExpTableHead from "./ExpTableHead";

export default function UngroupedSection({
  experiments,
  activeId,
  selectedIds,
  onToggleSelect,
  onDelete,
  onCancel,
  datasetMap,
}: {
  experiments: Experiment[];
  activeId: string | null;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
  datasetMap: Map<string, string>;
}) {
  const { isOver, setNodeRef } = useDroppable({ id: UNGROUPED_ID });

  const ids = experiments.map((e) => e.id);
  const allSelected = ids.length > 0 && ids.every((id) => selectedIds.has(id));
  const someSelected = !allSelected && ids.some((id) => selectedIds.has(id));

  function toggleAll() {
    if (allSelected) ids.forEach((id) => onToggleSelect(id));
    else ids.filter((id) => !selectedIds.has(id)).forEach((id) => onToggleSelect(id));
  }

  return (
    <div ref={setNodeRef}>
      <div className={cn(
        "flex items-center gap-2 py-2 border-b transition-colors",
        isOver ? "border-primary/60 bg-primary/5" : "border-border/60",
      )}>
        <span className="text-sm font-semibold text-muted-foreground">Bez grupy</span>
        <span className="text-xs text-muted-foreground">({experiments.length})</span>
      </div>

      <Table>
        <ExpTableHead allSelected={allSelected} someSelected={someSelected} onToggleAll={toggleAll} />
        <TableBody>
          {experiments.map((e) => (
            <DraggableRow
              key={e.id}
              experiment={e}
              groupId={null}
              isDragging={activeId === e.id}
              isSelected={selectedIds.has(e.id)}
              onToggleSelect={() => onToggleSelect(e.id)}
              onDelete={() => onDelete(e.id)}
              onCancel={() => onCancel(e.id)}
              datasetName={datasetMap.get(e.dataset_id)}
            />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
