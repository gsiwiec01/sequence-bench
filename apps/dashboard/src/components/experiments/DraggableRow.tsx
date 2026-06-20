import { useDraggable } from "@dnd-kit/core";
import { CSS } from "@dnd-kit/utilities";
import { Link } from "@tanstack/react-router";
import type { Experiment } from "@/api/experiments";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { TableCell, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { STATUS_LABEL, STATUS_VARIANT } from "@/lib/status";

export default function DraggableRow({
  experiment,
  groupId,
  onDelete,
  onCancel,
  onDetach,
  isDragging,
  isSelected,
  onToggleSelect,
  datasetName,
}: {
  experiment: Experiment;
  groupId: string | null;
  onDelete: () => void;
  onCancel: () => void;
  onDetach?: () => void;
  isDragging?: boolean;
  isSelected?: boolean;
  onToggleSelect?: () => void;
  datasetName?: string;
}) {
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: experiment.id,
    data: { experiment, fromGroupId: groupId },
  });

  const style = transform ? { transform: CSS.Translate.toString(transform) } : undefined;

  return (
    <TableRow
      ref={setNodeRef}
      style={style}
      className={cn("text-xs", isDragging && "opacity-40", isSelected && "bg-muted/50")}
    >
      <TableCell className="w-8" onClick={(e) => e.stopPropagation()}>
        <Checkbox
          checked={!!isSelected}
          onCheckedChange={onToggleSelect}
          aria-label="Zaznacz wiersz"
        />
      </TableCell>

      <TableCell
        className="w-6 pr-0 cursor-grab active:cursor-grabbing text-muted-foreground/40 hover:text-muted-foreground select-none"
        {...listeners}
        {...attributes}
      >
        ⠿
      </TableCell>

      <TableCell className="font-mono">
        <Link to="/experiments/$id" params={{ id: experiment.id }} className="hover:underline">
          {experiment.id.slice(0, 8)}
        </Link>
      </TableCell>

      <TableCell className="max-w-[8rem] truncate" title={datasetName}>
        {datasetName ?? <span className="text-muted-foreground">-</span>}
      </TableCell>

      <TableCell className="uppercase font-medium">{experiment.architecture}</TableCell>
      <TableCell>{experiment.k1}</TableCell>
      <TableCell>{experiment.k2}</TableCell>
      <TableCell>{experiment.seed}</TableCell>

      <TableCell>
        <Badge variant={STATUS_VARIANT[experiment.status] ?? "outline"} className="text-[10px] px-1.5 py-0">
          {STATUS_LABEL[experiment.status] ?? experiment.status}
        </Badge>
      </TableCell>

      <TableCell>
        {experiment.best_metric != null ? (
          <span className="font-mono">{experiment.best_metric.toFixed(4)}</span>
        ) : (
          <span className="text-muted-foreground">-</span>
        )}
        <span className="ml-1 text-[10px] text-muted-foreground">{experiment.early_stopping_metric}</span>
      </TableCell>

      <TableCell className="text-right space-x-2">
        {experiment.status === "running" && (
          <button onClick={onCancel} className="text-xs text-muted-foreground hover:text-foreground">
            Zatrzymaj
          </button>
        )}

        {onDetach && (
          <button onClick={onDetach} className="text-xs text-muted-foreground hover:text-amber-600">
            Odepnij
          </button>
        )}

        <button onClick={onDelete} className="text-xs text-muted-foreground hover:text-destructive">
          Usuń
        </button>
      </TableCell>
    </TableRow>
  );
}
