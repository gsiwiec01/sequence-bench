import { Checkbox } from "@/components/ui/checkbox";
import { TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function ExpTableHead({
  allSelected,
  someSelected,
  onToggleAll,
}: {
  allSelected?: boolean;
  someSelected?: boolean;
  onToggleAll?: () => void;
}) {
  return (
    <TableHeader>
      <TableRow className="text-xs">
        <TableHead className="h-8 w-8">
          {onToggleAll && (
            <Checkbox
              checked={allSelected ? true : someSelected ? "indeterminate" : false}
              onCheckedChange={onToggleAll}
              aria-label="Zaznacz wszystkie"
            />
          )}
        </TableHead>

        <TableHead className="h-8 w-6" />
        <TableHead className="h-8">ID</TableHead>
        <TableHead className="h-8">Dataset</TableHead>
        <TableHead className="h-8">Arch</TableHead>
        <TableHead className="h-8">k₁</TableHead>
        <TableHead className="h-8">k₂</TableHead>
        <TableHead className="h-8">Seed</TableHead>
        <TableHead className="h-8">Status</TableHead>
        <TableHead className="h-8">Najlepsza metryka</TableHead>
        <TableHead className="h-8" />
      </TableRow>
    </TableHeader>
  );
}
