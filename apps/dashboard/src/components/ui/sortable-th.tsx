import type { SortDir } from "@/lib/table-utils";

export default function SortableTh({
  label, colKey, sortKey, sortDir, onSort,
}: {
  label: string;
  colKey: string;
  sortKey: string;
  sortDir: SortDir;
  onSort: (k: string) => void;
}) {
  const active = colKey === sortKey;
  return (
    <th
      className="px-2 py-1 text-left font-medium text-muted-foreground cursor-pointer select-none whitespace-nowrap hover:text-foreground"
      onClick={() => onSort(colKey)}
    >
      {label}
      <span className="ml-1 text-[10px] opacity-60">
        {active ? (sortDir === "asc" ? "▲" : "▼") : "⇅"}
      </span>
    </th>
  );
}
