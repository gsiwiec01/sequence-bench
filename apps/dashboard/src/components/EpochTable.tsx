import {useState, useMemo} from "react";
import {useEpochMetrics} from "@/api/results";
import {Button} from "@/components/ui/button";
import SortableTh from "@/components/ui/sortable-th";
import {
  Table, TableBody, TableCell, TableHeader, TableRow,
} from "@/components/ui/table";
import {STANDARD_KEYS, formatNum, sortRows, downloadCsv} from "@/lib/table-utils";
import type {SortDir} from "@/lib/table-utils";

const PAGE_SIZE = 20;

export default function EpochTable({experimentId}: { experimentId: string }) {
  const {data, isLoading} = useEpochMetrics(experimentId);
  const [sortKey, setSortKey] = useState("epoch");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(0);

  const extraKeys = useMemo(() => {
    if (!data?.length) return [];
    return Object.keys(data[0]).filter((k) => !STANDARD_KEYS.has(k));
  }, [data]);

  const sorted = useMemo(() => {
    return (data ? sortRows(data, sortKey, sortDir) : [])
  }, [data, sortKey, sortDir],);

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageData = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function handleSort(key: string) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    }
    else {
      setSortKey(key);
      setSortDir("asc");
    }

    setPage(0);
  }

  function handleExport() {
    if (!sorted.length) return;
    const headers = [
      "Epoka", "Train loss", "Val loss",
      ...extraKeys,
      "Norma grad. (mean)", "Norma grad. (max)",
    ];

    const rows = sorted.map((r) => {
      const rec = r as Record<string, number | null>;
      return [
        rec.epoch, rec.train_loss ?? null, rec.val_loss ?? null,
        ...extraKeys.map((k) => rec[k] ?? null),
        rec.grad_norm_mean ?? null, rec.grad_norm_max ?? null,
      ];
    });

    downloadCsv(headers, rows, `epochs_${experimentId.slice(0, 8)}.csv`);
  }

  const cols = [
    {key: "epoch", label: "Epoka"},
    {key: "train_loss", label: "Train loss"},
    {key: "val_loss", label: "Val loss"},
    ...extraKeys.map((k) => ({key: k, label: k})),
    {key: "grad_norm_mean", label: "Norma grad. (mean)"},
    {key: "grad_norm_max", label: "Norma grad. (max)"},
  ];

  if (isLoading) return <p className="text-xs text-muted-foreground py-2">Ładowanie...</p>;
  if (!data?.length) return <p className="text-xs text-muted-foreground py-2">Brak danych epok.</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">{sorted.length} epok</span>
        <Button variant="outline" size="sm" className="h-6 text-xs px-2" onClick={handleExport}>
          Eksportuj jako CSV
        </Button>
      </div>

      <div className="overflow-auto rounded-md border text-sm">
        <Table>
          <TableHeader>
            <TableRow>
              {cols.map((c) => (
                <SortableTh
                  key={c.key}
                  label={c.label}
                  colKey={c.key}
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onSort={handleSort}
                />
              ))}
            </TableRow>
          </TableHeader>

          <TableBody>
            {pageData.map((row) => (
              <TableRow key={row.epoch}>
                <TableCell className="font-mono">{row.epoch}</TableCell>
                <TableCell className="font-mono">{formatNum(row.train_loss)}</TableCell>
                <TableCell className="font-mono">{formatNum(row.val_loss)}</TableCell>

                {extraKeys.map((k) => (
                  <TableCell key={k} className="font-mono">{formatNum(row[k])}</TableCell>
                ))}

                <TableCell className="font-mono">{formatNum(row.grad_norm_mean)}</TableCell>
                <TableCell className="font-mono">{formatNum(row.grad_norm_max)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-2">
          <Button
            variant="outline"
            size="sm"
            className="h-6 text-xs px-2"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            ← Poprzednia
          </Button>

          <span className="text-xs text-muted-foreground">
            {page + 1} / {totalPages}
          </span>

          <Button
            variant="outline"
            size="sm"
            className="h-6 text-xs px-2"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            Następna →
          </Button>
        </div>
      )}
    </div>
  );
}
