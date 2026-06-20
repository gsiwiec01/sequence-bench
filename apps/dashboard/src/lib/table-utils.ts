export const STANDARD_KEYS = new Set([
  "epoch", "train_loss", "val_loss",
  "epoch_time_s", "gpu_memory_mb",
  "grad_norm_mean", "grad_norm_max",
]);

export function formatNum(v: number | null | undefined): string {
  if (v == null) return "-";

  const abs = Math.abs(v);
  if (abs !== 0 && abs < 0.001) return v.toExponential(4);
  if (abs >= 10000) return v.toFixed(2);

  return v.toFixed(4);
}

export type SortDir = "asc" | "desc";

export function sortRows<T extends Record<string, unknown>>(data: T[], key: string, dir: SortDir): T[] {
  return [...data].sort((a, b) => {
    const av = a[key] as number | null;
    const bv = b[key] as number | null;

    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;

    return dir === "asc" ? av - bv : bv - av;
  });
}

export function downloadCsv(
  headers: string[],
  rows: (number | string | null)[][],
  filename: string,
) {
  const esc = (v: number | string | null) => {
    const s = v == null ? "" : String(v);
    return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const csv = [headers.join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
