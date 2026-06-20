import type { Experiment } from "@/api/experiments";

export const MAX_SELECTED = 8;

export const MIN_METRICS = new Set(["mse", "mae", "mape", "perplexity", "cross_entropy", "loss"]);

export const COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
  "#9333ea", "#0891b2", "#ea580c", "#0d9488",
];

export type ActiveMode = "overview" | "compare";
export type OverviewTab = "table" | "degradation" | "convergence";
export type CompareTab = "curves" | "gradients" | "metrics" | "trajectories";

export type ResultSortKey =
  | "architecture" | "k1" | "k2" | "seed" | "status"
  | "best_metric" | "convergence_epoch" | "total_training_time_s" | "created_at";

export type GradientHealth = {
  lastNorm: number | null;
  status: "vanishing" | "exploding" | "stabilne" | null;
};

export type MetricDef = {
  label: string;
  get: (e: Experiment) => number | null;
  fmt: (v: number) => string;
  lowerIsBetter?: boolean;
};

export const METRICS: MetricDef[] = [
  { label: "Najlepsza metryka",  get: (e) => e.best_metric,           fmt: (v) => v.toFixed(4) },
  { label: "Epoka konwergencji", get: (e) => e.convergence_epoch,     fmt: (v) => String(v), lowerIsBetter: true },
  { label: "Czas treningu",      get: (e) => e.total_training_time_s, fmt: (v) => `${v.toFixed(1)} s`, lowerIsBetter: true },
  { label: "Parametry modelu",   get: (e) => e.n_parameters,          fmt: (v) => v.toLocaleString() },
  { label: "Końcowy train loss", get: (e) => e.final_train_loss,      fmt: (v) => v.toFixed(4), lowerIsBetter: true },
  { label: "Końcowy val loss",   get: (e) => e.final_val_loss,        fmt: (v) => v.toFixed(4), lowerIsBetter: true },
];

export const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 83, 160], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

export const CELL_H = 14;
export const LABEL_W = 190;
export const MAX_HEATMAP_EPOCHS = 40;

export function viridisColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const i = Math.min(Math.floor(c * (VIRIDIS.length - 1)), VIRIDIS.length - 2);
  const f = c * (VIRIDIS.length - 1) - i;
  const [r1, g1, b1] = VIRIDIS[i];
  const [r2, g2, b2] = VIRIDIS[i + 1];
  return `rgb(${Math.round(r1 + (r2 - r1) * f)},${Math.round(g1 + (g2 - g1) * f)},${Math.round(b1 + (b2 - b1) * f)})`;
}

export function normToColor(v: number | null, logMin: number, logMax: number): string {
  if (v == null || v <= 0) return "rgb(30,30,30)";
  const range = logMax - logMin;
  const t = range > 0 ? Math.max(0, Math.min(1, (Math.log10(v) - logMin) / range)) : 0.5;
  return viridisColor(t);
}

export function getSortValue(e: Experiment, key: ResultSortKey): number | string {
  if (key === "best_metric") return e.best_metric ?? -Infinity;
  return (e as unknown as Record<string, string | number>)[key] ?? "";
}
