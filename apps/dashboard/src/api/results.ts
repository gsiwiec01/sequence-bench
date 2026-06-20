import {useQuery} from "@tanstack/react-query";
import {request} from "./client";

export interface EpochMetric {
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  epoch_time_s: number | null;
  gpu_memory_mb: number | null;
  grad_norm_mean: number | null;
  grad_norm_max: number | null;

  [key: string]: number | null | undefined;
}

export interface DegradationGroupData {
  k2_ratios: number[];
  k2_values: number[];
  baseline_k2_ratio: number;
  delta_mean: number[];
  delta_std: number[];
  n_per_ratio: number[];
  experiment_ids: string[][];
}

export interface DegradationMeta {
  T: number;
  total_fetched: number;
  with_metric: number;
  null_metric_ids: string[];
}

export interface DegradationResponse {
  groups: Record<string, DegradationGroupData>;
  meta: DegradationMeta;
}

export interface ConvergencePoint {
  k2: number;
  k2_ratio: number;
  architecture?: string;
  convergence_epoch_mean: number | null;
  convergence_epoch_std: number | null;
  convergence_epoch_min: number | null;
  convergence_epoch_max: number | null;
  n_seeds: number;
  n_converged: number;

  [key: string]: string | number | null | undefined;
}

export interface ConvergenceResponse {
  dataset_id: string;
  group_by: string;
  points: ConvergencePoint[];
  metric: string | null;
  threshold: number | null;
  threshold_mode: "min" | "max" | null;
}

export const resultKeys = {
  epochs: (experimentId: string) => ["results", "epochs", experimentId] as const,
  degradation: (datasetId: string, groupBy: string) =>
    ["results", "degradation", datasetId, groupBy] as const,
  compare: (ids: string[]) => ["results", "compare", ...ids] as const,
  convergence: (
    datasetId: string, groupBy: string, groupId?: string | null,
    metric?: string | null, threshold?: number | null,
  ) =>
    ["results", "convergence", datasetId, groupBy, groupId ?? null,
      metric ?? null, threshold ?? null] as const,
};

export function useEpochMetrics(experimentId: string, refetchInterval: number | false = false) {
  return useQuery({
    queryKey: resultKeys.epochs(experimentId),
    queryFn: () => request<EpochMetric[]>(`/api/results/${experimentId}/epochs`),
    enabled: !!experimentId,
    refetchInterval,
  });
}

export function useDegradationCurves(datasetId: string, groupBy: string, baselineK2?: number, metric?: string, baselineK1?: number, onlyConverged = false) {
  const params = new URLSearchParams({dataset_id: datasetId, group_by: groupBy});
  if (baselineK2 !== undefined) params.set("baseline_k2", String(baselineK2));
  if (baselineK1 !== undefined) params.set("baseline_k1", String(baselineK1));
  if (metric) params.set("metric", metric);
  if (onlyConverged) params.set("only_converged", "true");

  return useQuery({
    queryKey: [...resultKeys.degradation(datasetId, groupBy), baselineK2, baselineK1, metric, onlyConverged] as const,
    queryFn: () => request<DegradationResponse>(`/api/results/degradation?${params}`),
    enabled: !!datasetId,
  });
}

export function useConvergence(
  datasetId: string,
  groupBy = "architecture",
  groupId?: string | null,
  metric?: string | null,
  threshold?: number | null,
) {
  const params = new URLSearchParams({dataset_id: datasetId, group_by: groupBy});
  if (groupId) params.set("group_id", groupId);
  if (metric) params.set("metric", metric);
  if (threshold != null) params.set("metric_threshold", String(threshold));

  return useQuery({
    queryKey: resultKeys.convergence(datasetId, groupBy, groupId, metric, threshold),
    queryFn: () => request<ConvergenceResponse>(`/api/results/convergence?${params}`),
    enabled: !!datasetId,
  });
}