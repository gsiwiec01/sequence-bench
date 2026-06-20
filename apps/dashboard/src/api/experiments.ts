import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { request } from "./client";

export interface Experiment {
  id: string;
  dataset_id: string;
  group_id: string | null;
  architecture: string;
  k1: number;
  k2: number;
  seed: number;
  task_type: string;
  early_stopping_metric: string;
  early_stopping_mode: "min" | "max";
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  best_metric: number | null;
  created_at: string;
  finished_at: string | null;
  hyperparams: Record<string, number | string | null>;
  additional_metrics: string[];
  n_parameters: number | null;
  total_training_time_s: number | null;
  convergence_epoch: number | null;
  final_train_loss: number | null;
  final_val_loss: number | null;
  has_weight_trajectory: boolean;
}

export type GradClipMode = "norm" | "value" | "none";

export interface HyperParams {
  hidden_size: number;
  num_layers: number;
  dropout: number;
  learning_rate: number;
  batch_size: number;
  max_epochs: number;
  early_stopping_patience: number;
  gradient_clip: number;
  grad_clip_mode: GradClipMode;
  max_grad_value: number | null;
  gradient_log_interval: number;
  weight_track_enabled: boolean;
  weight_log_interval: number;
}

export interface WeightTrajectory {
  pairs: string[]; // ["PC1", "PC2"]
  trajectory: (number | null)[][];
  epochs: number[];
  explained_variance?: number | null;
}

export interface ExperimentCreate {
  dataset_id: string;
  architecture: string;
  k1: number;
  k2: number;
  seed?: number;
  task_type: string;
  early_stopping_metric?: string;
  hyperparams?: Partial<HyperParams>;
}

export interface MatrixCreate {
  dataset_id: string;
  task_type: string;
  architectures: string[];
  k2_values?: number[];
  k2_ratios?: number[];
  k1_values?: number[];
  k1_ratios?: number[];
  seeds: number[];
  early_stopping_metric?: string;
  hyperparams?: Partial<HyperParams>;
  clip_norms?: (number | null)[];
}

export type AdditionalMetricSeries = Record<string, { epoch: number; value: number }[]>;

export interface MatrixResponse {
  experiments: Experiment[];
  group_id: string;
}

export const experimentKeys = {
  all: ["experiments"] as const,
  list: (params?: Record<string, string>) => ["experiments", "list", params] as const,
  detail: (id: string) => ["experiments", "detail", id] as const,
  metrics: (id: string, names?: string, epochFrom?: number, epochTo?: number) =>
    ["experiments", "metrics", id, names, epochFrom, epochTo] as const,
};

export function useExperiments(params?: Record<string, string>, enabled = true) {
  return useQuery({
    queryKey: experimentKeys.list(params),
    queryFn: () => request<Experiment[]>(`/api/experiments?${new URLSearchParams(params)}`),
    enabled,
    refetchInterval: (query) => {
      const hasActive = query.state.data?.some(
        (e) => e.status === "running" || e.status === "pending"
      );
      return hasActive ? 5_000 : false;
    },
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: experimentKeys.detail(id),
    queryFn: () => request<Experiment>(`/api/experiments/${id}`),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "pending" ? 3_000 : false;
    },
  });
}

export function useAdditionalMetrics(
  experimentId: string,
  names?: string[],
  epochFrom = 0,
  epochTo?: number,
  enabled = true,
) {
  const params = new URLSearchParams();
  params.set("epoch_from", String(epochFrom));

  const namesStr = names?.join(",");
  if (namesStr) params.set("names", namesStr);
  if (epochTo !== undefined) params.set("epoch_to", String(epochTo));

  return useQuery({
    queryKey: experimentKeys.metrics(experimentId, namesStr, epochFrom, epochTo),
    queryFn: () =>
      request<AdditionalMetricSeries>(`/api/experiments/${experimentId}/metrics?${params}`),
    enabled: enabled && !!experimentId,
  });
}

export function useCreateExperiment(onSuccess?: (exp: Experiment) => void) {
  return useMutation({
    mutationFn: (body: ExperimentCreate) =>
      request<Experiment>("/api/experiments", { method: "POST", body: JSON.stringify(body) }),
    onSuccess,
  });
}

export function useCreateMatrix(onSuccess?: (response: MatrixResponse) => void) {
  return useMutation({
    mutationFn: (body: MatrixCreate) =>
      request<MatrixResponse>("/api/experiments/matrix", { method: "POST", body: JSON.stringify(body) }),
    onSuccess,
  });
}

export function useDeleteExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => request<void>(`/api/experiments/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: experimentKeys.all }),
  });
}

export function useWeightTrajectory(experimentId: string, enabled = true) {
  return useQuery({
    queryKey: ["weight_trajectory", experimentId],
    queryFn: () => request<WeightTrajectory>(`/api/experiments/${experimentId}/weight_trajectory`),
    enabled: enabled && !!experimentId,
    retry: false,
  });
}


export type LandscapeStatus =
  | "none" | "queued" | "pending" | "running" | "completed" | "failed";

export interface SurfaceParams {
  resolution: number;
  margin: number;
}

export interface LossLandscape {
  id?: string;
  status: LandscapeStatus;
  params?: { resolution: number; margin: number; method_version: string } | null;
  x_range?: [number, number] | null;
  y_range?: [number, number] | null;
  x_values?: number[];
  y_values?: number[];
  loss_grid?: number[][];
  a_traj?: number[];
  b_traj?: number[];
  explained_variance?: number | null;
  anchor_loss?: number | null;
  error_message?: string | null;
  created_at?: string | null;
}

export interface LandscapeCreate extends Partial<SurfaceParams> {
  force?: boolean;
}

export interface SurfaceCreateResult {
  id: string;
  status: LandscapeStatus;
  cached: boolean;
}

const surfaceLookupKey = (experimentId: string, p: SurfaceParams) =>
  ["loss_landscape_lookup", experimentId, p.resolution, p.margin] as const;

export function useCreateLossLandscape(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LandscapeCreate) =>
      request<SurfaceCreateResult>(
        `/api/experiments/${experimentId}/loss_landscape`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ["loss_landscape_lookup", experimentId] }),
  });
}

export function useLandscapeLookup(
  experimentId: string,
  params: SurfaceParams,
  enabled = true,
) {
  return useQuery({
    queryKey: surfaceLookupKey(experimentId, params),
    queryFn: () =>
      request<LossLandscape>(
        `/api/experiments/${experimentId}/loss_landscape?resolution=${params.resolution}&margin=${params.margin}`,
      ),
    enabled: enabled && !!experimentId,
    retry: false,
  });
}

export function useCancelExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      request<Experiment>(`/api/experiments/${id}/cancel`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: experimentKeys.all }),
  });
}

export function useCloneConfig(experimentId: string, enabled = true) {
  return useQuery({
    queryKey: ["experiments", "clone-config", experimentId],
    queryFn: () => request<ExperimentCreate>(`/api/experiments/${experimentId}/clone-config`),
    enabled: enabled && !!experimentId,
    retry: false,
  });
}

