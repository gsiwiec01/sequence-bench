import { useQuery } from "@tanstack/react-query";
import { request } from "./client";

export interface GradientNorms {
  [paramName: string]: number[];
}

export interface GradientEpochs {
  epochs: number[];
}

export interface GradientParamTrends {
  epochs: number[];
  params: Record<string, (number | null)[]>;
}

export const gradientKeys = {
  epochs: (experimentId: string) => ["gradients", "epochs", experimentId] as const,
  norms: (experimentId: string, epoch: number) =>
    ["gradients", "norms", experimentId, epoch] as const,
  paramTrends: (experimentId: string) =>
    ["gradients", "param-trends", experimentId] as const,
};

export function useGradientEpochs(experimentId: string) {
  return useQuery({
    queryKey: gradientKeys.epochs(experimentId),
    queryFn: () => request<GradientEpochs>(`/api/gradients/${experimentId}/epochs`),
    enabled: !!experimentId,
  });
}

export function useGradientNorms(experimentId: string, epoch: number | null) {
  return useQuery({
    queryKey: gradientKeys.norms(experimentId, epoch ?? -1),
    queryFn: () => request<GradientNorms>(`/api/gradients/${experimentId}/${epoch}`),
    enabled: !!experimentId && epoch != null,
  });
}

export function useGradientParamTrends(experimentId: string) {
  return useQuery({
    queryKey: gradientKeys.paramTrends(experimentId),
    queryFn: () =>
      request<GradientParamTrends>(`/api/gradients/${experimentId}/param-trends`),
    enabled: !!experimentId,
  });
}
