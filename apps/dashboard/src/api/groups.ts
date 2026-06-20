import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { request } from "./client";
import type { DegradationResponse } from "./results";

// --- Types ---

export interface ExperimentGroup {
  id: string;
  name: string;
  description: string | null;
  tags: string[];
  created_at: string;
  dataset: string | null;
  created_from_matrix: boolean;
  n_total: number;
  n_completed: number;
  n_running: number;
  n_pending: number;
  n_failed: number;
  n_cancelled: number;
  best_metric_by_arch: Record<string, number | null>;
}

export interface ExperimentGroupDetail extends ExperimentGroup {
  experiment_ids: string[];
}

export interface GroupCreate {
  name: string;
  description?: string;
  tags?: string[];
  dataset?: string;
}

export const groupKeys = {
  all: ["groups"] as const,
  list: () => ["groups", "list"] as const,
  detail: (id: string) => ["groups", "detail", id] as const,
  degradation: (id: string, groupBy: string) => ["groups", "degradation", id, groupBy] as const,
};

export function useGroups() {
  return useQuery({
    queryKey: groupKeys.list(),
    queryFn: () => request<ExperimentGroup[]>("/api/groups"),
    refetchInterval: (query) => {
      const hasActive = query.state.data?.some((g) => g.n_running > 0 || g.n_pending > 0);
      return hasActive ? 5_000 : false;
    },
  });
}

export function useCreateGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: GroupCreate) =>
      request<ExperimentGroupDetail>("/api/groups", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: groupKeys.all }),
  });
}

export function useAddExperimentsToGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ groupId, experimentIds }: { groupId: string; experimentIds: string[] }) =>
      request<{ added: number }>(`/api/groups/${groupId}/experiments`, {
        method: "POST",
        body: JSON.stringify({ experiment_ids: experimentIds }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: groupKeys.all });
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

export function useRemoveExperimentFromGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ groupId, expId }: { groupId: string; expId: string }) =>
      request<void>(`/api/groups/${groupId}/experiments/${expId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: groupKeys.all });
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

export function useRenameGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      request<ExperimentGroup>(`/api/groups/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: groupKeys.all }),
  });
}

export function useDeleteGroup() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, deleteExperiments = false }: { id: string; deleteExperiments?: boolean }) =>
      request<void>(
        `/api/groups/${id}${deleteExperiments ? "?delete_experiments=true" : ""}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: groupKeys.all });
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

export function useCancelAllRunning(groupId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      request<{ cancelled: number }>(`/api/groups/${groupId}/cancel_all`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: groupKeys.all }),
  });
}

export function useGroupDegradationCurves(groupId: string, groupBy: string, baselineK2?: number, metric?: string, baselineK1?: number, onlyConverged = false) {
  const params = new URLSearchParams({ group_by: groupBy });
  if (baselineK2 !== undefined) params.set("baseline_k2", String(baselineK2));
  if (baselineK1 !== undefined) params.set("baseline_k1", String(baselineK1));
  if (metric) params.set("metric", metric);
  if (onlyConverged) params.set("only_converged", "true");

  return useQuery({
    queryKey: [...groupKeys.degradation(groupId, groupBy), baselineK2, baselineK1, metric, onlyConverged] as const,
    queryFn: () => request<DegradationResponse>(`/api/groups/${groupId}/degradation?${params}`),
    enabled: !!groupId,
  });
}
