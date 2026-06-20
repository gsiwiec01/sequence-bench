import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BASE_URL, request } from "./client";

export interface Dataset {
  id: string;
  name: string;
  type: "builtin" | "custom";
  T: number;
  input_size: number;
  output_size: number;
  task_type: string;
}

export const datasetKeys = {
  all: ["datasets"] as const,
  list: () => ["datasets", "list"] as const,
  preview: (id: string) => ["datasets", "preview", id] as const,
};

export function useDatasets() {
  return useQuery({
    queryKey: datasetKeys.list(),
    queryFn: () => request<Dataset[]>("/api/datasets"),
  });
}

export function useUploadDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (formData: FormData) => {
      const res = await fetch(`${BASE_URL}/api/datasets/upload`, { method: "POST", body: formData });
      const body = await res.json();
      if (!res.ok) throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
      return body as Dataset;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: datasetKeys.all }),
  });
}

export function useDeleteDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => request<void>(`/api/datasets/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: datasetKeys.all }),
  });
}