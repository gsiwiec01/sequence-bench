import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { EpochMetric } from "@/api/results";

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export type SSEStatus = "idle" | "running" | "completed" | "failed" | "disconnected";

export function useSSE(experimentId: string | null) {
  const [metrics, setMetrics] = useState<EpochMetric[]>([]);
  const [status, setStatus] = useState<SSEStatus>("idle");
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!experimentId) return;
    setMetrics([]);
    setStatus("running");

    const es = new EventSource(`${BASE_URL}/api/experiments/${experimentId}/stream`);
    esRef.current = es;

    es.addEventListener("metric", (e) => {
      setMetrics((prev) => [...prev, JSON.parse((e as MessageEvent<string>).data) as EpochMetric]);
    });
    es.addEventListener("completed", () => { setStatus("completed"); es.close(); });
    es.addEventListener("failed",    () => { setStatus("failed");    es.close(); });
    es.onerror = () => {
      setStatus((prev) => (prev === "completed" ? "completed" : "disconnected"));
      es.close();
    };

    return () => es.close();
  }, [experimentId]);

  return { metrics, status };
}

export function useSurfaceSSE(
  experimentId: string | null,
  jobId: string | null,
  enabled = true,
): SSEStatus {
  const qc = useQueryClient();
  const [status, setStatus] = useState<SSEStatus>("idle");

  useEffect(() => {
    if (!experimentId || !jobId || !enabled) return;
    setStatus("running");

    const es = new EventSource(
      `${BASE_URL}/api/experiments/${experimentId}/loss_landscape/${jobId}/stream`,
    );

    const finish = (s: SSEStatus) => {
      setStatus(s);
      es.close();
      qc.invalidateQueries({ queryKey: ["loss_landscape_lookup", experimentId] });
    };

    es.addEventListener("status", () => setStatus("running"));
    es.addEventListener("completed", () => finish("completed"));
    es.addEventListener("failed", () => finish("failed"));
    es.onerror = () => {
      setStatus((prev) => (prev === "completed" ? "completed" : "disconnected"));
      es.close();
    };

    return () => es.close();
  }, [experimentId, jobId, enabled, qc]);

  return status;
}
