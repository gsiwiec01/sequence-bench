import { useCreateLossLandscape } from "@/api/experiments";
import type { LossLandscape, SurfaceParams } from "@/api/experiments";
import { Button } from "@/components/ui/button";

export const SURFACE_PARAMS: SurfaceParams = { resolution: 25, margin: 0.15 };

export default function LandscapePanel({
  experimentId,
  status,
  hasTrajectory,
  job,
}: {
  experimentId: string;
  status: string;
  hasTrajectory: boolean;
  job: LossLandscape | undefined;
}) {
  const create = useCreateLossLandscape(experimentId);

  if (status !== "completed") return null;

  const active = job?.status === "queued" || job?.status === "running" || create.isPending;
  const isDone = job?.status === "completed";

  return (
    <div>
      <Button
        size="xs"
        variant="outline"
        disabled={active || !hasTrajectory}
        onClick={() => create.mutate({ ...SURFACE_PARAMS, force: isDone })}
        className="h-8 text-xs"
      >
        {active ? "Obliczanie…" : isDone ? "Przelicz ponownie" : "Oblicz powierzchnię"}
      </Button>

      {!hasTrajectory && (
        <span className="text-xs text-muted-foreground">
          Brak trajektorii wag.
        </span>
      )}

      {job?.status === "queued"  && <span className="text-xs text-muted-foreground animate-pulse">Czeka w kolejce…</span>}
      {job?.status === "running" && <span className="text-xs text-muted-foreground animate-pulse">Obliczanie…</span>}
      {job?.status === "failed"  && <span className="text-xs text-destructive">Błąd: {job?.error_message}</span>}
    </div>
  );
}
