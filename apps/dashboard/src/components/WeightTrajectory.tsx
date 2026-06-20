import { useWeightTrajectory, useExperiment, useLandscapeLookup } from "@/api/experiments";
import { useSurfaceSSE } from "@/hooks/useSSE";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import LandscapePanel, { SURFACE_PARAMS } from "@/components/LandscapePanel";
import TrajectoryPlot from "@/components/TrajectoryPlot";

interface Props { experimentId: string; status: string; }

export default function WeightTrajectorySection({ experimentId, status }: Props) {
  const { data, isLoading, isError } = useWeightTrajectory(experimentId);
  const { data: exp } = useExperiment(experimentId);
  const hasTrajectory = !!exp?.has_weight_trajectory;

  const { data: job } = useLandscapeLookup(experimentId, SURFACE_PARAMS, hasTrajectory && status === "completed");
  const jobActive = job?.status === "queued" || job?.status === "running";
  useSurfaceSSE(experimentId, jobActive ? job?.id ?? null : null, jobActive);

  const surface = job?.status === "completed" ? job : undefined;
  const pngPath = surface?.id ? `/api/experiments/${experimentId}/loss_landscape/${surface.id}/surface.png` : undefined;

  if (isLoading) return null;
  if ((isError || !data || data.trajectory.length === 0) && !hasTrajectory) return null;

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Trajektoria wag i przestrzeń błędów</CardTitle>
        <LandscapePanel experimentId={experimentId} status={status} hasTrajectory={hasTrajectory} job={job} />
      </CardHeader>

      <CardContent className="space-y-5">
        {data && data.trajectory.length > 0 ? (
          <TrajectoryPlot data={data} landscape={surface} pngPath={pngPath} />
        ) : (
          <p className="text-sm text-muted-foreground">Trajektoria zostanie pokazana po zebraniu co najmniej 3 epok.</p>
        )}
      </CardContent>
    </Card>
  );
}
