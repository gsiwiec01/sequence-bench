import { useState } from "react";
import { createFileRoute, useNavigate, useParams } from "@tanstack/react-router";
import { useExperiment, useCloneConfig } from "@/api/experiments";
import { useDatasets } from "@/api/datasets";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TrainingMonitor from "@/components/TrainingMonitor";
import GradientSection from "@/components/GradientSection";
import WeightTrajectorySection from "@/components/WeightTrajectory";
import RawDataSection from "@/components/RawDataSection";

export const Route = createFileRoute("/experiments/$id")({
  component: RouteComponent,
});

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  completed: "default", running: "secondary", pending: "outline",
  failed: "destructive", cancelled: "outline",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "oczekuje",
  running: "trwa",
  completed: "ukończone",
  failed: "błąd",
  cancelled: "anulowane",
};

function RouteComponent() {
  const { id } = useParams({ from: "/experiments/$id" });
  const navigate = useNavigate();
  const { data: exp, isLoading } = useExperiment(id);
  const { data: datasets = [] } = useDatasets();
  const { refetch: fetchCloneConfig, isFetching: isCloning } = useCloneConfig(id, false);
  const [exporting, setExporting] = useState(false);

  if (isLoading) return <p className="text-muted-foreground">Ładowanie...</p>;
  if (!exp) return <p className="text-destructive">Eksperyment nie znaleziony.</p>;

  async function handleClone() {
    const result = await fetchCloneConfig();
    if (result.data) {
      localStorage.setItem("cloneConfig", JSON.stringify(result.data));
      navigate({ to: "/experiments/new" });
    }
  }

  async function handleExport() {
    setExporting(true);
    try {
      const res = await fetch(`/api/experiments/${id}/export`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = `exp_${id.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  const datasetName = datasets.find((d) => d.id === exp.dataset_id)?.name ?? exp.dataset_id;

  const rows: [string, React.ReactNode][] = [
    ["Dataset", datasetName],
    ["Architektura", <span className="font-mono uppercase">{exp.architecture}</span>],
    ["k₁ / k₂", <span className="font-mono">{exp.k1} / {exp.k2}</span>],
    ["Seed", exp.seed],
    ["Typ zadania", exp.task_type],
    ["Early stopping", <span className="font-mono">{exp.early_stopping_metric}</span>],
    ["Status", <Badge variant={STATUS_VARIANT[exp.status] ?? "outline"}>{STATUS_LABEL[exp.status] ?? exp.status}</Badge>],
    ["Najlepsza metryka", exp.best_metric?.toFixed(4) ?? "-"],
    ["Parametry modelu", exp.n_parameters?.toLocaleString() ?? "-"],
    ["Czas treningu", exp.total_training_time_s ? `${exp.total_training_time_s.toFixed(1)} s` : "-"],
    ["Epoka konwergencji", exp.convergence_epoch ?? "-"],
    ["Końcowy val loss", exp.final_val_loss?.toFixed(4) ?? "-"],
    ["Zakończono", exp.finished_at ? new Date(exp.finished_at).toLocaleString() : "-"],
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold font-mono">{exp.id.slice(0, 8)}</h1>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exp.status !== "completed" || exporting}
          >
            {exporting ? "Pobieranie..." : "Eksportuj JSON"}
          </Button>

          <Button variant="outline" size="sm" onClick={handleClone} disabled={isCloning}>
            {isCloning ? "Ładowanie..." : "Klonuj"}
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Szczegóły</CardTitle></CardHeader>

          <CardContent>
            <dl className="divide-y">
              {rows.map(([label, value]) => (
                <div key={label as string} className="flex justify-between py-2 text-sm">
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="font-medium">{value}</dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Hiperparametry</CardTitle></CardHeader>

          <CardContent>
            <dl className="divide-y">
              {Object.entries(exp.hyperparams)
                .filter(([k]) => k !== "grad_clip_mode" && k !== "max_grad_value")
                .map(([k, v]) => (
                  <div key={k} className="flex justify-between py-2 text-sm">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd className="font-mono">{String(v)}</dd>
                  </div>
                ))}
            </dl>
          </CardContent>
        </Card>
      </div>

      <TrainingMonitor
        experimentId={exp.id}
        status={exp.status}
        additionalMetricNames={exp.additional_metrics}
      />

      <GradientSection
        experimentId={exp.id}
        status={exp.status}
      />

      <WeightTrajectorySection
        experimentId={exp.id}
        status={exp.status}
      />

      <RawDataSection experimentId={exp.id} />
    </div>
  );
}
