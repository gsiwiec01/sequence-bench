import {useEffect, useMemo, useState} from "react";
import {useForm, Controller} from "react-hook-form";
import {zodResolver} from "@hookform/resolvers/zod";
import {z} from "zod";
import {createFileRoute, useNavigate} from "@tanstack/react-router";
import {
  useCreateExperiment,
  useCreateMatrix,
} from "@/api/experiments";
import type {HyperParams, MatrixCreate} from "@/api/experiments";
import {useDatasets} from "@/api/datasets";
import {Button} from "@/components/ui/button";
import {Field, FieldDescription, FieldError, FieldLabel} from "@/components/ui/field";
import {Input} from "@/components/ui/input";
import {Label} from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {Tabs, TabsList, TabsTrigger} from "@/components/ui/tabs";
import TBPTTSliders from "@/components/TBPTTSliders";
import {cn} from "@/lib/utils";
import Collapsible from "@/components/ui/collapsible";
import HyperParamsForm from "@/components/experiments/HyperParamsForm";
import WeightTrackSection from "@/components/experiments/WeightTrackSection";

export const Route = createFileRoute("/experiments/new")({
  component: RouteComponent,
});

const DEFAULT_HP: HyperParams = {
  hidden_size: 256,
  num_layers: 1,
  dropout: 0.0,
  learning_rate: 0.001,
  batch_size: 64,
  max_epochs: 100,
  early_stopping_patience: 10,
  gradient_clip: 1.0,
  grad_clip_mode: "norm",
  max_grad_value: null,
  gradient_log_interval: 1,
  weight_track_enabled: true,
  weight_log_interval: 1,
};

const TASK_TYPE_LABELS: Record<string, string> = {
  classification: "Klasyfikacja",
  regression: "Regresja",
  seq2seq: "Seq2Seq",
  language_model: "Language Model",
  forecasting: "Prognozowanie",
};

const TASK_TYPE_METRICS: Record<string, string[]> = {
  classification: ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "auc_macro", "mcc"],
  regression: ["mse", "mae", "r2", "mape"],
  language_model: ["perplexity", "cross_entropy"],
  seq2seq: ["accuracy", "f1_macro", "cross_entropy"],
  forecasting: ["mse", "mae", "mape", "r2"],
};

const formSchema = z.object({
  datasetId: z.string().min(1, "Wybierz dataset"),
  taskType: z.string().min(1, "Wybierz typ zadania"),
});
type FormValues = z.infer<typeof formSchema>;

function parseNums(s: string): number[] {
  return s
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n) && n >= 0);
}

function ArchToggle({
                      arch,
                      active,
                      onClick,
                    }: {
  arch: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded border px-4 py-1.5 text-sm font-medium transition-colors",
        active
          ? "border-primary bg-primary text-primary-foreground"
          : "border-border text-muted-foreground hover:border-foreground hover:text-foreground",
      )}
    >
      {arch.toUpperCase()}
    </button>
  );
}

const ARCHS = ["lstm", "gru", "rnn"] as const;

type Mode = "single" | "matrix";

function RouteComponent() {
  const navigate = useNavigate();

  const {
    control,
    watch,
    setValue,
    handleSubmit,
    formState: {errors: formErrors},
  } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {datasetId: "", taskType: ""},
  });
  const datasetId = watch("datasetId") ?? "";
  const taskType = watch("taskType") ?? "";

  const [mode, setMode] = useState<Mode>("single");
  const [hp, setHp] = useState<HyperParams>({...DEFAULT_HP});
  const [hpOpen, setHpOpen] = useState(false);
  const [earlyStoppingMetric, setEarlyStoppingMetric] = useState("val_loss");

  const [architecture, setArchitecture] = useState("lstm");
  const [k1, setK1] = useState(5);
  const [k2, setK2] = useState(50);
  const [seed, setSeed] = useState(42);

  const [matrixArchitectures, setMatrixArchitectures] = useState<string[]>(["lstm", "gru"]);
  const [k2Input, setK2Input] = useState("30, 60, 120");
  const [k1Input, setK1Input] = useState("1, 60");
  const [seedsInput, setSeedsInput] = useState("42, 43, 44");

  const {data: datasets = []} = useDatasets();
  const selectedDataset = datasets.find((d) => d.id === datasetId);

  useEffect(() => {
    const raw = localStorage.getItem("cloneConfig");
    if (!raw) return;

    localStorage.removeItem("cloneConfig");

    try {
      const cfg = JSON.parse(raw) as {
        dataset_id?: string;
        task_type?: string;
        architecture?: string;
        k1?: number;
        k2?: number;
        seed?: number;
        early_stopping_metric?: string;
        hyperparams?: Partial<HyperParams>;
      };

      if (cfg.dataset_id) setValue("datasetId", cfg.dataset_id);
      if (cfg.task_type) setValue("taskType", cfg.task_type);
      if (cfg.architecture) setArchitecture(cfg.architecture);
      if (cfg.k1 !== undefined) setK1(cfg.k1);
      if (cfg.k2 !== undefined) setK2(cfg.k2);
      if (cfg.seed !== undefined) setSeed(cfg.seed);
      if (cfg.early_stopping_metric) setEarlyStoppingMetric(cfg.early_stopping_metric);
      if (cfg.hyperparams) setHp((prev) => ({...prev, ...cfg.hyperparams}));

    } catch {
      // ignore malformed data
    }
  }, []);

  const k2Resolved = useMemo<number[]>(
    () => parseNums(k2Input).filter((v) => v >= 1).map(Math.floor),
    [k2Input],
  );

  const k1Resolved = useMemo<number[]>(
    () => parseNums(k1Input).filter((v) => v >= 1).map(Math.floor),
    [k1Input],
  );

  const seeds = useMemo(() => parseNums(seedsInput).filter(Number.isInteger), [seedsInput]);

  const validPairs = useMemo(() => {
    let valid = 0;

    for (const k2v of k2Resolved)
      for (const k1v of k1Resolved)
        if (k1v <= k2v) valid++;

    return valid;
  }, [k1Resolved, k2Resolved]);

  const totalRuns = matrixArchitectures.length * validPairs * seeds.length;

  const singleMutation = useCreateExperiment((exp) =>
    navigate({to: "/experiments/$id", params: {id: exp.id}}),
  );
  const matrixMutation = useCreateMatrix(() => navigate({to: "/"}));

  const error = singleMutation.error?.message ?? matrixMutation.error?.message;

  function setHpField(key: keyof HyperParams, value: number | string | null | boolean) {
    setHp((prev) => ({...prev, [key]: value}));
  }

  function pickDataset(id: string) {
    setValue("datasetId", id);
    const ds = datasets.find((d) => d.id === id);
    if (ds) {
      const newK2 = Math.min(k2, ds.T);
      setK2(newK2);
      setK1(Math.min(k1, newK2));
      setValue("taskType", ds.task_type);
    }
  }

  function toggleArch(arch: string) {
    setMatrixArchitectures((prev) =>
      prev.includes(arch) ? prev.filter((a) => a !== arch) : [...prev, arch],
    );
  }

  function submitSingle(values: FormValues) {
    singleMutation.mutate({
      dataset_id: values.datasetId,
      architecture,
      k1,
      k2,
      seed,
      task_type: values.taskType,
      early_stopping_metric: earlyStoppingMetric,
      hyperparams: hp,
    });
  }

  function submitMatrix() {
    matrixMutation.mutate({
      dataset_id: datasetId,
      task_type: taskType,
      architectures: matrixArchitectures,
      seeds,
      early_stopping_metric: earlyStoppingMetric,
      hyperparams: hp,
      k2_values: k2Resolved,
      k1_values: k1Resolved,
    });
  }

  const canSubmitSingle = !!datasetId && !!taskType && !singleMutation.isPending;
  const canSubmitMatrix =
    !!datasetId &&
    !!taskType &&
    matrixArchitectures.length > 0 &&
    k2Resolved.length > 0 &&
    k1Resolved.length > 0 &&
    seeds.length > 0 &&
    totalRuns > 0 &&
    !matrixMutation.isPending;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Nowy eksperyment</h1>

      <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
        <TabsList>
          <TabsTrigger value="single">Pojedynczy</TabsTrigger>
          <TabsTrigger value="matrix">Macierz</TabsTrigger>
        </TabsList>
      </Tabs>

      {error && (
        <div className="rounded-md border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="max-w-2xl space-y-5">
        <div className="grid grid-cols-2 gap-3">
          <Field>
            <FieldLabel>Dataset</FieldLabel>
            <Controller
              name="datasetId"
              control={control}
              render={({field}) => (
                <Select value={field.value} onValueChange={pickDataset}>
                  <SelectTrigger className="h-9">
                    <SelectValue placeholder="Wybierz dataset..."/>
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((d) => (
                      <SelectItem key={d.id} value={d.id}>
                        {d.name} -T={d.T}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
            <FieldError errors={[formErrors.datasetId]}/>
          </Field>

          {selectedDataset && (
            <Field>
              <FieldLabel>Typ zadania</FieldLabel>
              <Input disabled={true} value={TASK_TYPE_LABELS[selectedDataset.task_type] ?? selectedDataset.task_type}/>
            </Field>
          )}
        </div>

        <div className="space-y-2">
          <Label className="text-sm font-medium">
            {mode === "matrix" ? "Typy sieci (kilka = macierz)" : "Architektura"}
          </Label>
          <div className="flex gap-2">
            {ARCHS.map((arch) => (
              <ArchToggle
                key={arch}
                arch={arch}
                active={mode === "single" ? architecture === arch : matrixArchitectures.includes(arch)}
                onClick={() =>
                  mode === "single" ? setArchitecture(arch) : toggleArch(arch)
                }
              />
            ))}
          </div>
        </div>

        {mode === "single" ? (
          selectedDataset ? (
            <TBPTTSliders
              T={selectedDataset.T}
              k1={k1}
              k2={k2}
              onK1Change={setK1}
              onK2Change={setK2}
            />
          ) : (
            <div className="rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
              Wybierz dataset, aby skonfigurować k₁ i k₂
            </div>
          )
        ) : (
          <div className="rounded-lg border bg-card p-4 space-y-4">
            <h4 className="text-sm font-semibold">Parametry TBPTT</h4>
            <div className="grid grid-cols-2 gap-4">
              <Field>
                <FieldLabel className="text-xs">k₂ -głębokości propagacji</FieldLabel>
                <Input
                  value={k2Input}
                  onChange={(e) => setK2Input(e.target.value)}
                  placeholder="np. 30, 60, 120"
                  className="h-8"
                />
              </Field>
              <Field>
                <FieldLabel className="text-xs">k₁ -interwały aktualizacji</FieldLabel>

                <Input
                  value={k1Input}
                  onChange={(e) => setK1Input(e.target.value)}
                  placeholder="np. 1, 60, 120"
                  className="h-8"
                />
              </Field>
            </div>
          </div>
        )}

        <div className="flex items-end gap-3">
          <Field className="w-48">
            <FieldLabel className="text-xs">
              {mode === "matrix" ? "Seedy" : "Seed"}
            </FieldLabel>

            {mode === "matrix" ? (
              <Input
                value={seedsInput}
                onChange={(e) => setSeedsInput(e.target.value)}
                placeholder="np. 42, 43, 44"
                className="h-8"
              />
            ) : (
              <Input
                type="number"
                value={seed}
                onChange={(e) => setSeed(+e.target.value)}
                className="h-8"
              />
            )}
          </Field>

          {taskType && (
            <Field className="w-56">
              <FieldLabel className="text-xs">Early stopping</FieldLabel>
              <Select value={earlyStoppingMetric} onValueChange={setEarlyStoppingMetric}>
                <SelectTrigger className="h-9">
                  <SelectValue/>
                </SelectTrigger>
                <SelectContent>
                  {(["val_loss", ...(TASK_TYPE_METRICS[taskType] ?? [])]).map((m) => (
                    <SelectItem key={m} value={m} className="font-mono text-xs">
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          )}
        </div>

        <Collapsible
          title="Hiperparametry"
          open={hpOpen}
          onToggle={() => setHpOpen((v) => !v)}
        >
          <HyperParamsForm hp={hp} onChange={setHpField}/>
        </Collapsible>

        <WeightTrackSection hp={hp} onChange={setHpField}/>

        {mode === "single" ? (
          <Button disabled={!canSubmitSingle} onClick={handleSubmit(submitSingle)}>
            {singleMutation.isPending ? "Tworzę..." : "Utwórz eksperyment"}
          </Button>
        ) : (
          <Button disabled={!canSubmitMatrix} onClick={submitMatrix}>
            {matrixMutation.isPending ? "Tworzę macierz..." : `Utwórz macierz (${totalRuns} przebiegów)`}
          </Button>
        )}
      </div>
    </div>
  );
}
