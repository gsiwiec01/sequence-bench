import type { HyperParams } from "@/api/experiments";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";

export default function HyperParamsForm({
  hp,
  onChange,
}: {
  hp: HyperParams;
  onChange: (key: keyof HyperParams, value: number | string | null | boolean) => void;
}) {
  const intField = (label: string, key: keyof HyperParams, min: number, max: number) => (
    <Field key={key}>
      <FieldLabel className="text-xs">{label}</FieldLabel>
      <Input
        type="number"
        min={min}
        max={max}
        step={1}
        value={hp[key] as number}
        onChange={(e) => onChange(key, parseInt(e.target.value, 10) || min)}
        className="h-8"
      />
    </Field>
  );

  const floatField = (
    label: string,
    key: keyof HyperParams,
    min: number,
    max: number,
    step: number,
  ) => (
    <Field key={key}>
      <FieldLabel className="text-xs">{label}</FieldLabel>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        value={hp[key] as number}
        onChange={(e) => onChange(key, parseFloat(e.target.value) || min)}
        className="h-8"
      />
    </Field>
  );

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {intField("Hidden size", "hidden_size", 16, 4096)}
      {intField("Liczba warstw", "num_layers", 1, 8)}
      {floatField("Dropout", "dropout", 0.0, 0.9, 0.05)}
      {floatField("Learning rate", "learning_rate", 1e-6, 1.0, 1e-5)}
      {intField("Batch size", "batch_size", 1, 2048)}
      {intField("Max epochs", "max_epochs", 1, 2000)}
      {intField("Early stopping", "early_stopping_patience", 1, 200)}
      {floatField("Gradient clip", "gradient_clip", 0, 100, 0.1)}
      {intField("Interwał gradientów", "gradient_log_interval", 1, 1000)}
    </div>
  );
}
