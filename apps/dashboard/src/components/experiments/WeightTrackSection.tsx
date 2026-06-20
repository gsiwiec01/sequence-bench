import type { HyperParams } from "@/api/experiments";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

export default function WeightTrackSection({
  hp,
  onChange,
}: {
  hp: HyperParams;
  onChange: (key: keyof HyperParams, value: number | string | null | boolean) => void;
}) {
  return (
    <div className="rounded-md border">
      <div className="flex w-full items-center justify-between px-4 py-2.5">
        <label
          htmlFor="weight-track-switch"
          className="cursor-pointer text-sm font-medium"
        >
          Trajektoria wag
        </label>

        <div className="flex items-center gap-3">
          <Switch
            id="weight-track-switch"
            checked={hp.weight_track_enabled}
            onCheckedChange={(v) => onChange("weight_track_enabled", v)}
          />
        </div>
      </div>

      {hp.weight_track_enabled && (
        <div className="border-t px-4 py-4">
          <Field>
            <FieldLabel className="text-xs">
              Próbkowanie{" "}
              <span className="font-normal text-muted-foreground">(zapis pełnego wektora wag co N epok)</span>
            </FieldLabel>

            <Input
              type="number"
              min={1}
              max={100}
              step={1}
              value={hp.weight_log_interval}
              onChange={(e) =>
                onChange("weight_log_interval", parseInt(e.target.value, 10) || 1)
              }
              className="h-8 w-24"
            />
          </Field>
        </div>
      )}
    </div>
  );
}
