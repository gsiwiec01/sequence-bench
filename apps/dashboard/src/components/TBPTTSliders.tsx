import {Slider} from "@/components/ui/slider";
import {Badge} from "@/components/ui/badge";
import {Button} from "@/components/ui/button";
import {Label} from "@/components/ui/label";
import {cn} from "@/lib/utils";

interface Props {
  T: number;
  k1: number;
  k2: number;
  onK1Change: (v: number) => void;
  onK2Change: (v: number) => void;
}

const PRESETS = [
  {label: "TBPTT(1,1)", getK2: () => 1, getK1: () => 1},
  {label: "Segmentowy", getK2: (T: number) => Math.max(1, Math.round(0.1 * T)), getK1: (k2: number) => k2},
  {label: "Full BPTT", getK2: (T: number) => T, getK1: () => 1},
];

export default function TBPTTSliders({T, k1, k2, onK1Change, onK2Change}: Props) {
  const applyPreset = (preset: typeof PRESETS[number]) => {
    const newK2 = preset.getK2(T);
    const newK1 = preset.getK1(newK2);
    onK2Change(newK2);
    onK1Change(newK1);
  };

  return (
    <div className="rounded-lg border bg-card p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Parametry TBPTT</h3>

        <div className="flex gap-1">
          {PRESETS.map((p) => (
            <Button key={p.label} size="sm" variant="outline" className="h-7 text-xs"
                    onClick={() => applyPreset(p)}>
              {p.label}
            </Button>
          ))}
        </div>
      </div>

      <div className="space-y-5">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Interwał aktualizacji (k₁)</Label>
            <div className="flex items-center gap-2">
              <span className="text-sm font-mono font-medium">{k1}</span>
            </div>
          </div>

          <Slider
            min={1} max={T} step={1} value={[k1]}
            onValueChange={([v]) => {
              onK1Change(v);
              if (k2 < v) onK2Change(v);
            }}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Głębokość propagacji (k₂)</Label>
            <div className="flex items-center gap-2">
              <span className="text-sm font-mono font-medium">{k2}</span>
            </div>
          </div>

          <Slider
            min={k1} max={T} step={1} value={[k2]}
            onValueChange={([v]) => onK2Change(v)}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
}
