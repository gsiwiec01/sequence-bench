import { useState } from "react";
import type { Experiment } from "@/api/experiments";
import WeightTrajectorySection from "@/components/WeightTrajectory";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

export default function WeightTrajectoryTab({
  selected,
}: {
  selected: { exp: Experiment; color: string }[];
}) {
  const [primaryId, setPrimaryId] = useState<string>("");
  const effectivePrimary = primaryId || selected[0]?.exp.id || "";
  const primaryExp = selected.find((s) => s.exp.id === effectivePrimary)?.exp;

  if (selected.length === 0) {
    return <p className="py-12 text-center text-sm text-muted-foreground">Zaznacz eksperymenty w panelu po lewej</p>;
  }

  return (
    <div className="space-y-4">
      {selected.length > 1 && (
        <div className="flex items-center gap-3">
          <Label className="text-xs">Eksperyment</Label>
          <Select value={effectivePrimary} onValueChange={setPrimaryId}>
            <SelectTrigger className="h-8 w-72">
              <SelectValue />
            </SelectTrigger>

            <SelectContent>
              {selected.map(({ exp, color }) => (
                <SelectItem key={exp.id} value={exp.id}>
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ background: color }} />
                    {exp.id.slice(0, 8)} -{exp.architecture.toUpperCase()} k₁={exp.k1} k₂={exp.k2}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {primaryExp && (
        <WeightTrajectorySection experimentId={primaryExp.id} status={primaryExp.status} />
      )}
    </div>
  );
}
