import { Link } from "@tanstack/react-router";
import type { Experiment } from "@/api/experiments";
import { cn } from "@/lib/utils";
import { METRICS, MIN_METRICS } from "./constants";
import type { GradientHealth } from "./constants";

export default function MetricsTable({
  selected,
  additionalMetrics,
  additionalBests,
  gradientHealth,
}: {
  selected: { exp: Experiment; color: string }[];
  additionalMetrics: string[];
  additionalBests: Record<string, Record<string, number | null>>;
  gradientHealth: Record<string, GradientHealth>;
}) {
  if (selected.length === 0) {
    return (
      <p className="py-12 text-center text-sm text-muted-foreground">
        Zaznacz eksperymenty w panelu po lewej
      </p>
    );
  }

  function BestCell({ v, best, fmt }: { v: number | null; best: number | null; fmt: (n: number) => string }) {
    const isBest = v != null && v === best;
    return (
      <td className={cn("py-2 px-3 font-mono text-xs", isBest && "text-emerald-600 dark:text-emerald-400 font-semibold")}>
        {v != null ? fmt(v) : "-"}
      </td>
    );
  }

  function SectionRow({ label }: { label: string }) {
    return (
      <tr>
        <td colSpan={selected.length + 1} className="pt-5 pb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </td>
      </tr>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            <th className="py-2 pr-6 text-left text-xs text-muted-foreground font-medium w-44 sticky left-0 bg-card" />
            {selected.map(({ exp, color }) => (
              <th key={exp.id} className="py-2 px-3 text-left min-w-[150px]">
                <div className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: color }} />
                  <Link
                    to="/experiments/$id"
                    params={{ id: exp.id }}
                    className="font-mono text-xs hover:underline"
                  >
                    {exp.id.slice(0, 8)}
                  </Link>
                </div>

                <div className="text-xs text-muted-foreground font-normal mt-0.5">
                  {exp.architecture.toUpperCase()} k₁={exp.k1} k₂={exp.k2} seed={exp.seed}
                </div>
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          <SectionRow label="Wyniki" />
          {METRICS.map(({ label, get, fmt, lowerIsBetter }) => {
            const nums = selected.map(({ exp }) => get(exp)).filter((v): v is number => typeof v === "number");
            const best = nums.length > 1 ? (lowerIsBetter ? Math.min(...nums) : Math.max(...nums)) : null;

            return (
              <tr key={label} className="border-t">
                <td className="py-2 pr-6 text-xs text-muted-foreground sticky left-0 bg-card whitespace-nowrap">{label}</td>
                {selected.map(({ exp }) => <BestCell key={exp.id} v={get(exp)} best={best} fmt={fmt} />)}
              </tr>
            );
          })}

          {additionalMetrics.length > 0 && (
            <>
              <SectionRow label="Dodatkowe metryki" />
              {additionalMetrics.map((name) => {
                const nums = selected
                  .map(({ exp }) => additionalBests[exp.id]?.[name] ?? null)
                  .filter((v): v is number => typeof v === "number");

                const best = nums.length > 1
                  ? (MIN_METRICS.has(name) ? Math.min(...nums) : Math.max(...nums))
                  : null;

                return (
                  <tr key={name} className="border-t">
                    <td className="py-2 pr-6 text-xs font-mono text-muted-foreground sticky left-0 bg-card">{name}</td>
                    {selected.map(({ exp }) => (
                      <BestCell
                        key={exp.id}
                        v={additionalBests[exp.id]?.[name] ?? null}
                        best={best}
                        fmt={(v) => v.toFixed(4)}
                      />
                    ))}
                  </tr>
                );
              })}
            </>
          )}

          {Object.values(gradientHealth).some((g) => g.lastNorm != null) && (
            <>
              <SectionRow label="Gradienty" />
              <tr className="border-t">
                <td className="py-2 pr-6 text-xs text-muted-foreground sticky left-0 bg-card whitespace-nowrap">Norma (ostatnia)</td>

                {selected.map(({ exp }) => (
                  <td key={exp.id} className="py-2 px-3 font-mono text-xs">
                    {gradientHealth[exp.id]?.lastNorm != null ? gradientHealth[exp.id].lastNorm!.toExponential(2) : "-"}
                  </td>
                ))}
              </tr>

              <tr className="border-t">
                <td className="py-2 pr-6 text-xs text-muted-foreground sticky left-0 bg-card whitespace-nowrap">Status</td>
                {selected.map(({ exp }) => {
                  const s = gradientHealth[exp.id]?.status ?? null;
                  return (
                    <td key={exp.id} className={cn("py-2 px-3 text-xs font-medium",
                      s === "stabilne" && "text-emerald-600 dark:text-emerald-400",
                      (s === "vanishing" || s === "exploding") && "text-destructive",
                    )}>
                      {s ?? "-"}
                    </td>
                  );
                })}
              </tr>
            </>
          )}
        </tbody>
      </table>
    </div>
  );
}
