import type { Experiment } from "@/api/experiments";
import type { GradientParamTrends } from "@/api/gradients";
import { LABEL_W, CELL_H, MAX_HEATMAP_EPOCHS, normToColor } from "./constants";

export default function ParamHeatmap({
  exp,
  paramTrends,
  sharedLogRange,
}: {
  exp: Experiment;
  paramTrends: GradientParamTrends | null;
  sharedLogRange?: { logMin: number; logMax: number } | null;
}) {
  if (!paramTrends) {
    return <p className="text-xs text-muted-foreground py-2">Brak danych o parametrach.</p>;
  }

  const { epochs, params } = paramTrends;
  const paramNames = Object.keys(params);
  if (paramNames.length === 0 || epochs.length === 0) {
    return <p className="text-xs text-muted-foreground py-2">Brak danych o parametrach.</p>;
  }

  const step = Math.ceil(epochs.length / MAX_HEATMAP_EPOCHS);
  const sampledIdx = Array.from({ length: epochs.length }, (_, i) => i).filter((i) => i % step === 0);
  const sampledEpochs = sampledIdx.map((i) => epochs[i]);

  const allVals = paramNames.flatMap((p) =>
    sampledIdx.map((i) => params[p]?.[i]).filter((v): v is number => v != null && v > 0),
  );
  if (allVals.length === 0) {
    return <p className="text-xs text-muted-foreground py-2">Brak wartości gradientów.</p>;
  }

  const localLogMin = Math.log10(Math.min(...allVals));
  const localLogMax = Math.log10(Math.max(...allVals));
  const { logMin, logMax } = sharedLogRange ?? { logMin: localLogMin, logMax: localLogMax };
  const cellW = Math.max(6, Math.min(18, Math.floor(560 / sampledEpochs.length)));
  const labelStep = Math.max(1, Math.ceil(sampledEpochs.length / 8));
  const svgW = LABEL_W + cellW * sampledEpochs.length + 10;
  const svgH = CELL_H * paramNames.length + 24;

  return (
    <div className="space-y-1">
      <p className="text-xs font-mono font-medium">
        {exp.id.slice(0, 8)} -{exp.architecture.toUpperCase()} k₁={exp.k1} k₂={exp.k2}
      </p>

      <div className="overflow-x-auto">
        <svg width={svgW} height={svgH} overflow="visible">
          {paramNames.map((p, yi) => (
            <text key={p} x={LABEL_W - 6} y={yi * CELL_H + CELL_H / 2 + 4}
              textAnchor="end" fontSize={9} fill="#9ca3af">
              {p.length > 28 ? "…" + p.slice(-27) : p}
            </text>
          ))}

          {paramNames.map((p, yi) =>
            sampledIdx.map((si, xi) => (
              <rect
                key={`${p}-${xi}`}
                x={LABEL_W + xi * cellW} y={yi * CELL_H}
                width={cellW - 1} height={CELL_H - 1}
                fill={normToColor(params[p]?.[si] ?? null, logMin, logMax)}
                rx={1}
              />
            )),
          )}

          {sampledEpochs.map((e, xi) =>
            xi % labelStep === 0 ? (
              <text key={`e-${e}`}
                x={LABEL_W + xi * cellW + cellW / 2}
                y={CELL_H * paramNames.length + 16}
                textAnchor="middle" fontSize={9} fill="#9ca3af">
                {e}
              </text>
            ) : null,
          )}
        </svg>
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted-foreground" style={{ paddingLeft: LABEL_W }}>
        <span>{(10 ** logMin).toExponential(0)}</span>
        <div className="h-2 w-28 rounded" style={{ background: "linear-gradient(to right, rgb(68,1,84), rgb(38,130,142), rgb(253,231,37))" }} />
        <span>{(10 ** logMax).toExponential(0)}</span>
      </div>
    </div>
  );
}
