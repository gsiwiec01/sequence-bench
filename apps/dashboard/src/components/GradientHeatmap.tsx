import {useEffect, useRef, useState} from "react";
import {useGradientNorms} from "@/api/gradients";
import {downloadFile} from "@/api/client";

interface Props {
  experimentId: string;
  epoch: number;
  label?: string;
}

const PALETTE: [number, number, number][] = [
  [13, 2, 33], // #0d0221
  [45, 27, 105], // #2d1b69
  [26, 122, 74], // #1a7a4a 
  [126, 200, 160], // #7ec8a0
  [245, 245, 0], // #f5f500
];

function paletteColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const i = Math.min(Math.floor(c * (PALETTE.length - 1)), PALETTE.length - 2);
  const f = c * (PALETTE.length - 1) - i;
  const [r1, g1, b1] = PALETTE[i];
  const [r2, g2, b2] = PALETTE[i + 1];
  return `rgb(${Math.round(r1 + (r2 - r1) * f)},${Math.round(g1 + (g2 - g1) * f)},${Math.round(b1 + (b2 - b1) * f)})`;
}

const CELL_H = 24;
const Y_LABEL_W = 210;
const X_AXIS_H = 18;
const X_TICK_EVERY = 10;

interface ColorbarProps {
  logMin: number;
  logMax: number;
  canvasHeight: number;
}

function VerticalColorbar({logMin, logMax, canvasHeight}: ColorbarProps) {
  const logSpan = logMax - logMin || 1;
  const ticks: number[] = [];
  const first = Math.ceil(logMin);
  const last = Math.floor(logMax);
  const step = Math.max(1, Math.ceil((last - first) / 5));
  for (let v = first; v <= last; v += step) ticks.push(v);

  return (
    <div className="ml-2 flex items-start" style={{height: canvasHeight + X_AXIS_H}}>
      <div className="flex flex-col">
        <div className="w-3 rounded" style={{
          height: canvasHeight,
          background: "linear-gradient(to top, #0d0221, #2d1b69, #1a7a4a, #7ec8a0, #f5f500)"
        }}/>

        <div style={{height: X_AXIS_H}}/>
      </div>

      <div className="relative ml-1" style={{height: canvasHeight}}>
        {ticks.map((v) => (
          <div
            key={v}
            className="absolute flex items-center"
            style={{top: `${((logMax - v) / logSpan) * 100}%`, transform: "translateY(-50%)"}}
          >
            <span className="text-[9px] leading-none text-muted-foreground">
              10<sup>{v}</sup>
            </span>
          </div>
        ))}
      </div>

      <div className="relative ml-5 flex items-center justify-center" style={{height: canvasHeight}}>
        <span
          className="text-[10px] text-muted-foreground whitespace-nowrap"
          style={{writingMode: "vertical-rl", transform: "rotate(180deg)"}}
        >
          norma gradientu (log)
        </span>
      </div>
    </div>
  );
}

export default function GradientHeatmap({experimentId, epoch}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState<{ param: string; step: number; norm: number } | null>(null);

  const {data, isLoading, isError} = useGradientNorms(experimentId, epoch);

  const logRange = (() => {
    if (!data) return null;

    const all = Object.values(data).flat().filter((v) => v > 0);
    if (!all.length) return null;

    return {logMin: Math.log10(Math.min(...all)), logMax: Math.log10(Math.max(...all))};
  })();

  useEffect(() => {
    if (!data || !canvasRef.current || !logRange) return;

    const params = Object.keys(data);
    const maxSteps = Math.max(...params.map((p) => data[p].length), 1);
    const cellW = Math.max(2, Math.floor(Math.min(900, maxSteps * 6) / maxSteps));

    const canvas = canvasRef.current;
    canvas.width = cellW * maxSteps;
    canvas.height = CELL_H * params.length;
    const ctx = canvas.getContext("2d")!;

    const {logMin, logMax} = logRange;
    const logSpan = logMax - logMin || 1;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    params.forEach((param, pi) => {
      data[param].forEach((norm, si) => {
        if (norm <= 0) return;

        const t = (Math.log10(norm) - logMin) / logSpan;
        ctx.fillStyle = paletteColor(t);
        ctx.fillRect(si * cellW, pi * CELL_H, cellW, CELL_H);
      });
    });
  }, [data, logRange]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;

    const params = Object.keys(data);
    const maxSteps = Math.max(...params.map((p) => data[p].length), 1);

    const cellW = Math.max(2, Math.floor(Math.min(900, maxSteps * 6) / maxSteps));
    const rect = canvasRef.current.getBoundingClientRect();

    const pi = Math.floor((e.clientY - rect.top) / CELL_H);
    const si = Math.floor((e.clientX - rect.left) / cellW);
    const param = params[pi];
    if (param && si < data[param].length) {
      setHovered({param, step: si, norm: data[param][si]});
    }
  };

  if (isLoading) return <p className="animate-pulse text-sm text-muted-foreground">Ładowanie gradientów...</p>;
  if (isError || !data) return <p className="text-sm text-destructive">Brak danych gradientów dla epoki {epoch}.</p>;

  const params = Object.keys(data);
  if (params.length === 0) return <p className="text-sm text-muted-foreground">Brak danych gradientów.</p>;

  const maxSteps = Math.max(...params.map((p) => data[p].length), 1);
  const cellW = Math.max(2, Math.floor(Math.min(900, maxSteps * 6) / maxSteps));
  const canvasH = CELL_H * params.length;

  const xTicks: number[] = [];
  for (let s = 0; s < maxSteps; s += X_TICK_EVERY) xTicks.push(s);

  const handleExportPNG = () =>
    downloadFile(
      `/api/gradients/${experimentId}/${epoch}/heatmap.png`,
      `gradients_${experimentId}_epoch${epoch}.png`,
    );

  const handleExportCSV = () => {
    const header = ["parametr", ...Array.from({length: maxSteps}, (_, i) => `krok_${i}`)].join(",");
    const rows = params.map((p) => {
      const cells = data[p].map((v) => (v != null ? v.toExponential(6) : "0")).join(",");
      return `"${p.replace(/"/g, '""')}",${cells}`;
    });

    const blob = new Blob([[header, ...rows].join("\n")], {type: "text/csv;charset=utf-8;"});
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `gradients_${experimentId}_epoch${epoch}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="h-5 text-xs text-muted-foreground">
          {hovered ? (
            <span>
                Parametr: <strong className="text-foreground">{hovered.param}</strong>
              {" | "}Krok: <strong className="text-foreground">{hovered.step}</strong>
              {" | "}Norma: <strong
              className="font-mono text-foreground">{hovered.norm != null ? hovered.norm.toExponential(2) : "-"}</strong>
              </span>
          ) : (
            "Najedź kursorem na heatmapę"
          )}
        </div>

        <div className="flex shrink-0 gap-1.5">
          <button
            onClick={handleExportPNG}
            className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
            title="Pobierz heatmapę (PNG)"
          >PNG</button>

          <button
            onClick={handleExportCSV}
            className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
            title="Pobierz dane (CSV)"
          >CSV</button>
        </div>
      </div>


      <div className="flex items-start gap-0">
        <div className="min-w-0 flex-1 overflow-x-auto">
          <div className="flex items-start gap-0">
            <div className="shrink-0" style={{width: Y_LABEL_W}}>
              {params.map((p) => (
                <div
                  key={p}
                  style={{height: CELL_H, lineHeight: `${CELL_H}px`}}
                  className="truncate pr-2 text-right text-[10px] text-muted-foreground"
                  title={p}
                >
                  {p}
                </div>
              ))}
              <div style={{height: X_AXIS_H}}/>
            </div>

            <div className="flex flex-col">
              <canvas
                ref={canvasRef}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHovered(null)}
                className="cursor-crosshair rounded"
                style={{imageRendering: "pixelated"}}
              />

              <div className="relative" style={{height: X_AXIS_H, width: cellW * maxSteps}}>
                {xTicks.map((s) => (
                  <span
                    key={s}
                    className="absolute text-[9px] text-muted-foreground"
                    style={{left: s * cellW, transform: "translateX(-50%)"}}
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {logRange && (
          <div className="shrink-0">
            <VerticalColorbar
              logMin={logRange.logMin}
              logMax={logRange.logMax}
              canvasHeight={canvasH}
            />
          </div>
        )}
      </div>
    </div>
  );
}
