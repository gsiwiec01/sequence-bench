import { useMemo, useState } from "react";
import { contours } from "d3-contour";
import type { WeightTrajectory, LossLandscape } from "@/api/experiments";
import LossColorbar from "@/components/LossColorbar";
import { viridis } from "@/lib/viridis";

// SVG constants
const W = 600, H = 460, PAD = 50;
const AX_COL  = "#374151";
const MUT_COL = "#6b7280";
const TRAJ    = "#ef4444";

interface PlotPoint { x: number; y: number; epoch: number; wx: number; wy: number }
interface AxisBounds { wxMin: number; wxMax: number; wyMin: number; wyMax: number }

function makeScales(bounds: AxisBounds) {
  const { wxMin, wxMax, wyMin, wyMax } = bounds;
  const rx = wxMax - wxMin || 1;
  const ry = wyMax - wyMin || 1;
  return {
    scaleX: (v: number) => PAD + ((v - wxMin) / rx) * (W - 2 * PAD),
    scaleY: (v: number) => H - PAD - ((v - wyMin) / ry) * (H - 2 * PAD),
  };
}

function trajectoryBounds(trajectory: (number | null)[][], epochs: number[]): AxisBounds | null {
  const raw = epochs
    .map((_, i) => ({ wx: trajectory[i]?.[0] ?? null, wy: trajectory[i]?.[1] ?? null }))
    .filter((p): p is { wx: number; wy: number } => p.wx != null && p.wy != null);
  if (!raw.length) return null;
  const wxs = raw.map((p) => p.wx), wys = raw.map((p) => p.wy);
  const wxMin = Math.min(...wxs), wxMax = Math.max(...wxs);
  const wyMin = Math.min(...wys), wyMax = Math.max(...wys);
  const dx = (wxMax - wxMin) * 0.12 || 0.05;
  const dy = (wyMax - wyMin) * 0.12 || 0.05;
  return { wxMin: wxMin - dx, wxMax: wxMax + dx, wyMin: wyMin - dy, wyMax: wyMax + dy };
}

function computePoints(
  trajectory: (number | null)[][],
  epochs: number[],
  scaleX: (v: number) => number,
  scaleY: (v: number) => number,
): PlotPoint[] {
  return epochs
    .map((epoch, i) => ({ epoch, wx: trajectory[i]?.[0] ?? null, wy: trajectory[i]?.[1] ?? null }))
    .filter((p): p is { epoch: number; wx: number; wy: number } => p.wx != null && p.wy != null)
    .map((p) => ({ x: scaleX(p.wx), y: scaleY(p.wy), epoch: p.epoch, wx: p.wx, wy: p.wy }));
}

function contourRingsToPath(
  coordinates: number[][][][],
  col2x: (c: number) => number,
  row2y: (r: number) => number,
): string {
  const parts: string[] = [];
  for (const polygon of coordinates) {
    for (const ring of polygon) {
      if (ring.length < 2) continue;
      parts.push(ring.map(([c, r], i) => `${i === 0 ? "M" : "L"}${col2x(c).toFixed(1)},${row2y(r).toFixed(1)}`).join("") + "Z");
    }
  }

  return parts.join(" ");
}

function contourLabelPos(
  coordinates: number[][][][],
  col2x: (c: number) => number,
  row2y: (r: number) => number,
): { x: number; y: number } | null {
  for (const polygon of coordinates) {
    const ring = polygon[0];
    if (!ring || ring.length < 3) continue;

    const mid = ring[Math.floor(ring.length / 3)];
    if (!mid) continue;

    const x = col2x(mid[0]), y = row2y(mid[1]);
    if (x >= PAD + 4 && x <= W - PAD - 4 && y >= PAD + 4 && y <= H - PAD - 4) {
      return { x, y };
    }
  }
  return null;
}

function smooth2d(grid: number[][], passes = 2): number[][] {
  let g = grid;
  for (let p = 0; p < passes; p++) {
    const rows = g.length, cols = g[0]?.length ?? 0;
    if (rows < 3 || cols < 3) break;
    const next = g.map((r) => [...r]);
    for (let r = 1; r < rows - 1; r++) {
      for (let c = 1; c < cols - 1; c++) {
        next[r][c] = (
          g[r-1][c-1] + g[r-1][c]*2 + g[r-1][c+1] +
          g[r][c-1]*2 + g[r][c]*4   + g[r][c+1]*2 +
          g[r+1][c-1] + g[r+1][c]*2 + g[r+1][c+1]
        ) / 16;
      }
    }
    g = next;
  }
  return g;
}

export default function TrajectoryPlot({
  data,
  landscape,
  pngPath,
}: {
  data: WeightTrajectory;
  landscape?: LossLandscape;
  pngPath?: string;
}) {
  const [hovered, setHovered] = useState<{ epoch: number; wx: number; wy: number } | null>(null);

  const surfaceReady =
    landscape?.status === "completed" && !!landscape.loss_grid &&
    !!landscape.x_range && !!landscape.y_range;

  const trajSource: WeightTrajectory = useMemo(() => {
    if (surfaceReady && landscape?.a_traj && landscape?.b_traj) {
      return {
        pairs: ["PC1", "PC2"],
        trajectory: landscape.a_traj.map((a, i) => [a, landscape.b_traj![i] ?? null] as (number | null)[]),
        epochs: landscape.a_traj.map((_, i) => i),
      };
    }
    return data;
  }, [surfaceReady, landscape, data]);

  const bounds: AxisBounds = useMemo(() => {
    if (surfaceReady && landscape?.x_range && landscape?.y_range) {
      return { wxMin: landscape.x_range[0], wxMax: landscape.x_range[1], wyMin: landscape.y_range[0], wyMax: landscape.y_range[1] };
    }
    return trajectoryBounds(trajSource.trajectory, trajSource.epochs) ?? { wxMin: -1, wxMax: 1, wyMin: -1, wyMax: 1 };
  }, [surfaceReady, landscape, trajSource]);

  const { scaleX, scaleY } = useMemo(() => makeScales(bounds), [bounds]);
  const points = useMemo(() => computePoints(trajSource.trajectory, trajSource.epochs, scaleX, scaleY), [trajSource, scaleX, scaleY]);
  const explained = landscape?.explained_variance ?? data.explained_variance ?? null;

  const contourData = useMemo(() => {
    if (!surfaceReady || !landscape?.loss_grid || !landscape.x_range || !landscape.y_range) return null;
    const { loss_grid, x_range, y_range } = landscape;
    const [xMin, xMax] = x_range, [yMin, yMax] = y_range;
    const gs = loss_grid.length;
    const smoothed = smooth2d([...loss_grid].reverse());
    const flat = smoothed.flat() as number[];

    let lossMin = Infinity, lossMax = -Infinity;
    for (const v of flat) { if (v < lossMin) lossMin = v; if (v > lossMax) lossMax = v; }
    if (!isFinite(lossMin) || lossMax <= lossMin) return null;
    const lossSpan = lossMax - lossMin;

    const col2x = (c: number) => scaleX(xMin + (c / gs) * (xMax - xMin));
    const row2y = (r: number) => scaleY(yMax - (r / gs) * (yMax - yMin));
    const levels = contours().size([gs, gs]).thresholds(7)(flat);

    const paths: JSX.Element[] = [];
    const labels: JSX.Element[] = [];

    levels.forEach((level, li) => {
      const coords = level?.coordinates as number[][][][];
      if (!coords?.length) return;
      const t = (level.value - lossMin) / lossSpan;
      const color = viridis(t);
      const d = contourRingsToPath(coords, col2x, row2y);
      if (!d) return;

      paths.push(
        <path key={`f${li}`} d={d} fill={color} fillOpacity={1} stroke={color} strokeWidth={0.6} strokeOpacity={1} />,
      );

      const pos = contourLabelPos(coords, col2x, row2y);
      if (pos) {
        labels.push(
          <text key={`l${li}`} x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="middle" fontSize={9} fill={MUT_COL} opacity={0.9}>
            {li + 1}
          </text>,
        );
      }
    });

    return { paths, labels, lossMin, lossMax };
  }, [surfaceReady, landscape, scaleX, scaleY]);

  const xt = Array.from({ length: 5 }, (_, i) => {
    const v = bounds.wxMin + (i / 4) * (bounds.wxMax - bounds.wxMin);
    return { v, x: scaleX(v) };
  });
  const yt = Array.from({ length: 5 }, (_, i) => {
    const v = bounds.wyMin + (i / 4) * (bounds.wyMax - bounds.wyMin);
    return { v, y: scaleY(v) };
  });

  const xLabel = trajSource.pairs[0] ?? "PC1";
  const yLabel = trajSource.pairs[1] ?? "PC2";
  const startPt = points[0];
  const endPt   = points[points.length - 1];

  function downloadFile(path: string, name: string) {
    const a = document.createElement("a");
    a.href = path;
    a.download = name;
    a.click();
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="h-5 text-xs text-muted-foreground font-mono">
          {explained != null && (
            <div className="text-muted-foreground text-xs">
              Wariancja toru wag: <strong>{(explained * 100).toFixed(1)}%</strong>
              {landscape?.anchor_loss != null && <>, punkt odniesienia: {landscape.anchor_loss.toFixed(4)}</>}
            </div>
          )}
          {hovered
            ? <span>Epoka {hovered.epoch}: PC1={hovered.wx.toFixed(4)}, PC2={hovered.wy.toFixed(4)}</span>
            : "Najedź na punkt"}
        </div>

        {pngPath && (
          <button
            onClick={() => downloadFile(pngPath, "surface_pca.png")}
            className="rounded border px-2 py-0.5 text-xs hover:bg-muted shrink-0"
            title="Pobierz PNG (render na serwerze)"
          >
            PNG
          </button>
        )}
      </div>

      <div className="flex items-start gap-1 overflow-x-auto flex justify-center">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="select-none shrink-0">
          {xt.map(({ x }, i) => (
            <line key={i} x1={x} y1={PAD} x2={x} y2={H - PAD} stroke={AX_COL} strokeOpacity={0.08} strokeWidth={0.5} />
          ))}

          {yt.map(({ y }, i) => (
            <line key={i} x1={PAD} y1={y} x2={W - PAD} y2={y} stroke={AX_COL} strokeOpacity={0.08} strokeWidth={0.5} />
          ))}

          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke={AX_COL} strokeWidth={1} />
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke={AX_COL} strokeWidth={1} />

          {xt.map(({ v, x }) => (
            <g key={v}>
              <line x1={x} y1={H - PAD} x2={x} y2={H - PAD + 4} stroke={AX_COL} strokeWidth={1} />
              <text x={x} y={H - PAD + 14} textAnchor="middle" fontSize={9} fill={MUT_COL}>
                {Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(3) : v.toExponential(1)}
              </text>
            </g>
          ))}

          {yt.map(({ v, y }) => (
            <g key={v}>
              <line x1={PAD - 4} y1={y} x2={PAD} y2={y} stroke={AX_COL} strokeWidth={1} />
              <text x={PAD - 7} y={y} textAnchor="end" dominantBaseline="middle" fontSize={9} fill={MUT_COL}>
                {Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(3) : v.toExponential(1)}
              </text>
            </g>
          ))}

          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={11} fontWeight="bold" fill={AX_COL}>{xLabel}</text>
          <text x={13} y={H / 2} textAnchor="middle" fontSize={11} fontWeight="bold" fill={AX_COL} transform={`rotate(-90, 13, ${H / 2})`}>{yLabel}</text>

          {contourData?.paths}
          {contourData?.labels}
          {points.length > 1 && (
            <path
              d={points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
              fill="none" stroke={TRAJ} strokeWidth={1.5}
            />
          )}

          {points.map((p, i) => {
            const isEnd = i === points.length - 1;
            if (isEnd) return null;
            return (
              <circle
                key={i} cx={p.x} cy={p.y} r={3}
                fill={TRAJ} stroke="none"
                className="cursor-pointer"
                onMouseEnter={() => setHovered({ epoch: p.epoch, wx: p.wx, wy: p.wy })}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}

          {startPt && (
            <circle
              cx={startPt.x} cy={startPt.y} r={5}
              fill="none" stroke="white" strokeWidth={2}
              className="cursor-pointer"
              onMouseEnter={() => setHovered({ epoch: startPt.epoch, wx: startPt.wx, wy: startPt.wy })}
              onMouseLeave={() => setHovered(null)}
            />
          )}

          {endPt && (
            <text
              x={endPt.x} y={endPt.y}
              textAnchor="middle" dominantBaseline="central"
              fontSize={18} fill="#f5f500"
              className="cursor-pointer"
              onMouseEnter={() => setHovered({ epoch: endPt.epoch, wx: endPt.wx, wy: endPt.wy })}
              onMouseLeave={() => setHovered(null)}
            >★</text>
          )}
        </svg>

        {contourData && (
          <div style={{ marginTop: PAD, marginBottom: PAD - 50, marginLeft: -40 }}>
            <LossColorbar lossMin={contourData.lossMin} lossMax={contourData.lossMax} height={H - 2 * PAD} />
          </div>
        )}
      </div>

      <div className="flex justify-center items-center gap-6 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <svg width="28" height="12" className="shrink-0">
            <line x1="0" y1="6" x2="28" y2="6" stroke={TRAJ} strokeWidth="2" />
            <circle cx="14" cy="6" r="3" fill={TRAJ} />
          </svg>
          <span>trajektoria wag</span>
        </div>

        <div className="flex items-center gap-1.5">
          <svg width="16" height="16" className="shrink-0">
            <circle cx="8" cy="8" r="5" fill="none" stroke={AX_COL} strokeWidth="1.5" />
          </svg>
          <span>start</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-base leading-none" style={{ color: "#ca8a04" }}>★</span>
          <span>koniec</span>
        </div>
      </div>
    </div>
  );
}
