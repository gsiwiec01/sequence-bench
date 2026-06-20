import { VIRIDIS_CSS } from "@/lib/viridis";

export default function LossColorbar({
  lossMin,
  lossMax,
  height,
}: {
  lossMin: number;
  lossMax: number;
  height: number;
}) {
  const N = 8;
  const ticks = Array.from({ length: N + 1 }, (_, i) => lossMin + (i / N) * (lossMax - lossMin));

  return (
    <div className="flex items-start gap-1">
      <div
        className="w-3 rounded"
        style={{ height, background: `linear-gradient(to top, ${VIRIDIS_CSS})`, flexShrink: 0 }}
      />

      <div className="relative ml-1" style={{ height, minWidth: 34 }}>
        {ticks.map((v, i) => (
          <div
            key={i}
            className="absolute flex items-center"
            style={{ top: `${((N - i) / N) * 100}%`, transform: "translateY(-50%)" }}
          >
            <span className="text-[9px] leading-none text-muted-foreground whitespace-nowrap">
              {v.toFixed(1)}
            </span>
          </div>
        ))}
      </div>

      <div className="relative ml-1 flex items-center justify-center" style={{ height }}>
        <span
          className="text-[10px] text-muted-foreground whitespace-nowrap"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          val_loss
        </span>
      </div>
    </div>
  );
}
