export const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 83, 160], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];

export const VIRIDIS_CSS = VIRIDIS.map(([r, g, b]) => `rgb(${r},${g},${b})`).join(", ");

export function viridis(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const i = Math.min(Math.floor(c * (VIRIDIS.length - 1)), VIRIDIS.length - 2);
  const f = c * (VIRIDIS.length - 1) - i;
  const [r1, g1, b1] = VIRIDIS[i];
  const [r2, g2, b2] = VIRIDIS[i + 1];

  return `rgb(${Math.round(r1 + (r2 - r1) * f)},${Math.round(g1 + (g2 - g1) * f)},${Math.round(b1 + (b2 - b1) * f)})`;
}
