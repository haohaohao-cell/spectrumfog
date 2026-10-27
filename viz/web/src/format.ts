/** 纯展示用的格式化与配色（与 Python 侧的调色板保持一致）。 */

export type RGB = [number, number, number];

export const COLOR_JAMMED: RGB = [192, 57, 43];
export const COLOR_MARGINAL: RGB = [200, 135, 26];
export const COLOR_SEEN: RGB = [47, 111, 191];

/** Pd → 颜色：0 全红（被压制）、0.5 琥珀（临界）、1 蓝（看得见）。 */
export function pdColor(pd: number): RGB {
  const value = Math.min(1, Math.max(0, pd));
  const [from, to, t] =
    value < 0.5
      ? ([COLOR_JAMMED, COLOR_MARGINAL, value / 0.5] as const)
      : ([COLOR_MARGINAL, COLOR_SEEN, (value - 0.5) / 0.5] as const);
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ];
}

export function fmtKm(meters: number): string {
  if (!Number.isFinite(meters)) return "—";
  if (meters >= 1000) return `${(meters / 1000).toFixed(1)} km`;
  return `${meters.toFixed(0)} m`;
}

export function fmtNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return value.toExponential(2);
  if (value >= 1) return value.toFixed(2);
  return value.toExponential(2);
}
