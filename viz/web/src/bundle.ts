/**
 * 数据包解码与查表 —— 前端唯一"像物理"的部分，但这里没有任何公式。
 *
 * 契约（docs/web.md）：
 *   pd[i][j] 对应 Rt = geomspace(rt_m)[j]、Rj = geomspace(rj_m)[i]（行 = 干扰机距离）。
 *   两个轴都是 geomspace，所以插值必须在 log 空间做，否则近距段会明显失真。
 */

export const SCHEMA = "spectrumfog.web/1";

export interface Axis {
  min: number;
  max: number;
  n: number;
}

export interface PackedTable {
  dtype: string;
  shape: number[];
  data: string;
}

export interface RingSeries {
  jammer_range_m: number[];
  detection_m: number[];
  burn_through_m: number[];
}

export interface Bundle {
  schema: string;
  scenario: string;
  units: string;
  pfa: number;
  pd_threshold: number;
  noise_power_w: number;
  radar: { id: string; pos_m: [number, number] };
  jammer: { id: string; pos_m: [number, number] };
  target: { id: string; pos_m: [number, number]; range_m: number };
  grid: { rt_m: Axis; rj_m: Axis; spacing: string; units: string };
  pd: PackedTable;
  js: PackedTable;
  rings: RingSeries;
  target_curves: { jammer_range_m: number[]; pd: number[]; js: number[] };
}

export interface Table {
  values: Float32Array;
  rows: number; // Rj
  cols: number; // Rt
}

/** base64（小端 float32）→ 二维表。 */
export function decodeTable(packed: PackedTable): Table {
  if (packed.dtype !== "float32") {
    throw new Error(`不支持的 dtype: ${packed.dtype}`);
  }
  const binary = atob(packed.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const values = new Float32Array(bytes.buffer);
  const [rows, cols] = packed.shape;
  if (values.length !== rows * cols) {
    throw new Error(`表尺寸不符：${values.length} != ${rows}×${cols}`);
  }
  return { values, rows, cols };
}

/** 与 numpy.geomspace 对齐的生成方式（n 个点，含两端）。 */
export function geomspace(min: number, max: number, n: number): Float64Array {
  if (!(min > 0) || !(max > min) || n < 2) {
    throw new Error(`geomspace 参数非法：[${min}, ${max}] n=${n}`);
  }
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = 10 ** (lo + ((hi - lo) * i) / (n - 1));
  }
  out[0] = min;
  out[n - 1] = max;
  return out;
}

/** 在 log 轴上找夹住 value 的两个下标与插值系数（越界时夹到端点）。 */
function bracket(
  grid: Float64Array,
  value: number,
): { i0: number; i1: number; t: number } {
  const n = grid.length;
  if (!(value > 0)) return { i0: 0, i1: 0, t: 0 };
  if (value <= grid[0]) return { i0: 0, i1: 0, t: 0 };
  if (value >= grid[n - 1]) return { i0: n - 1, i1: n - 1, t: 0 };

  const lo = Math.log10(grid[0]);
  const hi = Math.log10(grid[n - 1]);
  const step = (hi - lo) / (n - 1);
  const pos = (Math.log10(value) - lo) / step;
  const i0 = Math.min(Math.floor(pos), n - 2);
  return { i0, i1: i0 + 1, t: pos - i0 };
}

/** 双线性取值：给定目标距离与干扰机距离，返回表里的值。 */
export function sampleTable(
  table: Table,
  rtGrid: Float64Array,
  rjGrid: Float64Array,
  rangeTarget: number,
  rangeJammer: number,
): number {
  const row = bracket(rjGrid, rangeJammer);
  const col = bracket(rtGrid, rangeTarget);
  const at = (i: number, j: number) => table.values[i * table.cols + j];

  const top = at(row.i0, col.i0) * (1 - col.t) + at(row.i0, col.i1) * col.t;
  const bottom = at(row.i1, col.i0) * (1 - col.t) + at(row.i1, col.i1) * col.t;
  return top * (1 - row.t) + bottom * row.t;
}

/** 固定干扰机距离，沿目标距离切一条曲线（给 Pd/J-S 图用）。 */
export function sliceAt(
  table: Table,
  rtGrid: Float64Array,
  rjGrid: Float64Array,
  rangeJammer: number,
): Float64Array {
  const out = new Float64Array(rtGrid.length);
  for (let j = 0; j < rtGrid.length; j += 1) {
    out[j] = sampleTable(table, rtGrid, rjGrid, rtGrid[j], rangeJammer);
  }
  return out;
}

/** 从 241 点的一维曲线里按干扰机距离取环半径（log 空间线性插值）。 */
export function ringAt(rings: RingSeries, rangeJammer: number): {
  detection: number;
  burnThrough: number;
} {
  const xs = rings.jammer_range_m;
  const n = xs.length;
  if (!(rangeJammer > 0) || n === 0) {
    return { detection: 0, burnThrough: 0 };
  }
  if (rangeJammer <= xs[0]) {
    return { detection: rings.detection_m[0], burnThrough: rings.burn_through_m[0] };
  }
  if (rangeJammer >= xs[n - 1]) {
    return {
      detection: rings.detection_m[n - 1],
      burnThrough: rings.burn_through_m[n - 1],
    };
  }
  const lo = Math.log10(xs[0]);
  const hi = Math.log10(xs[n - 1]);
  const pos = ((Math.log10(rangeJammer) - lo) / (hi - lo)) * (n - 1);
  const i0 = Math.min(Math.floor(pos), n - 2);
  const t = pos - i0;
  return {
    detection: rings.detection_m[i0] * (1 - t) + rings.detection_m[i0 + 1] * t,
    burnThrough:
      rings.burn_through_m[i0] * (1 - t) + rings.burn_through_m[i0 + 1] * t,
  };
}

/** 地图半宽：包住两个平台与最大的一圈环，固定下来避免拖动时画面跳动。 */
export function mapExtent(bundle: Bundle): number {
  const points = [bundle.radar.pos_m, bundle.jammer.pos_m, bundle.target.pos_m];
  let reach = 0;
  for (const [x, y] of points) {
    reach = Math.max(reach, Math.hypot(x, y));
  }
  const widestRing = Math.max(
    ...bundle.rings.burn_through_m,
    ...bundle.rings.detection_m,
  );
  return Math.max(reach, widestRing) * 1.12;
}
