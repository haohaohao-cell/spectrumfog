/**
 * 查表层的单元测试：这里不验物理（物理在 Python 里测过），
 * 只验"解码/插值/边界"没有把 Python 算好的数弄坏。
 */

import { describe, expect, it } from "vitest";
import {
  type Bundle,
  type PackedTable,
  decodeTable,
  geomspace,
  mapExtent,
  ringAt,
  sampleTable,
  sliceAt,
} from "./bundle";

function pack(values: number[], rows: number, cols: number): PackedTable {
  const floats = new Float32Array(values);
  const bytes = new Uint8Array(floats.buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return { dtype: "float32", shape: [rows, cols], data: btoa(binary) };
}

describe("decodeTable", () => {
  it("还原 float32 表的数值与形状", () => {
    const table = decodeTable(pack([0, 0.5, 1, 0.25], 2, 2));
    expect(table.rows).toBe(2);
    expect(table.cols).toBe(2);
    expect(Array.from(table.values)).toEqual([0, 0.5, 1, 0.25]);
  });

  it("拒绝未知 dtype", () => {
    expect(() =>
      decodeTable({ dtype: "float64", shape: [1, 1], data: "" }),
    ).toThrow(/dtype/);
  });

  it("拒绝尺寸不符的表", () => {
    expect(() => decodeTable(pack([1, 2, 3], 2, 2))).toThrow(/尺寸/);
  });
});

describe("geomspace", () => {
  it("端点精确、比例恒定、单调递增", () => {
    const grid = geomspace(100, 1_000_000, 5);
    expect(grid[0]).toBe(100);
    expect(grid[4]).toBe(1_000_000);
    const ratios = [];
    for (let index = 1; index < grid.length; index += 1) {
      ratios.push(grid[index] / grid[index - 1]);
    }
    for (const ratio of ratios) {
      expect(ratio).toBeCloseTo(ratios[0], 9);
    }
    for (let index = 1; index < grid.length; index += 1) {
      expect(grid[index]).toBeGreaterThan(grid[index - 1]);
    }
  });

  it("参数非法时报错", () => {
    expect(() => geomspace(0, 10, 5)).toThrow();
    expect(() => geomspace(10, 1, 5)).toThrow();
    expect(() => geomspace(1, 10, 1)).toThrow();
  });
});

describe("sampleTable", () => {
  const rt = geomspace(100, 10_000, 3); // 100, 1000, 10000
  const rj = geomspace(1_000, 1_000_000, 3); // 1k, 31.6k, 1M
  const table = decodeTable(pack([0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5], 3, 3));

  it("落在网格点上时等于表里的值（不引入插值误差）", () => {
    expect(sampleTable(table, rt, rj, rt[0], rj[0])).toBeCloseTo(0, 6);
    expect(sampleTable(table, rt, rj, rt[2], rj[2])).toBeCloseTo(0.5, 6);
    expect(sampleTable(table, rt, rj, rt[1], rj[1])).toBeCloseTo(0.8, 6);
    expect(sampleTable(table, rt, rj, rt[2], rj[0])).toBeCloseTo(0.4, 6);
  });

  it("在 log 空间的几何中点取值等于两行平均", () => {
    const mid = Math.sqrt(rt[0] * rt[1]); // 316 m，log 中点
    const expected = (0.0 + 0.2) / 2;
    expect(sampleTable(table, rt, rj, mid, rj[0])).toBeCloseTo(expected, 5);
  });

  it("越界时夹到端点，不返回 NaN", () => {
    const low = sampleTable(table, rt, rj, 1e-3, 1e-3);
    const high = sampleTable(table, rt, rj, 1e9, 1e9);
    expect(low).toBeCloseTo(0, 6);
    expect(high).toBeCloseTo(0.5, 6);
    expect(Number.isFinite(low)).toBe(true);
  });
});

describe("sliceAt", () => {
  it("沿目标距离切一条与逐点取值一致的曲线", () => {
    const rt = geomspace(100, 10_000, 7);
    const rj = geomspace(1_000, 1_000_000, 5);
    const values = Array.from({ length: 35 }, (_, index) => index / 35);
    const table = decodeTable(pack(values, 5, 7));
    const rangeJammer = 42_000;

    const curve = sliceAt(table, rt, rj, rangeJammer);
    expect(curve.length).toBe(rt.length);
    for (let index = 0; index < rt.length; index += 1) {
      expect(curve[index]).toBeCloseTo(
        sampleTable(table, rt, rj, rt[index], rangeJammer),
        9,
      );
    }
  });
});

describe("ringAt", () => {
  const rings = {
    jammer_range_m: [1_000, 10_000, 100_000],
    detection_m: [1_000, 5_000, 20_000],
    burn_through_m: [2_000, 8_000, 30_000],
  };

  it("网格点上取到原值", () => {
    expect(ringAt(rings, 10_000).detection).toBeCloseTo(5_000, 6);
    expect(ringAt(rings, 100_000).burnThrough).toBeCloseTo(30_000, 6);
  });

  it("网格之间按 log 距离插值", () => {
    const mid = Math.sqrt(1_000 * 10_000); // 3162 m
    expect(ringAt(rings, mid).detection).toBeCloseTo(3_000, 6);
  });

  it("越界夹到端点", () => {
    expect(ringAt(rings, 1).detection).toBe(1_000);
    expect(ringAt(rings, 1e9).detection).toBe(20_000);
    expect(ringAt(rings, 0).detection).toBe(0);
  });
});

describe("mapExtent", () => {
  const bundle = {
    radar: { pos_m: [0, 0] },
    jammer: { pos_m: [60_000, 0] },
    target: { pos_m: [9_830, 6_880] },
    rings: {
      jammer_range_m: [1_000, 10_000, 100_000],
      detection_m: [1_000, 5_000, 20_000],
      burn_through_m: [2_000, 8_000, 30_000],
    },
  } as unknown as Bundle;

  it("按初始干扰机位置取环，而不是整条曲线的最大值", () => {
    // rj = 60 km → 环在 20 km 与 30 km 之间插值，远小于 100 km 那一端的最大值
    const extent = mapExtent(bundle, 60_000);
    expect(extent).toBeGreaterThan(60_000);
    expect(extent).toBeLessThan(80_000);
  });

  it("平台在很远时仍要包住平台", () => {
    const far = {
      ...bundle,
      jammer: { pos_m: [200_000, 0] },
    } as unknown as Bundle;
    const extent = mapExtent(far, 200_000);
    expect(extent).toBeGreaterThan(200_000);
  });
});
