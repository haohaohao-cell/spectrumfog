/** 首屏冒烟测试：用真实数据包渲染，确保组件不会因为缺字段/NaN 崩掉。 */

import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import bundleJson from "../public/data.json";
import { Scene } from "./App";
import type { Bundle } from "./bundle";

const bundle = bundleJson as unknown as Bundle;

describe("Scene", () => {
  it("用真实数据包首屏渲染，给出关键读数", () => {
    const html = renderToString(
      <Scene
        bundle={bundle}
        jammer={[bundle.jammer.pos_m[0], bundle.jammer.pos_m[1]]}
        onJammer={() => {}}
      />,
    );

    expect(html).toContain("spectrum coverage explorer");
    expect(html).toContain("jammer range");
    expect(html).toContain("tracking boundary");
    // 60 km 处的干扰机压不住 12 km 的目标：首屏应当是 TRACKING
    expect(html).toContain("TRACKING");
    expect(html).not.toContain("NaN");
    expect(html).toContain("km");
  });

  it("干扰机贴脸时目标应当判为 JAMMED", () => {
    const html = renderToString(
      <Scene bundle={bundle} jammer={[1_000, 500]} onJammer={() => {}} />,
    );
    expect(html).toContain("JAMMED");
  });
});
