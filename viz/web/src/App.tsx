/**
 * 交互式频谱覆盖浏览器（T2）。
 *
 * 拖动红色干扰机 → 重新查表 → 热力图、两条曲线和两个环即时更新。
 * 这里没有任何物理公式：全部数值来自 data.json（由 `spectrumfog export-web` 生成）。
 */

import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type Bundle,
  type Table,
  SCHEMA,
  decodeTable,
  geomspace,
  mapExtent,
  ringAt,
  sampleTable,
  sliceAt,
} from "./bundle";
import { fmtKm, fmtNumber, pdColor } from "./format";

const VIEW = 640; // SVG 视图边长（正方形）
const HEAT = 220; // 热力图离线分辨率，CSS 放大显示

export default function App() {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jammer, setJammer] = useState<[number, number] | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("data.json")
      .then((response) => {
        if (!response.ok) throw new Error(`data.json 请求失败：${response.status}`);
        return response.json() as Promise<Bundle>;
      })
      .then((payload) => {
        if (cancelled) return;
        if (payload.schema !== SCHEMA) {
          throw new Error(`数据包版本不符：${payload.schema}（期望 ${SCHEMA}）`);
        }
        setBundle(payload);
        setJammer([payload.jammer.pos_m[0], payload.jammer.pos_m[1]]);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="fatal">
        <h1>data.json 载入失败</h1>
        <pre>{error}</pre>
        <p>
          先在仓库根目录跑：
          <code>
            spectrumfog export-web scenarios/02_standoff.yaml -o
            viz/web/public/data.json
          </code>
        </p>
      </div>
    );
  }
  if (!bundle || !jammer) {
    return <div className="loading">载入 data.json …</div>;
  }
  return <Scene bundle={bundle} jammer={jammer} onJammer={setJammer} />;
}

interface SceneProps {
  bundle: Bundle;
  jammer: [number, number];
  onJammer: (pos: [number, number]) => void;
}

export function Scene({ bundle, jammer, onJammer }: SceneProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const extent = useMemo(() => mapExtent(bundle), [bundle]);
  const scale = VIEW / (2 * extent);

  const rtGrid = useMemo(
    () => geomspace(bundle.grid.rt_m.min, bundle.grid.rt_m.max, bundle.grid.rt_m.n),
    [bundle],
  );
  const rjGrid = useMemo(
    () => geomspace(bundle.grid.rj_m.min, bundle.grid.rj_m.max, bundle.grid.rj_m.n),
    [bundle],
  );
  const pdTable: Table = useMemo(() => decodeTable(bundle.pd), [bundle]);
  const jsTable: Table = useMemo(() => decodeTable(bundle.js), [bundle]);

  const targetRange = bundle.target.range_m;
  const rangeJammer = Math.hypot(
    jammer[0] - bundle.radar.pos_m[0],
    jammer[1] - bundle.radar.pos_m[1],
  );

  const rings = useMemo(
    () => ringAt(bundle.rings, rangeJammer),
    [bundle.rings, rangeJammer],
  );
  const pdAtTarget = useMemo(
    () => sampleTable(pdTable, rtGrid, rjGrid, targetRange, rangeJammer),
    [pdTable, rtGrid, rjGrid, targetRange, rangeJammer],
  );
  const jsAtTarget = useMemo(
    () => sampleTable(jsTable, rtGrid, rjGrid, targetRange, rangeJammer),
    [jsTable, rtGrid, rjGrid, targetRange, rangeJammer],
  );
  const pdCurve = useMemo(
    () => sliceAt(pdTable, rtGrid, rjGrid, rangeJammer),
    [pdTable, rtGrid, rjGrid, rangeJammer],
  );
  const jsCurve = useMemo(
    () => sliceAt(jsTable, rtGrid, rjGrid, rangeJammer),
    [jsTable, rtGrid, rjGrid, rangeJammer],
  );

  // --- 热力图：低分辨率 canvas，CSS 放大，避免每帧几十万次查表 ---
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    canvas.width = HEAT;
    canvas.height = HEAT;
    const image = context.createImageData(HEAT, HEAT);
    for (let py = 0; py < HEAT; py += 1) {
      const wy = extent - ((py + 0.5) / HEAT) * 2 * extent;
      for (let px = 0; px < HEAT; px += 1) {
        const wx = ((px + 0.5) / HEAT) * 2 * extent - extent;
        const rt = Math.hypot(wx, wy);
        const rj = Math.hypot(wx - jammer[0], wy - jammer[1]);
        const [red, green, blue] = pdColor(
          sampleTable(pdTable, rtGrid, rjGrid, rt, rj),
        );
        const offset = 4 * (py * HEAT + px);
        image.data[offset] = red;
        image.data[offset + 1] = green;
        image.data[offset + 2] = blue;
        image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [jammer, extent, pdTable, rtGrid, rjGrid]);

  const toWorld = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const svg = svgRef.current;
      if (!svg) return [0, 0];
      const rect = svg.getBoundingClientRect();
      const vx = ((clientX - rect.left) / rect.width) * VIEW;
      const vy = ((clientY - rect.top) / rect.height) * VIEW;
      const clamp = (value: number) => Math.max(-extent, Math.min(extent, value));
      return [clamp(vx / scale - extent), clamp(extent - vy / scale)];
    },
    [extent, scale],
  );

  const handlePointer = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      onJammer(toWorld(event.clientX, event.clientY));
    },
    [onJammer, toWorld],
  );

  const toView = useCallback(
    (x: number, y: number): [number, number] => [
      (x + extent) * scale,
      (extent - y) * scale,
    ],
    [extent, scale],
  );

  const [radarX, radarY] = toView(bundle.radar.pos_m[0], bundle.radar.pos_m[1]);
  const [jammerX, jammerY] = toView(jammer[0], jammer[1]);
  const [targetX, targetY] = toView(
    bundle.target.pos_m[0],
    bundle.target.pos_m[1],
  );

  const jammed = pdAtTarget < bundle.pd_threshold;
  const referenceRings = [0.25, 0.5, 0.75].map((fraction) => fraction * extent);

  return (
    <div className="app">
      <header>
        <h1>SpectrumFog · spectrum coverage explorer</h1>
        <p className="sub">
          {bundle.scenario} · drag the red jammer · physics precomputed by Python,
          this page only interpolates
        </p>
      </header>

      <main>
        <div className="map">
          <canvas className="heat" ref={canvasRef} />
          <svg
            ref={svgRef}
            className="overlay"
            viewBox={`0 0 ${VIEW} ${VIEW}`}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              setDragging(true);
              handlePointer(event);
            }}
            onPointerMove={(event) => {
              if (dragging) handlePointer(event);
            }}
            onPointerUp={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
              setDragging(false);
            }}
            onPointerCancel={() => setDragging(false)}
          >
            <g className="range-rings">
              {referenceRings.map((radius) => (
                <circle key={radius} cx={radarX} cy={radarY} r={radius * scale} />
              ))}
            </g>

            <circle
              className="ring detection"
              cx={radarX}
              cy={radarY}
              r={Math.max(0.5, rings.detection * scale)}
            />
            <circle
              className="ring burn"
              cx={radarX}
              cy={radarY}
              r={Math.max(0.5, rings.burnThrough * scale)}
            />

            <line className="link" x1={radarX} y1={radarY} x2={jammerX} y2={jammerY} />
            <line className="link" x1={radarX} y1={radarY} x2={targetX} y2={targetY} />

            <g className="target" transform={`translate(${targetX} ${targetY})`}>
              <circle r={7} className={jammed ? "target-dot jammed" : "target-dot"} />
              <text x={11} y={4}>target</text>
            </g>

            <g className="jammer" transform={`translate(${jammerX} ${jammerY})`}>
              <rect x={-7} y={-7} width={14} height={14} rx={2} />
              <text x={12} y={4}>jammer</text>
            </g>

            <g className="radar" transform={`translate(${radarX} ${radarY})`}>
              <path d="M0,-11 L9,7 L-9,7 Z" />
              <text x={12} y={4}>radar</text>
            </g>

            <text className="scale-label" x={12} y={VIEW - 14}>
              {`0 – ${(extent / 1000).toFixed(0)} km`}
            </text>
          </svg>
        </div>

        <aside>
          <div className={jammed ? "badge jammed" : "badge tracking"}>
            {jammed ? "JAMMED" : "TRACKING"}
          </div>

          <dl className="readout">
            <dt>jammer range</dt>
            <dd>{fmtKm(rangeJammer)}</dd>
            <dt>target range</dt>
            <dd>{fmtKm(targetRange)}</dd>
            <dt>J/S at target</dt>
            <dd>{fmtNumber(jsAtTarget)}</dd>
            <dt>Pd at target</dt>
            <dd>{fmtNumber(pdAtTarget)}</dd>
            <dt>tracking boundary</dt>
            <dd>{fmtKm(rings.detection)}</dd>
            <dt>J/S = 1 ring</dt>
            <dd>{fmtKm(rings.burnThrough)}</dd>
          </dl>

          <Chart
            title="Pd vs target range"
            values={pdCurve}
            xs={rtGrid}
            targetRange={targetRange}
            targetValue={pdAtTarget}
            yMin={0}
            yMax={1}
            logY={false}
            threshold={bundle.pd_threshold}
            color="#2f6fbf"
          />
          <Chart
            title="J/S vs target range"
            values={jsCurve}
            xs={rtGrid}
            targetRange={targetRange}
            targetValue={jsAtTarget}
            logY
            threshold={1}
            color="#c0392b"
          />
        </aside>
      </main>

      <footer>
        fictional parameters · educational simulation · SpectrumFog T2 · Pfa{" "}
        {bundle.pfa.toExponential(0)} · Pd threshold {bundle.pd_threshold}
      </footer>
    </div>
  );
}

interface ChartProps {
  title: string;
  values: Float64Array;
  xs: Float64Array;
  targetRange: number;
  targetValue: number;
  yMin?: number;
  yMax?: number;
  logY: boolean;
  threshold: number;
  color: string;
}

function Chart({
  title,
  values,
  xs,
  targetRange,
  targetValue,
  yMin = 0,
  yMax,
  logY,
  threshold,
  color,
}: ChartProps) {
  const width = 420;
  const height = 132;
  const pad = { left: 52, right: 10, top: 12, bottom: 20 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const xLo = Math.log10(xs[0]);
  const xHi = Math.log10(xs[xs.length - 1]);
  const xAt = (range: number) =>
    pad.left +
    ((Math.log10(Math.max(range, xs[0])) - xLo) / (xHi - xLo)) * plotWidth;

  const finite = Array.from(values).filter((value) => Number.isFinite(value));
  const dataMax = finite.length ? Math.max(...finite) : 1;
  const hi = yMax ?? (dataMax * 1.1 || 1);
  const lo = yMin;

  const positive = finite.filter((value) => value > 0);
  const yLogLo = Math.max(
    1e-8,
    Math.min(positive.length ? Math.min(...positive) : 1e-3, threshold * 1e-2) * 0.8,
  );
  const yLogHi =
    Math.max(positive.length ? Math.max(...positive) : 1, threshold * 10) * 1.25;

  const yAt = (value: number): number => {
    if (logY) {
      const clamped = Math.max(value, yLogLo);
      const fraction =
        (Math.log10(clamped) - Math.log10(yLogLo)) /
        (Math.log10(yLogHi) - Math.log10(yLogLo));
      return pad.top + plotHeight * (1 - fraction);
    }
    return pad.top + plotHeight * (1 - (value - lo) / (hi - lo));
  };

  const points: string[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    points.push(
      `${index === 0 ? "M" : "L"}${xAt(xs[index]).toFixed(1)},${yAt(value).toFixed(1)}`,
    );
  }

  const targetX = xAt(targetRange);
  const targetY = yAt(targetValue);

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <rect
          className="plot-bg"
          x={pad.left}
          y={pad.top}
          width={plotWidth}
          height={plotHeight}
        />
        {[0, 0.5, 1].map((fraction) => {
          const y = pad.top + plotHeight * (1 - fraction);
          const value = logY
            ? Math.pow(
                10,
                Math.log10(yLogLo) +
                  fraction * (Math.log10(yLogHi) - Math.log10(yLogLo)),
              )
            : lo + fraction * (hi - lo);
          return (
            <g key={fraction}>
              <line
                className="gridline"
                x1={pad.left}
                x2={pad.left + plotWidth}
                y1={y}
                y2={y}
              />
              <text className="tick" x={pad.left - 6} y={y + 3} textAnchor="end">
                {logY ? value.toExponential(0) : value.toFixed(1)}
              </text>
            </g>
          );
        })}
        <line
          className="threshold"
          x1={pad.left}
          x2={pad.left + plotWidth}
          y1={yAt(threshold)}
          y2={yAt(threshold)}
        />
        <path className="curve" d={points.join(" ")} stroke={color} />
        <line
          className="cursor"
          x1={targetX}
          x2={targetX}
          y1={pad.top}
          y2={pad.top + plotHeight}
        />
        <circle className="cursor-dot" cx={targetX} cy={targetY} r={3.5} fill={color} />
        <text className="tick" x={pad.left} y={height - 5}>
          0.1 km
        </text>
        <text className="tick" x={pad.left + plotWidth} y={height - 5} textAnchor="end">
          {`${(xs[xs.length - 1] / 1000).toFixed(0)} km`}
        </text>
      </svg>
    </figure>
  );
}
