"""AAR 复盘：replay → 关键事件时间线 + 暴露时刻 + 定位误差归因 + 单页 HTML。

设计约束
--------
* **只读 replay**：不 import 引擎、不重算物理。所以报告里的每个数字都能由日志行复算。
* 归因不是"感觉哪台传感器有用"，而是**去掉它重解一次**：留一法（leave-one-out）。
  既不改变模型，也不需要额外参数，容易被审。
* 输出一句人话结论（PROJECT.md §9 T3a 的验收就是这个）：
  "你的 jammer_b 在 t = X s 被有效定位（CEP ≈ Y m），原因是……"
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .engine import fusion
from .metrics import cep_from_covariance, exposure_windows, t_expose
from .replay import Frame, Replay

__all__ = ["AarReport", "analyse", "render_html", "write_html"]


@dataclass(frozen=True)
class KeyEvent:
    t_s: float
    kind: str
    detail: str


@dataclass(frozen=True)
class AarReport:
    emitter: str
    threshold_m: float
    n_ticks: int
    n_fixes: int
    fix_rate: float
    first_fix_s: float | None
    first_effective_s: float | None
    first_effective_cep_m: float | None
    best_cep_m: float | None
    best_cep_s: float | None
    median_true_error_m: float | None
    t_expose_s: float
    cep_series: tuple[tuple[float, float, float], ...]  # (t, 自报 CEP, 真实误差)
    timeline: tuple[KeyEvent, ...]
    attribution: tuple[dict[str, Any], ...]
    sentence: str
    map_data: dict[str, Any] = field(default_factory=dict)


def _solve(measurements: Sequence[dict[str, Any]]) -> fusion.Fix | None:
    """用给定量测子集重解一次。"""
    if not measurements:
        return None
    return fusion.solve_fix(
        [
            fusion.Measure(
                sensor_id=str(item["sensor"]),
                sensor_pos=(
                    float(item["sensor_pos"][0]),
                    float(item["sensor_pos"][1]),
                ),
                aoa_rad=item.get("aoa_rad"),
                toa_s=item.get("toa_s"),
                sigma_aoa_rad=float(item.get("sigma_aoa_rad") or fusion.DEFAULT_SIGMA_AOA_RAD),
                sigma_toa_s=float(item.get("sigma_toa_s") or fusion.DEFAULT_SIGMA_TOA_S),
            )
            for item in measurements
        ]
    )


def analyse(
    replay: Replay,
    *,
    threshold_m: float | None = None,
    emitter: str | None = None,
) -> AarReport:
    """把一份 replay 变成可解释的复盘结论。"""
    meta = replay.meta
    # 被定位的是"辐射源"（localize objective），不是雷达要探测的目标（meta.target）
    target = (
        emitter
        or meta.get("localize_target")
        or str(meta.get("target", ""))
    )
    limit = float(
        threshold_m
        if threshold_m is not None
        else meta.get("cep_threshold_m", 500.0)
    )

    truth: dict[str, tuple[float, float]] = {}
    sensor_positions: dict[str, tuple[float, float]] = {}
    events: list[KeyEvent] = []
    cep_series: list[tuple[float, float, float]] = []
    fixes: list[tuple[float, float, dict[str, Any]]] = []
    first_detection: dict[str, float] = {}
    first_fix_frame: Frame | None = None

    for frame in replay.frames:
        for measurement in frame.measurements:
            sensor_id = str(measurement["sensor"])
            sensor_positions.setdefault(
                sensor_id,
                (
                    float(measurement["sensor_pos"][0]),
                    float(measurement["sensor_pos"][1]),
                ),
            )
            if measurement["detected"] and measurement.get("emitter") == target:
                first_detection.setdefault(sensor_id, frame.t)
        if target in frame.positions:
            truth[frame.t] = frame.positions[target]

        for estimate in frame.estimates:
            if str(estimate["target"]) != target:
                continue
            if first_fix_frame is None:
                first_fix_frame = frame
            reported = float(estimate["cep_cov_m"])
            position = (float(estimate["pos"][0]), float(estimate["pos"][1]))
            reference = truth.get(frame.t)
            error = (
                float(math.dist(position, reference))
                if reference is not None
                else float("nan")
            )
            cep_series.append((frame.t, reported, error))
            fixes.append((frame.t, reported, estimate))

    n_ticks = len(replay.frames)
    fix_rate = len(fixes) / n_ticks if n_ticks else 0.0

    # 首次"有效"定位：自报 CEP ≤ 门槛（这也就是操作员会认为"被定住了"的时刻）
    first_effective = next(((t, cep) for t, cep, _ in cep_series if cep <= limit), None)
    best = min(cep_series, key=lambda item: item[1]) if cep_series else None

    times = np.array([t for t, _, _ in cep_series], dtype=float)
    ceps = np.array([cep for _, cep, _ in cep_series], dtype=float)
    dt = float(meta.get("dt", 1.0))
    windows = exposure_windows(times, ceps, limit, dt=dt) if cep_series else ()
    exposure = t_expose(times, ceps, limit, dt=dt) if cep_series else 0.0

    for sensor_id, t_s in sorted(first_detection.items(), key=lambda item: item[1]):
        events.append(KeyEvent(t_s, "首次截获", f"{sensor_id} 收到 {target} 的辐射"))
    if fixes:
        events.append(KeyEvent(fixes[0][0], "首次交汇", "两条以上量测解出位置"))
    if first_effective is not None:
        events.append(
            KeyEvent(
                first_effective[0],
                "有效定位",
                f"CEP {first_effective[1]:.0f} m ≤ 门槛 {limit:.0f} m",
            )
        )
    if best is not None:
        events.append(KeyEvent(best[0], "最精确定位", f"CEP {best[1]:.0f} m"))
    for window in windows[:3]:
        events.append(
            KeyEvent(
                window.start_s,
                "暴露开始",
                f"持续到 {window.end_s:.0f} s（峰值 CEP {window.peak_cep_m:.0f} m）",
            )
        )
    events.sort(key=lambda item: item.t_s)

    attribution, sentence = _explain(
        replay=replay,
        target=target,
        limit=limit,
        first_fix=(fixes[0][0], fixes[0][2]) if fixes else None,
        first_effective=first_effective,
        best=best,
        truth=truth,
        fix_rate=fix_rate,
    )

    true_errors = [error for _, _, error in cep_series if math.isfinite(error)]
    return AarReport(
        emitter=target,
        threshold_m=limit,
        n_ticks=n_ticks,
        n_fixes=len(fixes),
        fix_rate=fix_rate,
        first_fix_s=fixes[0][0] if fixes else None,
        first_effective_s=first_effective[0] if first_effective else None,
        first_effective_cep_m=first_effective[1] if first_effective else None,
        best_cep_m=best[1] if best else None,
        best_cep_s=best[0] if best else None,
        median_true_error_m=(
            float(np.median(true_errors)) if true_errors else None
        ),
        t_expose_s=exposure,
        cep_series=tuple(cep_series),
        timeline=tuple(events),
        attribution=tuple(attribution),
        sentence=sentence,
        map_data={
            "sensors": sensor_positions,
            "truth": truth,
            "estimates": [
                (t, float(estimate["pos"][0]), float(estimate["pos"][1]), float(estimate["cep_cov_m"]))
                for t, _, estimate in fixes
            ],
            "radar": _radar_position(replay, meta),
        },
    )


def _radar_position(replay: Replay, meta: dict[str, Any]) -> tuple[float, float] | None:
    radar_id = str(meta.get("radar", ""))
    for frame in replay.frames:
        if radar_id in frame.positions:
            return frame.positions[radar_id]
    return None


def _explain(
    *,
    replay: Replay,
    target: str,
    limit: float,
    first_fix: tuple[float, dict[str, Any]] | None,
    first_effective: tuple[float, float] | None,
    best: tuple[float, float, float] | None,
    truth: dict[str, tuple[float, float]],
    fix_rate: float,
) -> tuple[list[dict[str, Any]], str]:
    """留一法归因 + 一句话结论。"""
    if first_effective is None:
        best_text = (
            f"最好的一次自报 CEP 是 {best[1]:.0f} m"
            if best is not None
            else "从未解出位置"
        )
        return [], (
            f"你的 {target} 全程没有被有效定位：{best_text}"
            f"（门槛 {limit:.0f} m）。"
        )

    moment, cep = first_effective
    frame = next((f for f in replay.frames if f.t == moment and f.estimates), None)
    measurements = (
        [
            item
            for item in frame.measurements
            if item["detected"] and item.get("emitter") == target
        ]
        if frame is not None
        else []
    )
    reference = truth.get(moment)
    base_fix = _solve(measurements)
    base_error = (
        float(math.dist(base_fix.pos, reference))
        if base_fix is not None and reference is not None
        else float("nan")
    )

    rows: list[dict[str, Any]] = []
    for index, measurement in enumerate(measurements):
        remaining = _solve(measurements[:index] + measurements[index + 1 :])
        row: dict[str, Any] = {"sensor": str(measurement["sensor"])}
        if remaining is None:
            row.update(
                {
                    "error_m": None,
                    "delta_m": None,
                    "verdict": "决定性：去掉后定不了位",
                }
            )
        else:
            error = (
                float(math.dist(remaining.pos, reference))
                if reference is not None
                else float("nan")
            )
            row.update(
                {
                    "error_m": error,
                    "delta_m": error - base_error,
                    "verdict": "去掉后误差变大" if error > base_error else "影响很小",
                }
            )
        rows.append(row)

    sensors = sorted({str(item["sensor"]) for item in measurements})
    geometry = (
        base_fix.geometry_deg if base_fix is not None else _geometry_of(measurements)
    )
    angle_text = f"（交会角 {geometry:.0f}°）" if geometry is not None else ""

    # 精度主要由哪类量测决定：用自报协方差算 CEP，比单次真实误差稳定。
    # 注意要**真的把另一类量测删掉**（字段置 None），否则子集里两类量测都还在。
    def subset_cep(keep_aoa: bool, keep_toa: bool) -> float | None:
        subset = _solve(
            [
                {
                    **item,
                    "aoa_rad": item.get("aoa_rad") if keep_aoa else None,
                    "toa_s": item.get("toa_s") if keep_toa else None,
                }
                for item in measurements
            ]
        )
        if subset is None:
            return None
        try:
            return cep_from_covariance(subset.covariance, samples=2000)[0]
        except ValueError:
            return None

    aoa_cep = subset_cep(True, False)
    tdoa_cep = subset_cep(False, True)
    full_cep = (
        cep_from_covariance(base_fix.covariance, samples=2000)[0]
        if base_fix is not None
        else None
    )

    worst = max(
        (row for row in rows if row["delta_m"] is not None),
        key=lambda row: row["delta_m"],
        default=None,
    )

    parts = [
        f"你的 {target} 在 t = {moment:.0f} s 被有效定位"
        f"（CEP ≈ {cep:.0f} m，门槛 {limit:.0f} m）。"
    ]
    if first_fix is not None and first_fix[0] < moment - 1e-9:
        early_t, early = first_fix
        early_sensors = early.get("sensors") or []
        early_geo = early.get("geometry_deg")
        parts.append(
            f"t = {early_t:.0f} s 其实已经首次交汇，但当时只有 "
            f"{len(early_sensors)} 台传感器参与"
            + (f"（交会角 {early_geo:.0f}°）" if early_geo is not None else "")
            + f"、CEP {float(early['cep_cov_m']):.0f} m 还没进门槛；"
        )
    parts.append(
        f"真正锁定它的是 {'、'.join(sensors)} 的 {len(measurements)} 条量测同时交汇{angle_text}"
    )
    if worst is not None and worst["delta_m"] is not None and worst["delta_m"] > 0:
        parts.append(
            f"；留一检验里去掉 {worst['sensor']} 会让误差从 "
            f"{base_error:.0f} m 变成 {worst['error_m']:.0f} m"
        )
    if aoa_cep is not None and tdoa_cep is not None and full_cep is not None:
        if max(aoa_cep, tdoa_cep) > 1.5 * min(aoa_cep, tdoa_cep):
            dominant = "TDOA" if tdoa_cep < aoa_cep else "测向"
            parts.append(
                f"；精度主要由 {dominant} 决定"
                f"（只留测向 CEP {aoa_cep:.0f} m，只留 TDOA {tdoa_cep:.0f} m）"
            )
    parts.append(f"。此后 {fix_rate * 100:.0f}% 的帧能解出位置。")
    return rows, "".join(parts)


def _geometry_of(measurements: Sequence[dict[str, Any]]) -> float | None:
    bearings = [item for item in measurements if item.get("aoa_rad") is not None]
    if len(bearings) != 2:
        return None
    separation = abs(
        (bearings[0]["aoa_rad"] - bearings[1]["aoa_rad"] + math.pi) % (2 * math.pi) - math.pi
    )
    return math.degrees(min(separation, math.pi - separation))


# --- HTML --------------------------------------------------------------------


def render_html(report: AarReport, replay: Replay) -> str:
    """自包含单页报告：内联 CSS + 手绘 SVG，不引外部资源。"""
    scene = html.escape(str(replay.meta.get("scenario", "replay")))
    sentence = html.escape(report.sentence)

    def stat(label: str, value: str, note: str = "") -> str:
        return (
            f'<div class="stat"><span class="v">{html.escape(value)}</span>'
            f'<span class="l">{html.escape(label)}</span>'
            f'<span class="n">{html.escape(note)}</span></div>'
        )

    stats = "".join(
        [
            stat("首次有效定位", f"{report.first_effective_s:.0f} s" if report.first_effective_s is not None else "—"),
            stat("当时 CEP", f"{report.first_effective_cep_m:.0f} m" if report.first_effective_cep_m is not None else "—", f"门槛 {report.threshold_m:.0f} m"),
            stat("最精确的一次", f"{report.best_cep_m:.0f} m" if report.best_cep_m is not None else "—"),
            stat("真实误差中位数", f"{report.median_true_error_m:.0f} m" if report.median_true_error_m is not None else "—"),
            stat("暴露时长", f"{report.t_expose_s:.0f} s", f"解出位置的帧占比 {report.fix_rate * 100:.0f}%"),
        ]
    )

    timeline_rows = "".join(
        f"<tr><td>{event.t_s:.0f}</td><td>{html.escape(event.kind)}</td>"
        f"<td>{html.escape(event.detail)}</td></tr>"
        for event in report.timeline
    )

    attribution_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(row["sensor"])),
            "—" if row.get("error_m") is None else "{:.0f} m".format(row["error_m"]),
            "—" if row.get("delta_m") is None else "{:+.0f} m".format(row["delta_m"]),
            html.escape(str(row.get("verdict", ""))),
        )
        for row in report.attribution
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AAR · {scene}</title>
<style>
  :root {{ --ink:#2b2f36; --muted:#6b7178; --line:#dfe3e8; --jam:#c0392b; --primary:#2f6fbf; --green:#2e8b57; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px 30px 40px; background:#f7f8fa; color:var(--ink);
         font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:12.5px; margin:0 0 18px; }}
  .verdict {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--jam);
              border-radius:10px; padding:14px 16px; font-size:14px; line-height:1.65; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 20px; }}
  .stat {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 14px;
           min-width:150px; display:flex; flex-direction:column; }}
  .stat .v {{ font-size:20px; font-weight:650; }}
  .stat .l {{ font-size:11.5px; color:var(--muted); }}
  .stat .n {{ font-size:11px; color:#8a8f98; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:16px; }}
  .card h2 {{ font-size:13px; margin:0 0 10px; color:var(--muted); font-weight:600; }}
  .card svg {{ display:block; width:100%; height:auto; }}
  .geom {{ max-width:560px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
  th,td {{ text-align:left; padding:5px 8px; border-bottom:1px solid #eef1f4; }}
  th {{ color:var(--muted); font-weight:600; }}
  .foot {{ color:#a0a5ad; font-size:11px; margin-top:8px; }}
</style>
</head>
<body>
<h1>AAR · {scene}</h1>
<p class="sub">目标 {html.escape(report.emitter)} · {report.n_ticks} 帧 · 解出位置 {report.n_fixes} 帧 · 门槛 {report.threshold_m:.0f} m</p>

<div class="verdict">{sentence}</div>
<div class="stats">{stats}</div>

<div class="card">
  <h2>关键事件</h2>
  <table><thead><tr><th>t [s]</th><th>事件</th><th>说明</th></tr></thead>
  <tbody>{timeline_rows}</tbody></table>
</div>

<div class="card">
  <h2>CEP 与真实误差</h2>
  {_cep_svg(report)}
</div>

<div class="card">
  <h2>几何</h2>
  <div class="geom">{_map_svg(report)}</div>
</div>

<div class="card">
  <h2>归因：留一法（去掉某台传感器重解一次）</h2>
  <table><thead><tr><th>传感器</th><th>去掉后的误差</th><th>相对全量</th><th>判定</th></tr></thead>
  <tbody>{attribution_rows or '<tr><td colspan="4">没有可用量测</td></tr>'}</tbody></table>
</div>

<p class="foot">fictional parameters · educational simulation · 由 spectrumfog aar 自动生成</p>
</body>
</html>
"""


def _cep_svg(report: AarReport) -> str:
    if not report.cep_series:
        return '<p class="foot">没有解出任何位置。</p>'
    width, height = 900, 240
    pad = {"l": 54, "r": 14, "t": 14, "b": 26}
    plot_w = width - pad["l"] - pad["r"]
    plot_h = height - pad["t"] - pad["b"]
    times = [t for t, _, _ in report.cep_series]
    ceps = [max(cep, 1.0) for _, cep, _ in report.cep_series]
    windows = exposure_windows(
        np.array(times, dtype=float),
        np.array([cep for _, cep, _ in report.cep_series], dtype=float),
        report.threshold_m,
    )
    t0 = min(times)
    t1 = max(times) if max(times) > min(times) else min(times) + 1.0

    # 对数纵轴：CEP 从十几米到十几公里，线性轴只会得到一堆尖刺
    y_min = 1.0
    y_max = max(max(ceps) * 1.1, report.threshold_m * 3.0)
    log_lo, log_hi = math.log10(y_min), math.log10(y_max)

    def x(t: float) -> float:
        return pad["l"] + (t - t0) / (t1 - t0) * plot_w

    def y(value: float) -> float:
        clamped = min(max(value, y_min), y_max)
        return pad["t"] + plot_h * (1.0 - (math.log10(clamped) - log_lo) / (log_hi - log_lo))

    cep_path = " ".join(
        f"{'M' if index == 0 else 'L'}{x(t):.1f},{y(cep):.1f}"
        for index, (t, cep, _) in enumerate(report.cep_series)
    )
    err_path = " ".join(
        f"{'M' if index == 0 else 'L'}{x(t):.1f},{y(err):.1f}"
        for index, (t, _, err) in enumerate(report.cep_series)
        if math.isfinite(err) and err > 0
    )

    # 暴露区间：合并间隔很小（< 2% 时长）的窗口，否则图上会变成一堆竖条纹
    merged: list[tuple[float, float]] = []
    for window in windows:
        start, end = window.start_s, window.end_s
        if merged and start - merged[-1][1] <= 0.02 * (t1 - t0):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    bands = []
    for start, end in merged:
        left = x(start)
        right = x(max(end, start + (t1 - t0) * 0.004))
        bands.append(
            f'<rect x="{left:.1f}" y="{pad["t"]}" width="{max(right - left, 2.0):.1f}" '
            f'height="{plot_h}" fill="#c0392b" opacity="0.10"/>'
        )

    ticks = []
    for power in range(0, 6):
        value = 10.0**power
        if value < y_min or value > y_max:
            continue
        ticks.append(
            f'<line x1="{pad["l"]}" y1="{y(value):.1f}" x2="{width - pad["r"]}" '
            f'y2="{y(value):.1f}" stroke="#eef1f4"/>'
            f'<text x="{pad["l"] - 6}" y="{y(value) + 3:.1f}" text-anchor="end" '
            f'font-size="9.5" fill="#8a8f98">{value:g} m</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="CEP">
  {''.join(bands)}
  {''.join(ticks)}
  <line x1="{pad['l']}" y1="{y(report.threshold_m):.1f}" x2="{width - pad['r']}" y2="{y(report.threshold_m):.1f}"
        stroke="#8a8f98" stroke-dasharray="3 4"/>
  <text x="{width - pad['r']}" y="{y(report.threshold_m) - 5:.1f}" text-anchor="end" font-size="10" fill="#8a8f98">门槛 {report.threshold_m:.0f} m</text>
  <path d="{cep_path}" fill="none" stroke="#2f6fbf" stroke-width="1.3" opacity="0.9"/>
  <path d="{err_path}" fill="none" stroke="#c0392b" stroke-width="1.0" opacity="0.6"/>
  <text x="{pad['l']}" y="{height - 6}" font-size="10" fill="#6b7178">{t0:.0f} s</text>
  <text x="{width - pad['r']}" y="{height - 6}" font-size="10" fill="#6b7178" text-anchor="end">{t1:.0f} s</text>
  <text x="{pad['l'] + 6}" y="{pad['t'] + 10}" font-size="10" fill="#2f6fbf">自报 CEP</text>
  <text x="{pad['l'] + 76}" y="{pad['t'] + 10}" font-size="10" fill="#c0392b">真实误差（淡红底 = 处于暴露窗口）</text>
</svg>"""


def _map_svg(report: AarReport) -> str:
    map_data = report.map_data
    truth: dict[float, tuple[float, float]] = map_data.get("truth", {})
    sensors: dict[str, tuple[float, float]] = map_data.get("sensors", {})
    estimates = map_data.get("estimates", [])
    radar = map_data.get("radar")
    if not truth and not estimates:
        return '<p class="foot">没有可画的几何数据。</p>'

    points = list(truth.values()) + list(sensors.values()) + [
        (x, y) for _, x, y, _ in estimates
    ]
    if radar is not None:
        points.append(radar)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 1.15
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    size = 520

    def project(x: float, y: float) -> tuple[float, float]:
        return (
            size / 2 + (x - cx) / span * size,
            size / 2 - (y - cy) / span * size,
        )

    truth_track = " ".join(
        f"{project(x, y)[0]:.1f},{project(x, y)[1]:.1f}"
        for _, (x, y) in sorted(truth.items())
    )
    estimate_dots = "".join(
        f'<circle cx="{project(x, y)[0]:.1f}" cy="{project(x, y)[1]:.1f}" r="2.6" fill="#2f6fbf" opacity="0.55"/>'
        for _, x, y, _ in estimates
    )
    sensor_marks = "".join(
        f'<g><rect x="{project(x, y)[0] - 5:.1f}" y="{project(x, y)[1] - 5:.1f}" width="10" height="10" '
        f'fill="#2e8b57" stroke="#fff"/><text x="{project(x, y)[0] + 9:.1f}" y="{project(x, y)[1] + 4:.1f}" '
        f'font-size="10" fill="#2b2f36">{html.escape(name)}</text></g>'
        for name, (x, y) in sensors.items()
    )
    radar_mark = ""
    if radar is not None:
        px, py = project(*radar)
        radar_mark = (
            f'<path d="M{px:.1f},{py - 9:.1f} L{px + 8:.1f},{py + 6:.1f} L{px - 8:.1f},{py + 6:.1f} Z" '
            f'fill="#2f6fbf" stroke="#fff"/>'
        )
    # 真值只有一个点时 polyline 画不出来，用十字标出来
    truth_mark = ""
    if truth:
        last = sorted(truth.items())[-1]
        px, py = project(*last[1])
        truth_mark = (
            f'<path d="M{px - 6:.1f},{py:.1f} L{px + 6:.1f},{py:.1f} M{px:.1f},{py - 6:.1f} '
            f'L{px:.1f},{py + 6:.1f}" stroke="#c0392b" stroke-width="1.6"/>'
            f'<text x="{px + 9:.1f}" y="{py - 8:.1f}" font-size="10" fill="#c0392b">'
            f'{html.escape(report.emitter)}</text>'
        )

    # 比例尺：取一个好看的整十公里
    scale_km = max(1.0, round(span / 1000.0 / 4.0))
    scale_px = scale_km * 1000.0 / span * size
    return f"""<svg viewBox="0 0 {size} {size}" role="img" aria-label="geometry">
  <rect width="{size}" height="{size}" fill="#fbfbfd" stroke="#dfe3e8"/>
  <polyline points="{truth_track}" fill="none" stroke="#c0392b" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.7"/>
  {truth_mark}
  {estimate_dots}
  {sensor_marks}
  {radar_mark}
  <line x1="16" y1="{size - 22}" x2="{16 + scale_px:.1f}" y2="{size - 22}" stroke="#6b7178" stroke-width="2"/>
  <text x="16" y="{size - 28}" font-size="10" fill="#6b7178">{scale_km:.0f} km</text>
  <text x="12" y="20" font-size="10" fill="#6b7178">绿方块 = ESM　蓝三角 = 雷达　红十字 = 真实位置　蓝点 = 定位解</text>
</svg>"""


def write_html(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return target
