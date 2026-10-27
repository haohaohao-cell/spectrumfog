"""T1 俯视动画：把 replay 画成 GIF。

只读回放（§2 决策 2）：本模块不 import 引擎、不重算任何物理量，
画面里的每个数字都来自日志行。所以"图对了但物理错了"不可能被动画掩盖。

画面结构
--------
左：俯视图。雷达、干扰机航迹、目标，三条同心圆
    （无干扰探测距离 / Pd=门槛 的检测边界 / J/S=1 的烧穿环）。
右：两张时间曲线。上 Pd、下 J/S，只画到当前时刻，末尾带游标。

坐标一律用 **米**（内核单位），刻度用 km 显示 —— 不要在半路换单位。
图内文字一律英文：CJK 字体不可移植，README 头图要在任何机器上成立。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from spectrumfog.replay import Replay  # noqa: E402

__all__ = ["animate_replay", "animate_file"]

PRIMARY = "#2f6fbf"
JAM = "#c0392b"
MUTED = "#8a8f98"
GREEN = "#2e8b57"
AMBER = "#c8871a"
INK = "#2b2f36"

FIGSIZE = (11.0, 5.0)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d0d4da",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": "#d0d4da",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.size": 9,
        }
    )


def _km_formatter(value: float, _pos: float) -> str:
    return f"{value / 1.0e3:.0f}"


def _map_legend(
    *, unjammed_m: float, threshold: float, has_rings: bool
) -> list[Line2D]:
    handles = [
        Line2D([], [], color=PRIMARY, marker="^", ms=10, lw=0, label="radar"),
        Line2D([], [], color=JAM, marker="s", ms=7, lw=0, label="jammer"),
        Line2D([], [], color=GREEN, marker="o", ms=7, lw=0, label="target"),
        Line2D(
            [],
            [],
            color=MUTED,
            ls=(0, (5, 4)),
            lw=1.1,
            label=f"unjammed range {unjammed_m / 1e3:.0f} km",
        ),
    ]
    if has_rings:
        handles.extend(
            [
                Line2D([], [], color=PRIMARY, lw=1.8, label=f"tracking boundary (Pd={threshold:.2f})"),
                Line2D([], [], color=JAM, ls=(0, (2, 3)), lw=1.0, label="J/S = 1"),
            ]
        )
    return handles


def animate_replay(
    replay: Replay,
    out_path: str | Path,
    *,
    fps: int = 12,
    dpi: int = 110,
    stride: int = 1,
) -> Path:
    """把一份回放渲染成 GIF，返回输出路径。"""
    meta = replay.meta
    frames = replay.frames[:: max(1, stride)]
    if not frames:
        raise ValueError("回放里没有任何帧")

    radar_id = str(meta["radar"])
    jammer_id = str(meta["jammer"])
    target_id = str(meta["target"])
    threshold = float(meta["pd_threshold"])
    unjammed_m = float(meta["unjammed_detection_range_m"])

    times = np.array([frame.t for frame in frames], dtype=float)
    pd_series = np.array(
        [float(frame.link["pd"]) for frame in frames if frame.link], dtype=float
    )
    js_series = np.array(
        [float(frame.link["js"]) for frame in frames if frame.link], dtype=float
    )
    scan_hits = np.array(
        [bool(frame.detection["detected"]) for frame in frames if frame.detection],
        dtype=float,
    )
    lost_index = next(
        (index for index, value in enumerate(pd_series) if value < threshold), None
    )

    # 画面范围（米）：包住所有位置与最大的一圈环，留 8% 边距
    coords = [value for frame in frames for value in frame.positions.values()]
    max_abs = max((max(abs(x), abs(y)) for x, y in coords), default=1.0)
    max_ring = max(
        (
            float(frame.link["burn_through_m"])
            for frame in frames
            if frame.link and frame.link.get("burn_through_m") is not None
        ),
        default=unjammed_m,
    )
    limit = max(max_abs, unjammed_m, max_ring) * 1.08

    _style()
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi, facecolor="white")
    ax_map = fig.add_axes([0.028, 0.075, 0.405, 0.845])
    ax_pd = fig.add_axes([0.520, 0.570, 0.450, 0.330])
    ax_js = fig.add_axes([0.520, 0.140, 0.450, 0.330])

    # --- 右侧时间曲线（整条只画一次，逐帧只移动游标） ---
    ax_pd.plot(times, pd_series, color=PRIMARY, lw=1.6, label="Pd (model)")
    ax_pd.plot(
        times,
        scan_hits,
        color=MUTED,
        lw=0.6,
        alpha=0.35,
        drawstyle="steps-post",
        label="single-scan result (sampled)",
    )
    ax_pd.axhline(threshold, color=MUTED, ls=":", lw=1.0)
    ax_pd.set_ylim(-0.03, 1.03)
    ax_pd.set_xlim(0.0, float(times[-1]))
    ax_pd.set_ylabel("$P_d$")
    ax_pd.legend(loc="lower left", ncol=2)
    cursor_pd = ax_pd.axvline(times[0], color=AMBER, lw=1.2)
    dot_pd = ax_pd.plot([times[0]], [pd_series[0]], "o", ms=5, color=AMBER)[0]
    if lost_index is not None:
        lost_t = float(times[lost_index])
        ax_pd.axvline(lost_t, color=JAM, ls="--", lw=1.0)
        ax_pd.annotate(
            "target lost",
            xy=(lost_t, threshold),
            xytext=(7, 30),
            textcoords="offset points",
            fontsize=7.5,
            color=JAM,
        )

    ax_js.plot(times, js_series, color=JAM, lw=1.6, label="J/S")
    ax_js.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    ax_js.set_yscale("log")
    ax_js.set_xlim(0.0, float(times[-1]))
    ax_js.set_ylim(max(js_series.min() * 0.6, 1e-4), max(js_series.max() * 1.6, 2.0))
    ax_js.set_xlabel("time [s]")
    ax_js.set_ylabel("J/S  [linear]")
    cursor_js = ax_js.axvline(times[0], color=AMBER, lw=1.2)
    dot_js = ax_js.plot([times[0]], [js_series[0]], "o", ms=5, color=AMBER)[0]

    fig.text(
        0.028,
        0.945,
        f"{meta['scenario']} · standoff jammer closing in",
        fontsize=11.5,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.520,
        0.917,
        "probability the radar detects the target   ·   J/S at the target",
        fontsize=7.2,
        color="#6b7178",
    )
    fig.text(
        0.998,
        0.010,
        "fictional parameters · educational simulation · SpectrumFog T1",
        ha="right",
        fontsize=6.5,
        color="#a0a5ad",
    )

    def draw_map(index: int) -> None:
        frame = frames[index]
        link = frame.link or {}
        pd_value = float(link.get("pd", 0.0))
        jamming = pd_value < threshold
        radar_pos = frame.positions.get(radar_id, (0.0, 0.0))
        jammer_pos = frame.positions.get(jammer_id, (np.nan, np.nan))
        target_pos = frame.positions.get(target_id, (np.nan, np.nan))

        ax_map.clear()
        ax_map.set_xlim(-limit, limit)
        ax_map.set_ylim(-limit, limit)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.xaxis.set_major_formatter(FuncFormatter(_km_formatter))
        ax_map.yaxis.set_major_formatter(FuncFormatter(_km_formatter))
        ax_map.set_xlabel("east [km]")
        ax_map.set_ylabel("north [km]")

        # 参考圈：每 20 km 一圈
        radius = 20_000.0
        while radius <= limit:
            ax_map.add_patch(
                Circle(radar_pos, radius, fill=False, ec="#e6e9ed", lw=0.7, zorder=0)
            )
            radius += 20_000.0

        # 无干扰探测距离
        ax_map.add_patch(
            Circle(
                radar_pos,
                unjammed_m,
                fill=False,
                ec=MUTED,
                lw=1.0,
                ls=(0, (5, 4)),
                zorder=1,
            )
        )

        # J/S = 1 烧穿环（细虚线）
        burn_through_m = link.get("burn_through_m")
        if burn_through_m is not None:
            ax_map.add_patch(
                Circle(
                    radar_pos,
                    float(burn_through_m),
                    fill=False,
                    ec=JAM,
                    lw=1.0,
                    ls=(0, (2, 3)),
                    zorder=2,
                )
            )

        # Pd = 门槛 的检测边界（实线 + 淡填充）
        ring_m = link.get("detection_ring_m")
        if ring_m is not None:
            ax_map.add_patch(
                Circle(
                    radar_pos,
                    float(ring_m),
                    facecolor=PRIMARY,
                    alpha=0.07,
                    ec="none",
                    zorder=2,
                )
            )
            ax_map.add_patch(
                Circle(radar_pos, float(ring_m), fill=False, ec=PRIMARY, lw=2.0, zorder=3)
            )

        # 航迹（单位：米）
        trail = frames[: index + 1]
        jammer_trail = np.array(
            [f.positions.get(jammer_id, (np.nan, np.nan)) for f in trail], dtype=float
        )
        target_trail = np.array(
            [f.positions.get(target_id, (np.nan, np.nan)) for f in trail], dtype=float
        )
        if len(trail) > 1:
            ax_map.plot(
                jammer_trail[:, 0],
                jammer_trail[:, 1],
                color=JAM,
                lw=1.1,
                alpha=0.45,
                zorder=4,
            )
            ax_map.plot(
                target_trail[:, 0],
                target_trail[:, 1],
                color=GREEN if not jamming else JAM,
                lw=1.0,
                ls=(0, (1, 2)),
                alpha=0.8,
                zorder=4,
            )

        # 雷达—干扰机、雷达—目标连线
        for other in (jammer_pos, target_pos):
            ax_map.plot(
                [radar_pos[0], other[0]],
                [radar_pos[1], other[1]],
                color="#cdd2d9",
                lw=0.7,
                zorder=3,
            )

        ax_map.plot(
            radar_pos[0],
            radar_pos[1],
            marker="^",
            ms=13,
            color=PRIMARY,
            mec="white",
            mew=1.2,
            zorder=6,
        )
        ax_map.plot(
            jammer_pos[0],
            jammer_pos[1],
            marker="s",
            ms=8,
            color=JAM,
            mec="white",
            mew=1.0,
            zorder=6,
        )
        ax_map.plot(
            target_pos[0],
            target_pos[1],
            marker="o",
            ms=9,
            color=JAM if jamming else GREEN,
            mec="white",
            mew=1.2,
            zorder=7,
        )
        ax_map.annotate(
            "target",
            xy=(target_pos[0], target_pos[1]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=7,
            color=JAM if jamming else GREEN,
        )

        ax_map.legend(
            handles=_map_legend(
                unjammed_m=unjammed_m,
                threshold=threshold,
                has_rings=ring_m is not None,
            ),
            loc="lower left",
            borderaxespad=0.5,
            handlelength=1.6,
        )

        # HUD
        coverage = "none" if ring_m is None else f"{float(ring_m) / 1e3:5.1f} km"
        burn = (
            "  n/a"
            if burn_through_m is None
            else f"{float(burn_through_m) / 1e3:5.1f} km"
        )
        ax_map.text(
            0.028,
            0.975,
            "\n".join(
                [
                    f"t         {frame.t:6.0f} s",
                    f"jammer    {float(link.get('range_jammer_m', 0.0)) / 1e3:6.1f} km",
                    f"target    {float(link.get('range_target_m', 0.0)) / 1e3:6.1f} km",
                    f"J/S       {float(link.get('js', 0.0)):6.3f}",
                    f"Pd        {pd_value:6.3f}",
                    f"boundary  {coverage}",
                    f"J/S=1     {burn}",
                ]
            ),
            transform=ax_map.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.0,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#dfe3e8",
                "alpha": 0.93,
            },
            zorder=10,
        )

        ax_map.text(
            0.975,
            0.975,
            "JAMMED" if jamming else "TRACKING",
            transform=ax_map.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            weight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": JAM if jamming else GREEN,
                "edgecolor": "none",
            },
            zorder=10,
        )

    def update(index: int):
        draw_map(index)
        current_t = float(times[index])
        for cursor in (cursor_pd, cursor_js):
            cursor.set_xdata([current_t, current_t])
        dot_pd.set_data([current_t], [pd_series[index]])
        dot_js.set_data([current_t], [js_series[index]])
        return ()

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / max(1, fps), blit=False
    )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(destination), writer=PillowWriter(fps=max(1, fps)), dpi=dpi)
    plt.close(fig)
    return destination


def animate_file(
    replay_path: str | Path,
    out_path: str | Path,
    *,
    fps: int = 12,
    dpi: int = 110,
    stride: int = 1,
) -> Path:
    """读回放文件并出 GIF。"""
    return animate_replay(
        Replay.load(replay_path), out_path, fps=fps, dpi=dpi, stride=stride
    )
