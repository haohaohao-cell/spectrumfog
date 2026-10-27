"""生成 GitHub 社交预览图（1280×640）。

一次性资产，不属于产品代码：只依赖本仓库的物理层 + matplotlib。
重跑：

    python tools/make_social_preview.py

产物：docs/images/social_preview.png
（GitHub 的 social preview 目前只能在上传 Settings 里手动传 —— REST API 没有这个端点。）

图内先用中文名 + 英文副标题；找不到 CJK 字体时自动退化成纯英文，
所以换机器重跑不会出现豆腐块。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spectrumfog.model.scenario import load_scenario  # noqa: E402
from spectrumfog.physics import radar  # noqa: E402

INK = "#2b2f36"
MUTED = "#6b7178"
PRIMARY = "#2f6fbf"
JAM = "#c0392b"
GREEN = "#2e8b57"
BG = "#ffffff"

PALETTE_STOPS = np.array([0.0, 0.5, 1.0])
PALETTE_RGB = np.array([[192, 57, 43], [200, 135, 26], [47, 111, 191]], dtype=float)

CJK_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)


def pick_cjk_font() -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            return name
    return None


def colorize(values: np.ndarray) -> np.ndarray:
    """Pd → RGB，与网页端同一套配色。"""
    clipped = np.clip(values, 0.0, 1.0)
    channels = [
        np.interp(clipped, PALETTE_STOPS, PALETTE_RGB[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def main() -> Path:
    scenario = load_scenario(REPO_ROOT / "scenarios" / "02_standoff.yaml")
    params = scenario.link_params()
    target = scenario.platform("target_c")
    jammer_pos = scenario.platform("jammer_b").pos

    pfa = 1.0e-6
    extent_km = 74.0  # 地图半幅
    resolution = 460
    axis = np.linspace(-extent_km, extent_km, resolution)
    xx, yy = np.meshgrid(axis, axis)
    range_target = np.hypot(xx, yy) * 1.0e3
    range_jammer = np.hypot(xx - jammer_pos[0] / 1e3, yy - jammer_pos[1] / 1e3) * 1.0e3

    # 1 m 处的功率即系数：Pr ∝ 1/Rt⁴、Pj_rx ∝ 1/Rj²（与 docs/physics.md §3.5 一致）
    pr_coef = float(
        radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=1.0,
            loss=params["loss"],
        )
    )
    pj_coef = float(
        radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=1.0,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
    )
    pn = float(
        radar.noise_power(
            bw_hz=params["bw_r_hz"],
            noise_figure=scenario.environment.noise_figure,
            temperature_k=scenario.environment.temperature_k,
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pd = np.asarray(
            radar.pd_from_powers(
                pfa,
                pr_coef / np.maximum(range_target, 1.0) ** 4,
                pn,
                pj_coef / np.maximum(range_jammer, 1.0) ** 2,
            )
        )
    pd = np.nan_to_num(pd, nan=0.0)

    detection = radar.detection_range(
        0.5,
        pfa,
        pj_w=params["pj_w"],
        gj=params["gj"],
        bw_j_hz=params["bw_j_hz"],
        jammer_range_m=float(np.hypot(*jammer_pos)),
        pt_w=params["pt_w"],
        gain=params["gain"],
        sigma_m2=params["sigma_m2"],
        lam=params["lam"],
        bw_r_hz=params["bw_r_hz"],
        noise_figure=scenario.environment.noise_figure,
        temperature_k=scenario.environment.temperature_k,
        loss=params["loss"],
    )
    burn = radar.burn_through_range_standoff(
        pt_w=params["pt_w"],
        gain=params["gain"],
        sigma_m2=params["sigma_m2"],
        pj_w=params["pj_w"],
        gj=params["gj"],
        bw_r_hz=params["bw_r_hz"],
        bw_j_hz=params["bw_j_hz"],
        loss=params["loss"],
        jammer_range_m=float(np.hypot(*jammer_pos)),
    )

    cjk = pick_cjk_font()

    # 目标处的 Pd（与地图用的是同一套系数）
    radar_pos = scenario.platform("radar_a").pos
    target_range_m = float(np.hypot(target.pos[0] - radar_pos[0], target.pos[1] - radar_pos[1]))
    # 干扰功率只取决于"干扰机→雷达"的路径，与目标在哪无关（目标位置只影响回波）
    jammer_range_m = float(np.hypot(jammer_pos[0] - radar_pos[0], jammer_pos[1] - radar_pos[1]))
    pd_at_target = float(
        radar.pd_from_powers(
            pfa,
            pr_coef / target_range_m**4,
            pn,
            pj_coef / jammer_range_m**2,
        )
    )

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)

    # --- 左：覆盖图 ---
    ax = fig.add_axes([0.030, 0.085, 0.400, 0.800])
    ax.imshow(
        colorize(pd),
        extent=[-extent_km, extent_km, -extent_km, extent_km],
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(
        Circle((0, 0), float(burn) / 1e3, fill=False, ec="#37414d", lw=1.3,
               ls=(0, (4, 4)), zorder=2)
    )
    ax.add_patch(
        Circle((0, 0), float(detection) / 1e3, fill=False, ec="#ffffff", lw=4.2,
               alpha=0.85, zorder=3)
    )
    ax.add_patch(
        Circle((0, 0), float(detection) / 1e3, fill=False, ec=PRIMARY, lw=2.2, zorder=4)
    )
    ax.plot(0, 0, marker="^", ms=15, color=PRIMARY, mec="white", mew=1.4, zorder=6)
    ax.plot(
        jammer_pos[0] / 1e3, jammer_pos[1] / 1e3,
        marker="s", ms=11, color=JAM, mec="white", mew=1.2, zorder=6,
    )
    ax.plot(
        target.pos[0] / 1e3, target.pos[1] / 1e3,
        marker="o", ms=11, color=GREEN, mec="white", mew=1.4, zorder=7,
    )
    ax.set_xlim(-extent_km, extent_km)
    ax.set_ylim(-extent_km, extent_km)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.03, 0.035,
        "Pd map  ·  60 km standoff jammer  ·  target at 12 km",
        transform=ax.transAxes, fontsize=9.5, color="white",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#00000066", "edgecolor": "none"},
        zorder=10,
    )

    # --- 右：文字 ---
    tx = fig.add_axes([0.455, 0.085, 0.520, 0.800])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)

    if cjk:
        tx.text(0, 0.97, "频谱迷雾", fontsize=30, color=INK, weight="bold",
                family=cjk, va="top")
    tx.text(0, 0.845, "SpectrumFog", fontsize=46, color=INK, weight="bold", va="top")
    tx.text(
        0, 0.695,
        "partially observable spectrum-electromagnetic simulation",
        fontsize=13.5, color=MUTED, va="top",
    )
    tx.text(
        0, 0.648,
        "deterministic replay  ·  physics in one place  ·  no backend",
        fontsize=12, color=MUTED, va="top",
    )

    # --- 三个读数：数值在上、标签在下，行距按字号留够 ---
    stats = [
        ("burn-through ring", f"{float(burn) / 1e3:.1f} km", PRIMARY),
        ("coverage boundary", f"{float(detection) / 1e3:.1f} km", PRIMARY),
        ("Pd at target", f"{pd_at_target:.2f}", GREEN),
    ]
    top = 0.525
    for index, (label, value, color) in enumerate(stats):
        row = top - index * 0.135
        tx.text(0, row, value, fontsize=24, color=color, weight="bold", va="top")
        tx.text(0, row - 0.058, label, fontsize=10.5, color=MUTED, va="top")

    tx.text(
        0, 0.135,
        "drag the jammer: the coverage ring collapses",
        fontsize=12.5, color=INK, va="top",
    )
    tx.text(
        0, 0.085,
        "github.com/haohaohao-cell/spectrumfog",
        fontsize=10.5, color=MUTED, va="top",
    )
    tx.text(
        0, 0.038,
        "fictional parameters · educational simulation",
        fontsize=10, color="#a0a5ad", va="top",
    )

    out = REPO_ROOT / "docs" / "images" / "social_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = main()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
