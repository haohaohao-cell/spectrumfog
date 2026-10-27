"""T0 静态出图（matplotlib → PNG）。

设计约束
--------
* 图内文字**一律英文**：CJK 字体不可移植，README 头图必须在任何机器上都不缺字形。
  CLI 的中文输出在终端，图里的英文在图片里，两边分开。
* 本模块只调用 ``physics.radar`` 的纯函数，不回写内核状态（§2 决策 2）。
* 参数统一走 ``Scenario.link_params()`` 的那一份 dict，避免前后端各写一套物理。

三张图对应 T0 验收：
  burn-through   烧穿距离 vs 干扰功率（README 头图候选）
  pd-vs-range    检测概率 vs 距离（有 / 无干扰两条线）
  js-vs-range    干信比 vs 距离（J/S = 1 的穿点）
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前：无显示环境下也能出图

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectrumfog.physics import radar  # noqa: E402
from spectrumfog.units import linear_to_db  # noqa: E402

__all__ = [
    "plot_burn_through_curve",
    "plot_pd_vs_range",
    "plot_js_vs_range",
    "plot_all",
]

DPI = 150
FIGSIZE = (7.4, 4.4)

PRIMARY = "#2f6fbf"
JAM = "#c0392b"
MUTED = "#8a8f98"
GREEN = "#2e8b57"

_FOOTNOTE = "fictional parameters · educational simulation · SpectrumFog T0"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d0d4da",
            "axes.labelcolor": "#2b2f36",
            "axes.titlecolor": "#2b2f36",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": "#d0d4da",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "xtick.color": "#4b5159",
            "ytick.color": "#4b5159",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
        }
    )


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.008, _FOOTNOTE, ha="right", va="bottom", fontsize=6.5, color="#a0a5ad")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _burn_through_kwargs(params: Mapping[str, float]) -> dict:
    """``burn_through_range`` 不接受 ``lam``（自屏蔽下波长相消），这里剔掉。"""
    return {key: value for key, value in params.items() if key != "lam"}


def _detection_kwargs(
    params: Mapping[str, float], *, noise_figure: float, temperature_k: float
) -> dict:
    return {
        "pt_w": params["pt_w"],
        "gain": params["gain"],
        "sigma_m2": params["sigma_m2"],
        "lam": params["lam"],
        "bw_r_hz": params["bw_r_hz"],
        "noise_figure": noise_figure,
        "temperature_k": temperature_k,
        "loss": params["loss"],
    }


# --- 图 1：烧穿曲线 ----------------------------------------------------------


def plot_burn_through_curve(
    params: Mapping[str, float],
    *,
    out_path: Path,
    title: str = "Burn-through range vs jammer power",
    pfa: float = 1.0e-6,
    pd_threshold: float = 0.5,
    noise_figure: float = 1.0,
    temperature_k: float = 290.0,
    sweep_db: float = 10.0,
    points: int = 161,
) -> Path:
    """R_bt 随干扰功率变化（x 轴 dBm，y 轴 km，双对数下是 -1/2 斜率的直线）。

    曲线以上 = J/S < 1，雷达能烧穿；曲线以下 = 干扰占优。
    """
    _style()
    pj_nominal = float(params["pj_w"])
    pj = np.geomspace(
        pj_nominal * 10.0 ** (-sweep_db / 10.0),
        pj_nominal * 10.0 ** (sweep_db / 10.0),
        points,
    )
    r_bt = np.asarray(
        radar.burn_through_range(**_burn_through_kwargs({**params, "pj_w": pj}))
    )
    r_bt_nominal = float(radar.burn_through_range(**_burn_through_kwargs(params)))
    r_det = float(
        radar.detection_range(
            pd_threshold,
            pfa,
            **_detection_kwargs(
                params, noise_figure=noise_figure, temperature_k=temperature_k
            ),
        )
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x_db = np.asarray(linear_to_db(pj * 1.0e3))  # W → dBm
    y_km = r_bt / 1.0e3
    ax.plot(x_db, y_km, color=PRIMARY, linewidth=2.0, label=r"$R_{bt}$ (J/S = 1)")
    ax.scatter(
        [float(linear_to_db(pj_nominal * 1.0e3))],
        [r_bt_nominal / 1.0e3],
        s=64,
        color=JAM,
        zorder=5,
        label="this scenario",
    )
    ax.annotate(
        f"nominal: {r_bt_nominal / 1.0e3:.2f} km",
        xy=(float(linear_to_db(pj_nominal * 1.0e3)), r_bt_nominal / 1.0e3),
        xytext=(14, 16),
        textcoords="offset points",
        fontsize=8.5,
        color=JAM,
        arrowprops={"arrowstyle": "-", "color": JAM, "linewidth": 0.8},
    )
    # 纵轴只框住曲线本身：无干扰检测距离（几十 km）留作文字标注，避免把曲线压扁。
    y_bottom, y_top = float(y_km.min()) * 0.8, float(y_km.max()) * 1.35
    ax.set_ylim(y_bottom, y_top)
    ax.fill_between(
        x_db,
        y_bottom,
        y_km,
        color=JAM,
        alpha=0.07,
        linewidth=0,
        label=r"J/S > 1 (radar jammed)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Jammer transmit power  $P_j$  [dBm]")
    ax.set_ylabel("Burn-through range  $R_{bt}$  [km]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.96,
        f"unjammed detection range: {r_det / 1.0e3:.0f} km  (above this window)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=MUTED,
    )
    return _save(fig, out_path)


# --- 图 2：Pd vs 距离 --------------------------------------------------------


def plot_pd_vs_range(
    params: Mapping[str, float],
    *,
    out_path: Path,
    title: str = "Detection probability vs range (Swerling I)",
    pfa: float = 1.0e-6,
    pd_threshold: float = 0.5,
    noise_figure: float = 1.0,
    temperature_k: float = 290.0,
    points: int = 400,
) -> Path:
    """两条 Pd–R 曲线：仅热噪声 / 含噪声压制干扰。

    竖线分别标出烧穿距离与"无干扰时的检测距离"，两条线的落差就是干扰的效果。
    """
    _style()
    detection = _detection_kwargs(
        params, noise_figure=noise_figure, temperature_k=temperature_k
    )
    r_bt = float(radar.burn_through_range(**_burn_through_kwargs(params)))
    r_det = float(radar.detection_range(pd_threshold, pfa, **detection))
    try:
        r_det_jammed: float | None = float(
            radar.detection_range(
                pd_threshold,
                pfa,
                pj_w=params["pj_w"],
                gj=params["gj"],
                bw_j_hz=params["bw_j_hz"],
                **detection,
            )
        )
    except ValueError:
        r_det_jammed = None

    r = np.geomspace(min(1.0e2, r_bt / 50.0), max(3.0e5, r_det * 3.0), points)
    pr = np.asarray(
        radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=r,
            loss=params["loss"],
        )
    )
    pn = float(
        radar.noise_power(
            bw_hz=params["bw_r_hz"],
            noise_figure=noise_figure,
            temperature_k=temperature_k,
        )
    )
    pj_rx = np.asarray(
        radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=r,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
    )
    pd_clean = np.asarray(radar.pd_from_powers(pfa, pr, pn))
    pd_jammed = np.asarray(radar.pd_from_powers(pfa, pr, pn, pj_rx))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(r / 1.0e3, pd_clean, color=PRIMARY, linewidth=2.0, label="thermal noise only")
    ax.plot(r / 1.0e3, pd_jammed, color=JAM, linewidth=2.0, label="with noise jamming")
    ax.axhline(
        pd_threshold,
        color=MUTED,
        linestyle=":",
        linewidth=1.2,
        label=f"$P_d$ = {pd_threshold:.2f}",
    )
    ax.axvline(
        r_bt / 1.0e3,
        color=JAM,
        linestyle="--",
        linewidth=1.0,
        label=f"$R_{{bt}}$ = {r_bt / 1.0e3:.2f} km",
    )
    ax.axvline(
        r_det / 1.0e3,
        color=PRIMARY,
        linestyle="--",
        linewidth=1.0,
        label=f"$R_{{det}}$ = {r_det / 1.0e3:.0f} km",
    )
    if r_det_jammed is not None:
        ax.axvline(
            r_det_jammed / 1.0e3,
            color=JAM,
            linestyle=":",
            linewidth=1.2,
            label=f"jammed $R_{{det}}$ = {r_det_jammed / 1.0e3:.2f} km",
        )
    ax.set_xscale("log")
    ax.set_xlim(r[0] / 1.0e3, r[-1] / 1.0e3)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Range  $R$  [km]")
    ax.set_ylabel("Detection probability  $P_d$")
    ax.set_title(title)
    ax.legend(loc="lower left", ncol=2)
    return _save(fig, out_path)


# --- 图 3：J/S vs 距离 -------------------------------------------------------


def plot_js_vs_range(
    params: Mapping[str, float],
    *,
    out_path: Path,
    title: str = "Jamming-to-signal ratio vs range",
    points: int = 400,
) -> Path:
    """J/S 随距离变化，双对数下斜率为 +2（干扰 1/R²、回波 1/R⁴）。"""
    _style()
    r_bt = float(radar.burn_through_range(**_burn_through_kwargs(params)))
    r = np.geomspace(r_bt / 100.0, 3.0e5, points)
    js = np.asarray(radar.js_ratio_self_screen(range_m=r, **params))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(r / 1.0e3, js, color=PRIMARY, linewidth=2.0, label="J/S(R)")
    ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=1.2, label="J/S = 1")
    ax.axvline(
        r_bt / 1.0e3,
        color=JAM,
        linestyle="--",
        linewidth=1.0,
        label=f"$R_{{bt}}$ = {r_bt / 1.0e3:.2f} km",
    )
    ax.fill_between(
        r / 1.0e3,
        js,
        1.0,
        where=js >= 1.0,
        color=JAM,
        alpha=0.10,
        linewidth=0,
        label="jammer-dominant (J/S > 1)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Range  $R$  [km]")
    ax.set_ylabel("Jamming-to-signal ratio  J/S  [linear]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.text(
        0.015,
        0.04,
        r"slope $+2$: $J/S \propto R^2$",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5159",
    )
    return _save(fig, out_path)


# --- 一次出三张 --------------------------------------------------------------


def plot_all(
    params: Mapping[str, float],
    *,
    out_dir: str | Path,
    stem: str,
    title_prefix: str | None = None,
    pfa: float = 1.0e-6,
    pd_threshold: float = 0.5,
    noise_figure: float = 1.0,
    temperature_k: float = 290.0,
) -> list[Path]:
    """出 T0 三张图，返回生成的文件路径列表。"""
    out_dir = Path(out_dir)
    prefix = title_prefix or stem
    common = {
        "pfa": pfa,
        "pd_threshold": pd_threshold,
        "noise_figure": noise_figure,
        "temperature_k": temperature_k,
    }
    return [
        plot_burn_through_curve(
            params,
            out_path=out_dir / f"{stem}_burnthrough.png",
            title=f"{prefix} · burn-through range vs jammer power",
            **common,
        ),
        plot_pd_vs_range(
            params,
            out_path=out_dir / f"{stem}_pd_vs_range.png",
            title=f"{prefix} · detection probability vs range",
            **common,
        ),
        plot_js_vs_range(
            params,
            out_path=out_dir / f"{stem}_js_vs_range.png",
            title=f"{prefix} · jamming-to-signal ratio vs range",
        ),
    ]
