"""把想定编译成 Web 前端要的数据包（T2 的"无后端"前提）。

为什么要有这一层
----------------
PROJECT.md §2 决策 3 要求"同一份物理，不许前后端各写一套"，而 T2 又要求无后端。
两条同时成立的唯一做法：**物理只在 Python 里算一次**，前端只做查表 + 插值 + 画图。

导出的东西
----------
1. ``pd`` / ``js`` 二维表：自变量是 (目标距离 Rt, 干扰机距离 Rj)，单位米，对数网格。
   因为模型对各向同性（只跟距离有关），地图上任意一点都能由 (Rt, Rj) 查出来。
2. 三条一维曲线：检测边界 R_pd(Rj)、烧穿环 R_bt(Rj)、以及目标处的 Pd/J-S 随 Rj 的变化。
3. 场景元数据：雷达/干扰机/目标的参数与位置、Pfa、门槛。

编码：表用 base64 的 float32 小端序，避免 JSON 浮点数组把体积放大 3–5 倍；
一维曲线直接用普通数组（几百个数，没必要编码）。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model.scenario import Scenario
from .physics import radar

__all__ = ["SCHEMA", "pack_array", "unpack_array", "build_web_bundle", "write_web_bundle"]

#: 数据包格式版本；前端遇到不认识的版本应当直接报错而不是猜
SCHEMA = "spectrumfog.web/1"

_DTYPE = np.dtype("<f4")  # 小端 float32

#: 直接喂给 ``radar.js_ratio_standoff`` 的那一组键（在 link_params 里）
_JS_KEYS = (
    "pt_w",
    "gain",
    "sigma_m2",
    "lam",
    "pj_w",
    "gj",
    "bw_r_hz",
    "bw_j_hz",
    "loss",
)


def pack_array(array: np.ndarray) -> dict[str, Any]:
    """把二维/一维数组压成 base64 float32。"""
    values = np.ascontiguousarray(np.asarray(array, dtype=_DTYPE))
    return {
        "dtype": "float32",
        "shape": list(values.shape),
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def unpack_array(packed: dict[str, Any]) -> np.ndarray:
    """``pack_array`` 的逆运算（测试与前端文档用）。"""
    raw = base64.b64decode(packed["data"])
    values = np.frombuffer(raw, dtype=_DTYPE)
    return values.reshape(tuple(packed["shape"]))


def _log_grid(low_m: float, high_m: float, samples: int) -> np.ndarray:
    if not 0.0 < low_m < high_m:
        raise ValueError(f"网格范围非法：[{low_m}, {high_m}]")
    if samples < 3:
        raise ValueError(f"samples 至少 3，收到 {samples}")
    return np.geomspace(low_m, high_m, samples)


def build_web_bundle(
    scenario: Scenario,
    *,
    pfa: float = 1.0e-6,
    pd_threshold: float = 0.5,
    radar_id: str | None = None,
    jammer_id: str | None = None,
    samples: int = 97,
    ring_samples: int = 241,
    rt_bounds_m: tuple[float, float] = (1.0e2, 5.0e5),
    rj_bounds_m: tuple[float, float] = (1.0e2, 1.0e6),
) -> dict[str, Any]:
    """生成可直接 ``json.dumps`` 的数据包。"""
    radar_platform, radar_emitter = scenario.primary_radar(radar_id)
    jammer_platform, jammer_emitter = scenario.primary_jammer(jammer_id)
    params = scenario.link_params(radar_platform.id, jammer_platform.id)
    environment = scenario.environment

    # T2 的交互目标是"雷达看某个目标、干扰机在别处"，所以缺省用 detect objective
    target_platform = None
    for objective in scenario.objectives:
        if objective.type == "detect":
            target_platform = scenario.platform(objective.target)
            break
    if target_platform is None:
        target_platform = jammer_platform  # 自屏蔽：目标就是干扰机

    rt = _log_grid(*rt_bounds_m, samples)  # 目标距离
    rj = _log_grid(*rj_bounds_m, samples)  # 干扰机距离

    pn_w = float(
        radar.noise_power(
            bw_hz=params["bw_r_hz"],
            noise_figure=environment.noise_figure,
            temperature_k=environment.temperature_k,
        )
    )

    # --- 二维表：Pd(Rt, Rj) 与 JS(Rt, Rj) ---
    # Pr 只跟 Rt 有关（一行），Pj_rx 只跟 Rj 有关（一列），广播成矩阵。
    pr_row = np.asarray(
        radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=rt,
            loss=params["loss"],
        )
    )[None, :]
    pj_col = np.asarray(
        radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=rj,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
    )[:, None]
    pd_table = np.asarray(radar.pd_from_powers(pfa, pr_row, pn_w, pj_col))

    js_table = np.asarray(
        radar.js_ratio_standoff(
            range_m=rt[None, :],
            jammer_range_m=rj[:, None],
            pt_w=params["pt_w"],
            gain=params["gain"],
            sigma_m2=params["sigma_m2"],
            lam=params["lam"],
            pj_w=params["pj_w"],
            gj=params["gj"],
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
            loss=params["loss"],
        )
    )

    # --- 一维曲线（按干扰机距离扫） ---
    rj_curve = _log_grid(*rj_bounds_m, ring_samples)
    jammer_kwargs = {
        "pj_w": params["pj_w"],
        "gj": params["gj"],
        "bw_j_hz": params["bw_j_hz"],
        "lam": params["lam"],
        "pt_w": params["pt_w"],
        "gain": params["gain"],
        "sigma_m2": params["sigma_m2"],
        "bw_r_hz": params["bw_r_hz"],
        "noise_figure": environment.noise_figure,
        "temperature_k": environment.temperature_k,
        "loss": params["loss"],
    }
    detection_ring = np.array(
        [
            float(
                radar.detection_range(
                    pd_threshold, pfa, jammer_range_m=float(value), **jammer_kwargs
                )
            )
            for value in rj_curve
        ]
    )
    burn_through = np.asarray(
        radar.burn_through_range_standoff(
            pt_w=params["pt_w"],
            gain=params["gain"],
            sigma_m2=params["sigma_m2"],
            pj_w=params["pj_w"],
            gj=params["gj"],
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
            loss=params["loss"],
            jammer_range_m=rj_curve,
        )
    )

    # 目标处（固定位置）的 Pd / J-S 随干扰机距离的变化
    target_range_m = float(
        np.hypot(
            target_platform.pos[0] - radar_platform.pos[0],
            target_platform.pos[1] - radar_platform.pos[1],
        )
    )
    pr_target = float(
        radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=params["gain"],
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=target_range_m,
            loss=params["loss"],
        )
    )
    pj_curve = np.asarray(
        radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=params["gain"],
            lam=params["lam"],
            range_j_m=rj_curve,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
    )
    pd_target = np.asarray(radar.pd_from_powers(pfa, pr_target, pn_w, pj_curve))
    js_target = np.asarray(
        radar.js_ratio_standoff(
            range_m=target_range_m,
            jammer_range_m=rj_curve,
            **{key: params[key] for key in _JS_KEYS},
        )
    )

    return {
        "schema": SCHEMA,
        "scenario": scenario.meta.name,
        "units": "SI: m, W, Hz, s",
        "pfa": float(pfa),
        "pd_threshold": float(pd_threshold),
        "noise_power_w": pn_w,
        "radar": {
            "id": radar_platform.id,
            "pos_m": [float(radar_platform.pos[0]), float(radar_platform.pos[1])],
            "freq_hz": float(radar_emitter.freq_hz),
            "bw_hz": float(radar_emitter.bw_hz),
            "pt_w": float(radar_emitter.pt_w),
            "gain_db": float(radar_emitter.gain_db),
            "sigma_m2": float(radar_emitter.sigma_m2),
            "loss_db": float(radar_emitter.loss_db),
        },
        "jammer": {
            "id": jammer_platform.id,
            "pos_m": [float(jammer_platform.pos[0]), float(jammer_platform.pos[1])],
            "freq_hz": float(jammer_emitter.freq_hz),
            "bw_hz": float(jammer_emitter.bw_hz),
            "pj_w": float(jammer_emitter.pj_w),
            "gain_db": float(jammer_emitter.gain_db),
        },
        "target": {
            "id": target_platform.id,
            "pos_m": [float(target_platform.pos[0]), float(target_platform.pos[1])],
            "range_m": target_range_m,
        },
        "grid": {
            "rt_m": {"min": float(rt[0]), "max": float(rt[-1]), "n": int(rt.size)},
            "rj_m": {"min": float(rj[0]), "max": float(rj[-1]), "n": int(rj.size)},
            "spacing": "log10",
            "units": "m",
        },
        "pd": pack_array(pd_table),
        "js": pack_array(js_table),
        "rings": {
            "jammer_range_m": [float(value) for value in rj_curve],
            "detection_m": [float(value) for value in detection_ring],
            "burn_through_m": [float(value) for value in burn_through],
        },
        "target_curves": {
            "jammer_range_m": [float(value) for value in rj_curve],
            "pd": [float(value) for value in pd_target],
            "js": [float(value) for value in js_target],
        },
    }


def write_web_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    """写 JSON（紧凑分隔符：体积敏感，前端不需要缩进）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(bundle, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    return target
