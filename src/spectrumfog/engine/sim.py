"""确定性主循环。

每个 tick 做四件事：

1. 按航路点求各平台位置（纯时间函数，无随机）；
2. 结算物理：回波功率 → 干扰功率 → J/S → Pd，以及两条半径
   （J/S = 1 的烧穿半径、Pd = 门槛的检测边界）；
3. 用 seeded RNG 掷一次检测（Pd 是概率，检测与否是抽样结果）；
4. 写事件：每个平台一条 ``state``，再加一条 ``link`` 与一条 ``detection``。

时间只来自 ``Clock``（整数倍 dt），随机只来自 ``np.random.default_rng(seed)``，
所以**同一份想定 + 同一 seed ⇒ 逐字节相同的 replay**（§2 决策 1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import metrics
from ..model.platform import Platform
from ..model.scenario import Scenario
from ..physics import radar
from . import detect, fusion
from .clock import Clock

__all__ = ["Simulation", "SimulationResult"]


@dataclass(frozen=True)
class SimulationResult:
    """一次仿真的产物：meta + 事件流 + 人读摘要。"""

    meta: dict[str, Any]
    events: list[dict[str, Any]]
    summary: dict[str, Any]


class Simulation:
    """单雷达 / 单干扰机 / 单目标的自屏蔽或站距干扰仿真（v1）。"""

    def __init__(
        self,
        scenario: Scenario,
        *,
        pfa: float = 1.0e-6,
        pd_threshold: float = 0.5,
    ) -> None:
        if not 0.0 < pfa < 1.0:
            raise ValueError(f"pfa 必须落在 (0, 1)，收到 {pfa!r}")
        if not 0.0 < pd_threshold < 1.0:
            raise ValueError(f"pd_threshold 必须落在 (0, 1)，收到 {pd_threshold!r}")
        self.scenario = scenario
        self.pfa = float(pfa)
        self.pd_threshold = float(pd_threshold)

    # --- 主循环 -----------------------------------------------------------

    def run(self) -> SimulationResult:
        scenario = self.scenario
        radar_platform, radar_emitter = scenario.primary_radar()
        jammer_platform, jammer_emitter = scenario.primary_jammer()
        target_platform = self._target_platform(jammer_platform)
        params = scenario.link_params(radar_platform.id, jammer_platform.id)
        environment = scenario.environment

        pn_w = float(
            radar.noise_power(
                bw_hz=params["bw_r_hz"],
                noise_figure=environment.noise_figure,
                temperature_k=environment.temperature_k,
            )
        )
        detection_kwargs = {
            "pt_w": params["pt_w"],
            "gain": params["gain"],
            "sigma_m2": params["sigma_m2"],
            "lam": params["lam"],
            "bw_r_hz": params["bw_r_hz"],
            "noise_figure": environment.noise_figure,
            "temperature_k": environment.temperature_k,
            "loss": params["loss"],
        }
        rbt_kwargs = {
            key: params[key]
            for key in ("pt_w", "gain", "sigma_m2", "pj_w", "gj", "bw_r_hz", "bw_j_hz", "loss")
        }
        jammer_kwargs = {
            "pj_w": params["pj_w"],
            "gj": params["gj"],
            "bw_j_hz": params["bw_j_hz"],
        }

        unjammed_range_m = float(
            radar.detection_range(self.pd_threshold, self.pfa, **detection_kwargs)
        )
        burn_through_self_m = float(radar.burn_through_range(**rbt_kwargs))

        clock = Clock(scenario.meta.dt, scenario.meta.duration)
        rng = np.random.default_rng(scenario.meta.seed)

        meta = {
            "kind": "meta",
            "t": 0.0,
            "scenario": scenario.meta.name,
            "seed": int(scenario.meta.seed),
            "dt": float(scenario.meta.dt),
            "duration": float(scenario.meta.duration),
            "pfa": self.pfa,
            "pd_threshold": self.pd_threshold,
            "radar": radar_platform.id,
            "jammer": jammer_platform.id,
            "target": target_platform.id,
            "units": "SI",
            "unjammed_detection_range_m": unjammed_range_m,
            "burn_through_self_m": burn_through_self_m,
            "noise_power_w": pn_w,
            "cep_threshold_m": self._cep_threshold(),
            "localize_target": self._localize_target(),
        }

        events: list[dict[str, Any]] = [meta]
        detected_series: list[bool] = []
        pd_series: list[float] = []
        jammer_ranges: list[float] = []

        for tick in clock:
            positions = {
                platform.id: platform.position_at(tick.t)
                for platform in scenario.platforms
            }
            for platform in scenario.platforms:
                events.append(
                    self._state_event(
                        t=tick.t, platform=platform, pos=positions[platform.id]
                    )
                )

            radar_pos = positions[radar_platform.id]
            target_pos = positions[target_platform.id]
            jammer_pos = positions[jammer_platform.id]
            range_target_m = _distance(radar_pos, target_pos)
            range_jammer_m = _distance(radar_pos, jammer_pos)

            js = float(
                radar.js_ratio_standoff(
                    range_m=range_target_m,
                    jammer_range_m=range_jammer_m,
                    **params,
                )
            )
            burn_through_m = float(
                radar.burn_through_range_standoff(
                    jammer_range_m=range_jammer_m, **rbt_kwargs
                )
            )
            pr_w = float(
                radar.received_power(
                    pt_w=params["pt_w"],
                    gt=params["gain"],
                    gr=params["gain"],
                    lam=params["lam"],
                    sigma_m2=params["sigma_m2"],
                    range_m=range_target_m,
                    loss=params["loss"],
                )
            )
            pj_rx_w = float(
                radar.jammer_received_power(
                    pj_w=params["pj_w"],
                    gj=params["gj"],
                    gr=params["gain"],
                    lam=params["lam"],
                    range_j_m=range_jammer_m,
                    bw_r_hz=params["bw_r_hz"],
                    bw_j_hz=params["bw_j_hz"],
                )
            )
            pd = float(radar.pd_from_powers(self.pfa, pr_w, pn_w, pj_rx_w))

            detection_ring_m = self._detection_ring(
                jammer_range_m=range_jammer_m,
                detection_kwargs=detection_kwargs,
                jammer_kwargs=jammer_kwargs,
            )

            draw = float(rng.random())
            detected = draw < pd

            events.append(
                {
                    "kind": "link",
                    "t": tick.t,
                    "radar": radar_platform.id,
                    "jammer": jammer_platform.id,
                    "target": target_platform.id,
                    "range_target_m": range_target_m,
                    "range_jammer_m": range_jammer_m,
                    "pr_w": pr_w,
                    "pj_rx_w": pj_rx_w,
                    "js": js,
                    "pd": pd,
                    "burn_through_m": burn_through_m,
                    "detection_ring_m": detection_ring_m,
                }
            )
            events.append(
                {
                    "kind": "detection",
                    "t": tick.t,
                    "sensor": radar_platform.id,
                    "target": target_platform.id,
                    "pd": pd,
                    "draw": draw,
                    "detected": detected,
                }
            )

            detected_series.append(detected)
            pd_series.append(pd)
            jammer_ranges.append(range_jammer_m)

            # --- ESM 观测 + 融合（T3a）---
            if scenario.sensors:
                observations = self._observe_tick(
                    t_s=tick.t, positions=positions, rng=rng
                )
                for observation in observations:
                    events.append(observation.to_event())
                for estimate_event in self._fuse_tick(
                    t_s=tick.t, observations=observations
                ):
                    events.append(estimate_event)

        summary = self._summarise(
            detected_series=detected_series,
            pd_series=pd_series,
            jammer_ranges=jammer_ranges,
            dt=scenario.meta.dt,
            unjammed_range_m=unjammed_range_m,
            n_fixes=sum(1 for event in events if event["kind"] == "estimate"),
        )
        return SimulationResult(meta=meta, events=events, summary=summary)

    # --- 内部 -------------------------------------------------------------

    def _observe_tick(
        self,
        *,
        t_s: float,
        positions: dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> list[detect.EsmMeasurement]:
        """本 tick 所有 ESM 传感器对敌方辐射源的观测（含虚假目标）。"""
        observations: list[detect.EsmMeasurement] = []
        for sensor in self.scenario.sensors:
            if sensor.kind != "esm":
                continue
            for platform in self.scenario.platforms:
                if platform.side == sensor.side:
                    continue
                for emitter in platform.emitters:
                    power = getattr(emitter, "pt_w", None) or getattr(emitter, "pj_w", None)
                    if power is None:
                        continue
                    observations.append(
                        detect.observe(
                            sensor_id=sensor.id,
                            sensor_pos=sensor.pos,
                            sensor_bw_hz=sensor.bw_hz,
                            sensor_noise_figure=sensor.noise_figure,
                            sensor_gain=sensor.gain,
                            emitter_id=platform.id,
                            emitter_pos=positions[platform.id],
                            emitter_power_w=float(power),
                            emitter_gain=float(emitter.gain),
                            emitter_freq_hz=float(emitter.freq_hz),
                            t_s=t_s,
                            pfa=self.pfa,
                            rng=rng,
                        )
                    )
            alarm = detect.false_alarm(
                sensor_id=sensor.id,
                sensor_pos=sensor.pos,
                t_s=t_s,
                pfa_scan=sensor.pfa_scan,
                rng=rng,
            )
            if alarm is not None:
                observations.append(alarm)
        return observations

    def _fuse_tick(
        self, *, t_s: float, observations: list[detect.EsmMeasurement]
    ) -> list[dict[str, Any]]:
        """把本 tick 的检出量测按辐射源分组，做加权最小二乘定位。"""
        grouped: dict[str, list[detect.EsmMeasurement]] = {}
        for observation in observations:
            if observation.detected and observation.emitter_id is not None:
                grouped.setdefault(observation.emitter_id, []).append(observation)

        estimates: list[dict[str, Any]] = []
        for emitter_id, items in grouped.items():
            fix = fusion.solve_fix(
                [
                    fusion.Measure(
                        sensor_id=item.sensor_id,
                        sensor_pos=item.sensor_pos,
                        aoa_rad=item.aoa_rad,
                        toa_s=item.toa_s,
                        sigma_aoa_rad=item.sigma_aoa_rad or fusion.DEFAULT_SIGMA_AOA_RAD,
                        sigma_toa_s=item.sigma_toa_s or fusion.DEFAULT_SIGMA_TOA_S,
                    )
                    for item in items
                ]
            )
            if fix is None:
                continue
            cep50, cep90 = metrics.cep_from_covariance(fix.covariance, samples=2000)
            estimates.append(
                {
                    "kind": "estimate",
                    "t": t_s,
                    "target": emitter_id,
                    "pos": [float(fix.pos[0]), float(fix.pos[1])],
                    "cov": [
                        [float(fix.covariance[0, 0]), float(fix.covariance[0, 1])],
                        [float(fix.covariance[1, 0]), float(fix.covariance[1, 1])],
                    ],
                    "cep_cov_m": cep50,
                    "cep90_cov_m": cep90,
                    "n_measurements": fix.n_measurements,
                    "sensors": list(fix.sensors),
                    "geometry_deg": fix.geometry_deg,
                }
            )
        return estimates

    def _cep_threshold(self) -> float:
        """"被有效定位"的 CEP 门槛 [m]：取想定里第一个 localize 目标，缺省 500 m。"""
        for objective in self.scenario.objectives:
            if objective.type == "localize" and objective.cep_m is not None:
                return float(objective.cep_m)
        return 500.0

    def _localize_target(self) -> str | None:
        """被定位的目标平台 id（localize objective）。

        注意它和 ``meta["target"]`` 不是一回事：后者是雷达要**探测**的平台，
        AAR 关心的是被对方 ESM **定位**的那个辐射源。
        """
        for objective in self.scenario.objectives:
            if objective.type == "localize":
                return objective.target
        return None

    def _target_platform(self, jammer_platform: Platform) -> Platform:
        """目标 = 想定里 ``type: detect`` 的 objective 指向的平台。

        没有这样的 objective 时退化为"干扰机就是目标"（自屏蔽），
        与 ``01_duel`` 那种单雷达 vs 单干扰机的想定一致。
        """
        for objective in self.scenario.objectives:
            if objective.type == "detect":
                return self.scenario.platform(objective.target)
        return jammer_platform

    def _state_event(
        self, *, t: float, platform: Platform, pos: tuple[float, float]
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "kind": "state",
            "t": t,
            "platform": platform.id,
            "pos": [float(pos[0]), float(pos[1])],
            "emitter": None,
            "on": False,
        }
        if platform.emitters:
            emitter = platform.emitters[0]
            power = getattr(emitter, "pt_w", None) or getattr(emitter, "pj_w", None)
            event.update(
                {
                    "emitter": emitter.kind,
                    "on": True,
                    "freq_hz": float(emitter.freq_hz),
                    "power_w": float(power) if power is not None else None,
                }
            )
        return event

    def _detection_ring(
        self,
        *,
        jammer_range_m: float,
        detection_kwargs: dict[str, float],
        jammer_kwargs: dict[str, float],
    ) -> float | None:
        """当前干扰态势下 Pd = 门槛的边界半径；完全被压制时返回 None。"""
        try:
            return float(
                radar.detection_range(
                    self.pd_threshold,
                    self.pfa,
                    jammer_range_m=jammer_range_m,
                    **jammer_kwargs,
                    **detection_kwargs,
                )
            )
        except ValueError:
            # 全距离都没达到门槛：边界不存在（HUD 显示 coverage: none）
            return None

    def _summarise(
        self,
        *,
        detected_series: list[bool],
        pd_series: list[float],
        jammer_ranges: list[float],
        dt: float,
        unjammed_range_m: float,
        n_fixes: int = 0,
    ) -> dict[str, Any]:
        """两种口径分开报，别混：

        * ``pd_below_threshold_at_s`` —— **确定性**事件：Pd 掉到门槛以下的那一刻，
          动画和"雷达失效"的判断都用它；
        * ``scan_detection_fraction`` / ``first_miss_s`` —— **抽样**结果：单次照射
          掷点没命中，只反映 Pd 的概率，不能当成"目标丢失"（漏检一两次不代表丢批）。
        """
        n = len(detected_series)
        scan_fraction = sum(detected_series) / n if n else 0.0
        first_miss = next(
            (index for index, hit in enumerate(detected_series) if not hit), None
        )
        below = next(
            (index for index, pd in enumerate(pd_series) if pd < self.pd_threshold), None
        )
        return {
            "n_ticks": n,
            "duration_s": (n - 1) * dt,
            "pd_below_threshold_at_s": None if below is None else below * dt,
            "scan_detection_fraction": scan_fraction,
            "first_miss_s": None if first_miss is None else first_miss * dt,
            "pd_min": min(pd_series) if pd_series else 0.0,
            "pd_max": max(pd_series) if pd_series else 0.0,
            "unjammed_detection_range_m": unjammed_range_m,
            "jammer_range_start_m": jammer_ranges[0] if jammer_ranges else None,
            "jammer_range_end_m": jammer_ranges[-1] if jammer_ranges else None,
            "n_fixes": int(n_fixes),
        }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))
