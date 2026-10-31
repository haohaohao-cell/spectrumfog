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
import math
from typing import Any, Sequence

import numpy as np

from .. import metrics
from ..bands import in_band
from ..model.emitter import CommsEmitter, DecoyEmitter
from ..model.platform import Platform
from ..model.scenario import Scenario
from ..physics import comms, propagation, radar
from . import detect, fusion
from .imm import IMMTracker
from .tracker import Tracker
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
        tracker: str = "cv",
    ) -> None:
        if not 0.0 < pfa < 1.0:
            raise ValueError(f"pfa 必须落在 (0, 1)，收到 {pfa!r}")
        if not 0.0 < pd_threshold < 1.0:
            raise ValueError(f"pd_threshold 必须落在 (0, 1)，收到 {pd_threshold!r}")
        if tracker not in ("cv", "imm"):
            raise ValueError(f"tracker 只能是 'cv' 或 'imm'，收到 {tracker!r}")
        self.scenario = scenario
        self.pfa = float(pfa)
        self.pd_threshold = float(pd_threshold)
        self.tracker_kind = tracker

    # --- 主循环 -----------------------------------------------------------

    def run(self) -> SimulationResult:
        scenario = self.scenario
        radar_platform, radar_emitter = scenario.primary_radar()
        # 每个接收机选"自己频段内"的干扰机；找不到才退回主干扰机
        radar_jammer = self._jammer_for(radar_emitter.freq_hz, radar_emitter.bw_hz)
        jammer_platform, jammer_emitter = (
            radar_jammer if radar_jammer is not None else scenario.primary_jammer()
        )
        target_platform = self._target_platform(jammer_platform)
        params = dict(scenario.link_params(radar_platform.id, jammer_platform.id))
        # 处理增益（默认 0 dB = 1.0，既有想定完全不受影响）。开启后等效于
        # 把干扰机功率除以 Gp：J/S_eff = (Pj_rx/Pr)/Gp，见 docs/physics.md §3.9。
        processing_gain = float(radar_emitter.processing_gain)
        params["pj_w"] = params["pj_w"] / processing_gain
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
            "processing_gain_db": float(radar_emitter.processing_gain_db),
        }

        events: list[dict[str, Any]] = [meta]
        detected_series: list[bool] = []
        pd_series: list[float] = []
        jammer_ranges: list[float] = []
        comms_states: list[bool] = []
        trackers: dict[str, Tracker] = {}
        # 雷达的量测精度比 ESM 高一个量级，所以门限也收紧
        radar_tracker = Tracker(
            dt=scenario.meta.dt,
            assoc_radius_m=1500.0,
            update_radius_m=1500.0,
            confirm_hits=3,
            max_coast=4,
        )
        radar_track_ticks = 0
        false_track_ticks = 0
        first_false_track_s: float | None = None
        decoy_pd_series: list[float] = []
        decoy_detected: list[bool] = []

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

            # 频段不重叠就不构成干扰：多频段想定（03_triad）靠这条区分谁打谁
            radar_band_ok = in_band(
                radar_emitter.freq_hz,
                radar_emitter.bw_hz,
                jammer_emitter.freq_hz,
                jammer_emitter.bw_hz,
            )
            js = (
                float(
                    radar.js_ratio_standoff(
                        range_m=range_target_m,
                        jammer_range_m=range_jammer_m,
                        **params,
                    )
                )
                if radar_band_ok
                else 0.0
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
            pj_rx_w = (
                float(
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
                if radar_band_ok
                else 0.0
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
                    "processing_gain_db": float(radar_emitter.processing_gain_db),
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

            # --- 通信链路（T3b）：只读几何 + 链路预算 ---
            for platform in scenario.platforms:
                for emitter in platform.emitters:
                    if not isinstance(emitter, CommsEmitter):
                        continue
                    comms_jammer = self._jammer_for(emitter.freq_hz, emitter.bw_hz)
                    comms_event = self._comms_event(
                        t_s=tick.t,
                        platform=platform,
                        emitter=emitter,
                        positions=positions,
                        jammer=comms_jammer,
                    )
                    events.append(comms_event)
                    comms_states.append(bool(comms_event["available"]))

            # --- 诱饵：雷达看到的假目标（T3b）---
            decoy_events, decoy_powers = self._decoy_tick(
                t_s=tick.t,
                positions=positions,
                radar_platform=radar_platform,
                radar_emitter=radar_emitter,
                radar_pos=radar_pos,
                target_echo_w=pr_w,
                params=params,
                pn_w=pn_w,
                pj_rx_w=pj_rx_w,
                rng=rng,
            )
            for decoy_event in decoy_events:
                events.append(decoy_event)
                decoy_pd_series.append(float(decoy_event["pd"]))
                decoy_detected.append(bool(decoy_event["detected"]))

            # --- 雷达航迹（假航迹的来源）：真目标 + 每个诱饵的视在位置 ---
            radar_measurements = []
            if detected:
                radar_measurements.append(
                    self._radar_measurement(
                        radar_pos, target_pos, params["bw_r_hz"]
                    )
                )
            for decoy_event in decoy_events:
                if not decoy_event["detected"]:
                    continue
                radar_measurements.append(
                    self._radar_measurement(
                        radar_pos,
                        (decoy_event["apparent_pos"][0], decoy_event["apparent_pos"][1]),
                        params["bw_r_hz"],
                    )
                )
            for radar_event in radar_tracker.step(radar_measurements, tick.t):
                radar_event["source"] = "radar"
                radar_event["phantom"] = _closer_to_decoy(
                    radar_event["pos"],
                    target_pos,
                    [event["apparent_pos"] for event in decoy_events],
                )
                events.append(radar_event)
                radar_track_ticks += 1
                if radar_event["phantom"]:
                    false_track_ticks += 1
                    if first_false_track_s is None:
                        first_false_track_s = float(tick.t)

            # --- ESM 观测 + 融合（T3a）---
            if scenario.sensors:
                observations = self._observe_tick(
                    t_s=tick.t,
                    positions=positions,
                    rng=rng,
                    effective_powers=decoy_powers,
                )
                for observation in observations:
                    events.append(observation.to_event())
                estimate_events = self._fuse_tick(
                    t_s=tick.t, observations=observations
                )
                events.extend(estimate_events)
                # --- 航迹层（T3b+）：每个辐射源一条独立航迹 ---
                by_emitter: dict[str, list[tuple[list[float], list[list[float]]]]] = {}
                for estimate_event in estimate_events:
                    by_emitter.setdefault(estimate_event["target"], []).append(
                        (estimate_event["pos"], estimate_event["cov"])
                    )
                for emitter_id, fixes in by_emitter.items():
                    tracker = trackers.setdefault(
                        emitter_id, self._make_tracker(scenario.meta.dt)
                    )
                    for track_event in tracker.step(fixes, tick.t):
                        track_event["emitter"] = emitter_id
                        events.append(track_event)

        summary = self._summarise(
            detected_series=detected_series,
            pd_series=pd_series,
            jammer_ranges=jammer_ranges,
            dt=scenario.meta.dt,
            unjammed_range_m=unjammed_range_m,
            n_fixes=sum(1 for event in events if event["kind"] == "estimate"),
            comms_states=comms_states,
            decoy_pd=decoy_pd_series,
            decoy_detected=decoy_detected,
            track_events=[
                event
                for event in events
                if event["kind"] == "track" and event.get("source") != "radar"
            ],
            radar_track_ticks=radar_track_ticks,
            false_track_ticks=false_track_ticks,
            first_false_track_s=first_false_track_s,
            localize_target=self._localize_target(),
            n_ticks=int(round(scenario.meta.duration / scenario.meta.dt)) + 1,
        )
        return SimulationResult(meta=meta, events=events, summary=summary)

    # --- 内部 -------------------------------------------------------------

    def _radar_measurement(
        self,
        radar_pos: tuple[float, float],
        target_pos: tuple[float, float],
        bw_hz: float,
    ) -> tuple[tuple[float, float], Any]:
        """雷达对某个点的量测协方差：距离分辨率 + 测角精度，投到直角坐标。"""
        range_m = _distance(radar_pos, target_pos)
        sigma_r = max(fusion.SPEED_OF_LIGHT_MPS / (2.0 * float(bw_hz)), 1.0) / 2.0
        sigma_cross = max(range_m * math.radians(0.5), 1.0)  # 0.5° 测角
        ux = (target_pos[0] - radar_pos[0]) / max(range_m, 1e-9)
        uy = (target_pos[1] - radar_pos[1]) / max(range_m, 1e-9)
        radial = np.array([[ux * ux, ux * uy], [uy * ux, uy * uy]]) * sigma_r**2
        cross = np.array([[uy * uy, -uy * ux], [-ux * uy, ux * ux]]) * sigma_cross**2
        return (float(target_pos[0]), float(target_pos[1])), radial + cross

    def _make_tracker(self, dt: float):
        """按 ``tracker`` 选择跟踪器：cv = 单模型最近邻，imm = 双模型 + JPDA。"""
        if self.tracker_kind == "imm":
            return IMMTracker(dt=dt)
        return Tracker(dt=dt)

    def _jammer_for(self, freq_hz: float, bw_hz: float):
        """挑一台与给定频段重叠的干扰机；没有就返回 None。"""
        for platform, emitter in self.scenario.jammers():
            if in_band(freq_hz, bw_hz, emitter.freq_hz, emitter.bw_hz):
                return platform, emitter
        return None

    def _decoy_tick(
        self,
        *,
        t_s: float,
        positions: dict[str, tuple[float, float]],
        radar_platform: Platform,
        radar_emitter,
        radar_pos: tuple[float, float],
        target_echo_w: float,
        params: dict[str, float],
        pn_w: float,
        pj_rx_w: float,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """本 tick 每个诱饵在雷达处造成的假目标，以及它自己的转发功率。

        返回 ``(事件列表, {平台 id: 转发功率})``——后者给 ESM 用：
        诱饵只在该帧辐射，所以它的事后可发现性由照射强度决定。
        """
        events: list[dict[str, Any]] = []
        powers: dict[str, float] = {}
        for platform in self.scenario.platforms:
            for emitter in platform.emitters:
                if not isinstance(emitter, DecoyEmitter):
                    continue
                if not in_band(
                    radar_emitter.freq_hz,
                    radar_emitter.bw_hz,
                    emitter.freq_hz,
                    emitter.bw_hz,
                ):
                    continue

                decoy_pos = positions[platform.id]
                range_m = _distance(radar_pos, decoy_pos)
                # 距离欺骗：雷达会把回波放到更远处（沿视线方向外推 offset）
                unit = (
                    (decoy_pos[0] - radar_pos[0]) / range_m,
                    (decoy_pos[1] - radar_pos[1]) / range_m,
                )
                apparent_pos = (
                    decoy_pos[0] + unit[0] * emitter.range_offset_m,
                    decoy_pos[1] + unit[1] * emitter.range_offset_m,
                )
                tx_power = float(
                    radar.decoy_transmit_power(
                        pt_w=params["pt_w"],
                        gt=params["gain"],
                        decoy_gain=emitter.gain,
                        amplification=emitter.amplification,
                        lam=params["lam"],
                        range_m=range_m,
                    )
                )
                echo_w = float(
                    radar.decoy_echo_power(
                        pt_w=params["pt_w"],
                        gt=params["gain"],
                        gr=params["gain"],
                        decoy_gain=emitter.gain,
                        amplification=emitter.amplification,
                        lam=params["lam"],
                        range_m=range_m,
                    )
                )
                sigma_eq = float(
                    radar.decoy_equivalent_rcs(
                        lam=params["lam"],
                        gain=emitter.gain,
                        amplification=emitter.amplification,
                        loss=params["loss"],
                    )
                )
                pd_decoy = float(
                    radar.pd_from_powers(self.pfa, echo_w, pn_w, pj_rx_w)
                )
                draw = float(rng.random())

                powers[platform.id] = tx_power
                events.append(
                    {
                        "kind": "decoy",
                        "t": t_s,
                        "decoy": platform.id,
                        "radar": radar_platform.id,
                        "pos": [float(decoy_pos[0]), float(decoy_pos[1])],
                        "apparent_pos": [
                            float(apparent_pos[0]),
                            float(apparent_pos[1]),
                        ],
                        "range_offset_m": float(emitter.range_offset_m),
                        "range_m": range_m,
                        "sigma_eq_m2": sigma_eq,
                        "tx_power_w": tx_power,
                        "echo_w": echo_w,
                        "target_echo_w": target_echo_w,
                        "echo_ratio": (
                            echo_w / target_echo_w if target_echo_w > 0.0 else None
                        ),
                        "pd": pd_decoy,
                        "draw": draw,
                        "detected": draw < pd_decoy,
                    }
                )
        return events, powers

    def _observe_tick(
        self,
        *,
        t_s: float,
        positions: dict[str, tuple[float, float]],
        rng: np.random.Generator,
        effective_powers: dict[str, float] | None = None,
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
                    if effective_powers and platform.id in effective_powers:
                        # 诱饵的辐射功率取决于本帧被照得多亮，不是固定值
                        power = effective_powers[platform.id]
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

    @staticmethod
    def _track_summary(
        track_events: list[dict[str, Any]],
        localize_target: str | None,
        n_ticks: int | None,
    ) -> dict[str, Any]:
        """航迹统计：只统计"被定位目标"的航迹，其余辐射源只出现在事件流里。"""
        if localize_target is None or n_ticks is None:
            return {
                "track_confirmed_fraction": None,
                "first_track_s": None,
                "track_lost_s": None,
                "n_tracks": 0,
            }
        mine = [event for event in track_events if event.get("emitter") == localize_target]
        confirmed_ticks = {
            event["t"] for event in mine if event["status"] == "confirmed"
        }
        first_confirmed = min(confirmed_ticks) if confirmed_ticks else None
        lost = [event["t"] for event in mine if event["status"] == "lost"]
        return {
            "track_confirmed_fraction": len(confirmed_ticks) / max(1, n_ticks),
            "first_track_s": first_confirmed,
            "track_lost_s": min(lost) if lost else None,
            "n_tracks": len({event["track_id"] for event in mine}),
        }

    def _comms_event(
        self,
        *,
        t_s: float,
        platform: Platform,
        emitter: CommsEmitter,
        positions: dict[str, tuple[float, float]],
        jammer: tuple[Platform, Any] | None,
    ) -> dict[str, Any]:
        """一条链路的接收功率 / Eb/N0 / 实际 J/S / 抗干扰余量。"""
        environment = self.scenario.environment
        receiver = self.scenario.platform(emitter.to_platform)
        tx_pos = positions[platform.id]
        rx_pos = positions[receiver.id]
        lam = float(radar.wavelength(emitter.freq_hz))

        ps_rx_w = float(
            propagation.received_power(
                ptx_w=emitter.ptx_w,
                gtx=emitter.gain,
                grx=1.0,  # 接收端按全向处理（与 ESM 同一约定）
                freq_hz=emitter.freq_hz,
                range_m=float(np.hypot(rx_pos[0] - tx_pos[0], rx_pos[1] - tx_pos[1])),
            )
        )
        # 只有同频段的干扰机才打得到这条链路
        pj_rx_w = 0.0
        if jammer is not None:
            jammer_platform, jammer_emitter = jammer
            jammer_pos = positions[jammer_platform.id]
            pj_rx_w = float(
                radar.jammer_received_power(
                    pj_w=float(jammer_emitter.pj_w),
                    gj=float(jammer_emitter.gain),
                    gr=1.0,
                    lam=lam,
                    range_j_m=float(
                        np.hypot(rx_pos[0] - jammer_pos[0], rx_pos[1] - jammer_pos[1])
                    ),
                    bw_r_hz=emitter.bw_hz,
                    bw_j_hz=float(jammer_emitter.bw_hz),
                )
            )
        js = pj_rx_w / ps_rx_w if ps_rx_w > 0.0 else 0.0
        noise_psd = float(
            comms.noise_psd(
                noise_figure=environment.noise_figure,
                temperature_k=environment.temperature_k,
            )
        )
        ebn0 = float(comms.ebn0(ps_rx_w, noise_psd, emitter.bit_rate_bps))
        aj_margin_db = float(
            emitter.processing_gain_db
            + emitter.coding_gain_db
            - 10.0 * np.log10(max(js, 1e-30))
        )
        return {
            "kind": "comms",
            "t": t_s,
            "emitter": platform.id,
            "receiver": receiver.id,
            "range_m": float(
                np.hypot(rx_pos[0] - tx_pos[0], rx_pos[1] - tx_pos[1])
            ),
            "ps_rx_w": ps_rx_w,
            "pj_rx_w": pj_rx_w,
            "js": js,
            "ebn0": ebn0,
            "aj_margin_db": aj_margin_db,
            "available": aj_margin_db > 0.0,
        }

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
        comms_states: list[bool] | None = None,
        decoy_pd: list[float] | None = None,
        decoy_detected: list[bool] | None = None,
        track_events: list[dict[str, Any]] | None = None,
        localize_target: str | None = None,
        n_ticks: int | None = None,
        radar_track_ticks: int = 0,
        false_track_ticks: int = 0,
        first_false_track_s: float | None = None,
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
            "link_avail": (
                sum(comms_states) / len(comms_states) if comms_states else None
            ),
            "n_comms_samples": len(comms_states or []),
            "decoy_pd_mean": (
                sum(decoy_pd) / len(decoy_pd) if decoy_pd else None
            ),
            "decoy_detected_fraction": (
                sum(decoy_detected) / len(decoy_detected) if decoy_detected else None
            ),
            "n_decoy_samples": len(decoy_pd or []),
            **self._track_summary(track_events or [], localize_target, n_ticks),
            "radar_track_ticks": int(radar_track_ticks),
            "false_track_ticks": int(false_track_ticks),
            "false_track_fraction": (
                float(false_track_ticks) / radar_track_ticks
                if radar_track_ticks
                else None
            ),
            "first_false_track_s": first_false_track_s,
        }


def _closer_to_decoy(
    pos: Sequence[float],
    target_pos: Sequence[float],
    decoy_positions: Sequence[Sequence[float]],
) -> bool:
    """这条雷达航迹离诱饵比离真目标更近吗？（事后归因用，不参与滤波）"""
    if not decoy_positions:
        return False
    to_target = float(np.hypot(pos[0] - target_pos[0], pos[1] - target_pos[1]))
    to_decoy = min(
        float(np.hypot(pos[0] - decoy[0], pos[1] - decoy[1]))
        for decoy in decoy_positions
    )
    return to_decoy < to_target


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


