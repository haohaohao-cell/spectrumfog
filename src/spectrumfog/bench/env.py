"""Gymnasium 环境：把想定变成"红方干扰机"的决策问题。

**部分可观测体现在哪**
蓝方的定位结果对红方不可见。红方只有一个**延迟的告警**：最近 ``warning_window_s``
秒里是否被"有效定位"（CEP ≤ 门槛）。真正的 CEP 数字不进观测，策略只能靠告警推断
自己暴露了没有——这正是 AAR 里那句"你被定位了"的反面：当局者不知道。

**动作**（Discrete(15)）
``action = 方向 * 3 + 功率档``：方向 0=停, 1=+x, 2=-x, 3=+y, 4=-y；
功率档 0=关机, 1=半功率, 2=满功率（相对想定里的 ``pj_w``）。

**奖励**
+1：目标处 Pd < 门槛（干扰有效）
−1：本帧被有效定位（暴露）——与"干扰有效"同权，逼策略在两者间做取舍
−0.02 · 功率档：开机成本（鼓励"该关就关"）

一步 = 想定的一个 tick，物理量与 ``spectrumfog sim`` 完全同一套函数。
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..bands import in_band
from ..engine import detect, fusion
from ..metrics import cep_from_covariance
from ..model.scenario import Scenario
from ..physics import comms, propagation, radar

__all__ = ["SpectrumFogEnv", "POWER_LEVELS", "DIRECTIONS", "decode_action"]

#: 功率档：相对想定里标称 pj_w 的比例
POWER_LEVELS = (0.0, 0.5, 1.0)

#: 方向：0=停, 1=+x, 2=-x, 3=+y, 4=-y
DIRECTIONS = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))

_ACTIONS = len(DIRECTIONS) * len(POWER_LEVELS)


def decode_action(action: int) -> tuple[int, int]:
    """Discrete 动作 → (方向下标, 功率档下标)。"""
    if not 0 <= int(action) < _ACTIONS:
        raise ValueError(f"动作越界：{action}")
    return int(action) // len(POWER_LEVELS), int(action) % len(POWER_LEVELS)


class SpectrumFogEnv(gym.Env):
    """单红方干扰机、单蓝方目标 + ESM 星座的决策环境。"""

    metadata = {"render_modes": [], "name": "SpectrumFogEnv"}

    def __init__(
        self,
        scenario: Scenario,
        *,
        move_speed_mps: float = 500.0,
        warning_window_s: float = 10.0,
        bounds_m: float | None = None,
        pd_threshold: float = 0.5,
        pfa: float = 1e-6,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.move_speed_mps = float(move_speed_mps)
        self.warning_window_s = float(warning_window_s)
        self.pd_threshold = float(pd_threshold)
        self.pfa = float(pfa)

        self.radar_platform, self.radar_emitter = scenario.primary_radar()
        self.jammer_platform, self.jammer_emitter = scenario.primary_jammer()
        self.params = scenario.link_params(self.radar_platform.id, self.jammer_platform.id)
        self.target_platform = self._localize_target()
        self.threshold_m = self._cep_threshold()

        self.dt = float(scenario.meta.dt)
        self.max_steps = int(round(scenario.meta.duration / self.dt))
        self.bounds_m = float(
            bounds_m
            if bounds_m is not None
            else max(1.5 * math.hypot(*self.jammer_platform.pos), 80_000.0)
        )

        # 频段决定"这台干扰机打谁"：同频打雷达，否则找一条它打得到的通信链路
        self.radar_band_ok = in_band(
            self.radar_emitter.freq_hz,
            self.radar_emitter.bw_hz,
            self.jammer_emitter.freq_hz,
            self.jammer_emitter.bw_hz,
        )
        self.link = None if self.radar_band_ok else self._find_link()
        self.mode = "link" if self.link is not None else "radar"

        self.action_space = spaces.Discrete(_ACTIONS)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(6,), dtype=np.float32
        )

        # 运行期状态
        self._rng = np.random.default_rng(0)
        self._pos = np.zeros(2)
        self._initial_pos = np.asarray(self.jammer_platform.position_at(0.0), dtype=float)
        self._power = 1.0
        self._step_index = 0
        self._warning: list[bool] = []
        self._last = {"pd": 0.0, "cep": float("nan"), "localized": False, "js": 0.0}

    # --- gym API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else 0)
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._pos = np.asarray(self.jammer_platform.position_at(0.0), dtype=float)
        self._initial_pos = self._pos.copy()
        self._power = POWER_LEVELS[-1]
        self._step_index = 0
        self._warning = []
        self._last = {"pd": 0.0, "cep": float("nan"), "localized": False, "js": 0.0}
        return self._observation(), self._info()

    def step(self, action: int):
        direction_index, power_index = decode_action(action)
        self._power = POWER_LEVELS[power_index]

        dx, dy = DIRECTIONS[direction_index]
        self._pos = np.clip(
            self._pos + np.array([dx, dy]) * self.move_speed_mps * self.dt,
            -self.bounds_m,
            self.bounds_m,
        )

        t = self._step_index * self.dt
        self._step_index += 1

        radar_pos = np.asarray(self.radar_platform.position_at(t), dtype=float)
        target_pos = np.asarray(self.target_platform.position_at(t), dtype=float)
        range_target = float(np.linalg.norm(target_pos - radar_pos))
        range_jammer = float(np.linalg.norm(self._pos - radar_pos))

        pd_value, js_value = self._range_physics(range_target, range_jammer)
        link_state = self._link_state() if self.mode == "link" else None
        cep = self._sense()
        localized = bool(np.isfinite(cep) and cep <= self.threshold_m)

        if self.mode == "link":
            effective = link_state is not None and not link_state["available"]
        else:
            effective = pd_value < self.pd_threshold

        reward = 0.0
        if effective:
            reward += 1.0
        if localized:
            reward -= 1.0
        reward -= 0.02 * self._power

        self._warning.append(localized)
        keep = max(1, int(round(self.warning_window_s / self.dt)))
        self._warning = self._warning[-keep:]

        self._last = {
            "pd": pd_value,
            "cep": cep,
            "localized": localized,
            "js": js_value,
            "range_jammer": range_jammer,
            "power": self._power,
            "mode": self.mode,
            "link": link_state,
        }

        terminated = self._step_index >= self.max_steps
        truncated = False
        return self._observation(), float(reward), terminated, truncated, self._info()

    # --- 内部 ------------------------------------------------------------

    def _range_physics(self, range_target: float, range_jammer: float) -> tuple[float, float]:
        """目标处 Pd 与 J/S —— 与 ``radar`` 同一套函数，只是把功率乘上档位。"""
        pn_w = float(
            radar.noise_power(
                bw_hz=self.params["bw_r_hz"],
                noise_figure=self.scenario.environment.noise_figure,
                temperature_k=self.scenario.environment.temperature_k,
            )
        )
        pr_w = float(
            radar.received_power(
                pt_w=self.params["pt_w"],
                gt=self.params["gain"],
                gr=self.params["gain"],
                lam=self.params["lam"],
                sigma_m2=self.params["sigma_m2"],
                range_m=range_target,
                loss=self.params["loss"],
            )
        )
        # 处理增益（默认 0 dB = 1.0）：等效于把干扰机功率除以 Gp
        pj_w = self.params["pj_w"] * self._power / float(self.radar_emitter.processing_gain)
        # 关机、或这台干扰机根本不在雷达频段内 → 无干扰
        if pj_w <= 0.0 or not self.radar_band_ok:
            return float(radar.pd_from_powers(self.pfa, pr_w, pn_w)), 0.0
        pj_rx = float(
            radar.jammer_received_power(
                pj_w=pj_w,
                gj=self.params["gj"],
                gr=self.params["gain"],
                lam=self.params["lam"],
                range_j_m=range_jammer,
                bw_r_hz=self.params["bw_r_hz"],
                bw_j_hz=self.params["bw_j_hz"],
            )
        )
        pd_value = float(radar.pd_from_powers(self.pfa, pr_w, pn_w, pj_rx))
        return pd_value, pj_rx / pr_w

    def _sense(self) -> float:
        """ESM 观测 + 融合，返回本帧的 CEP（解不出来为 nan）。"""
        if self._power <= 0.0:
            return float("nan")  # 关机：没有辐射，也就没有被截获

        t = self._step_index * self.dt
        emitter_power = float(self.jammer_emitter.pj_w) * self._power
        measurements = []
        for sensor in self.scenario.sensors:
            if sensor.kind != "esm" or sensor.side == self.jammer_platform.side:
                continue
            observation = detect.observe(
                sensor_id=sensor.id,
                sensor_pos=sensor.pos,
                sensor_bw_hz=sensor.bw_hz,
                sensor_noise_figure=sensor.noise_figure,
                sensor_gain=sensor.gain,
                emitter_id=self.jammer_platform.id,
                emitter_pos=(float(self._pos[0]), float(self._pos[1])),
                emitter_power_w=emitter_power,
                emitter_gain=float(self.jammer_emitter.gain),
                emitter_freq_hz=float(self.jammer_emitter.freq_hz),
                t_s=t,
                pfa=1e-6,
                rng=self._rng,
            )
            if observation.detected:
                measurements.append(
                    fusion.Measure(
                        sensor_id=observation.sensor_id,
                        sensor_pos=observation.sensor_pos,
                        aoa_rad=observation.aoa_rad,
                        toa_s=observation.toa_s,
                        sigma_aoa_rad=observation.sigma_aoa_rad or fusion.DEFAULT_SIGMA_AOA_RAD,
                        sigma_toa_s=observation.sigma_toa_s or fusion.DEFAULT_SIGMA_TOA_S,
                    )
                )
        fix = fusion.solve_fix(measurements)
        if fix is None:
            return float("nan")
        return float(cep_from_covariance(fix.covariance, samples=600)[0])

    def _find_link(self):
        """找一条这台干扰机"打得到"的通信链路（同频段）。"""
        from ..model.emitter import CommsEmitter

        for platform in self.scenario.platforms:
            if platform.id == self.jammer_platform.id:
                continue
            for emitter in platform.emitters:
                if not isinstance(emitter, CommsEmitter):
                    continue
                if in_band(
                    emitter.freq_hz,
                    emitter.bw_hz,
                    self.jammer_emitter.freq_hz,
                    self.jammer_emitter.bw_hz,
                ):
                    return (
                        platform,
                        emitter,
                        self.scenario.platform(emitter.to_platform),
                    )
        return None

    def _link_state(self) -> dict:
        """链路当前状态：接收功率、J/S、抗干扰余量、是否可用。"""
        platform, emitter, receiver = self.link  # type: ignore[misc]
        t = self._step_index * self.dt
        tx = np.asarray(platform.position_at(t), dtype=float)
        rx = np.asarray(receiver.position_at(t), dtype=float)
        lam = float(radar.wavelength(emitter.freq_hz))

        ps_rx_w = float(
            propagation.received_power(
                ptx_w=emitter.ptx_w,
                gtx=emitter.gain,
                grx=1.0,
                freq_hz=emitter.freq_hz,
                range_m=float(np.linalg.norm(rx - tx)),
            )
        )
        pj_rx_w = 0.0
        if self._power > 0.0:
            pj_rx_w = float(
                radar.jammer_received_power(
                    pj_w=float(self.jammer_emitter.pj_w) * self._power,
                    gj=float(self.jammer_emitter.gain),
                    gr=1.0,
                    lam=lam,
                    range_j_m=float(np.linalg.norm(rx - self._pos)),
                    bw_r_hz=emitter.bw_hz,
                    bw_j_hz=float(self.jammer_emitter.bw_hz),
                )
            )
        js = pj_rx_w / ps_rx_w if ps_rx_w > 0.0 else 0.0
        margin_db = (
            emitter.processing_gain_db
            + emitter.coding_gain_db
            - 10.0 * math.log10(max(js, 1e-30))
        )
        return {
            "available": margin_db > 0.0,
            "js": js,
            "margin_db": margin_db,
            "ps_rx_w": ps_rx_w,
            "pj_rx_w": pj_rx_w,
        }

    def _observation(self) -> np.ndarray:
        x, y = float(self._pos[0]), float(self._pos[1])
        # 观测里的距离用"相对初始距离"表示：1.0 = 出发位置，策略不需要知道米数
        initial = max(1.0, float(np.linalg.norm(self._initial_pos)))
        return np.array(
            [
                x / self.bounds_m,
                y / self.bounds_m,
                self._power,
                self._step_index / max(1, self.max_steps),
                float(np.linalg.norm(self._pos)) / initial,
                1.0 if any(self._warning) else 0.0,
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict:
        return dict(self._last)

    def _cep_threshold(self) -> float:
        for objective in self.scenario.objectives:
            if objective.type == "localize" and objective.cep_m is not None:
                return float(objective.cep_m)
        return 500.0

    def _localize_target(self):
        # 目标 = 被雷达探测的那个平台（detect objective）；没有就用干扰机自己
        for objective in self.scenario.objectives:
            if objective.type == "detect":
                return self.scenario.platform(objective.target)
        return self.jammer_platform
