"""双人环境：红方干扰机 vs 蓝方雷达（EMCON）。

前面所有想定里蓝方都是环境的一部分——雷达一直开着。这一层把蓝方也变成决策者：

    红方：15 个动作（5 方向 × 3 功率档）
    蓝方：3 个动作（关机 / 半功率 / 满功率）

**只在蓝方开机时结算**，这是整个博弈的关键：

* 蓝方开机 → 雷达有检测概率；同时被红方 ESM 看见，可能被定位；
* 蓝方关机 → 什么都看不见，但也没人能定位它；
* 红方开机 → 压得住目标，同时被蓝方 ESM 看见。

所以两边都在做同一件事：**挑时间**。

奖励（每 tick）：

| 事件 | 红方 | 蓝方 |
|---|---|---|
| 蓝方开机且 Pd < 0.5（被压住） | +1 | 0 |
| 蓝方开机且 Pd ≥ 0.5（看得见） | 0 | +1 |
| 蓝方关机 | 0 | 0 |
| 自己被有效定位（CEP ≤ 门槛） | −1 | −1 |
| 开机的功率成本 | −0.02·功率档 | −0.02·功率档 |

两边的观测都是 7 维、都是部分可观测的：红方不知道蓝方什么时候会开机
（只能事后看到"上一帧它有没有辐射"），蓝方也不知道红方在哪。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..engine import detect, fusion
from ..engine.tracker import Tracker
from ..metrics import cep_from_covariance
from ..model.scenario import Scenario
from ..physics import radar

__all__ = [
    "DuelEnv",
    "DuelResult",
    "BLUE_POWERS",
    "play_duel",
    "RED_BASELINES",
    "BLUE_BASELINES",
]

#: 蓝方的三个动作档位（相对标称功率）
BLUE_POWERS = (0.0, 0.5, 1.0)

#: 红方方向：0=停, 1=+x, 2=-x, 3=+y, 4=-y
_RED_DIRECTIONS = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
_RED_POWERS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class DuelResult:
    blue_policy: str
    red_policy: str
    episodes: int
    red_reward: float
    blue_reward: float
    red_exposed_fraction: float
    blue_exposed_fraction: float
    blue_on_fraction: float
    pd_mean: float

    def as_row(self) -> dict:
        return {
            "blue": self.blue_policy,
            "red": self.red_policy,
            "red_reward": round(self.red_reward, 1),
            "blue_reward": round(self.blue_reward, 1),
            "blue_exposed": round(self.blue_exposed_fraction, 2),
            "red_exposed": round(self.red_exposed_fraction, 2),
            "blue_on": round(self.blue_on_fraction, 2),
            "pd_mean": round(self.pd_mean, 3),
        }


class DuelEnv(gym.Env):
    """单步双人环境：两个智能体每 tick 同时出动作。"""

    metadata = {"name": "SpectrumFogDuelEnv", "render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        *,
        pd_threshold: float = 0.5,
        pfa: float = 1e-6,
        move_speed_mps: float = 500.0,
        bounds_m: float | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.radar_platform, self.radar_emitter = scenario.primary_radar()
        self.jammer_platform, self.jammer_emitter = scenario.primary_jammer()
        self.params = scenario.link_params(
            self.radar_platform.id, self.jammer_platform.id
        )
        self.target_platform = self._target()
        self.threshold_m = self._cep_threshold()
        self.pd_threshold = float(pd_threshold)
        self.pfa = float(pfa)
        self.dt = float(scenario.meta.dt)
        self.max_steps = int(round(scenario.meta.duration / self.dt))
        self.move_speed_mps = float(move_speed_mps)
        self.bounds_m = float(
            bounds_m
            if bounds_m is not None
            else max(1.5 * math.hypot(*self.jammer_platform.pos), 80_000.0)
        )

        self.red_action_space = spaces.Discrete(15)
        self.blue_action_space = spaces.Discrete(len(BLUE_POWERS))
        self.red_observation_space = spaces.Box(-2.0, 2.0, shape=(7,), dtype=np.float32)
        self.blue_observation_space = spaces.Box(-2.0, 2.0, shape=(7,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._pos = np.zeros(2)
        self._red_power = 1.0
        self._blue_power = 1.0
        self._step = 0
        self._red_warned = False
        self._blue_warned = False
        self._blue_seen = False
        self._red_seen = False
        self._last: dict = {}
        self._trackers = self._new_trackers()

    # --- gym API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else 0)
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._pos = np.asarray(self.jammer_platform.position_at(0.0), dtype=float)
        self._red_power = 1.0
        self._blue_power = 1.0
        self._step = 0
        self._red_warned = False
        self._blue_warned = False
        self._blue_seen = False
        self._red_seen = False
        self._last = {}
        self._trackers = self._new_trackers()
        return self._observations(), self._info()

    def step(self, red_action: int, blue_action: int):
        red_action = int(red_action)
        blue_action = int(blue_action)
        if not 0 <= red_action < self.red_action_space.n:
            raise ValueError(f"红方动作越界：{red_action}")
        if not 0 <= blue_action < self.blue_action_space.n:
            raise ValueError(f"蓝方动作越界：{blue_action}")

        self._red_power = _RED_POWERS[red_action % 3]
        self._blue_power = BLUE_POWERS[blue_action]

        direction = _RED_DIRECTIONS[red_action // 3]
        self._pos = np.clip(
            self._pos + np.array(direction) * self.move_speed_mps * self.dt,
            -self.bounds_m,
            self.bounds_m,
        )

        t = self._step * self.dt
        self._step += 1

        radar_pos = np.asarray(self.radar_platform.position_at(t), dtype=float)
        target_pos = np.asarray(self.target_platform.position_at(t), dtype=float)
        range_target = float(np.linalg.norm(target_pos - radar_pos))
        range_jammer = float(np.linalg.norm(self._pos - radar_pos))

        pd_value, js_value = self._link(range_target, range_jammer)
        radiating = self._blue_power > 0.0

        # 结算：只有蓝方开机时才有"看见/没看见"
        red_reward = 0.0
        blue_reward = 0.0
        if radiating:
            if pd_value < self.pd_threshold:
                red_reward += 1.0
            else:
                blue_reward += 1.0

        # 定位：各自只在辐射时才会被对方看见
        red_fix = self._localize(
            side="blue", emitter_pos=(float(self._pos[0]), float(self._pos[1])),
            power=self._red_power * float(self.jammer_emitter.pj_w),
            gain=float(self.jammer_emitter.gain),
            freq=float(self.jammer_emitter.freq_hz), t=t,
        )
        blue_fix = None
        if radiating:
            blue_fix = self._localize(
                side="red", emitter_pos=(float(radar_pos[0]), float(radar_pos[1])),
                power=self._blue_power * float(self.radar_emitter.pt_w),
                gain=float(self.radar_emitter.gain),
                freq=float(self.radar_emitter.freq_hz), t=t,
            )

        # "被发现一次"不等于"被干掉"：只有确认跟踪（M-of-N）才算暴露。
        # 这一步让蓝方的 EMCON 有了甜点：开得够短，雷达看得见，红方跟不住。
        red_track_events = self._trackers["blue"].step(
            [(red_fix[1], red_fix[2])] if red_fix is not None else [], t
        )
        blue_track_events = self._trackers["red"].step(
            [(blue_fix[1], blue_fix[2])] if blue_fix is not None else [], t
        )
        red_exposed = any(event["status"] == "confirmed" for event in red_track_events)
        blue_exposed = any(event["status"] == "confirmed" for event in blue_track_events)
        red_cep = red_fix[0] if red_fix is not None else float("nan")
        blue_cep = blue_fix[0] if blue_fix is not None else float("nan")
        if red_exposed:
            red_reward -= 1.0
        if blue_exposed:
            blue_reward -= 1.0

        red_reward -= 0.02 * self._red_power
        blue_reward -= 0.02 * self._blue_power

        # 观测用的"上一帧被看见了吗"
        self._red_seen = bool(radiating)
        self._blue_seen = bool(self._red_power > 0.0)
        self._red_warned = red_exposed
        self._blue_warned = blue_exposed

        self._last = {
            "pd": pd_value,
            "js": js_value,
            "radiating": radiating,
            "red_exposed": red_exposed,
            "blue_exposed": blue_exposed,
            "red_cep_m": red_cep,
            "blue_cep_m": blue_cep,
        }

        terminated = self._step >= self.max_steps
        return (
            self._observations(),
            {"red": float(red_reward), "blue": float(blue_reward)},
            terminated,
            False,
            self._info(),
        )

    def _new_trackers(self) -> dict[str, Tracker]:
        """两侧各一个跟踪器：确认跟踪才等于暴露。"""
        return {
            "blue": Tracker(
                dt=self.dt,
                confirm_hits=3,
                max_coast=3,
                assoc_radius_m=3000.0,
                update_radius_m=3000.0,
            ),
            "red": Tracker(
                dt=self.dt,
                confirm_hits=3,
                max_coast=3,
                assoc_radius_m=3000.0,
                update_radius_m=3000.0,
            ),
        }

    # --- 物理 ------------------------------------------------------------

    def _link(self, range_target: float, range_jammer: float) -> tuple[float, float]:
        if self._blue_power <= 0.0:
            return 0.0, 0.0

        environment = self.scenario.environment
        pn_w = float(
            radar.noise_power(
                bw_hz=self.params["bw_r_hz"],
                noise_figure=environment.noise_figure,
                temperature_k=environment.temperature_k,
            )
        )
        pr_w = float(
            radar.received_power(
                pt_w=self.params["pt_w"] * self._blue_power,
                gt=self.params["gain"],
                gr=self.params["gain"],
                lam=self.params["lam"],
                sigma_m2=self.params["sigma_m2"],
                range_m=range_target,
                loss=self.params["loss"],
            )
        )
        pj_w = self.params["pj_w"] * self._red_power / float(
            self.radar_emitter.processing_gain
        )
        pj_rx_w = 0.0
        if pj_w > 0.0:
            pj_rx_w = float(
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
        pd_value = float(radar.pd_from_powers(self.pfa, pr_w, pn_w, pj_rx_w))
        return pd_value, (pj_rx_w / pr_w if pr_w > 0 else 0.0)

    def _localize(
        self,
        *,
        side: str,
        emitter_pos: tuple[float, float],
        power: float,
        gain: float,
        freq: float,
        t: float,
    ):
        """某一侧的 ESM 联合定位（≥2 条量测才解），返回 (CEP, 位置, 协方差) 或 None。"""
        if power <= 0.0:
            return None

        measurements = []
        for sensor in self.scenario.sensors:
            if sensor.kind != "esm" or sensor.side != side:
                continue
            observation = detect.observe(
                sensor_id=sensor.id,
                sensor_pos=sensor.pos,
                sensor_bw_hz=sensor.bw_hz,
                sensor_noise_figure=sensor.noise_figure,
                sensor_gain=sensor.gain,
                emitter_id=f"{side}-target",
                emitter_pos=emitter_pos,
                emitter_power_w=power,
                emitter_gain=gain,
                emitter_freq_hz=freq,
                t_s=t,
                pfa=self.pfa,
                rng=self._rng,
            )
            if not observation.detected:
                continue
            measurements.append(
                fusion.Measure(
                    sensor_id=observation.sensor_id,
                    sensor_pos=observation.sensor_pos,
                    aoa_rad=observation.aoa_rad,
                    toa_s=observation.toa_s,
                    sigma_aoa_rad=observation.sigma_aoa_rad
                    or fusion.DEFAULT_SIGMA_AOA_RAD,
                    sigma_toa_s=observation.sigma_toa_s or fusion.DEFAULT_SIGMA_TOA_S,
                )
            )
        fix = fusion.solve_fix(measurements)
        if fix is None:
            return None
        cep = float(cep_from_covariance(fix.covariance, samples=400)[0])
        return cep, np.asarray(fix.pos, dtype=float), np.asarray(fix.covariance)

    # --- 观测 ------------------------------------------------------------

    def _observations(self) -> dict[str, np.ndarray]:
        x, y = float(self._pos[0]), float(self._pos[1])
        t_frac = self._step / max(1, self.max_steps)
        range_km = math.hypot(x, y) / 100_000.0

        red = np.array(
            [
                x / self.bounds_m,
                y / self.bounds_m,
                self._red_power,
                t_frac,
                range_km,
                1.0 if self._red_warned else 0.0,
                1.0 if self._blue_seen else 0.0,
            ],
            dtype=np.float32,
        )
        blue = np.array(
            [
                self._blue_power,
                t_frac,
                math.hypot(
                    self.target_platform.position_at(self._step * self.dt)[0]
                    - self.radar_platform.position_at(self._step * self.dt)[0],
                    self.target_platform.position_at(self._step * self.dt)[1]
                    - self.radar_platform.position_at(self._step * self.dt)[1],
                )
                / 100_000.0,
                float(self._last.get("pd", 0.0)),
                1.0 if float(self._last.get("js", 0.0)) >= 1.0 else 0.0,
                1.0 if self._blue_warned else 0.0,
                1.0 if self._red_seen else 0.0,
            ],
            dtype=np.float32,
        )
        return {"red": red, "blue": blue}

    def _info(self) -> dict:
        return dict(self._last)

    # --- 想定辅助 --------------------------------------------------------

    def _target(self):
        for objective in self.scenario.objectives:
            if objective.type == "detect":
                return self.scenario.platform(objective.target)
        return self.scenario.platform(self.scenario.primary_jammer()[0].id)

    def _cep_threshold(self) -> float:
        for objective in self.scenario.objectives:
            if objective.type == "localize" and objective.cep_m is not None:
                return float(objective.cep_m)
        return 500.0


# --- baseline 策略 ------------------------------------------------------------


class _RedPolicy:
    name = "red-?"
    def reset(self, *, seed: int | None = None) -> None: ...
    def act(self, observation: np.ndarray) -> int: ...


class RedStayJam(_RedPolicy):
    """原地满功率：压得住，但一直在暴露。"""

    name = "red-stay-jam"

    def reset(self, *, seed=None) -> None:
        return None

    def act(self, observation) -> int:
        return 0 * 3 + 2


class RedSilent(_RedPolicy):
    """全程关机：不暴露，也不压制。"""

    name = "red-silent"

    def reset(self, *, seed=None) -> None:
        return None

    def act(self, observation) -> int:
        return 0 * 3 + 0


class RedRaider(_RedPolicy):
    """被看见就关机，安全了就开机。"""

    name = "red-raider"

    def reset(self, *, seed=None) -> None:
        return None

    def act(self, observation) -> int:
        warned = float(observation[5]) > 0.5
        return 0 * 3 + (0 if warned else 2)


class _BluePolicy:
    name = "blue-?"
    def reset(self, *, seed: int | None = None) -> None: ...
    def act(self, observation: np.ndarray) -> int: ...


class BlueAlwaysOn(_BluePolicy):
    name = "blue-always-on"

    def reset(self, *, seed=None) -> None:
        return None

    def act(self, observation) -> int:
        return 2


class BluePulse(_BluePolicy):
    """每 5 帧开一帧：用时间换生存。"""

    name = "blue-pulse"

    def __init__(self) -> None:
        self._tick = 0

    def reset(self, *, seed=None) -> None:
        self._tick = 0

    def act(self, observation) -> int:
        self._tick += 1
        return 2 if self._tick % 5 == 0 else 0


class BlueReactive(_BluePolicy):
    """被定位就躲，躲够了再出来看一眼。"""

    name = "blue-reactive"

    def __init__(self) -> None:
        self._cooldown = 0

    def reset(self, *, seed=None) -> None:
        self._cooldown = 0

    def act(self, observation) -> int:
        if float(observation[5]) > 0.5:
            self._cooldown = 5
        if self._cooldown > 0:
            self._cooldown -= 1
            return 0
        return 2


RED_BASELINES: tuple[type, ...] = (RedStayJam, RedSilent, RedRaider)
BLUE_BASELINES: tuple[type, ...] = (BlueAlwaysOn, BluePulse, BlueReactive)


def _make(policy_cls, name: str):
    for candidate in (*RED_BASELINES, *BLUE_BASELINES):
        if candidate.name == name:
            return candidate()
    available = ", ".join(candidate.name for candidate in (*RED_BASELINES, *BLUE_BASELINES))
    raise ValueError(f"未知策略 {name!r}；可用：{available}")


def play_duel(
    scenario: Scenario,
    *,
    red_policy: str = "red-stay-jam",
    blue_policy: str = "blue-always-on",
    episodes: int = 3,
    seed0: int = 0,
) -> DuelResult:
    """跑若干局，返回双方平均奖励。"""
    if episodes < 1:
        raise ValueError("episodes 至少 1")
    if scenario.sensors is None:
        raise ValueError("双人想定需要 ESM")

    red_totals: list[float] = []
    blue_totals: list[float] = []
    red_exposed_ticks = 0
    blue_exposed_ticks = 0
    on_ticks = 0
    pd_values: list[float] = []
    steps_total = 0

    for episode in range(episodes):
        env = DuelEnv(scenario)
        seed = seed0 + episode
        observations, _ = env.reset(seed=seed)
        red_obs, blue_obs = observations["red"], observations["blue"]
        red = _make(None, red_policy)
        blue = _make(None, blue_policy)
        red.reset(seed=seed)
        blue.reset(seed=seed)
        red_total = 0.0
        blue_total = 0.0
        while True:
            red_action = red.act(red_obs)
            blue_action = blue.act(blue_obs)
            observations, rewards, terminated, truncated, info = env.step(
                red_action, blue_action
            )
            red_obs, blue_obs = observations["red"], observations["blue"]
            red_total += rewards["red"]
            blue_total += rewards["blue"]
            steps_total += 1
            pd_values.append(info["pd"])
            if info["radiating"]:
                on_ticks += 1
            if info["red_exposed"]:
                red_exposed_ticks += 1
            if info["blue_exposed"]:
                blue_exposed_ticks += 1
            if terminated or truncated:
                break
        red_totals.append(red_total)
        blue_totals.append(blue_total)

    return DuelResult(
        blue_policy=blue_policy,
        red_policy=red_policy,
        episodes=episodes,
        red_reward=sum(red_totals) / len(red_totals),
        blue_reward=sum(blue_totals) / len(blue_totals),
        red_exposed_fraction=red_exposed_ticks / max(1, steps_total),
        blue_exposed_fraction=blue_exposed_ticks / max(1, steps_total),
        blue_on_fraction=on_ticks / max(1, steps_total),
        pd_mean=sum(pd_values) / max(1, len(pd_values)),
    )
