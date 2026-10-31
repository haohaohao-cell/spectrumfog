"""IMM + JPDA：让跟踪器能应对机动和杂波。

`engine/tracker.py` 是单模型 CV + 最近邻关联——够用，但有两个已知短板：

1. **机动**：一个过程噪声定死了"目标能拐多急"。定小了拐弯就丢，定大了直线时抖。
   IMM 的解法是同时跑两个模型（低噪声 = 平稳、高噪声 = 机动），按每个模型的
   **似然**动态混合，输出加权状态。
2. **杂波**：最近邻只挑一个最像的量测，剩下的全丢。虚警率高时（04_fog 是 5e-3）
   容易挑错。JPDA 的解法是把门内的多条量测按似然**加权**，算一个"合成量测"再更新，
   同时把"这些都不是目标"的概率也算进去。

两者可以叠加：IMM 管状态，JPDA 管关联。

数学（IMM 一个周期）：

    混合：c̄_j = Σ_i p_ij μ_i,  w_ij = p_ij μ_i / c̄_j
          x⁰_j = Σ_i w_ij x_i,  P⁰_j = Σ_i w_ij (P_i + (x_i−x⁰_j)(x_i−x⁰_j)ᵀ)
    滤波：每个模型各自 predict（F, Q(σ_j)）→ update（JPDA 合成量测）
    混合概率：μ_j ∝ Λ_j c̄_j（Λ_j 是该模型的似然）
    输出：x = Σ_j μ_j x_j,  P = Σ_j μ_j (P_j + (x_j−x)(x_j−x)ᵀ)

不做：模型集合自适应、变结构 IMM、MHT。这些是 v3 的事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .tracker import CHI2_2D_99, Track, smoothing_noise

__all__ = ["IMMTrack", "IMMTracker", "MODE_NAMES"]

#: 两个模型的含义：平稳 / 机动（过程噪声不同）
MODE_NAMES = ("steady", "maneuver")


def _gaussian_likelihood(innovation: np.ndarray, covariance: np.ndarray) -> float:
    """二维高斯似然（只用相对大小，常数项不影响归一化后的权重）。"""
    try:
        det = float(np.linalg.det(covariance))
        if det <= 0.0:
            return 0.0
        exponent = -0.5 * float(innovation @ np.linalg.solve(covariance, innovation))
        if exponent < -700.0:
            return 0.0
        return float(np.exp(exponent) / (2.0 * np.pi * np.sqrt(det)))
    except np.linalg.LinAlgError:  # pragma: no cover - 奇异协方差
        return 0.0


@dataclass
class IMMTrack:
    """一条 IMM 航迹：两个模型各有一套状态与协方差。"""

    track_id: int
    states: np.ndarray  # (n_models, 4)
    covariances: np.ndarray  # (n_models, 4, 4)
    mode_probabilities: np.ndarray  # (n_models,)
    status: str = "tentative"
    hits: int = 1
    misses: int = 0
    age_ticks: int = 0

    @property
    def state(self) -> np.ndarray:
        """按模型概率加权的状态。"""
        return self.mode_probabilities @ self.states

    @property
    def covariance(self) -> np.ndarray:
        mean = self.state
        spread = np.zeros((4, 4))
        for index, probability in enumerate(self.mode_probabilities):
            delta = (self.states[index] - mean).reshape(4, 1)
            spread += probability * (self.covariances[index] + delta @ delta.T)
        return spread

    @property
    def position(self) -> tuple[float, float]:
        state = self.state
        return float(state[0]), float(state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        state = self.state
        return float(state[2]), float(state[3])

    def event(self, t_s: float, dt: float) -> dict[str, Any]:
        state = self.state
        covariance = self.covariance
        return {
            "kind": "track",
            "t": float(t_s),
            "track_id": int(self.track_id),
            "status": self.status,
            "pos": [float(state[0]), float(state[1])],
            "vel": [float(state[2]), float(state[3])],
            "hits": int(self.hits),
            "misses": int(self.misses),
            "age_s": float(self.age_ticks) * float(dt),
            "sigma_m": float(np.sqrt(max(covariance[0, 0], covariance[1, 1]))),
            "filter": "imm",
            "mode": int(np.argmax(self.mode_probabilities)),
            "mode_prob": float(np.max(self.mode_probabilities)),
        }


@dataclass
class IMMTracker:
    """IMM 滤波 + JPDA 关联的航迹管理器。"""

    dt: float
    #: 两个模型的过程噪声（加速度标准差），[平稳, 机动]
    accel_sigmas: tuple[float, float] = (5.0, 60.0)
    #: 模型转移矩阵：p[i, j] = P(下一个是 j | 当前是 i)
    transition: np.ndarray = field(
        default_factory=lambda: np.array([[0.97, 0.03], [0.10, 0.90]])
    )
    confirm_hits: int = 3
    tentative_max_miss: int = 2
    max_coast: int = 5
    gate_chi2: float = CHI2_2D_99
    assoc_radius_m: float = 12000.0
    update_radius_m: float = 3000.0
    measurement_sigma_inflation: float = 2.0
    #: 杂波项（相对最强量测）：越大越倾向"门内的点都不是目标"。
    #: 0.05 意味着一条好量测有 95% 的关联概率；0.5 会让每次更新都保留一半旧协方差，
    #: PDA 的 spread 项会把 P 越推越大（这个坑踩过）。
    clutter_weight: float = 0.05
    max_tracks: int = 8

    tracks: list[IMMTrack] = field(default_factory=list)
    _next_id: int = 1
    _lost_ids: list[int] = field(default_factory=list)

    # --- 运动学 ---------------------------------------------------------

    def _transition_matrix(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _process_noises(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(smoothing_noise(self.dt, sigma) for sigma in self.accel_sigmas)  # type: ignore[return-value]

    def _mix_track(self, track: IMMTrack) -> np.ndarray:
        probabilities = track.mode_probabilities
        c_bar = self.transition.T @ probabilities  # c̄_j = Σ_i p_ij μ_i
        c_bar = np.maximum(c_bar, 1e-12)
        weights = self.transition * probabilities[:, None] / c_bar[None, :]
        mixed_states = weights.T @ track.states
        mixed_covariances = np.zeros_like(track.covariances)
        for target in range(len(probabilities)):
            accumulator = np.zeros((4, 4))
            for source in range(len(probabilities)):
                delta = (track.states[source] - mixed_states[target]).reshape(4, 1)
                accumulator += weights[source, target] * (
                    track.covariances[source] + delta @ delta.T
                )
            mixed_covariances[target] = accumulator
        track.states = mixed_states
        track.covariances = mixed_covariances
        return c_bar

    # --- 一帧 -----------------------------------------------------------

    def predict(self) -> None:
        f = self._transition_matrix()
        process_noises = self._process_noises()
        for track in self.tracks:
            self._mix_track(track)
            for index in range(track.states.shape[0]):
                track.states[index] = f @ track.states[index]
                track.covariances[index] = (
                    f @ track.covariances[index] @ f.T + process_noises[index]
                )
            track.age_ticks += 1

    def start(self, position: Sequence[float], covariance: np.ndarray) -> IMMTrack:
        n_models = len(self.accel_sigmas)
        states = np.zeros((n_models, 4))
        states[:, 0] = position[0]
        states[:, 1] = position[1]
        covariances = np.zeros((n_models, 4, 4))
        for index in range(n_models):
            covariances[index, 0, 0] = max(float(covariance[0, 0]), 1.0)
            covariances[index, 1, 1] = max(float(covariance[1, 1]), 1.0)
            covariances[index, 0, 1] = covariances[index, 1, 0] = float(covariance[0, 1])
            covariances[index, 2, 2] = covariances[index, 3, 3] = 300.0**2
        track = IMMTrack(
            track_id=self._next_id,
            states=states,
            covariances=covariances,
            mode_probabilities=np.full(n_models, 1.0 / n_models),
            hits=1,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _measurement_covariance(self, covariance: np.ndarray) -> np.ndarray:
        return np.asarray(covariance, dtype=float) * (self.measurement_sigma_inflation**2)

    def _gate(self, track: IMMTrack, position: np.ndarray, covariance: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """返回门内的 (似然, 新息, S) 列表；S 取加权后的协方差。"""
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = self._measurement_covariance(covariance)
        z = np.asarray(position, dtype=float)
        state = track.state
        innovation = z - h @ state
        s = h @ track.covariance @ h.T + r
        metric = float(np.hypot(innovation[0], innovation[1]))
        mahalanobis = float(innovation @ np.linalg.solve(s, innovation))
        if metric > self.assoc_radius_m and mahalanobis > self.gate_chi2:
            return []
        return [(_gaussian_likelihood(innovation, s), innovation, s)]

    def _pda_update(
        self,
        track: IMMTrack,
        candidates: list[tuple[float, np.ndarray, np.ndarray]],
        measurement_covariance: np.ndarray,
    ) -> None:
        """JPDA 简化版：门内多条量测按似然加权成"合成量测"；模型概率按各自似然更新。"""
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = self._measurement_covariance(measurement_covariance)

        # 1) 数据关联概率（相对最强量测归一化）
        likelihoods = np.array(
            [max(likelihood, 1e-30) for likelihood, _, _ in candidates]
        )
        scaled = likelihoods / float(likelihoods.max())
        denominator = scaled.sum() + self.clutter_weight
        betas = scaled / denominator
        beta_0 = self.clutter_weight / denominator

        # 2) 合成新息与关联不确定度
        innovations = np.stack([innovation for _, innovation, _ in candidates])
        combined = betas @ innovations
        spread = sum(
            beta * np.outer(innovation - combined, innovation - combined)
            for beta, innovation in zip(betas, innovations)
        )
        combined_measurement = h @ track.state + combined  # 绝对量测（合成后）

        # 3) 每个模型各自更新（这一步才让模型概率有意义）
        mode_likelihoods = []
        for index in range(track.states.shape[0]):
            innovation = combined_measurement - h @ track.states[index]
            s = h @ track.covariances[index] @ h.T + r
            gain = track.covariances[index] @ h.T @ np.linalg.inv(s)
            identity = np.eye(4)
            updated = (identity - gain @ h) @ track.covariances[index]
            track.states[index] = track.states[index] + gain @ innovation
            track.covariances[index] = (
                beta_0 * track.covariances[index]
                + (1.0 - beta_0) * updated
                + gain @ spread @ gain.T
            )
            mode_likelihoods.append(_gaussian_likelihood(innovation, s))

        # 4) μ_j ∝ Λ_j · c̄_j（c̄ 已在 predict 的混合步里用掉，这里按先验比例近似）
        mode_likelihoods = np.maximum(np.asarray(mode_likelihoods), 1e-30)
        track.mode_probabilities = track.mode_probabilities * mode_likelihoods
        total = track.mode_probabilities.sum()
        if total > 0:
            track.mode_probabilities = track.mode_probabilities / total
        else:
            track.mode_probabilities = np.full_like(
                track.mode_probabilities, 1.0 / len(mode_likelihoods)
            )
        # 给每个模型留一个下限：概率一旦变成 0，之后无论多像也回不来了
        # （IMM 的标准做法；否则目标一机动，机动模型已经被"饿死"）
        track.mode_probabilities = np.maximum(track.mode_probabilities, 1e-3)
        track.mode_probabilities /= track.mode_probabilities.sum()

        track.hits += 1
        track.misses = 0
        if track.status == "tentative" and track.hits >= self.confirm_hits:
            track.status = "confirmed"

    def step(
        self, fixes: Iterable[tuple[Sequence[float], np.ndarray]], t_s: float
    ) -> list[dict[str, Any]]:
        fixes = [
            (np.asarray(position, dtype=float), np.asarray(covariance, dtype=float))
            for position, covariance in fixes
        ]
        self.predict()

        # 关联：每条航迹收集门内所有量测（JPDA 不急着挑一个）
        gated: dict[int, list[tuple[float, np.ndarray, np.ndarray]]] = {}
        gated_covariance: dict[int, np.ndarray] = {}
        used_by: dict[int, list[int]] = {}
        for track_index, track in enumerate(self.tracks):
            if track.status == "lost":
                continue
            for fix_index, (position, covariance) in enumerate(fixes):
                candidates = self._gate(track, position, covariance)
                if candidates:
                    gated.setdefault(track_index, []).extend(candidates)
                    gated_covariance[track_index] = covariance
                    used_by.setdefault(fix_index, []).append(track_index)

        lost_now: list[IMMTrack] = []
        for track_index, track in enumerate(self.tracks):
            if track.status == "lost":
                continue
            candidates = gated.get(track_index, [])
            if candidates:
                self._pda_update(track, candidates, gated_covariance[track_index])
                continue
            track.misses += 1
            limit = (
                self.tentative_max_miss
                if track.status == "tentative"
                else self.max_coast
            )
            if track.misses > limit:
                track.status = "lost"
                self._lost_ids.append(track.track_id)
                lost_now.append(track)

        # 起批：没被任何航迹门限接住的量测
        for fix_index, (position, covariance) in enumerate(fixes):
            if fix_index in used_by:
                continue
            if sum(1 for track in self.tracks if track.status != "lost") >= self.max_tracks:
                break
            self.start(position, covariance)

        events = [
            track.event(t_s, self.dt)
            for track in self.tracks
            if track.status != "lost"
        ]
        events.extend(track.event(t_s, self.dt) for track in lost_now)
        return events
