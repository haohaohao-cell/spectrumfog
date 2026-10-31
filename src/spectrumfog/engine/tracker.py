"""航迹层：预测 → 门限 → 关联 → 卡尔曼更新 → 起批 / 确认 / 滑行 / 删除。

这一层解决的是"检测 ≠ 航迹"：单帧的定位点只是**量测**，蓝方真正关心的是
"有没有一条稳定的航迹、它什么时候丢的"。没有这一层，诱饵只能制造"更亮的点"，
而不能制造"假航迹"；AAR 也只能说"这一帧解出来了"，不能说"你被跟踪了 200 秒"。

模型很简单，但每一条都在测试里：

* 状态 ``[x, y, vx, vy]``，匀速（CV）模型，离散白噪声加速度驱动；
* 关联用马氏距离门限（卡方 99%，2 自由度 = 13.8），最近邻贪心；
* 起批：累计 3 次命中转"确认"；未确认前最多漏 2 帧（第 3 帧删）；
* 确认后允许滑行 5 帧（只预测不更新），再漏就判"航迹丢失"。

不做：多模型（IMM）、JPDA / MHT、航迹合并。这些是 v2 的事，
现在这一层的目标是**让"丢失"和"假航迹"成为可观测的事实**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["Track", "Tracker", "CHI2_2D_99", "smoothing_noise"]

#: 卡方分布 99% 分位，2 自由度（位置门限）
CHI2_2D_99 = 13.816

#: 默认的过程噪声：加速度标准差 [m/s²]。虚构值，比任何真实目标都宽，
#: 用来保证"机动"不会因为模型太紧而整条航迹丢掉。
DEFAULT_ACCEL_SIGMA_MPS2 = 20.0


def smoothing_noise(dt: float, accel_sigma_mps2: float) -> np.ndarray:
    """离散白噪声加速度模型的过程噪声矩阵 Q（4×4）。"""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    sigma2 = float(accel_sigma_mps2) ** 2
    return sigma2 * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ]
    )


@dataclass
class Track:
    track_id: int
    x: np.ndarray  # [x, y, vx, vy]
    p: np.ndarray  # 4×4 协方差
    status: str = "tentative"  # tentative | confirmed | lost
    hits: int = 0
    misses: int = 0
    age_ticks: int = 0

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def event(self, t_s: float, dt: float) -> dict[str, Any]:
        return {
            "kind": "track",
            "t": float(t_s),
            "track_id": int(self.track_id),
            "status": self.status,
            "pos": [float(self.x[0]), float(self.x[1])],
            "vel": [float(self.x[2]), float(self.x[3])],
            "hits": int(self.hits),
            "misses": int(self.misses),
            "age_s": float(self.age_ticks) * float(dt),
            "sigma_m": float(np.sqrt(max(self.p[0, 0], self.p[1, 1]))),
        }


@dataclass
class Tracker:
    """单目标的航迹管理器（多目标靠"一条航迹一个实例"扩展）。"""

    dt: float
    confirm_hits: int = 3
    #: 未确认航迹允许漏的帧数（超过即删）
    tentative_max_miss: int = 2
    #: 已确认航迹允许滑行的帧数（超过即判丢失）
    max_coast: int = 5
    gate_chi2: float = CHI2_2D_99
    #: 关联半径（宽）：这个半径内的点算"同一个目标"，不会另起航迹。
    #: 定得宽是因为单帧定位的尾巴很重（01_duel 里 90 分位真实误差 4 km）。
    assoc_radius_m: float = 12000.0
    #: 更新半径（紧）：关联上了但超出这个半径的点不参与更新，只当漏检处理——
    #: 这样离群点既不污染状态，也不会另起一条航迹。
    update_radius_m: float = 3000.0
    #: 更新门限的马氏距离阈值（卡方 95%，2 自由度）
    update_gate_chi2: float = 5.991
    #: 量测噪声膨胀系数（标准差）：自报协方差偏乐观，按这个系数放大后再进门限。
    measurement_sigma_inflation: float = 2.0
    accel_sigma_mps2: float = DEFAULT_ACCEL_SIGMA_MPS2
    initial_velocity_sigma_mps: float = 300.0
    max_tracks: int = 8

    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _lost_ids: list[int] = field(default_factory=list)

    # --- 运动学 ---------------------------------------------------------

    def _transition(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def predict(self) -> None:
        f = self._transition()
        q = smoothing_noise(self.dt, self.accel_sigma_mps2)
        for track in self.tracks:
            track.x = f @ track.x
            track.p = f @ track.p @ f.T + q
            track.age_ticks += 1

    # --- 量测 -----------------------------------------------------------

    def start(self, position: Sequence[float], covariance: np.ndarray) -> Track:
        x = np.array([position[0], position[1], 0.0, 0.0], dtype=float)
        p = np.zeros((4, 4))
        p[0, 0] = max(float(covariance[0, 0]), 1.0)
        p[1, 1] = max(float(covariance[1, 1]), 1.0)
        p[0, 1] = p[1, 0] = float(covariance[0, 1])
        v_var = self.initial_velocity_sigma_mps**2
        p[2, 2] = p[3, 3] = v_var
        track = Track(track_id=self._next_id, x=x, p=p, hits=1)
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _accept_update(
        self, track: Track, position: Sequence[float], covariance: np.ndarray
    ) -> bool:
        """离群点判据：米制半径内，或马氏距离过门限。"""
        metric = float(np.hypot(position[0] - track.x[0], position[1] - track.x[1]))
        if metric <= self.update_radius_m:
            return True
        return self._mahalanobis(track, position, covariance) <= self.update_gate_chi2

    def update(self, track: Track, position: Sequence[float], covariance: np.ndarray) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = np.asarray(covariance, dtype=float) * (self.measurement_sigma_inflation**2)
        z = np.array([position[0], position[1]], dtype=float)

        innovation = z - h @ track.x
        s = h @ track.p @ h.T + r
        s_inv = np.linalg.inv(s)
        gain = track.p @ h.T @ s_inv
        track.x = track.x + gain @ innovation
        identity = np.eye(4)
        track.p = (identity - gain @ h) @ track.p
        track.hits += 1
        track.misses = 0
        if track.status == "tentative" and track.hits >= self.confirm_hits:
            track.status = "confirmed"

    def _mahalanobis(self, track: Track, position: Sequence[float], covariance: np.ndarray) -> float:
        innovation = np.array([position[0] - track.x[0], position[1] - track.x[1]])
        s = track.p[:2, :2] + np.asarray(covariance, dtype=float)
        try:
            return float(innovation @ np.linalg.solve(s, innovation))
        except np.linalg.LinAlgError:  # pragma: no cover - 奇异协方差
            return float("inf")

    # --- 一帧 -----------------------------------------------------------

    def step(
        self, fixes: Iterable[tuple[Sequence[float], np.ndarray]], t_s: float
    ) -> list[dict[str, Any]]:
        """推进一帧：预测 → 关联 → 更新/滑行 → 起批/删除，返回本帧的 track 事件。"""
        fixes = [(np.asarray(pos, dtype=float), np.asarray(cov, dtype=float)) for pos, cov in fixes]

        # 1) 预测（先验）
        self.predict()

        # 2) 关联：贪心最近邻，马氏距离过门限才算
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            if track.status == "lost":
                continue
            for fix_index, (position, covariance) in enumerate(fixes):
                distance = self._mahalanobis(track, position, covariance)
                metric = float(
                    np.hypot(position[0] - track.x[0], position[1] - track.x[1])
                )
                if distance <= self.gate_chi2 or metric <= self.assoc_radius_m:
                    pairs.append((min(distance, metric), track_index, fix_index))
        pairs.sort(key=lambda item: item[0])

        assigned: dict[int, int] = {}
        used_fixes: set[int] = set()
        for _, track_index, fix_index in pairs:
            if track_index in assigned or fix_index in used_fixes:
                continue
            assigned[track_index] = fix_index
            used_fixes.add(fix_index)

        # 3) 更新 / 滑行 / 删除
        lost_now: list[Track] = []
        for track_index, track in enumerate(self.tracks):
            if track.status == "lost":
                continue
            fix_index = assigned.get(track_index)
            if fix_index is not None:
                position, covariance = fixes[fix_index]
                if self._accept_update(track, position, covariance):
                    self.update(track, position, covariance)
                    continue
                # 关联上了但离得太远：当作漏检，不拿它更新状态
            track.misses += 1
            limit = self.tentative_max_miss if track.status == "tentative" else self.max_coast
            if track.misses > limit:
                track.status = "lost"
                self._lost_ids.append(track.track_id)
                lost_now.append(track)

        # 4) 没被任何航迹接住的定位点：起一条新航迹
        for fix_index, (position, covariance) in enumerate(fixes):
            if fix_index in used_fixes:
                continue
            if sum(1 for track in self.tracks if track.status != "lost") >= self.max_tracks:
                break
            self.start(position, covariance)

        # 5) 事件：活着的航迹每帧一条；刚丢失的航迹补一条 status="lost"
        events = [
            track.event(t_s, self.dt)
            for track in self.tracks
            if track.status != "lost"
        ]
        events.extend(track.event(t_s, self.dt) for track in lost_now)
        return events

    # --- 汇总 -----------------------------------------------------------

    def confirmed(self) -> list[Track]:
        return [track for track in self.tracks if track.status == "confirmed"]

    def lost_ids(self) -> list[int]:
        return list(self._lost_ids)
