"""雷达方程、干信比（J/S）、烧穿距离与 Swerling I 检测概率。

内部一律 SI：功率 W、长度 m、频率 Hz、时间 s、角度 rad。
天线增益与损耗以**线性**值传入（dB → 线性用 ``spectrumfog.units.db_to_linear``）。

符号表
------
======  ==========================================  ======
符号    含义                                        单位
======  ==========================================  ======
Pt      雷达发射功率                                  W
Pj      干扰机发射功率                                W
Gt      雷达发射天线增益（线性）                       -
Gr      雷达接收天线增益（线性）                       -
Gj      干扰机天线在雷达方向上的增益（线性）            -
lam     波长                                        m
sigma   目标雷达截面积 RCS                            m^2
R       雷达到目标距离                                m
Rj      雷达到干扰机距离                              m
Br      雷达接收机带宽                                Hz
Bj      干扰机噪声带宽                                Hz
L       系统损耗（线性，>= 1）                         -
======  ==========================================  ======

烧穿距离里的增益指数（规格歧义，已与需求方确认）
------------------------------------------------
PROJECT.md §6.1 写的是::

    R_bt = sqrt(Pt * G**2 * sigma * Bj / (4*pi * Pj * Gj * Br * L))

但按同一节 Pr 与 Pj_rx 的定义推导，Gr 会被约掉::

    JS = Pj_rx / Pr
       = [Pj*Gj*Gr*lam^2 / ((4*pi)^2 * R^2) * (Br/Bj)]
         / [Pt*Gt*Gr*lam^2*sigma / ((4*pi)^3 * R^4 * L)]
       = 4*pi*Pj*Gj*R^2*L*Br / (Pt*Gt*sigma*Bj)      # Gr 相消

令 JS = 1 解出::

    R_bt = sqrt(Pt*Gt*sigma*Bj / (4*pi*Pj*Gj*Br*L))   # 只有一个 G

即 G² 是笔误，物理上自屏蔽干扰的烧穿距离只含**发射**增益 Gt，
接收增益在干扰支路和回波支路里同时出现、互相抵消。
因此本模块的 ``burn_through_range`` 只接受 ``gain``(=Gt)，不带 ``lam``
（波长同样在两条支路里相消，见 ``test_self_screening_js_is_wavelength_agnostic``）。
"""

from __future__ import annotations

import math

import numpy as np

from ..units import K_BOLTZMANN, SPEED_OF_LIGHT, T0_K

__all__ = [
    "wavelength",
    "noise_power",
    "received_power",
    "jammer_received_power",
    "js_ratio",
    "js_ratio_standoff",
    "js_ratio_self_screen",
    "burn_through_range",
    "burn_through_range_standoff",
    "range_at_js",
    "snr_thermal",
    "sinr",
    "decoy_transmit_power",
    "decoy_echo_power",
    "decoy_equivalent_rcs",
    "pd_swerling1",
    "pd_from_powers",
    "detection_range",
]

_FOUR_PI = 4.0 * np.pi
_FOUR_PI_SQ = _FOUR_PI**2
_FOUR_PI_CUBE = _FOUR_PI**3


# --- 输入校验 ----------------------------------------------------------------


def _require_positive(**values: object) -> None:
    """拒绝非正值。内部是 SI，没有"负数功率"或"零米距离"这种合法输入。"""
    for name, value in values.items():
        if np.any(np.asarray(value, dtype=float) <= 0.0):
            raise ValueError(f"{name} 必须为正数（SI 单位），收到 {value!r}")


def _require_probability(name: str, value: object) -> None:
    arr = np.asarray(value, dtype=float)
    if np.any((arr <= 0.0) | (arr >= 1.0)):
        raise ValueError(f"{name} 必须落在 (0, 1) 开区间，收到 {value!r}")


# --- 波长与噪声 --------------------------------------------------------------


def wavelength(freq_hz):
    """lam = c / f。"""
    _require_positive(freq_hz=freq_hz)
    return SPEED_OF_LIGHT / np.asarray(freq_hz, dtype=float)


def noise_power(bw_hz, noise_figure=1.0, temperature_k=T0_K):
    """噪声底 Pn = k * T0 * B * F，单位 W。

    noise_figure 是**线性**噪声系数（3 dB → 1.995...），不是 dB 值。
    """
    _require_positive(bw_hz=bw_hz, noise_figure=noise_figure, temperature_k=temperature_k)
    return (
        K_BOLTZMANN
        * np.asarray(temperature_k, dtype=float)
        * np.asarray(bw_hz, dtype=float)
        * np.asarray(noise_figure, dtype=float)
    )


# --- 雷达方程与干扰功率 ------------------------------------------------------


def received_power(
    pt_w,
    gt,
    gr,
    lam,
    sigma_m2,
    range_m,
    loss=1.0,
):
    """雷达回波功率 Pr = Pt * Gt * Gr * lam^2 * sigma / ((4*pi)^3 * R^4 * L)，单位 W。

    单站雷达（PROJECT.md §6.1 的 G**2）即 gt = gr = G；
    分开传参是为了以后能做双站，同时让"Gr 相消"这件事可以被单测直接验证。
    """
    _require_positive(
        pt_w=pt_w, gt=gt, gr=gr, lam=lam, sigma_m2=sigma_m2, range_m=range_m, loss=loss
    )
    return (
        np.asarray(pt_w, dtype=float)
        * np.asarray(gt, dtype=float)
        * np.asarray(gr, dtype=float)
        * np.asarray(lam, dtype=float) ** 2
        * np.asarray(sigma_m2, dtype=float)
        / (
            _FOUR_PI_CUBE
            * np.asarray(range_m, dtype=float) ** 4
            * np.asarray(loss, dtype=float)
        )
    )


def jammer_received_power(
    pj_w,
    gj,
    gr,
    lam,
    range_j_m,
    bw_r_hz,
    bw_j_hz,
):
    """干扰进入接收机的功率 Pj_rx = Pj*Gj*Gr*lam^2/((4*pi)^2*Rj^2) * (Br/Bj)，单位 W。

    噪声压制型干扰机：只有落进接收带宽 Br 的那部分功率 (Br/Bj) 起作用。
    """
    _require_positive(
        pj_w=pj_w,
        gj=gj,
        gr=gr,
        lam=lam,
        range_j_m=range_j_m,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
    )
    return (
        np.asarray(pj_w, dtype=float)
        * np.asarray(gj, dtype=float)
        * np.asarray(gr, dtype=float)
        * np.asarray(lam, dtype=float) ** 2
        / (_FOUR_PI_SQ * np.asarray(range_j_m, dtype=float) ** 2)
        * (np.asarray(bw_r_hz, dtype=float) / np.asarray(bw_j_hz, dtype=float))
    )


def js_ratio(pr_w, pj_rx_w):
    """干信比 JS = Pj_rx / Pr（线性，非 dB）。"""
    _require_positive(pr_w=pr_w)
    if np.any(np.asarray(pj_rx_w, dtype=float) < 0.0):
        raise ValueError(f"pj_rx_w 不能为负，收到 {pj_rx_w!r}")
    return np.asarray(pj_rx_w, dtype=float) / np.asarray(pr_w, dtype=float)


def js_ratio_standoff(
    range_m,
    jammer_range_m,
    *,
    pt_w,
    gain,
    sigma_m2,
    lam,
    pj_w,
    gj,
    bw_r_hz,
    bw_j_hz,
    loss=1.0,
):
    """站距干扰（干扰机与目标分离）的干信比 JS(R, Rj)。

    干扰支路走 ``jammer_range_m``，回波支路走 ``range_m``，两者独立：

        JS = 4*pi*Pj*Gj*L*Br / (Pt*Gt*sigma*Bj) · R^4 / Rj^2

    自屏蔽（Rj = R）时退化为 ``js_ratio_self_screen``，即 JS ∝ R²。
    """
    pr = received_power(
        pt_w=pt_w,
        gt=gain,
        gr=gain,
        lam=lam,
        sigma_m2=sigma_m2,
        range_m=range_m,
        loss=loss,
    )
    pj_rx = jammer_received_power(
        pj_w=pj_w,
        gj=gj,
        gr=gain,
        lam=lam,
        range_j_m=jammer_range_m,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
    )
    return js_ratio(pr_w=pr, pj_rx_w=pj_rx)


def js_ratio_self_screen(
    pt_w,
    gain,
    sigma_m2,
    lam,
    range_m,
    pj_w,
    gj,
    bw_r_hz,
    bw_j_hz,
    loss=1.0,
):
    """自屏蔽干扰（干扰机与目标同处，Rj = R）的干信比 JS(R)。

    就是 ``js_ratio_standoff`` 在 Rj = R 时的特例；保留独立入口是因为它是
    PROJECT.md §6.1 的定义，且 ``range_at_js`` 的验收就是围着它写的。
    """
    return js_ratio_standoff(
        range_m=range_m,
        jammer_range_m=range_m,
        pt_w=pt_w,
        gain=gain,
        sigma_m2=sigma_m2,
        lam=lam,
        pj_w=pj_w,
        gj=gj,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
        loss=loss,
    )


def burn_through_range(pt_w, gain, sigma_m2, pj_w, gj, bw_r_hz, bw_j_hz, loss=1.0):
    """自屏蔽干扰的烧穿距离 R_bt（JS = 1 的解析解），单位 m。

    R_bt = sqrt(Pt*Gt*sigma*Bj / (4*pi*Pj*Gj*Br*L))

    注意：只含发射增益 Gt，且与波长无关 —— 见模块 docstring 的推导。
    """
    _require_positive(
        pt_w=pt_w,
        gain=gain,
        sigma_m2=sigma_m2,
        pj_w=pj_w,
        gj=gj,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
        loss=loss,
    )
    return np.sqrt(
        np.asarray(pt_w, dtype=float)
        * np.asarray(gain, dtype=float)
        * np.asarray(sigma_m2, dtype=float)
        * np.asarray(bw_j_hz, dtype=float)
        / (
            _FOUR_PI
            * np.asarray(pj_w, dtype=float)
            * np.asarray(gj, dtype=float)
            * np.asarray(bw_r_hz, dtype=float)
            * np.asarray(loss, dtype=float)
        )
    )


def burn_through_range_standoff(
    pt_w, gain, sigma_m2, pj_w, gj, bw_r_hz, bw_j_hz, jammer_range_m, loss=1.0
):
    """干扰机在 ``jammer_range_m`` 处时，雷达对它前方目标的烧穿半径 [m]。

    由 JS(R, Rj) = R⁴ / (R_bt² · Rj²) = 1 解出：

        R_bt(Rj) = sqrt( R_bt · Rj )

    即"**围绕雷达**的烧穿半径随干扰机距离开平方收缩"：干扰机从 200 km 压到 50 km，
    半径缩一半。干扰机越近，雷达能看穿的圈越小 —— 这就是 T1 动画要画的东西。

    与 ``burn_through_range`` 的关系：Rj = R_bt 时两者相等（自屏蔽），
    这条恒等式是单测里的交叉验证点。
    """
    _require_positive(jammer_range_m=jammer_range_m)
    r_bt_self = burn_through_range(
        pt_w=pt_w,
        gain=gain,
        sigma_m2=sigma_m2,
        pj_w=pj_w,
        gj=gj,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
        loss=loss,
    )
    return np.sqrt(r_bt_self * np.asarray(jammer_range_m, dtype=float))


def range_at_js(
    target_js=1.0,
    *,
    pt_w,
    gain,
    sigma_m2,
    lam,
    pj_w,
    gj,
    bw_r_hz,
    bw_j_hz,
    loss=1.0,
    jammer_range_m=None,
    r_min_m=1.0,
    r_max_m=1.0e9,
    rtol=1.0e-10,
    max_iter=200,
):
    """数值反解 JS(R) = target_js 对应的距离，单位 m。

    只用 JS 模型本身的取值做对数空间二分（JS 随 R 单调递增），
    不碰 R_bt 的闭式解，所以能和 ``burn_through_range`` 互相独立地交叉验证。

    参数
    ----
    target_js : 目标干信比（线性）。默认 1.0，即烧穿点。
    jammer_range_m : None（默认）→ 自屏蔽；给了值 → 站距干扰（Rj 固定）。
    r_min_m / r_max_m : 二分区间，必须跨过解；否则抛 ValueError。
    rtol : 对数空间区间宽度收敛阈值，等价于相对精度。
    """
    _require_positive(target_js=target_js, r_min_m=r_min_m, r_max_m=r_max_m)
    if float(r_max_m) <= float(r_min_m):
        raise ValueError(f"r_max_m({r_max_m}) 必须大于 r_min_m({r_min_m})")
    if jammer_range_m is not None:
        _require_positive(jammer_range_m=jammer_range_m)

    js_kwargs = dict(
        pt_w=pt_w,
        gain=gain,
        sigma_m2=sigma_m2,
        lam=lam,
        pj_w=pj_w,
        gj=gj,
        bw_r_hz=bw_r_hz,
        bw_j_hz=bw_j_hz,
        loss=loss,
    )

    def js_at(range_m: float) -> float:
        if jammer_range_m is None:
            return float(js_ratio_self_screen(range_m=range_m, **js_kwargs))
        return float(
            js_ratio_standoff(
                range_m=range_m, jammer_range_m=jammer_range_m, **js_kwargs
            )
        )

    def residual(range_m: float) -> float:
        return js_at(range_m) - float(target_js)

    lo = float(np.log(float(r_min_m)))
    hi = float(np.log(float(r_max_m)))
    f_lo = residual(np.exp(lo))
    f_hi = residual(np.exp(hi))

    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError(
            "区间 [r_min_m, r_max_m] 未跨过 JS 目标值："
            f"JS({r_min_m:g} m)={f_lo + target_js:.4g}, "
            f"JS({r_max_m:g} m)={f_hi + target_js:.4g}, target={target_js:.4g}"
        )

    iterations = 0
    while iterations < max_iter and (hi - lo) > rtol:
        mid = 0.5 * (lo + hi)
        if residual(np.exp(mid)) > 0.0:
            hi = mid
        else:
            lo = mid
        iterations += 1

    return np.float64(np.exp(0.5 * (lo + hi)))


# --- 检测概率 ----------------------------------------------------------------


def snr_thermal(pr_w, pn_w):
    """仅热噪声时的信噪比 SNR = Pr / Pn（线性）。"""
    _require_positive(pr_w=pr_w, pn_w=pn_w)
    return np.asarray(pr_w, dtype=float) / np.asarray(pn_w, dtype=float)


def sinr(pr_w, pn_w, pj_rx_w=0.0):
    """含干扰的信干噪比 SINR = Pr / (Pn + Pj_rx)（线性）。

    这是进入检测器的实际信噪比：干扰压制 = 分母变大 = 检测概率下降。
    """
    _require_positive(pr_w=pr_w, pn_w=pn_w)
    if np.any(np.asarray(pj_rx_w, dtype=float) < 0.0):
        raise ValueError(f"pj_rx_w 不能为负，收到 {pj_rx_w!r}")
    return np.asarray(pr_w, dtype=float) / (
        np.asarray(pn_w, dtype=float) + np.asarray(pj_rx_w, dtype=float)
    )


def pd_swerling1(pfa, snr):
    """Swerling I 单脉冲检测概率闭式：Pd = Pfa ** (1 / (1 + SNR))。

    SNR 为**线性**值（不是 dB）。边界行为：SNR=0 时 Pd=Pfa，SNR→∞ 时 Pd→1。
    """
    _require_probability("pfa", pfa)
    if np.any(np.asarray(snr, dtype=float) < 0.0):
        raise ValueError(f"snr 不能为负（线性值），收到 {snr!r}")
    return np.asarray(pfa, dtype=float) ** (
        1.0 / (1.0 + np.asarray(snr, dtype=float))
    )


def pd_from_powers(pfa, pr_w, pn_w, pj_rx_w=0.0):
    """由功率直接算检测概率：Pd(Pfa, Pr / (Pn + Pj_rx))。

    pj_rx_w = 0 时退化为纯热噪声情形。
    """
    return pd_swerling1(pfa, sinr(pr_w=pr_w, pn_w=pn_w, pj_rx_w=pj_rx_w))


def detection_range(
    target_pd,
    pfa,
    *,
    pt_w,
    gain,
    sigma_m2,
    lam,
    bw_r_hz,
    noise_figure=1.0,
    temperature_k=T0_K,
    loss=1.0,
    pj_w=0.0,
    gj=1.0,
    bw_j_hz=None,
    jammer_range_m=None,
):
    """满足 Pd(R) = target_pd 的距离（"检测距离"），单位 m。闭式解，不做迭代。

    推导（记 A = Pr(1 m)、B = Pj_rx(1 m)，于是 Pr(R) = A/R⁴、Pj_rx(R) = B/R²）::

        Pd = Pfa ** (1/(1+SINR)) = target_pd
        ⇒ 1/(1+SINR) = ln(target_pd)/ln(Pfa) ≡ k
        ⇒ SINR = S0 = 1/k - 1                        # 门槛对应的信干噪比

    站距干扰（Rj 固定）：A/R⁴ = S0·(Pn + B/Rj²)  ⇒  R = [A / (S0·(Pn + B/Rj²))]^(1/4)

    自屏蔽（Rj = R）：A/R⁴ = S0·(Pn + B/R²)，令 u = R²::

        S0·Pn·u² + S0·B·u - A = 0
        u = (-S0·B + sqrt((S0·B)² + 4·S0·Pn·A)) / (2·S0·Pn)

    系数 A、B 由 ``received_power`` / ``jammer_received_power`` 在 1 m 处取值得到，
    所以这里没有第二套物理公式；O(1) 也避免了每个 tick 跑二分。

    pj_w = 0（默认）时只算热噪声，bw_j_hz 可省略。
    jammer_range_m：None → 自屏蔽；给值 → 站距干扰（T1 动画用的几何）。
    """
    _require_probability("pfa", pfa)
    _require_probability("target_pd", target_pd)
    if float(target_pd) <= float(pfa):
        raise ValueError(
            f"target_pd({target_pd:g}) 必须大于 pfa({pfa:g})：Pd 的下限就是 Pfa，无解"
        )
    _require_positive(
        pt_w=pt_w,
        gain=gain,
        sigma_m2=sigma_m2,
        lam=lam,
        bw_r_hz=bw_r_hz,
        noise_figure=noise_figure,
        temperature_k=temperature_k,
        loss=loss,
    )
    if jammer_range_m is not None:
        _require_positive(jammer_range_m=jammer_range_m)

    if float(pj_w) < 0.0:
        raise ValueError(f"pj_w 不能为负，收到 {pj_w!r}")
    jammed = float(pj_w) > 0.0
    if jammer_range_m is not None and not jammed:
        raise ValueError("给了 jammer_range_m 却没给 pj_w：干扰机距离没有意义")
    if jammed:
        _require_positive(pj_w=pj_w, gj=gj)
        if bw_j_hz is None:
            raise ValueError("pj_w > 0 时必须给出 bw_j_hz（干扰机噪声带宽）")
        _require_positive(bw_j_hz=bw_j_hz)

    # 门槛对应的 SINR：k 必须落在 (0, 1)，即 pfa < target_pd < 1
    k = math.log(float(target_pd)) / math.log(float(pfa))
    s0 = 1.0 / k - 1.0

    pn_w = float(
        noise_power(
            bw_hz=bw_r_hz, noise_figure=noise_figure, temperature_k=temperature_k
        )
    )
    # 1 m 处的功率即系数（Pr ∝ 1/R⁴、Pj_rx ∝ 1/R²），复用同一套功率函数
    a = float(
        received_power(
            pt_w=pt_w,
            gt=gain,
            gr=gain,
            lam=lam,
            sigma_m2=sigma_m2,
            range_m=1.0,
            loss=loss,
        )
    )
    if not jammed:
        return np.float64((a / (s0 * pn_w)) ** 0.25)

    b = float(
        jammer_received_power(
            pj_w=pj_w,
            gj=gj,
            gr=gain,
            lam=lam,
            range_j_m=1.0,
            bw_r_hz=bw_r_hz,
            bw_j_hz=bw_j_hz,
        )
    )
    if jammer_range_m is not None:
        pj_rx = b / float(jammer_range_m) ** 2
        return np.float64((a / (s0 * (pn_w + pj_rx))) ** 0.25)

    # 自屏蔽。用共轭形式避免 (-b + sqrt(b²+4ac)) 的灾难性抵消：
    #   u = 2A / (sqrt((S0·B)² + 4·S0·Pn·A) + S0·B)
    qa = s0 * pn_w
    qb = s0 * b
    u = (2.0 * a) / (math.sqrt(qb * qb + 4.0 * qa * a) + qb)
    return np.float64(math.sqrt(u))


# --- 诱饵：转发式假目标（T3b） -----------------------------------------------


def decoy_transmit_power(pt_w, gt, decoy_gain, amplification, lam, range_m):
    """诱饵转发出去的功率 [W]：照到它的功率 × 放大倍数。

    对 ESM 来说，诱饵就是一个"只在该帧辐射"的发射源——所以它能不能被发现，
    取决于雷达照得有多强，而不是诱饵自己设定的功率。
    """
    _require_positive(
        pt_w=pt_w,
        gt=gt,
        decoy_gain=decoy_gain,
        amplification=amplification,
        lam=lam,
        range_m=range_m,
    )
    illumination = (
        np.asarray(pt_w, dtype=float)
        * np.asarray(gt, dtype=float)
        * np.asarray(decoy_gain, dtype=float)
        * np.asarray(lam, dtype=float) ** 2
        / (_FOUR_PI_SQ * np.asarray(range_m, dtype=float) ** 2)
    )
    return illumination * np.asarray(amplification, dtype=float)


def decoy_echo_power(
    pt_w,
    gt,
    gr,
    decoy_gain,
    amplification,
    lam,
    range_m,
):
    """转发式诱饵在雷达处形成的回波功率 [W]。

    链路是两段单程：

        P_r = [Pt·Gt·G_d·λ²/((4π)²·R²)] · A · [G_d·Gr·λ²/((4π)²·R²)]
            = Pt·Gt·Gr·λ⁴·G_d²·A / ((4π)⁴·R⁴)

    其中 ``A`` 是诱饵的放大倍数（线性）。诱饵自身的插入损耗折进 ``A``，
    所以这里没有单独的 ``loss`` 参数——与雷达方程的 ``L`` 是两件事，
    混在一起会让"等效 RCS"失去意义。
    """
    _require_positive(
        pt_w=pt_w,
        gt=gt,
        gr=gr,
        decoy_gain=decoy_gain,
        amplification=amplification,
        lam=lam,
        range_m=range_m,
    )
    return (
        decoy_transmit_power(
            pt_w=pt_w,
            gt=gt,
            decoy_gain=decoy_gain,
            amplification=amplification,
            lam=lam,
            range_m=range_m,
        )
        * np.asarray(decoy_gain, dtype=float)
        * np.asarray(gr, dtype=float)
        * np.asarray(lam, dtype=float) ** 2
        / (_FOUR_PI_SQ * np.asarray(range_m, dtype=float) ** 2)
    )


def decoy_equivalent_rcs(lam, gain, amplification, loss=1.0):
    """把诱饵折算成"等效 RCS" [m²]，便于和目标放在一起比较。

    令诱饵回波与同距离上的目标回波相等，解出 σ：

        σ_eq = λ²·G_d²·A·L / (4π)

    与 ``decoy_echo_power`` 严格一致：
    ``received_power(..., sigma_m2=σ_eq, ..., loss=L)`` 得到的就是同样的回波功率。
    """
    _require_positive(lam=lam, gain=gain, amplification=amplification, loss=loss)
    return (
        np.asarray(lam, dtype=float) ** 2
        * np.asarray(gain, dtype=float) ** 2
        * np.asarray(amplification, dtype=float)
        * np.asarray(loss, dtype=float)
        / _FOUR_PI
    )
