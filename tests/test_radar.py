"""T0 验收单测：``spectrumfog.physics.radar`` 的物理自洽性。

对应 PROJECT.md §6.3 的两条硬验收：
  1. J/S == 1 时数值解出的 R 与解析 R_bt 相对误差 < 1%
  2. R_bt 处 J/S 落在 1.00 ± 0.01

另外钉死几条容易被"顺手改回来"的性质：
  * 自屏蔽干扰下接收增益 Gr 相消 → R_bt 只含发射增益 G¹（规格书 G² 是笔误）
  * 波长在干扰支路与回波支路相消 → R_bt 与频率无关
  * 噪声底 Pn = k·T0·B·F
  * Pd 的边界、单调性，以及干扰确实压低 Pd（SINR 口径）
"""

from __future__ import annotations

import numpy as np
import pytest

from spectrumfog.physics import radar
from spectrumfog.units import db_to_linear, ghz_to_hz, mhz_to_hz

# ``burn_through_range`` 的入参：刻意不含 lam —— 波长相消，见模块 docstring。
_RBT_KEYS = ("pt_w", "gain", "sigma_m2", "pj_w", "gj", "bw_r_hz", "bw_j_hz", "loss")


def _rbt_kwargs(params: dict) -> dict:
    return {key: params[key] for key in _RBT_KEYS}


def _duel() -> dict:
    """01_duel 想定参数（PROJECT.md §7.1）原样换算成 SI。

    强干扰机压 30 km 外的雷达：R_bt 会很小（几十米量级），这是参数本身的结果。
    """
    return dict(
        pt_w=1.0e5,
        gain=float(db_to_linear(35.0)),
        sigma_m2=1.0,
        pj_w=2000.0,
        gj=float(db_to_linear(20.0)),
        bw_r_hz=float(mhz_to_hz(2.0)),
        bw_j_hz=float(mhz_to_hz(20.0)),
        loss=float(db_to_linear(3.0)),
        lam=float(radar.wavelength(ghz_to_hz(3.0))),
    )


def _weak_jammer() -> dict:
    """弱干扰机 + 强雷达：R_bt 拉到几百公里，验证二分区间与数量级。"""
    return dict(
        pt_w=1.0e5,
        gain=float(db_to_linear(30.0)),
        sigma_m2=10.0,
        pj_w=1.0e-3,
        gj=float(db_to_linear(0.0)),
        bw_r_hz=float(mhz_to_hz(1.0)),
        bw_j_hz=float(mhz_to_hz(1.0)),
        loss=float(db_to_linear(2.0)),
        lam=float(radar.wavelength(ghz_to_hz(1.0))),
    )


def _x_band_radar() -> dict:
    """X 波段大孔径雷达 vs 宽带噪声干扰：R_bt 落在几公里。"""
    return dict(
        pt_w=1.0e5,
        gain=float(db_to_linear(40.0)),
        sigma_m2=10.0,
        pj_w=50.0,
        gj=float(db_to_linear(10.0)),
        bw_r_hz=float(mhz_to_hz(1.0)),
        bw_j_hz=float(mhz_to_hz(100.0)),
        loss=float(db_to_linear(4.0)),
        lam=float(radar.wavelength(ghz_to_hz(10.0))),
    )


SCENARIOS = {
    "duel_01": _duel(),
    "weak_jammer": _weak_jammer(),
    "x_band": _x_band_radar(),
}
SCENARIO_IDS = list(SCENARIOS)


# --- 验收断言 ----------------------------------------------------------------


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_numeric_range_at_js1_equals_analytic_burn_through(params):
    """J/S == 1 时数值解出的 R 等于解析 R_bt，相对误差 < 1%（PROJECT.md §6.3）。"""
    r_bt_analytic = float(radar.burn_through_range(**_rbt_kwargs(params)))
    r_bt_numeric = float(radar.range_at_js(1.0, **params))

    rel_err = abs(r_bt_numeric - r_bt_analytic) / r_bt_analytic

    assert rel_err < 0.01, (
        f"相对误差 {rel_err:.3e} 超过 1% 验收线："
        f"解析 {r_bt_analytic:.6g} m vs 数值 {r_bt_numeric:.6g} m"
    )
    # 两路解的是同一个方程，实际精度应远好于验收线；这条防止"宽松通过"。
    assert rel_err < 1.0e-6


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_js_at_burn_through_range_is_unity(params):
    """R_bt 处 J/S 落在 1.00 ± 0.01（PROJECT.md §6.3）。"""
    r_bt = float(radar.burn_through_range(**_rbt_kwargs(params)))
    js = float(radar.js_ratio_self_screen(range_m=r_bt, **params))

    assert js == pytest.approx(1.0, abs=0.01)
    assert 1.00 - 0.01 <= js <= 1.00 + 0.01


# --- 钉死规格里那个 G 指数 ----------------------------------------------------


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_self_screening_js_is_independent_of_receive_gain(params):
    """Gr 在干扰支路与回波支路同时出现 → J/S 与 Gr 无关。

    这是 R_bt 只含 G¹ 的直接依据；若有人把 R_bt 改回 G²，这条会失败。
    """
    range_m = 5.0 * float(radar.burn_through_range(**_rbt_kwargs(params)))

    def js_with_gr(gr: float) -> float:
        pr = radar.received_power(
            pt_w=params["pt_w"],
            gt=params["gain"],
            gr=gr,
            lam=params["lam"],
            sigma_m2=params["sigma_m2"],
            range_m=range_m,
            loss=params["loss"],
        )
        pj_rx = radar.jammer_received_power(
            pj_w=params["pj_w"],
            gj=params["gj"],
            gr=gr,
            lam=params["lam"],
            range_j_m=range_m,
            bw_r_hz=params["bw_r_hz"],
            bw_j_hz=params["bw_j_hz"],
        )
        return float(radar.js_ratio(pr_w=pr, pj_rx_w=pj_rx))

    reference = js_with_gr(params["gain"])
    for gr in (1.0, 100.0, 1.0e4):
        assert js_with_gr(gr) == pytest.approx(reference, rel=1.0e-12)


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_self_screening_js_is_wavelength_agnostic(params):
    """自屏蔽时两条支路共享 lam² → J/S 与频率无关，故 R_bt 公式里没有 lam。"""
    range_m = 2.0 * float(radar.burn_through_range(**_rbt_kwargs(params)))
    js_at_lam = float(radar.js_ratio_self_screen(range_m=range_m, **params))

    for factor in (0.5, 2.0, 8.0):
        shifted = dict(params, lam=params["lam"] * factor)
        assert float(radar.js_ratio_self_screen(range_m=range_m, **shifted)) == (
            pytest.approx(js_at_lam, rel=1.0e-12)
        )


def test_burn_through_range_api_rejects_wavelength():
    """R_bt 不含波长，接口上就不该能传 lam（传了是 TypeError，不是静默忽略）。"""
    with pytest.raises(TypeError):
        radar.burn_through_range(lam=1.0, **_rbt_kwargs(SCENARIOS["duel_01"]))


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_js_increases_with_range(params):
    """干扰支路 1/R²、回波支路 1/R⁴ → J/S 随距离单调递增，二分法前提成立。"""
    r_bt = float(radar.burn_through_range(**_rbt_kwargs(params)))
    ranges = r_bt * np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    js = np.asarray(radar.js_ratio_self_screen(range_m=ranges, **params))

    assert np.all(np.diff(js) > 0.0)
    # 翻倍距离 → J/S 乘以 4（R² 律）
    assert js[3] / js[2] == pytest.approx(4.0, rel=1.0e-12)


# --- 噪声底与检测概率 --------------------------------------------------------


def test_noise_power_matches_ktbf():
    """Pn = k·T0·B·F，F 为线性噪声系数。"""
    bw_hz = mhz_to_hz(1.0)  # 1 MHz
    noise_figure = float(db_to_linear(3.0))
    pn = float(radar.noise_power(bw_hz=bw_hz, noise_figure=noise_figure))

    expected = 1.380649e-23 * 290.0 * 1.0e6 * noise_figure  # 常量照抄 spec，不复用模块常量
    assert pn == pytest.approx(expected, rel=1.0e-12)
    assert pn == pytest.approx(7.9882e-15, rel=1.0e-4)


def test_pd_bounds_and_limits():
    """Pd = Pfa**(1/(1+SNR))：SNR=0 退回虚警率，SNR→∞ 趋于 1，且始终夹在 [Pfa, 1]。"""
    pfa = 1.0e-6

    assert float(radar.pd_swerling1(pfa, 0.0)) == pytest.approx(pfa, rel=1.0e-12)
    assert float(radar.pd_swerling1(pfa, 1.0e6)) == pytest.approx(1.0, abs=1.0e-4)

    snr = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1.0e4])
    pd = np.asarray(radar.pd_swerling1(pfa, snr))
    assert np.all(np.diff(pd) > 0.0)  # 关于 SNR 严格单调
    assert np.all(pd >= pfa) and np.all(pd <= 1.0)


def test_jamming_lowers_detection_probability():
    """SINR 口径：同样的回波功率，加了干扰后 Pd 必须下降。"""
    pfa = 1.0e-6
    pr_w = 1.0e-12
    pn_w = 1.0e-14  # 热 SNR = 100 (20 dB)；Swerling I 闭式下 Pd ≈ 0.87
    pj_rx_w = 1.0e-11  # SINR 掉到 ≈ 0.1，Pd → 1e-5 量级

    pd_thermal = float(radar.pd_from_powers(pfa, pr_w, pn_w))
    pd_jammed = float(radar.pd_from_powers(pfa, pr_w, pn_w, pj_rx_w=pj_rx_w))

    # 闭式 Pd = Pfa**(1/(1+SNR)) 对 SNR 比较"钝"：Pfa=1e-6 时 SNR=20 dB 才 0.87
    assert pd_thermal > 0.8
    assert pd_jammed < 1.0e-3
    assert pd_jammed < pd_thermal


def test_js_ratio_and_sinr_are_consistent_with_power_functions():
    """js_ratio 就是 Pj_rx/Pr，sinr 就是 Pr/(Pn+Pj_rx)，不做任何隐藏修正。"""
    pr = radar.received_power(
        pt_w=1.0e5, gt=1000.0, gr=1000.0, lam=0.1, sigma_m2=1.0, range_m=1.0e4
    )
    pj_rx = radar.jammer_received_power(
        pj_w=100.0,
        gj=10.0,
        gr=1000.0,
        lam=0.1,
        range_j_m=1.0e4,
        bw_r_hz=1.0e6,
        bw_j_hz=1.0e7,
    )
    assert float(radar.js_ratio(pr_w=pr, pj_rx_w=pj_rx)) == pytest.approx(
        float(pj_rx) / float(pr), rel=1.0e-15
    )
    assert float(radar.sinr(pr_w=pr, pn_w=2.0 * pr, pj_rx_w=2.0 * pr)) == (
        pytest.approx(0.25, rel=1.0e-15)
    )


def test_vectorized_inputs_preserve_shape():
    """物理函数按 numpy 数组逐元素计算（性能目标：不写 Python 循环）。"""
    params = SCENARIOS["duel_01"]
    ranges = np.array([1.0e3, 5.0e3, 1.0e4, 3.0e4])
    js_vec = np.asarray(radar.js_ratio_self_screen(range_m=ranges, **params))
    assert js_vec.shape == ranges.shape

    js_scalar = np.array(
        [float(radar.js_ratio_self_screen(range_m=r, **params)) for r in ranges]
    )
    assert np.allclose(js_vec, js_scalar, rtol=1.0e-12)

    pd_vec = np.asarray(radar.pd_swerling1(1.0e-6, np.array([0.1, 1.0, 10.0])))
    assert pd_vec.shape == (3,)


def test_invalid_inputs_raise():
    """内部 SI 约定下这些输入没有物理意义，必须报错而不是静默算出一个数。"""
    params = SCENARIOS["duel_01"]

    with pytest.raises(ValueError):
        radar.burn_through_range(**dict(_rbt_kwargs(params), pj_w=0.0))

    with pytest.raises(ValueError):
        radar.received_power(
            pt_w=1.0e5, gt=1000.0, gr=1000.0, lam=0.1, sigma_m2=1.0, range_m=-1.0
        )

    with pytest.raises(ValueError):
        radar.js_ratio_self_screen(range_m=0.0, **params)

    with pytest.raises(ValueError):
        radar.pd_swerling1(pfa=1.5, snr=1.0)

    with pytest.raises(ValueError):
        radar.pd_swerling1(pfa=1.0e-6, snr=-1.0)

    # 二分区间没有跨过解时必须显式失败，而不是返回端点
    with pytest.raises(ValueError):
        radar.range_at_js(1.0, r_min_m=1.0e7, r_max_m=1.0e9, **params)


# --- 检测距离（Pd 反解） ------------------------------------------------------


def _detection_kwargs(params: dict) -> dict:
    return {
        "pt_w": params["pt_w"],
        "gain": params["gain"],
        "sigma_m2": params["sigma_m2"],
        "lam": params["lam"],
        "bw_r_hz": params["bw_r_hz"],
        "loss": params["loss"],
    }


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_detection_range_lands_on_target_pd(params):
    """R_det 的定义就是 Pd(R_det) = target_pd；代回去必须回到门槛。"""
    pfa, target = 1.0e-6, 0.5
    detection = _detection_kwargs(params)
    r_det = float(radar.detection_range(target, pfa, **detection))

    pn = radar.noise_power(bw_hz=params["bw_r_hz"])
    pr = radar.received_power(
        pt_w=params["pt_w"],
        gt=params["gain"],
        gr=params["gain"],
        lam=params["lam"],
        sigma_m2=params["sigma_m2"],
        range_m=r_det,
        loss=params["loss"],
    )
    assert float(radar.pd_from_powers(pfa, pr, pn)) == pytest.approx(target, abs=1e-8)

    # 检测距离与烧穿距离是两件事，但必须由同一组参数互洽：J/S(R_det) = (R_det/R_bt)^2
    r_bt = float(radar.burn_through_range(**_rbt_kwargs(params)))
    js_at_det = float(radar.js_ratio_self_screen(range_m=r_det, **params))
    assert js_at_det == pytest.approx((r_det / r_bt) ** 2, rel=1.0e-9)


def test_duel_detection_range_exceeds_burn_through():
    """01_duel：检测距离（几十 km）远大于烧穿距离（几 km），这是想定要讲的对比。"""
    params = SCENARIOS["duel_01"]
    r_det = float(
        radar.detection_range(0.5, 1.0e-6, **_detection_kwargs(params))
    )
    r_bt = float(radar.burn_through_range(**_rbt_kwargs(params)))
    assert r_det > 5.0 * r_bt


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_jammed_detection_range_is_shorter_and_consistent(params):
    """加了干扰，检测距离一定变短；且反解出的距离仍满足 Pd = 门槛。"""
    pfa, target = 1.0e-6, 0.5
    detection = _detection_kwargs(params)

    r_clean = float(radar.detection_range(target, pfa, **detection))
    r_jammed = float(
        radar.detection_range(
            target,
            pfa,
            pj_w=params["pj_w"],
            gj=params["gj"],
            bw_j_hz=params["bw_j_hz"],
            **detection,
        )
    )
    assert r_jammed < r_clean

    pn = radar.noise_power(bw_hz=params["bw_r_hz"])
    pr = radar.received_power(
        pt_w=params["pt_w"],
        gt=params["gain"],
        gr=params["gain"],
        lam=params["lam"],
        sigma_m2=params["sigma_m2"],
        range_m=r_jammed,
        loss=params["loss"],
    )
    pj_rx = radar.jammer_received_power(
        pj_w=params["pj_w"],
        gj=params["gj"],
        gr=params["gain"],
        lam=params["lam"],
        range_j_m=r_jammed,
        bw_r_hz=params["bw_r_hz"],
        bw_j_hz=params["bw_j_hz"],
    )
    assert float(radar.pd_from_powers(pfa, pr, pn, pj_rx)) == pytest.approx(
        target, abs=1e-8
    )


def test_detection_range_input_errors():
    params = SCENARIOS["duel_01"]
    detection = _detection_kwargs(params)

    # 给了干扰功率却没给干扰带宽
    with pytest.raises(ValueError, match="bw_j_hz"):
        radar.detection_range(
            0.5, 1.0e-6, pj_w=params["pj_w"], gj=params["gj"], **detection
        )

    # Pd 的下限就是 Pfa：门槛不高于 Pfa 时无解（闭式解要求 pfa < target_pd）
    with pytest.raises(ValueError, match="target_pd"):
        radar.detection_range(1.0e-7, 1.0e-6, **detection)

    # 概率参数必须在 (0, 1) 内
    with pytest.raises(ValueError):
        radar.detection_range(1.5, 1.0e-6, **detection)
    with pytest.raises(ValueError):
        radar.detection_range(0.5, 1.5, **detection)

    # 负干扰功率没有意义
    with pytest.raises(ValueError):
        radar.detection_range(0.5, 1.0e-6, pj_w=-1.0, bw_j_hz=1.0e8, **detection)


# --- 站距干扰（T1 动画用的几何：干扰机与目标分离） ---------------------------


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_standoff_burn_through_equals_self_screen_at_rbt(params):
    """Rj = R_bt 时，站距烧穿半径必须退回自屏蔽烧穿距离（定义上的一致点）。"""
    rbt = float(radar.burn_through_range(**_rbt_kwargs(params)))
    r_standoff = float(
        radar.burn_through_range_standoff(**_rbt_kwargs(params), jammer_range_m=rbt)
    )
    assert r_standoff == pytest.approx(rbt, rel=1.0e-12)


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_standoff_burn_through_has_unity_js(params):
    """站距烧穿半径处 J/S = 1，且与独立数值反解一致。"""
    for jammer_range_m in (1.0e4, 6.0e4, 2.0e5):
        r_analytic = float(
            radar.burn_through_range_standoff(
                **_rbt_kwargs(params), jammer_range_m=jammer_range_m
            )
        )
        js = float(
            radar.js_ratio_standoff(
                range_m=r_analytic, jammer_range_m=jammer_range_m, **params
            )
        )
        assert js == pytest.approx(1.0, rel=1.0e-12)

        r_numeric = float(
            radar.range_at_js(1.0, jammer_range_m=jammer_range_m, **params)
        )
        assert r_numeric == pytest.approx(r_analytic, rel=1.0e-6)


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_standoff_radius_scales_as_sqrt_of_jammer_range(params):
    """R_bt(Rj) ∝ sqrt(Rj)：干扰机距离翻 4 倍，烧穿半径翻 2 倍。"""
    near = float(
        radar.burn_through_range_standoff(**_rbt_kwargs(params), jammer_range_m=5.0e4)
    )
    far = float(
        radar.burn_through_range_standoff(**_rbt_kwargs(params), jammer_range_m=2.0e5)
    )
    assert far == pytest.approx(2.0 * near, rel=1.0e-12)


@pytest.mark.parametrize("params", SCENARIOS.values(), ids=SCENARIO_IDS)
def test_standoff_js_reduces_to_self_screen(params):
    """Rj = R 时，站距 J/S 与自屏蔽 J/S 是同一个数。"""
    range_m = 2.0e4
    assert float(
        radar.js_ratio_standoff(
            range_m=range_m, jammer_range_m=range_m, **params
        )
    ) == pytest.approx(
        float(radar.js_ratio_self_screen(range_m=range_m, **params)), rel=1.0e-15
    )


def test_detection_range_shrinks_as_jammer_approaches():
    """干扰机越近，Pd = 门槛 的边界越小；边界处 Pd 仍等于门槛。"""
    params = SCENARIOS["duel_01"]
    detection = _detection_kwargs(params)
    jammer_kwargs = {
        "pj_w": params["pj_w"],
        "gj": params["gj"],
        "bw_j_hz": params["bw_j_hz"],
    }
    pfa, target = 1.0e-6, 0.5

    r_far = float(
        radar.detection_range(
            target, pfa, jammer_range_m=2.0e5, **jammer_kwargs, **detection
        )
    )
    r_near = float(
        radar.detection_range(
            target, pfa, jammer_range_m=2.0e4, **jammer_kwargs, **detection
        )
    )
    assert r_near < r_far

    # 代回 Pd：边界处必须还是 0.5
    pn = radar.noise_power(bw_hz=params["bw_r_hz"])
    pr = radar.received_power(
        pt_w=params["pt_w"],
        gt=params["gain"],
        gr=params["gain"],
        lam=params["lam"],
        sigma_m2=params["sigma_m2"],
        range_m=r_near,
        loss=params["loss"],
    )
    pj_rx = radar.jammer_received_power(
        pj_w=params["pj_w"],
        gj=params["gj"],
        gr=params["gain"],
        lam=params["lam"],
        range_j_m=2.0e4,
        bw_r_hz=params["bw_r_hz"],
        bw_j_hz=params["bw_j_hz"],
    )
    assert float(radar.pd_from_powers(pfa, pr, pn, pj_rx)) == pytest.approx(
        target, abs=1.0e-8
    )


def test_standoff_without_jammer_power_raises():
    """给了干扰机距离却没给功率，是调用错误，不能静默忽略。"""
    params = SCENARIOS["duel_01"]
    with pytest.raises(ValueError, match="jammer_range_m"):
        radar.detection_range(
            0.5, 1.0e-6, jammer_range_m=3.0e4, **_detection_kwargs(params)
        )
