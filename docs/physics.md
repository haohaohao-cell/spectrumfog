# 物理内核：公式、假设、单位与已知缺口

> 本文件记录 `spectrumfog.physics` **实际实现**了什么、为什么这么实现。
> 规格源头是 [PROJECT.md](../PROJECT.md) §6；两边不一致时以本文件的"裁决记录"为准，
> 并且必须回来改 PROJECT.md。

## 1. 单位约定

内部一律 SI，且**增益/损耗一律线性**：

| 量 | 内部单位 | 允许的输入写法（YAML / CLI） |
|---|---|---|
| 功率 | W | `pt_w` / `pt_kw` / `pt_dbm` / `pt_dbw`（同 `pj_*`） |
| 长度 | m | `pos`（米）/ `pos_km` / `pos_nmi` |
| 频率 | Hz | `freq_hz` / `freq_khz` / `freq_mhz` / `freq_ghz`（同 `bw_*`） |
| 时间 | s | `dt` / `dt_ms`，`duration` / `duration_ms` |
| 增益、损耗、噪声系数 | 线性 | YAML 写 `gain_db` / `loss_db` / `noise_figure_db`，**加载时**换成线性 |
| 角度 | rad | 尚未使用（T1 的 AOA 会用到） |

换算只发生在 `model.scenario.normalize_units`（单位后缀）与 pydantic 的 before-validator（dB→线性）。
`physics` 层的函数**不接受 dB**：传 dB 进去会得到静默错误的数，所以接口层就用线性值。

## 2. 符号

| 符号 | 含义 | 单位 | 代码里的名字 |
|---|---|---|---|
| Pt | 雷达发射功率 | W | `pt_w` |
| Gt / Gr | 雷达发射 / 接收增益 | — | `gt` / `gr`（单站用同一个 `gain`） |
| Pj | 干扰机发射功率 | W | `pj_w` |
| Gj | 干扰机在雷达方向上的增益 | — | `gj` |
| λ | 波长 = c/f | m | `lam`（`wavelength()`） |
| σ | 目标 RCS | m² | `sigma_m2` |
| R | 雷达到目标距离 | m | `range_m` |
| Rj | 雷达到干扰机距离 | m | `range_j_m` |
| Br | 雷达接收带宽 | Hz | `bw_r_hz` |
| Bj | 干扰机噪声带宽 | Hz | `bw_j_hz` |
| L | 系统损耗（≥ 1） | — | `loss` |
| F | 噪声系数（线性） | — | `noise_figure` |
| T0 | 标准噪声温度 = 290 K | K | `temperature_k`（默认 `units.T0_K`） |
| k | 玻尔兹曼常数 = 1.380649e-23 | J/K | `units.K_BOLTZMANN` |

## 3. 公式

### 3.1 雷达方程与噪声

```
Pr    = Pt · Gt · Gr · λ² · σ / ((4π)³ · R⁴ · L)          # radar.received_power
Pn    = k · T0 · B · F                                     # radar.noise_power
```

### 3.2 噪声压制干扰

```
Pj_rx = Pj · Gj · Gr · λ² / ((4π)² · Rj²) · (Br / Bj)      # radar.jammer_received_power
JS    = Pj_rx / Pr                                         # radar.js_ratio
```

`Br/Bj` 是"只有落进接收带宽的那部分干扰功率算数"。自屏蔽干扰（干扰机与目标同处，Rj = R）时：

```
JS(R) = 4π · Pj · Gj · R² · L · Br / (Pt · Gt · σ · Bj)    # radar.js_ratio_self_screen
```

### 3.3 烧穿距离

令 JS = 1 解出（自屏蔽，Rj = R）：

```
R_bt = sqrt( Pt · Gt · σ · Bj / (4π · Pj · Gj · Br · L) )    # radar.burn_through_range
```

**站距干扰**（干扰机在 Rj、目标在 R ≠ Rj）时，JS ∝ R⁴/Rj²，令 JS = 1：

```
R_bt(Rj) = sqrt( R_bt · Rj )        # radar.burn_through_range_standoff
```

即"围绕雷达的烧穿半径随干扰机距离开平方根收缩"：干扰机从 200 km 压到 50 km，
半径缩一半；Rj = R_bt 时退回自屏蔽。这是 T1 动画的主线（`02_standoff`）。

### 3.4 检测概率（Swerling I 单脉冲闭式）

```
SNR = Pr / (Pn + Pj_rx)          # radar.sinr，含干扰；无干扰时退化为 Pr/Pn
Pd  = Pfa ^ (1 / (1 + SNR))      # radar.pd_swerling1
```

`SNR` 是**线性**值。边界行为：SNR = 0 → Pd = Pfa；SNR → ∞ → Pd → 1。

### 3.5 检测距离（闭式解，不是迭代）

`radar.detection_range(target_pd, pfa, ...)` 解 `Pd(R) = target_pd`。记 A = Pr(1 m)、
B = Pj_rx(1 m)，则 Pr(R) = A/R⁴、Pj_rx(R) = B/R²（自屏蔽）或 B/Rj²（站距）。

```
Pd = Pfa ** (1/(1+SINR)) = target_pd
⇒ k = ln(target_pd)/ln(Pfa) = 1/(1+SINR)  ⇒  S0 = 1/k - 1

站距：  A/R⁴ = S0·(Pn + B/Rj²)         ⇒ R = [A / (S0·(Pn + B/Rj²))]^(1/4)
自屏蔽：A/R⁴ = S0·(Pn + B/R²)           ⇒ 令 u=R²: S0·Pn·u² + S0·B·u - A = 0
                                        u = 2A / (sqrt((S0·B)² + 4·S0·Pn·A) + S0·B)
```

注意自屏蔽那一行用的是**共轭形式**：直接写 `(-b + sqrt(b²+4ac))/(2a)` 在
`Pn ≪ B` 的常见参数下会发生灾难性抵消（实测跑出负数根），所以必须改写。

A、B 由 `received_power` / `jammer_received_power` 在 1 m 处取值得到，
所以这里没有第二套物理公式；O(1) 也避免了引擎每个 tick 跑二分
（T1 实测：600 tick 的想定从 14 s 降到 1.3 s）。

### 3.6 传播与链路预算

```
Prx      = Ptx · Gtx · Grx · (λ / (4πd))²                   # propagation.received_power
FSPL     = (4πd / λ)²                                       # propagation.free_space_path_loss
Eb/N0    = Prx / (N0 · Rb),  N0 = k · T · F                 # comms.ebn0 / comms.noise_psd
AJ_margin_dB = Gp + Mc - required_JS_dB                     # comms.aj_margin_db
```

### 3.7 ESM 观测与定位（T3a）

传感器只看**对方阵营**的辐射源。单程 Friis 得到到达功率，噪声底用传感器自己的
带宽与噪声系数，再用同一个 Swerling I 闭式给单次扫描的检出概率：

```
Prx   = Ptx · Gtx · Grx · (λ / 4πR)²            # 单程，engine/detect.py
Pn    = k · T0 · Bs · Fs                        # 传感器带宽 / 噪声系数
Pd    = Pfa ^ (1/(1 + SNR))                     # 与雷达同一个函数
σ_aoa = 2° · sqrt(100 / SNR)                    # 夹在 [0.2°, 20°]
```

检出后生成带噪声的 AOA、TOA（σ = 100 ns ≈ 30 m）、频率、功率；按 `pfa_scan`
生成无源量测（虚假目标，`emitter = null`）。

定位用加权最小二乘（`engine/fusion.py`），未知量只有 (x, y)，残差两类：

```
测向：  wrap(aoa_i − atan2(p − s_i)) / σ_aoa
TDOA：  [ (|p − s_i| − |p − s_ref|)/c − (toa_i − toa_ref) ] / √(σ_toa,i² + σ_toa,ref²)
```

协方差 = (JᵀWJ)⁻¹，有冗余时用残差估方差因子（chi²/dof）。少于两条测向、
又不足三个 TOA 时几何上定不了位，直接返回 None——不猜。

AAR 的归因用的就是这套残差函数：留一法重解、单独喂测向 / TDOA，
所以"原因是 X"这句话是可以被复算的，不是写出来的。

### 3.8 诱饵（转发式假目标，T3b）

诱饵被雷达照到后放大转发，雷达收到的是两段单程：

```
P_r  = [Pt·Gt·G_d·λ²/((4π)²R²)] · A · [G_d·Gr·λ²/((4π)²R²)]
     = Pt·Gt·Gr·λ⁴·G_d²·A / ((4π)⁴·R⁴)          # radar.decoy_echo_power
σ_eq = λ²·G_d²·A·L / (4π)                        # radar.decoy_equivalent_rcs
```

`σ_eq` 的定义就是"同距离上给出同样回波功率的目标 RCS"，所以诱饵可以直接和真实目标
放在一张表里比较（06_decoy 里诱饵 σ_eq 6.3 m²，回波是 70 km 外真目标的 325 倍）。

诱饵对 ESM 的可见性由**转发功率**决定：`P_tx = 照射功率 × A`。
所以它只在该帧辐射、功率随距离变化——这条写进了 `radar.decoy_transmit_power`。

注意：诱饵不能对抗噪声压制。03_triad 里 90 W 的同频干扰机把雷达整个压住时，
诱饵（等效 6 m²）和真目标一起消失，`decoy_detected_fraction` 为 0——
诱饵骗的是"看得见的雷达"，不是"被致盲的雷达"。

## 4. 裁决记录

### 4.1 烧穿距离里的增益指数：G¹，不是 G²

PROJECT.md §6.1 写的是

```
R_bt = sqrt(Pt * G**2 * sigma * Bj / (4*pi * Pj * Gj * Br * L))
```

但与同一节的 Pr 与 Pj_rx 定义联立，Gr 会被约掉：

```
JS = [Pj·Gj·Gr·λ²/((4π)²R²) · (Br/Bj)] / [Pt·Gt·Gr·λ²σ/((4π)³R⁴L)]
   = 4π·Pj·Gj·R²·L·Br / (Pt·Gt·σ·Bj)        # Gr 相消
```

令 JS = 1：

```
R_bt = sqrt( Pt·Gt·σ·Bj / (4π·Pj·Gj·Br·L) )   # 只有一个 G
```

**结论：§6.1 的 `G**2` 是笔误，实现以 G¹ 为准**（2026-10-27 与需求方确认）。
物理上：自屏蔽干扰的接收增益同时出现在回波支路与干扰支路，互相抵消；
烧穿距离只由**发射**增益决定。

同理，λ² 在两条支路里也相消，所以 `burn_through_range` **签名里没有 `lam`** ——
传进去是 `TypeError`，不是静默忽略。

`tests/test_radar.py` 里两条测试直接钉住这两件事：
`test_self_screening_js_is_independent_of_receive_gain` 与
`test_self_screening_js_is_wavelength_agnostic`。

### 4.2 Pd 用 SINR 口径

PROJECT.md §6.1 只给了 `Pd = Pfa**(1/(1+SNR))`，没定义 SNR 是否含干扰。
实现取 `SNR = Pr/(Pn + Pj_rx)`：干扰压制 = 分母变大 = 检测概率下降，
这才是"干扰真的降低检测能力"的物理。纯热噪声 SNR 仍可单独取用（`snr_thermal`）。

## 5. 数值方法

| 问题 | 方法 | 参数 |
|---|---|---|
| JS(R) = 目标值 → R | 对数空间二分，只用 `js_ratio_self_screen` / `js_ratio_standoff` 的取值 | `rtol=1e-10`（对数区间宽度），区间 `[1 m, 1e9 m]`，未跨过解则抛 ValueError |
| Pd(R) = 门槛 → R | **闭式解**（§3.5），不迭代 | — |

`range_at_js` 保留二分是故意的：它是唯一"不信任解析式"的路径，
所以能对 `burn_through_range` 做独立交叉验证（见 §6 的实测误差）。
检测距离则不需要二分 —— 它的验收标准是"代回去 Pd = 门槛"，测试直接查这个。

## 6. 精度验收（PROJECT.md §6.3）

要求：`R_bt` 数值解与解析解误差 < 1%；`R_bt` 处 J/S 落在 1.00 ± 0.01。

实测（三组量级不同的参数，见 `tests/test_radar.py`）：

| 场景 | R_bt 解析 | 相对误差 | JS(R_bt) |
|---|---|---|---|
| 01_duel 原参数（强干扰） | 25.112 m | 7.3e-12 | 1.000000000000 |
| 弱干扰 + 强雷达 | 224.08 km | 2.4e-11 | 1.000000000000 |
| X 波段大孔径 | 7.96 km | 9.6e-12 | 1.000000000000 |

误差比验收线小 9 个数量级；单测里除了 1% 的宽松线，还钉了 `rel < 1e-6` 的紧线，
防止实现被改松后仍然"刚好通过"。

## 7. 假设与已知缺口

建模时明确接受的简化，改动前请先读这一节：

1. **单站、无处理增益**。没有脉压、相干积累、MTI/PD 滤波。
   后果：干扰效能被高估——等效"够用的干扰功率"偏小。
   若加 20 dB 处理增益，J/S 降 100 倍，R_bt 涨约 10 倍。
2. **Swerling I 单脉冲闭式对 SNR 很钝**：Pfa = 1e-6 时 Pd = 0.9 需要约 21 dB SNR，
   而不是经验直觉的 13 dB。这不是 bug，是闭式的形状；做检测距离估算时别按 13 dB 拍。
   推论：**"烧穿"不等于"能检测"**。01_duel 里 R_bt = 3.24 km，但干扰下 Pd = 0.5 的距离只有 0.75 km
   ——因为 J/S = 1 只是把信噪比砍一半（3 dB），而检测要的是绝对 SNR。
3. **干扰只建模了两种**：噪声压制与转发式诱饵。欺骗式干扰（假航迹）、脉冲、
   扫频干扰未建模；诱饵只做到"假回波 + 各自的检测概率"，没有航迹关联。
4. **自由空间传播**。无多径、无地形遮蔽、无大气吸收；B 只按带宽比例 `Br/Bj` 计，无频率选择性。
5. **无极化、无旁瓣**：Gj 是"朝雷达方向"的单一增益，天线方向图在 T1 才进来。
6. **单站假设**：`received_power` 接口支持 Gt ≠ Gr（为将来双站留的），
   但 `js_ratio_self_screen` / `burn_through_range` 只按单站用。
7. **参数全虚构**：所有数值都是为了让曲线可读而选的，不代表任何真实装备。
8. **检测是单次照射的抽样**：`detected` 事件只是"这一帧掷点是否 < Pd"，
   没有航迹维持、没有 M-of-N 确认。Pd = 0.9 时连续两次漏检是正常现象，
   **不能**把单次漏检当成"目标丢失"——动画里的 TRACKING/JAMMED 用的是 Pd 门槛，
   掷点只作为灰色阶梯线展示（见 docs/metrics.md §1）。
9. **运动是分段线性的航路点**：无加速度、无机动的物理约束；超出航路点范围时
   停在端点（不外推，否则 replay 里的位置会依赖仿真时长）。
10. **雷达链路仍是"一台雷达 + 一台干扰机 + 一个目标"**：多干扰机的功率叠加未建模，
    链路只按"与雷达同频的那台干扰机"结算。ESM 侧已经能观测多辐射源（04_fog），
    但融合仍是逐辐射源独立解算，没有联合关联。
11. **通信链路是单条、接收端全向**：无多径、无编码细节，抗干扰余量用
    `Gp + Mc − J/S` 的简化口径；频段判据只按矩形谱是否重叠。

## 8. 动态与回放（T1）

时间、随机、日志三件事的契约：

| 环节 | 实现 | 保证 |
|---|---|---|
| 时间 | `engine.clock.Clock`：`t = index · dt` | 不累加浮点，任意机器上时间戳一致 |
| 随机 | `np.random.default_rng(scenario.meta.seed)` | 同一 seed ⇒ 逐字节相同的 replay |
| 日志 | `replay.write_replay`：一行一事件、`sort_keys=True` | 可 diff / 可哈希 / 每行独立可解析 |
| 出图 | `viz/static/animate.py`：只读 replay | 画面数字全部来自日志，不重算物理 |

两条命令把这条链串起来：

```bash
spectrumfog sim scenarios/02_standoff.yaml --gif   # 跑仿真 + 写 replay + 出 GIF
spectrumfog animate out/02_standoff.jsonl -o x.gif # 只读 replay，重出动画
```

`02_standoff` 的实测：121 帧（dt=2 s，240 s），12 fps ⇒ 10.08 s GIF；
干扰机 60 → 4 km，检测边界 20.5 → 5.3 km，Pd 在 t = 170 s 跌破 0.5。
