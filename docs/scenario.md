# 想定 YAML schema（v1 · T0）

> 实现：`spectrumfog/model/scenario.py`（pydantic v2）。加载即校验，`extra="forbid"`，
> 拼错的字段会直接报错而不是被静默忽略。

## 1. 完整示例

```yaml
meta:
  name: 01_duel          # 必填，用于出图文件名
  seed: 42               # 默认 42，决定 RNG（确定性优先，§2）
  dt: 1.0                # s，固定步长，默认 1.0
  duration: 600          # s，默认 600

environment:
  noise_figure_db: 3.0   # dB，默认 3.0；加载后存线性
  temperature_k: 290     # K，默认 290

platforms:               # 至少一个
  - id: radar_a          # 必填，全想定唯一
    side: blue           # blue | red，必填
    pos: [0, 0]          # m，[east, north]，默认 [0, 0]
    emitters:            # 可为空（静默平台）
      - kind: radar
        freq_hz: 3.0e9
        bw_hz: 2.0e6
        pt_w: 1.5e5
        gain_db: 35
        sigma_m2: 1.0
        loss_db: 3

  - id: jammer_b
    side: red
    pos: [30000, 0]
    emitters:
      - kind: noise_jammer
        freq_hz: 3.0e9
        bw_hz: 1.0e8
        pj_w: 90.0
        gain_db: 0

sensors: []              # v1 只有 ESM，见下方字段表

objectives:              # 可空；target 必须指向存在的平台
  - {type: localize, target: jammer_b, cep_m: 500}
```

## 2. 字段表

### meta

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | 想定名，也是出图文件名的前缀 |
| `seed` | int ≥ 0 | 42 | RNG 种子；同一 seed 必须复现同一 replay |
| `dt` | float > 0 | 1.0 | 固定步长 [s] |
| `duration` | float > 0 | 600 | 仿真时长 [s] |

### environment

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `noise_figure_db` | float | 3.0 | 接收机噪声系数 [dB]，加载后存线性 |
| `temperature_k` | float > 0 | 290 | 噪声温度 [K] |

### platforms[]

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `id` | str | 必填 | 全想定唯一 |
| `side` | `blue` \| `red` | 必填 | 阵营 |
| `pos` | `[float, float]` | `[0, 0]` | `[east, north]`，米；也是 t=0 的位置 |
| `waypoints` | list | `[]` | 航路点（T1），见下 |
| `emitters` | list | `[]` | 辐射源，见下 |

### platforms[].waypoints[]（T1）

```yaml
waypoints:
  - {t: 0, pos: [60000, 0]}     # t=0 在 60 km
  - {t: 240, pos: [4000, 0]}    # t=240 s 压到 4 km
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `t` | float ≥ 0 | 时刻 [s] |
| `pos` | `[float, float]` | 该时刻的位置 [m] |

语义（纯确定性，无随机、无加速度）：

- 相邻航路点之间**分段线性插值**；重复时刻直接报错。
- 早于第一个航路点 → 停在第一点；晚于最后一个 → 停在最后一点
  （不做外推，否则 replay 里的位置会依赖仿真时长）。
- 空 `waypoints` = 静止在 `pos`。
- 位置完全由时间决定，所以同一 tick 在同一 seed 下永远落在同一坐标。

### emitters[] — `kind: radar`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `freq_hz` | float > 0 | 必填 | 中心频率 [Hz] |
| `bw_hz` | float > 0 | 必填 | 接收带宽 [Hz]，即雷达方程里的 Br |
| `pt_w` | float > 0 | 必填 | 峰值发射功率 [W] |
| `gain_db` | float | 必填 | 天线增益 [dBi]，单站收发共用 |
| `sigma_m2` | float > 0 | 必填 | 目标 RCS [m²]（虚构值） |
| `loss_db` | float ≥ 0 | 0 | 系统损耗 [dB] |

### emitters[] — `kind: noise_jammer`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `freq_hz` | float > 0 | 必填 | 中心频率 [Hz] |
| `bw_hz` | float > 0 | 必填 | 噪声带宽 [Hz]，即 Bj；只有 Br/Bj 的份额进接收机 |
| `pj_w` | float > 0 | 必填 | 发射功率 [W] |
| `gain_db` | float | 必填 | 天线在雷达方向上的增益 [dBi] |

### sensors[]

ESM 传感器从 T3a 起真正参与解算：每 tick 只看**对方阵营**的辐射源，产出带噪声的
AOA / TOA / 频率 / 功率，交给 `engine/fusion.py` 定位。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `id` | str | 必填 | 唯一，会出现在 replay 的 `measurement.sensor` 里 |
| `kind` | `esm` \| `radar` | `esm` | 现在只有 `esm` 参与解算 |
| `side` | `blue` \| `red` | `blue` | 决定它观察哪一方 |
| `pos` | `[float, float]` | `[0, 0]` | 米 |
| `noise_figure_db` | float | 6.0 | dB；直接决定灵敏度 |
| `bw_hz` | float > 0 | 1e6 | 瞬时带宽 [Hz]；窄带宽 = 更灵敏 |
| `gain_db` | float | 0.0 | 天线增益 [dBi]，v1 按全向 |
| `pfa_scan` | float ∈ (0,1) | 1e-4 | 单次扫描虚警率，虚假目标的来源 |

**灵敏度是调出来的，不是猜的**：`02_standoff` 里的干扰机只有 1 W，
用 1 MHz / 12 dB 的 ESM 在 60 km 处 SNR 只有 0.3（Pd ≈ 1e-5，什么都收不到），
所以那份想定改成 100 kHz / 6 dB。想定里的参数是可解释的虚构值，不是真实装备。

### objectives[]

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `type` | `localize` \| `detect` \| `protect_link` | 必填 | 验收类型 |
| `target` | str | 必填 | 平台 id，必须存在 |
| `cep_m` | float > 0 | null | 定位精度门槛 [m] |

两种 type 的含义**不同**，别混：

* `detect`：雷达要探测这个平台（T1 动画用它挑目标，写进 `meta.target`）；
* `localize`：对方 ESM 要定位这个辐射源（AAR 用它挑对象，写进 `meta.localize_target`，
  门槛 `cep_m` 写进 `meta.cep_threshold_m`）。

## 3. 单位后缀（加载时换算，§6.2）

| 量 | 可用写法 |
|---|---|
| 频率 / 带宽 | `freq_hz` `freq_khz` `freq_mhz` `freq_ghz`（`bw_*` 同理） |
| 功率 | `pt_w` `pt_kw` `pt_mw` `pt_dbm` `pt_dbw`（`pj_*` 同理） |
| 位置 | `pos`（米）`pos_km` `pos_nmi` |
| 时间 | `dt` / `dt_ms`，`duration` / `duration_ms` |

同一个量给了两种单位（例如同时写 `freq_hz` 和 `freq_mhz`）会**报错**，不会静默挑一个。

## 4. 校验规则（都会在加载时抛 `ValueError`）

- 至少一个平台；平台 id、传感器 id 各自唯一。
- `objectives[].target` 必须指向存在的平台。
- 未知字段（拼写错误）直接拒绝：`extra="forbid"`。
- 功率、带宽、频率、RCS、损耗必须为正（损耗可以为 0）。
- 未知 `kind` 报错（pydantic 判别联合）。
- 想定对象构造后不可变（`frozen`），防止运行期被引擎改写。

## 5. 与后续里程碑的关系

- **T1**：`dt` / `duration` / `seed` 才真正被主循环使用（固定步长 + 事件日志）。
- **T1+**：平台移动（`waypoints` / `velocity`）、波束指向、诱饵字段进来，
  届时本文档与 `docs/scenario.md` 的字段表同步更新。
- **T3b**：`objectives` 的 `cep_m` 等门槛会变成 `metrics.py` 的评分输入。
