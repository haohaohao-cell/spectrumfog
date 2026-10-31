# SpectrumFog / 频谱迷雾 — 项目规格书

> 本文件是项目的唯一事实来源（single source of truth）。
> 新对话开始时，先读本文件，从"里程碑"的当前档位继续。

---

## 1. 一句话定位

**一个部分可观测的电磁对抗仿真与博弈平台**：双方都只能看到被干扰污染、被聚合过的频谱态势，必须靠定位与识别反推对手的辐射源在哪、是什么、想干什么。

不是"算杀伤"的兵棋，是"算认知"的环境。

### 简历定位（对外一律用这句，不要提"游戏"和"军事"）

> 部分可观测多智能体仿真平台，含确定性回放内核、Gymnasium 接口与 Web 可视化。

---

### 项目组合（四个角色，别混为一谈）

| 角色 | 是什么 | 落在哪一档 | 说明 |
|---|---|---|---|
| **主项目** | 频谱迷雾内核 + 玩法 | T0 → T2 | 唯一的必做主线 |
| **内嵌件** | AAR 复盘分析 | T3a | 复用 replay 日志，是 T3b 的前置 |
| **门面** | 频谱瀑布（可视化） | 合并进 T1 / T2 | 不独立成项，但 T1 后必须发布 |
| **学术件** | EW-Gym 接口 | T3b（**可选分支**） | 要论文 / 进实验室才做，不是必经项 |

> 原则：**不做六个项目，做一个内核、三个面。** 上表四行是唯一允许的范围，超出即砍。

---

## 2. 已冻结的设计决策

这六条**不再讨论**，除非出现颠覆性问题：

1. **确定性优先**：固定步长 + seeded RNG + 事件日志 → 任意 replay 可逐帧复现。这是学术可信度的地基，妥协一次就全废。
2. **引擎与表现严格分离**：`core` 不 import 任何 UI 库；`viz` 只读 replay 文件，不回写。
3. **一个内核，三个面**：CLI（实验）、Web（传播）、Gym（研究）。同一份物理，不许前后端各写一套。
4. **只用公开教科书物理**（Skolnik 雷达方程、Friis、ITU-R 传播），参数全虚构。
5. **观测是核心资产**：不做"看得见/看不见"的二元判定，而做概率化测量 + 融合 + 虚假目标。这是本项目和所有竞品的分界线。
6. **单人可完成**：v1 不做联网对战，不做装备数据库，不做 3D。

---

## 3. 非目标（明确不做，防止范围膨胀）

- ❌ 真实装备参数库（既拿不到也不该有，且是最容易踩雷的部分）
- ❌ 3D 拟真渲染、粒子特效、军事 UI 皮肤
- ❌ 联网多人对战
- ❌ 完整兵棋（六角格、回合制、CRT 表）
- ❌ v1 不接 LLM
- ❌ 不做 SDR 硬件闭环（留作 v2 插件）

---

## 4. 核心循环

```
tick (默认 1s) →
  1. 双方提交动作：开机/关机、改频、改功率、改波束指向、放诱饵、机动
  2. 引擎结算物理：传播 → 接收功率 → J/S → 检测概率 → 测量值
  3. 生成观测：频率、AOA、TOA、功率，叠加噪声 + 虚假目标 + 漏检
  4. 融合定位：TDOA/AOA → 位置估计 + 协方差 → CEP
  5. 写入 replay（JSONL）
  6. 结算指标：Pd、CEP、暴露时长、识别正确率
```

胜负不用"血量"，用**多目标评分向量**：定位精度、识别正确率、自身暴露时长、链路可用率。

---

## 5. 目录结构

```
spectrumfog/
├── pyproject.toml
├── README.md
├── PROJECT.md
├── docs/
│   ├── physics.md          公式、假设、单位
│   ├── design.md           建模取舍
│   ├── scenario.md         YAML schema
│   ├── metrics.md          指标定义
│   ├── web.md              前端数据包
│   ├── bench.md            Gym 环境与评测
│   └── launch.md           发布文案
├── src/spectrumfog/
│   ├── units.py            SI 常量 + 单位换算
│   ├── physics/
│   │   ├── radar.py        雷达方程、J/S、烧穿距离
│   │   ├── comms.py        链路预算、Eb/N0、抗干扰余量
│   │   └── propagation.py  自由空间 + 可选绕射
│   ├── model/
│   │   ├── platform.py · emitter.py · sensor.py
│   │   └── scenario.py     pydantic 模型 + YAML 加载
│   ├── engine/
│   │   ├── clock.py
│   │   ├── detect.py       检测概率、测量生成
│   │   ├── fusion.py       TDOA/AOA 定位、协方差、CEP
│   │   └── sim.py          主循环
│   ├── replay.py           JSONL 读写
│   ├── metrics.py
│   └── cli.py
├── viz/
│   ├── static/             matplotlib 出图（T0/T1）
│   └── web/                Vite + React + deck.gl（T2）
├── scenarios/
│   ├── 01_duel.yaml        单雷达 vs 单干扰机（自屏蔽，静态）
│   ├── 02_standoff.yaml    站距干扰 + 移动干扰机（T1 动画想定）
│   ├── 03_triad.yaml       雷达 + 通信链路 + 双频段干扰机
│   ├── 04_fog.yaml         三辐射源 + 虚假目标
│   ├── 05_mobile.yaml       雷达 / 干扰机 / 目标三方机动
│   └── 06_decoy.yaml        转发式诱饵（等效 RCS 6.3 m²）
├── bench/                  T3：baseline 策略 + 评测脚本
└── tests/
```

---

## 6. 物理内核规格

### 6.1 必implement的公式

```python
# 雷达回波功率
Pr = Pt * G**2 * lam**2 * sigma / ((4*pi)**3 * R**4 * L)

# 噪声底
Pn = k * T0 * B * F                      # k=1.380649e-23, T0=290

# 干扰进入接收机的功率（噪声压制）
Pj_rx = Pj * Gj * Gr * lam**2 / ((4*pi)**2 * Rj**2) * (Br / Bj)

# 干信比
JS = Pj_rx / Pr

# 自屏蔽干扰的烧穿距离（JS = 1 时解出；Gr 与 lam 在两条支路里相消，故只有 G¹）
R_bt = sqrt(Pt * G * sigma * Bj / (4*pi * Pj * Gj * Br * L))

# 检测概率（Swerling I 闭式，v1 用这个，够用）
Pd = Pfa ** (1 / (1 + SNR))

# 链路预算
Prx = Ptx * Gtx * Grx * (lam / (4*pi*d))**2
EbN0 = Prx / (N0 * Rb)
AJ_margin_dB = Gp + Mc - required_JS_dB     # Gp=处理增益, Mc=编码增益
```

> 修正记录（2026-10-27）：`R_bt` 原写 `G**2`，与本节的 `Pr` / `Pj_rx` 推导矛盾
> （Gr 在两条支路里相消），已改为 `G`。推导与单测见 `docs/physics.md` §4.1。
> `Pd` 的 `SNR` 取 SINR 口径：`SNR = Pr / (Pn + Pj_rx)`，见同文件 §4.2。

### 6.2 单位约定

**内部一律 SI**：W、m、Hz、s、rad。只在 CLI 输出和 YAML 里允许 dB / km / MHz，加载时立刻转换。

| 量 | 内部单位 | 允许的输入单位 |
|---|---|---|
| 功率 | W | W / dBm / dBW |
| 长度 | m | m / km / nmi |
| 频率 | Hz | Hz / kHz / MHz / GHz |
| 时间 | s | s / ms |

### 6.3 精度验收

`R_bt` 数值解与解析解误差 **< 1%**；`R_bt` 处 J/S 必须落在 `1.00 ± 0.01`。这两个断言写进单测。

---

## 7. 数据格式

### 7.1 想定 YAML

```yaml
meta:
  name: 01_duel
  seed: 42
  dt: 1.0              # s
  duration: 600        # s

environment:
  noise_figure_db: 3.0
  temperature_k: 290

platforms:
  - id: radar_a
    side: blue
    pos: [0, 0]
    emitters:
      - kind: radar
        freq_hz: 3.0e9
        bw_hz: 2.0e6
        pt_w: 1.0e5
        gain_db: 35
        sigma_m2: 1.0
        loss_db: 3
  - id: jammer_b
    side: red
    pos: [30000, 0]
    emitters:
      - kind: noise_jammer
        freq_hz: 3.0e9
        bw_hz: 20.0e6
        pj_w: 2000
        gain_db: 20

sensors: []            # v1 只有 ESM，见 scenario.md

objectives:
  - {type: localize, target: jammer_b, cep_m: 500}
```

### 7.2 回放 JSONL

一行一个事件，**必须可独立解析**，不许依赖前一行：

```json
{"t":12.0,"kind":"detection","sensor":"esm_1","emitter":"radar_a","pd":0.83,"measured":{"freq_hz":3.0e9,"aoa_rad":0.31,"power_dbm":-72.1}}
{"t":12.0,"kind":"estimate","target":"jammer_b","pos":[30120,880],"cep_m":410,"n_measurements":3}
{"t":12.0,"kind":"state","emitter":"jammer_b","on":true,"freq_hz":3.0e9,"pj_w":2000}
```

---

## 8. 指标定义（写进 docs/metrics.md，全项目统一）

| 指标 | 定义 | 用途 |
|---|---|---|
| Pd | 单次照射检测概率 | 物理正确性 |
| CEP50 / CEP90 | 定位误差圆半径（50%/90% 分位） | 核心玩法指标 |
| T_expose | 己方辐射源被对方有效定位的持续时间 | 生存性 |
| Link_avail | 链路可用时间占比（AJ margin > 0） | 通信抗扰 |
| ID_acc | 辐射源型号识别正确率 | 认知维度 |
| Score | 上述加权向量，权重写进想定 | 排行榜 |

---

## 9. 里程碑与验收标准

总工期：**6–7 周**（T0–T3a 约 2.5 周，T3b 另加 4–5 周）。
比最初的"四周路线"长，是因为 T3 被加厚、且补回了 AAR。**这是范围决定的，不是执行变慢。**

### T0 — 物理闭环（半天，已完成）

产出：`physics/` + `model/scenario.py` + `cli.py` + `scenarios/01_duel.yaml` + `viz/static/plot.py` + 5 个单测。

**验收**：
- `spectrumfog run scenarios/01_duel.yaml` 一条命令出三张图（烧穿曲线、Pd–距离、J/S–距离）
- J/S=1 时 R 与解析解误差 < 1%
- 至少一张图可以直接当 README 头图

### T1 — 2D 动态 + 首次发布（+1–2 天，已完成）

产出：俯视动画，干扰机移动 → 烧穿半径收缩 → 雷达失效。

**验收**：导出一段 10 秒 GIF/MP4，不需要旁白就能看懂。

**🚩 发布闸门：T1 一结束就公开仓库、发 README、发渠道，不等 T2。**
这是"门面"真正的价值——不是多一个独立项目，是**早一次发布**。

### T2 — Web 交互（+5–8 天，已完成）

产出：Vite + React + deck.gl，GitHub Pages 静态部署。

**验收**：打开网页可拖动干扰机，热力图与曲线 < 100ms 更新；无后端。

### T3a — AAR 复盘分析（+3–4 天，已完成）

产出：`replay.jsonl` → 关键事件时间线、暴露时刻、定位误差归因、单页 HTML 报告。

**验收**：对 `01_duel` 自动生成"你的辐射源在第 12 秒被定位，原因是 X"。

**为什么排在 T3b 前面**：没有归因的排行榜只是一个分数列表；有了 AAR，它才是训练工具。顺序不能反。

### T3b — 平台化（+4–5 周，**可选分支**，已完成）

产出：Gymnasium 接口、3 个 baseline 策略、评测脚本、排行榜、5 个想定、文档站。

**验收**：`pip install -e . && spectrumfog bench --all` 三条命令跑完，产出可比表格。

**分支决策点**：若不打算写论文 / 进实验室，**停在这里即可**——T0–T3a 完成时项目已经完整。

---

## 10. 技术栈

```
语言     Python 3.11+
依赖     pydantic · pyyaml · numpy · matplotlib · pytest
可选     gymnasium · networkx · pint
前端     Vite + React + TypeScript + canvas 自绘（deck.gl 留到 T3b 的大规模轨迹）
部署     GitHub Pages（纯静态）/ PyPI（库）
协议     Apache-2.0（代码）+ CC-BY-4.0（文档与想定）
```

**性能目标**：10³ 平台 × 10⁴ tick 在单机 60 秒内跑完（numpy 向量化，别写 Python 循环）。

---

## 11. 发布计划

star 主要来自发布，不是来自代码量。

**发布闸门：T1 结束就发。** 有 GIF、有仓库就发，不等 T2 做完。这是"门面"真正的价值——不是多一个独立项目，是**早一次发布**，心理上和风险上都比憋到最后健康。

1. **README 三段式**：一句话 + GIF + 一条命令跑起来。第一屏必须有一张会动的图。
2. **渠道**：r/computerwargames、r/RadioFrequency、Hacker News（"Show HN"）、B站/知乎（中文长文，讲"看不见的战争怎么建模"）。
3. **可信度**：一篇 `docs/design.md` 讲建模取舍，比十篇 README 有用。
4. **进课堂**：找一门课用它做实验，比追 star 有效。

---

## 12. 合规红线（写进 README 顶部）

- 仅使用公开教科书物理公式与**虚构参数**，不包含任何真实装备性能数据、番号或部署信息。
- 定位为 educational / research simulation，非作战软件，不提供战术建议。
- 命名与文案避免"作战/打击/杀伤"等表述，统一用 simulation / analysis / wargame（学术义）。

---

## 13. 下一个对话的开场指令（直接复制粘贴）

```
读 PROJECT.md。T0–T3b 已全部完成，项目处于"可发布、可扩展"状态。
先跑一遍确认没坏：
1. python -m pytest -q                 （应 182 passed）
2. spectrumfog bench --suite scenarios/suite_v1.yaml   （应 18 行，baselines 得分约 45.8）
3. python tools/build_docs.py --check   （文档站产物应与源码一致）

接下来按 §11 做渠道投放（文案在 docs/launch.md），
或从 §3 的非目标清单之外挑新方向（多智能体、假航迹关联、Gym 向量化）。
```

---

## 14. 当前状态

- [x] 选题与规格确定
- [x] 路线对齐（四角色 × T0–T3b）
- [x] T0 物理闭环（2026-10-27：`spectrumfog run scenarios/01_duel.yaml` 出三图，56 项单测）
- [x] T1 2D 动态 + **首次发布**（2026-10-27：`sim --gif` 出 10 s 动画 + 确定性 replay；仓库已公开）
- [x] T2 Web 交互（2026-10-27：`export-web` 数据包 + Vite/React 交互页 + Pages 上线）
- [x] T3a AAR 复盘分析（2026-10-27：`spectrumfog aar` 出时间线 + 暴露时长 + 留一法归因 + 单页 HTML）
- [x] T3b 平台化（2026-10-27：Gymnasium 环境 + 3 个 baseline + 6 个想定 + 文档站 + 提交物校验 + 排行榜）
- [x] 后续增量 A（2026-10-31）：处理增益 `processing_gain_db`（默认关，`07_gain` 用 40 dB 做对照）+ 提交物改按内容指纹校验
- [x] 后续增量 B（2026-10-31）：航迹层（卡尔曼 + 马氏门限 + 米制半径关联 + 起批/确认/滑行/丢失），AAR 增加航迹时间线
- [x] 后续增量 C（2026-10-31）：假航迹——雷达航迹层 + 诱饵距离欺骗（RGPO），`false_track_fraction` 进入摘要与 AAR
- [x] 后续增量 D（2026-10-31）：双人对局 `08_duel_2p`——蓝方雷达 EMCON 三档动作，暴露按确认跟踪判定，`bench/duel.py` 含 3×3 baseline 矩阵

### 交付索引

| 东西 | 在哪 |
|---|---|
| 交互演示（拖干扰机） | <https://haohaohao-cell.github.io/spectrumfog/> |
| 文档站（9 页） | <https://haohaohao-cell.github.io/spectrumfog/docs/> |
| 排行榜 | <https://haohaohao-cell.github.io/spectrumfog/docs/leaderboard.html> |
| 仓库 / 发布 | <https://github.com/haohaohao-cell/spectrumfog> · v0.2.0 |
| 想定集合（排行榜基准） | `scenarios/suite_v1.yaml`（6 想定 × 5 种子 × 3 策略） |
| 测试 | 182 项 Python + 16 项前端，CI 全绿 |
| 发布文案 | `docs/launch.md`（五个渠道） |
