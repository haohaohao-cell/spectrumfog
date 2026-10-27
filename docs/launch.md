# 发布文案（launch kit）

> 五个渠道、五种切入角度，不要一稿多投。
> 所有文案都已按 [PROJECT.md](../PROJECT.md) §12 调过措辞：不出现"作战/打击/杀伤"，
> 不暗示任何真实装备，主动交代模型缺口。
> 代码与数据全部是虚构参数，公开可查。

## 0. 发布顺序与时机

| # | 渠道 | 语言 | 建议时机（北京时间） | 目的 |
|---|---|---|---|---|
| 1 | Show HN | 英文 | 周二–周四 20:00–22:00 | 拿技术反馈 + 星 |
| 2 | r/computerwargames | 英文 | 同上（美东上午） | 找懂兵棋的人聊设计 |
| 3 | r/RadioFrequency | 英文 | 隔 1–2 天 | 让懂射频的人挑物理毛病 |
| 4 | 知乎 | 中文长文 | 工作日 20:00–22:00 | 中文深度读者、简历可见度 |
| 5 | B站 | 中文 | 工作日 19:00–22:00 | 用 GIF/视频做传播 |

原则：

* **先 HN 再 Reddit**：HN 的评论区能帮你把"该被问倒的问题"先过一遍。
* **r/RadioFrequency 放最后**：那一轮大概率会指出模型简化，先让前两轮把明显的错挑完。
* 每个渠道发完**至少守 2 小时评论区**。HN 前 2 小时决定它能不能上首页。

---

## 1. Show HN

**标题**（66 字符，HN 标题上限 80）

```
Show HN: SpectrumFog – deterministic radar and jamming simulation
```

**正文**（贴进 text 框，纯文本，不要 markdown 加粗）

```
I built a small simulation core for spectrum scenarios where the interesting part
isn't the outcome, but the picture each side can build from a jammed spectrum.

What's actually there:

- Textbook physics with exactly one implementation: radar equation, J/S (both
  self-screening and standoff), burn-through range, Swerling I detection
  probability, and a closed-form detection range. SI units internally; dB, km
  and MHz exist only at the YAML and terminal boundary.

- Deterministic replay. Fixed-step clock where time is computed as index*dt
  (never accumulated), one seeded RNG, and a JSONL log where every line is a
  self-contained event. Same scenario plus same seed gives byte-identical logs;
  there is a test that asserts exactly that.

- A static web page where dragging the jammer recomputes a coverage heatmap.
  The page contains no physics and no backend: Python precomputes a
  (target range, jammer range) table and the browser interpolates it. That was
  the only way I could keep "one implementation of the physics" and "no server"
  both true.

- An after-action report. The engine emits ESM observations (angle of arrival,
  time of arrival, power, misses, false targets), fuses them into a position fix,
  and the report names which sensors made the fix possible - by re-running the
  fusion with each one removed. The sentence it produces is "your emitter was
  effectively localised at t = N", not "you lost".

- A gymnasium environment around the same physics: three baselines, a versioned
  scenario set, and submissions that carry a sha256 per scenario file, so a
  leaderboard can be checked without a server.

Deliberate omissions: no processing gain (so the jammer powers look small - 1 W
spread over 100 MHz against a 2 MHz receiver), no multipath or terrain, and no
track state machine. The radar link handles one jammer at a time; ESM sees
several emitters. The gap list is in docs/physics.md.

Two things the model taught me:

1) Burn-through is not detectability. J/S = 1 only halves the SNR, while this
   Swerling I closed form needs about 20 dB for Pd = 0.5. A target can sit inside
   the burn-through ring and still be undetectable - in the duel scenario the
   two radii differ by a factor of four.

2) With a standoff jammer, the coverage ring shrinks as the square root of the
   jammer's range. Halving the standoff range multiplies the ring radius by 0.7.

Repo: https://github.com/haohaohao-cell/spectrumfog
Demo: https://haohaohao-cell.github.io/spectrumfog/
Docs: https://haohaohao-cell.github.io/spectrumfog/docs/
Leaderboard: https://haohaohao-cell.github.io/spectrumfog/docs/leaderboard.html

I would rather fix the model than the copy, so critique of the physics is the
most useful feedback.
```

**发布后自己补一条评论**（HN 惯例：主贴不宜太长，细节放评论）

```
A few implementation notes that didn't fit above:

- The closed-form detection range was worth the detour. The first version
  bisected Pd(R) per tick, which cost 14 s for a 600-tick scenario. Solving
  Pd(R) = threshold analytically (it's a quartic in R, or a quadratic in R^2 for
  self-screening) brought it to 1.3 s. The coefficients come from evaluating the
  same power functions at 1 m, so there is still only one physics
  implementation.

- The quadratic needed the conjugate form. Writing (-b + sqrt(b^2 + 4ac)) / 2a
  directly produced a negative root because Pn is tiny compared to the jamming
  term; 2A / (sqrt(b^2 + 4ac) + b) fixed it.

- The spec originally had R_bt proportional to G^2. Deriving it from the same
  section's Pr and Pj_rx shows the receive gain cancels, so it is G, not G^2 -
  a factor of sqrt(G), about 56x at 35 dB. I changed the formula, wrote down the
  derivation, and added two regression tests (J/S independent of receive gain,
  and independent of wavelength).
```

---

## 2. r/computerwargames

**标题**

```
A partially-observable EW sim where both sides only see a jammed spectrum picture
```

**正文**

```markdown
Most wargames resolve to hit/miss. I wanted to model the part before that: what
each side can actually *know* about the other's emitters.

So the output of a tick isn't damage, it's a set of probabilistic measurements
(frequency, angle of arrival, time of arrival, power) with misses and false
targets, which then get fused into a position estimate and a covariance.

What's working today (v0.2, Python + a static web page):

- Radar equation, jamming-to-signal ratio, burn-through range, Swerling I
  detection probability. Fictional parameters only.
- A deterministic engine: fixed step, seeded RNG, JSONL replay where one line is
  one self-contained event. Same scenario + seed reproduces byte-identical logs.
  That's the part I care most about: without it, "why did I lose" is unanswerable
  and any AAR is fiction.
- Six scenarios: a duel (single radar vs a self-screening jammer); a standoff case
  where a jammer closes from 60 km to 4 km and the radar's tracking boundary
  collapses from 20.5 km to 5.3 km (this one is the 10 s GIF); a radar + comms
  link + two-band jamming case where the link availability drops to 31%; a
  three-emitter case with a 5e-3 per-scan false alarm rate; a case where radar,
  jammer and target all move; and a relay decoy whose echo is 325x the real
  target's at 70 km.

What I deliberately left out: track state machines (M-of-N), treating a decoy as
a *false track* rather than a bright false echo, and terrain. Those are the next
milestones, and I'd rather ship a small honest core than a big vague one.

The thing I'm least sure about is the observation model: right now a "detection"
is a single-scan Bernoulli draw at Pd, with no persistence. Real sensors hold
tracks. I'm planning an M-of-N track layer, but I'd be interested in how people
here handle the gap between "detection" and "track" in their own models.

Demo (drag the jammer): https://haohaohao-cell.github.io/spectrumfog/
Code: https://github.com/haohaohao-cell/spectrumfog
Design notes: https://github.com/haohaohao-cell/spectrumfog/blob/main/docs/design.md
Docs: https://haohaohao-cell.github.io/spectrumfog/docs/
Leaderboard: https://haohaohao-cell.github.io/spectrumfog/docs/leaderboard.html
```

---

## 3. r/RadioFrequency

> 这一版的核心是**请人挑错**。别写成推广，写成求验证。

**标题**

```
Sanity-checking a noise-jamming model: burn-through vs detection range
```

**正文**

```markdown
I've been building a small open simulation (fictional parameters, no real
equipment data) and I'd like the RF crowd to tell me where the model is wrong.

The chain is the textbook one:

    Pr    = Pt·G²·λ²·σ / ((4π)³·R⁴·L)
    Pj_rx = Pj·Gj·Gr·λ² / ((4π)²·Rj²) · (Br/Bj)
    J/S   = Pj_rx / Pr
    R_bt  = sqrt(Pt·G·σ·Bj / (4π·Pj·Gj·Br·L))     # self-screening, J/S = 1
    Pd    = Pfa^(1/(1+SNR)),  SNR = Pr / (Pn + Pj_rx)

With a 150 kW / 35 dBi S-band radar, 2 MHz receiver, 3 dB loss, 1 m² target,
and a 90 W noise jammer (0 dBi, 100 MHz bandwidth, so about 1.8 W in-band):

- burn-through range (J/S = 1): 3.2 km
- unjammed detection range (Pd = 0.5, Pfa = 1e-6): 59.5 km
- jammed detection range: 0.75 km
- at 30 km standoff, J/S = 85.6 (19.3 dB), Pd = 1.2e-6

Two things I want to be told are wrong:

1) The jammer power story. There is no processing gain in this model, so 90 W of
   noise spreads into 100 MHz and only ~1.8 W reaches a 2 MHz receiver - and that
   is enough to deny a 150 kW radar at 30 km. Is the arithmetic self-consistent
   given that omission, or is there a second error hiding behind it?

2) Burn-through ≠ detectability. J/S = 1 only halves the SNR, and this Swerling I
   closed form needs ~20 dB for Pd = 0.5, so the detectability radius ends up
   about 4x smaller than the J/S = 1 radius. Is that ratio plausible, or is it an
   artifact of using the Swerling I closed form instead of Marcum Q?

Everything is in SI internally and the gap list is public, so if a number looks
wrong, it probably is - I'd rather fix it now.

Code: https://github.com/haohaohao-cell/spectrumfog
Physics write-up: https://github.com/haohaohao-cell/spectrumfog/blob/main/docs/physics.md
```

---

## 4. 知乎（长文）

标题（避开"战争"字样，避免与合规条款擦边；你若坚持可用 §11 的原句）：

```
看不见的电磁环境，怎么建模？—— 一个部分可观测的频谱仿真
```

结构（正文约 2500–3500 字，配图 3 张）：

1. **开头：一个反直觉的数字**
   两台设备相距 30 km，一台功率 150 kW 的雷达看不见目标，因为 90 W 的干扰机在中间。
   这不是段子，是模型算出来的。引出"看不见"这件事本身值得建模。

2. **为什么"命中"不是好终点**
   兵棋结算到命中/未命中，等于假设双方都看得见。真实情况是：双方都在猜。
   把观测（频率、AOA、TOA、功率 + 噪声 + 漏检 + 虚假目标）当核心资产，
   是这个项目和别的东西的分界线。

3. **三层拆解：物理 / 观测 / 认知**
   - 物理层：雷达方程、J/S、烧穿距离、Swerling I —— 全是公开教科书公式，参数全虚构。
   - 观测层：概率化测量，不是"看得见/看不见"的二元判定。
   - 认知层：融合定位、协方差、CEP，以及"你以为的位置"和"真实位置"的差。

4. **三个数字，和它们各自骗人的地方**
   - J/S = 1 不代表能检测（只砍掉 3 dB，Swerling I 要 20 dB）——配 `01_duel` 的 Pd–距离图。
   - 干扰机逼近，边界按 √Rj 收缩（60 km → 4 km，边界 20.5 → 5.3 km）——配 GIF。
   - 没有处理增益时，1 W 干扰机就能压制——这是模型缺口，不是战果。

5. **确定性 replay 为什么是地基**
   固定步长（`t = index·dt`，不累加）+ 单一 seed + 一行一事件的 JSONL。
   同一想定同一 seed，`sha256` 完全一致。没有这一条，"为什么输"永远说不清，
   复盘工具也只能是编故事。

6. **一个前端不写物理的网站**
   T2 要求"无后端"，又要求"同一份物理"。解法是 Python 预编译查表，
   浏览器只插值。代价是数据包 126 KB、误差几 %；收益是网页和 CLI 的数不可能对不上。

7. **我刻意没做的五件事**
   真实装备库、多径地形、假航迹关联、航迹状态机、3D 与联网。
   （诱饵已经做了，但它只到"假回波"——没有让雷达锁上一条假航迹。）
   附上 `docs/physics.md` §7 的缺口清单链接。

8. **结尾**
   仓库 + 文档 + 在线试用 + 一句话邀请："如果你想拿它做课程实验、或者需要一个能逐行审的对比基线，欢迎聊。"

**开头三段（可直接用）**

> 先说一个反直觉的数字。
>
> 一台 150 kW、35 dBi 的 S 波段雷达，对一架 1 m² 的目标，无干扰时能在 59 km 外发现它。
> 但只要在 30 km 处放一台 90 W 的噪声干扰机，这台雷达对同一目标就基本失明了——
> 此时 J/S ≈ 85，检测概率 1e-6 量级。
>
> 这个数字来自一个我自己写的仿真内核，参数全是虚构的。但它说明了一件事：
> 电磁环境里，"看得见"是算出来的，不是给定的。

---

## 5. B站

**标题**（B 站可长一些）

```
【仿真】干扰机逼近时，雷达的"视野"是怎么塌掉的｜做了个部分可观测的电磁仿真
```

**视频简介**

```
一个自己写的电磁仿真：雷达 + 干扰机 + 目标，全部虚构参数。

干扰机从 60 km 压到 4 km，雷达的跟踪边界从 20.5 km 缩到 5.3 km，
12 km 处的目标在 t = 170 s 丢失。右侧是检测概率和干信比曲线。

几个关键设计：
· 物理只有一份（Python），网页只做查表插值，所以网页和命令行的数字不会对不上
· 确定性 replay：同一想定 + 同一 seed，日志逐字节一致
· 明确不做：真实装备参数、多径地形、假航迹（诱饵只做到假回波）

仓库：github.com/haohaohao-cell/spectrumfog
在线试用（拖动干扰机）：haohaohao-cell.github.io/spectrumfog
物理与缺口：docs/physics.md

#仿真 #雷达 #信号处理 #开源 #Python
```

**动态文案**（配 GIF，控制在 120 字内）

```
干扰机从 60 km 压到 4 km，雷达的"看得见的圈"从 20.5 km 缩到 5.3 km。
物理全在 Python 里，网页只做查表——所以两边数字不会打架。
在线可拖：haohaohao-cell.github.io/spectrumfog
```

**置顶评论**

```
几个可能被问到的点，先答了：
1. 为什么 1 W 干扰机就能压制？因为这个模型没有处理增益（脉压、相干积累），
   落进 2 MHz 接收机的只有 20 mW。这是已知缺口，不是"战果"。
2. 为什么数据这么整齐？参数全是虚构的，只为了让曲线可读。
3. 代码怎么读？docs/physics.md 有公式和缺口清单，docs/design.md 讲取舍。
```

---

## 6. 复用素材

**一句话**

> 一个部分可观测的电磁仿真：双方都只能看到被干扰污染的频谱，必须靠定位与识别反推对手在哪。

**三句话**

> 物理用公开教科书公式、参数全虚构；引擎固定步长 + 种子随机，replay 逐字节可复现。
> 前端不写物理，Python 预编译查表、浏览器只插值，所以网页和 CLI 的数字不会打架。
> 明确不做真实装备参数、多径地形、欺骗干扰——缺口清单公开写在 docs/physics.md。

**数字卡**（发图时用）

| 数 | 含义 |
|---|---|
| 59.5 km | 无干扰时 Pd = 0.5 的检测距离 |
| 3.24 km | J/S = 1 的烧穿距离（同一组参数） |
| 0.75 km | 干扰下 Pd = 0.5 的距离——比烧穿半径小 4 倍 |
| 85.6 | 30 km 处目标受到的 J/S（19.3 dB） |
| 6 | 想定数量（含通信链路、多辐射源、诱饵） |
| 325× | 诱饵回波 / 70 km 处真目标回波 |
| 45.76 | 三个 baseline 在排行榜上的得分 |
| 182 + 16 | Python / 前端单测数 |
| 0 | 后端进程数 |

**图片素材**

| 文件 | 用途 |
|---|---|
| `docs/images/02_standoff.gif` | HN/Reddit 首图、B站封面动图 |
| `docs/images/web_explorer.png` | 知乎第 6 节、Reddit 附图 |
| `docs/images/01_duel_pd_vs_range.png` | 知乎第 4 节"J/S=1 不代表能检测" |
| `docs/images/social_preview.png` | 链接卡片（需手动上传到 GitHub Settings） |

---

## 7. 发布前合规清单（逐条核）

- [ ] 文案里没有"作战/打击/杀伤/命中率"这类表述
- [ ] 没有出现任何真实装备型号、番号、部署地点
- [ ] 明确写了"参数为虚构值"
- [ ] 明确写了"educational / research simulation，不提供战术建议"
- [ ] 缺口（无处理增益、无多径、无航迹）在正文里主动交代，不是等人问
- [ ] 许可证与引用信息在仓库可见（Apache-2.0 + CC BY 4.0 + CITATION.cff）
- [x] GitHub 仓库的 Social preview 已上传（2026-10-27）
- [x] 数字与链接按 v0.2.0 核对（2026-10-27：6 想定 / 182+16 测试 / 三处线上地址）

---

## 8. 评论区应答口径

| 会被问 | 怎么答（别防御，给数） |
|---|---|
| "这不是军事软件吗？" | 定位是 educational / research simulation；参数全虚构，不含任何真实装备数据；Apache-2.0 公开可审。 |
| "1 W 干扰机就能压制 150 kW 雷达？假的吧" | 对，看着不合理，因为这个模型没有处理增益。落进 2 MHz 的只有 20 mW，而 12 km 处回波是 -67 dBm。缺口写在 docs/physics.md §7 第 1 条，加 20 dB 处理增益的话烧穿距离会涨约 10 倍。 |
| "Pd 公式太粗糙" | 用的 Swerling I 单脉冲闭式，是为了能反解检测距离。它对 SNR 偏钝（Pfa=1e-6 时要 21 dB 才到 0.9），这是选择的代价，不是笔误。 |
| "为什么不直接调现成的雷达库？" | 这个项目的卖点是"能逐行审的物理"和确定性 replay，引入第三方物理就没法保证了。 |
| "Python 跑得动吗？" | 热路径是闭式解，600 tick 的想定 1.3 s；向量化，不写逐点循环。 |
| "有验证吗？" | 182 + 16 项测试；关键一致性有专门断言（J/S=1 时 R=R_bt、同一 seed 字节级相同、接收增益与波长相消）。 |
| "为什么不用 deck.gl / three.js？" | 二维几百个图元，canvas 够了；打包 230 KB / gzip 73 KB。等有上万条轨迹再换。 |
| "做这个能干嘛？" | 课程实验、算法对比、复盘工具。T3a 在做 replay → 归因报告。 |
