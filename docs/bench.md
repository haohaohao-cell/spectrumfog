# Bench：Gym 接口、baseline 与评测（T3b）

## 这一层是什么

把想定变成一个**决策问题**，然后给几个"不动脑子"的策略打分：

```
想定 + 物理（Python，唯一实现）
        │
        ├── spectrumfog sim    → replay / 动画 / AAR
        └── spectrumfog bench  → Gym 环境 → 策略 → 表格
```

环境和 `sim` 调的是同一批函数（`radar` / `detect` / `fusion`），
区别只是"谁来决定下一步动作"。所以表格里的数字和 AAR 里的数字是同一套物理。

## 环境契约

| 项 | 内容 |
|---|---|
| 智能体 | 红方干扰机（单智能体） |
| 观测 | `Box(6)`：`[x, y, 功率档, 时间进度, 相对距离, 告警位]` |
| 动作 | `Discrete(15)` = 方向(5) × 功率档(3)：停/±x/±y × 关机/半功率/满功率 |
| 一步 | 想定的一个 tick（01_duel 1 s、其余 2 s） |
| 终止 | 达到想定时长（`duration / dt` 步） |
| 奖励 | `+1` 达成目标／`−1` 被有效定位（CEP ≤ 门槛）／`−0.02 × 功率档` |

"达成目标"看干扰机打的是谁：与雷达同频 → **干扰有效**（目标处 Pd < 0.5）；
与雷达不同频、但与某条通信链路同频 → **链路失效**（抗干扰余量 < 0）。
这就是 03_triad 那行 J/S 为 0 的原因：它在打链路，不在打雷达。

**部分可观测落在哪**：蓝方的定位结果对红方不可见。红方只有一个**延迟告警**——
最近 `warning_window_s` 秒内是否被"有效定位"。真实 CEP 不进观测。
这正是 AAR 那句"你被定位了"的反面：当局者不知道自己被定到多少米。

`相对距离 = |位置| / 初始距离`，1.0 表示还在出发位置。这样策略不用知道米数，
换一个想定也不用改阈值。

## 三个 baseline

| 名字 | 行为 | 想验证什么 |
|---|---|---|
| `random` | 均匀随机动作 | 下限 |
| `always-jam` | 原地满功率 | "只管压制、不管暴露"能走多远 |
| `stealth` | 静默接近 → 贴脸开火 → 收到告警就关机关后撤 5 帧 | "隐蔽优先"是否划算 |

三个都只吃观测（`act(observation)`），不读环境私有状态——否则等于用未来数据作弊。

## 跑

```bash
spectrumfog bench --all --episodes 5
spectrumfog bench --scenario scenarios/01_duel.yaml --episodes 10
spectrumfog bench --all --out out/bench.json --markdown out/leaderboard.md
```

同一 seed 下所有策略看到**同一串随机数**，所以差异来自决策，不来自运气。

## 当前结果（2026-10-27，5 个想定 × 3 策略 × 2 局）

```
scenario       policy             reward  jammed  exposed  Pd mean   J/S mean     首告警[s]
----------------------------------------------------------------------------------------
01_duel        always-jam     281.0± 16    100%      51%    0.000      85.63          2
01_duel        random         223.9± 13     68%      30%    0.305      54.71          5
01_duel        stealth          3.5±  3      4%       3%    0.917      53.66         46
02_standoff    stealth          3.8±  3      7%       3%    0.952       0.01         95
02_standoff    random          -1.6±  1      0%       0%    0.957       0.00        206
02_standoff    always-jam      -3.4±  0      0%       1%    0.919       0.01        127
03_triad       stealth          0.0±  0      0%       0%    0.861       0.00          —
03_triad       random         -37.5±  4      0%      62%    0.861       0.00          4
03_triad       always-jam     -61.2±  0      0%     100%    0.861       0.00          2
04_fog         always-jam      49.3±  4    100%      16%    0.000      43.25         11
04_fog         random          35.0±  3     62%       3%    0.351      22.70         51
04_fog         stealth          1.9±  2      7%       3%    0.875      55.13         48
05_mobile      always-jam      81.7±  3    100%       7%    0.002       1.52         25
05_mobile      random          51.2±  3     64%       6%    0.362       0.73         37
05_mobile      stealth          0.4±  0      4%       3%    0.952       1.24         88
06_decoy       stealth         60.0±  0    100%       0%    0.363       0.00          —
06_decoy       random          59.5±  0    100%       0%    0.137       0.55          —
06_decoy       always-jam      58.8±  0    100%       0%    0.000       1.17          —
```

**怎么读**：

* `01_duel` 里 90 W 的干扰机在 30 km 就足以把 Pd 压到 1e-6，所以"根本不用靠近"——
  蛮干（always-jam）赢；`stealth` 非要偷偷摸到 7.5 km 才开机，白白浪费 40 秒。
* `02_standoff` 里只有 1 W，60 km 处压不住任何东西，必须贴脸——
  但贴脸就意味着被三站交汇定位，所以它只能"打一下就跑"，成功率 7%。
* `03_triad` 是**链路场景**：主干扰机（300 MHz）和雷达（3 GHz）不同频，
  环境因此切到"打链路"模式（J/S 列是 0，Pd 是未被干扰的 0.861）。
  三个 baseline 都是雷达思路，谁也不会往接收机那边靠，所以全是负分——
  这不是环境坏了，是 baseline 不对口，提升空间就在这里。
* `04_fog` 有 3 个红方辐射源、虚警率 5e-3，蛮干反而能压住（Pd 0）且暴露只有 16%，
  因为它的干扰机离雷达最近、离 ESM 最远。
* `05_mobile` 三方都在动，蛮干仍然赢——说明在这组参数下"机动"没有改变力量对比。

**没有一个策略通吃**。这不是 bug：几份想定的能量差两个数量级，
"最优打法"本来就不同——这也是为什么要有 benchmark，而不是一个调好的 demo。

## 排行榜与提交（无服务器）

排行榜只有在**同一批想定、同一组种子、同一套策略**下才可比。
这把尺子固定在 [`scenarios/suite_v1.yaml`](../scenarios/suite_v1.yaml)：

```yaml
version: 1
seeds: [0, 1, 2, 3, 4]
policies: [random, always-jam, stealth]
scenarios: [六个想定]
```

```bash
spectrumfog bench --suite scenarios/suite_v1.yaml                 # 按尺子跑
spectrumfog bench --suite scenarios/suite_v1.yaml --submit --label 你的名字
spectrumfog bench --verify submissions/你的名字-*.json            # 自检
spectrumfog bench --leaderboard submissions --write-md docs/leaderboard.md
```

提交物是自包含的 JSON：`suite` 版本、种子、策略、**每份想定文件的 sha256**、
18 行结果、运行环境的 Python / numpy / gymnasium 版本。没有服务器，靠三件事保证可比：

1. `--verify` 用当前文件重算 sha256，改一个数字就报"想定内容已改动"；
2. 版本号写进提交物，换尺子必然对不上；
3. 排名页由 `submissions/*.json` 生成，不手写。

当前榜单：[docs/leaderboard.md](leaderboard.md)（文档站“排行榜”页）。

## 排行榜

`--markdown out/leaderboard.md` 生成：

| 想定 | 策略 | 奖励（均值±标准差） | 干扰有效率 | 暴露率 | Pd 均值 | 首告警 |
|---|---|---|---|---|---|---|
| 01_duel | `always-jam` | 289.3 ± 18 | 100% | 50% | 0.000 | 2 s |

现在只有本地排行榜（自己跑、自己贴表）。公开提交（带签名的 JSON + PR）属于 T3b 的后半段，
还没做——因为那需要先固定想定集合的版本号，否则表不可比。

## 加一个策略

实现两个方法即可：

```python
class MyPolicy:
    name = "my-policy"
    def reset(self, *, seed=None) -> None: ...
    def act(self, observation) -> int:
        # observation = [x, y, power, t, relative_range, warning]
        return 0 * 3 + 2   # 停 + 满功率
```

然后 `spectrumfog bench --all --policies random,always-jam,stealth,my-policy`
（策略要先注册进 `bench/policies.py` 的 `POLICIES`）。

## 已知限制

1. **单智能体**：只控制红方干扰机；蓝方（雷达 + ESM 星座）是环境的一部分。
2. **动作离散**：没有连续功率/航向；也没做动作屏蔽（不合法的动作不会出现，但也没优化）。
3. **环境不写 replay**：训练用不到逐帧回放；要复盘就改用 `sim` + `aar`。
4. **没做向量化环境**：跑 5 个种子是串行的，单局 120 步在 1 秒以内，暂时够用。
5. **5 个想定已就位**（01_duel / 02_standoff / 03_triad / 04_fog / 05_mobile）。
   诱饵（欺骗式假目标）仍未建模，03_triad 用"双频段干扰"替代了它。
