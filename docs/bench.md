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
| 一步 | 想定的一个 tick（01_duel 1 s、02_standoff 2 s） |
| 终止 | 达到想定时长（`duration / dt` 步） |
| 奖励 | `+1` 干扰有效（目标处 Pd < 0.5）／`−1` 被有效定位（CEP ≤ 门槛）／`−0.02 × 功率档` |

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

## 当前结果（2026-10-27，3 局/组合）

```
scenario       policy             reward  jammed  exposed  Pd mean   J/S mean     首告警[s]
----------------------------------------------------------------------------------------
01_duel        always-jam     289.3± 18    100%      50%    0.000      85.63          2
01_duel        random         221.3± 11     67%      29%    0.314      59.61          6
01_duel        stealth          3.2±  2      4%       3%    0.917      53.28         46
02_standoff    stealth          4.5±  3      7%       3%    0.948       0.01         94
02_standoff    random          -1.5±  0      0%       0%    0.957       0.00        206
02_standoff    always-jam      -3.1±  0      0%       1%    0.919       0.01        127
```

**怎么读**：

* `01_duel` 里 90 W 的干扰机在 30 km 就足以把 Pd 压到 1e-6，所以"根本不用靠近"——
  蛮干（always-jam）赢；`stealth` 非要偷偷摸到 7.5 km 才开机，白白浪费 40 秒。
* `02_standoff` 里只有 1 W，60 km 处压不住任何东西，必须贴脸——
  但贴脸就意味着被三站交汇定位，所以它只能"打一下就跑"，成功率只有 7%。
* **没有一个策略通吃**。这不是 bug：两份想定的能量差 90 倍，
  "最优打法"本来就不同——这也是为什么要有 benchmark，而不是一个调好的 demo。

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
5. **想定只有 2 个**（目标 5 个）。缺的三个需要先扩模型（通信链路、诱饵、多辐射源），
   不是"再写三个 YAML"就能解决的。
