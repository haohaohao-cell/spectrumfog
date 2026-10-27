# 排行榜

同一把尺子：想定集合、种子、策略都固定（见 `scenarios/suite_v1.yaml`）。
得分 = 各（想定 × 策略）平均奖励；奖励 = 干扰/压制有效 +1、被有效定位 −1。

| # | 提交 | 得分 | 想定数 | 环境 | 日期 |
|---|---|---|---|---|---|
| 1 | `baselines` | 45.76 | 6 | py3.12.6 / gym1.4.0 | 2026-10-27 |

## 怎么提交

```bash
spectrumfog bench --suite scenarios/suite_v1.yaml --submit --label 你的名字
spectrumfog bench --verify submissions/你的名字-*.json   # 自检
```

把生成的 JSON 放进 `submissions/` 提 PR。没有服务器：
提交物自带每份想定文件的 sha256，`--verify` 一跑就知道结果是不是在**当前**想定上跑出来的。
