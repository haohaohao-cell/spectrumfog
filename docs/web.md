# Web 层：数据包格式（T2）

## 为什么不是"前端再算一遍"

PROJECT.md §2 决策 3：一个内核三个面，同一份物理，不许前后端各写一套。
T2 又要求无后端。两条同时成立的唯一做法：

```
spectrumfog（Python，唯一物理实现）
        │  export-web 编译一次
        ▼
data.json（Pd/JS 的 (Rt, Rj) 二维表 + 三条 1D 曲线 + 元数据）
        │  前端只做：查表、双线性插值、画图
        ▼
浏览器（拖动干扰机 → 重新查表 → <100 ms 刷新）
```

前端**不含任何公式**。想改物理必须改 Python，然后重新导出——
这样不会出现"网页上的数和 CLI 的数对不上"这种最伤可信度的问题。

## 生成

```bash
spectrumfog export-web scenarios/02_standoff.yaml -o viz/web/public/data.json
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--out` | `out/<name>.web.json` | 输出路径 |
| `--pfa` | 1e-6 | 虚警概率 |
| `--pd-threshold` | 0.5 | 检测门槛（画环与判"可检测"用） |
| `--samples` | 97 | 二维表每轴采样数；体积 ∝ n² |

97×97 的 `02_standoff` 数据包约 126 KB（两张表各 ~76 KB base64 前的原始 float32 是 37 KB）。

## 结构

```jsonc
{
  "schema": "spectrumfog.web/1",     // 前端不认识的版本必须报错，不许猜
  "units": "SI: m, W, Hz, s",
  "scenario": "02_standoff",
  "pfa": 1e-06,
  "pd_threshold": 0.5,
  "noise_power_w": 1.5978e-14,

  "radar":  { "id": "radar_a", "pos_m": [0,0], "freq_hz": 3e9, "bw_hz": 2e6,
              "pt_w": 1.5e5, "gain_db": 35, "sigma_m2": 1.0, "loss_db": 3 },
  "jammer": { "id": "jammer_b", "pos_m": [6e4,0], "bw_hz": 1e8, "pj_w": 1.0, "gain_db": 0 },
  "target": { "id": "target_c", "pos_m": [9830,6880], "range_m": 11998 },

  "grid": {
    "rt_m": { "min": 100, "max": 5e5, "n": 97 },   // 目标距离轴
    "rj_m": { "min": 100, "max": 1e6, "n": 97 },   // 干扰机距离轴
    "spacing": "log10",                            // 轴上的点是 geomspace
    "units": "m"
  },

  "pd": { "dtype": "float32", "shape": [97,97], "data": "<base64 小端>" },  // [Rj, Rt]
  "js": { "dtype": "float32", "shape": [97,97], "data": "<base64 小端>" },

  "rings": {
    "jammer_range_m": [...],       // 241 点
    "detection_m":  [...],         // Pd = 门槛 的边界半径，随 Rj 增大
    "burn_through_m": [...]        // J/S = 1 的环，= sqrt(R_bt · Rj)
  },

  "target_curves": { "jammer_range_m": [...], "pd": [...], "js": [...] }
}
```

### 查表约定

* `pd[i][j]` 对应 `Rt = geomspace(rt_m)*[j]`、`Rj = geomspace(rj_m)*[i]`（行 = 干扰机距离）。
* 地图上任意一点 `p`：`Rt = |p - radar|`、`Rj = |p - jammer|`，双线性插值取值。
* 插值在 **log 空间**做（两个轴都是 `geomspace`），否则近距段的曲线会明显失真。
* `rings.*` 用普通数组，不需要插值：前端按 `Rj` 找最近的网格点即可（241 点足够密）。

## 前端约定

* 页面只读 `data.json`，不发请求、不写状态到服务器（GitHub Pages 纯静态）。
* 拖动干扰机时，每帧最多做 97×97 次查表，不需要 Web Worker；
  目标 <100 ms，实测在开发机上 <10 ms。
* 版本不符（`schema !== "spectrumfog.web/1"`）直接显示错误，不要降级渲染。
