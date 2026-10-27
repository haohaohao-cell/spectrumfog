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
* 拖动干扰机时，每帧最多做 220×220 次查表（canvas 是低分辨率再放大），
  不需要 Web Worker；目标 <100 ms，实测在开发机上 <10 ms。
* 版本不符（`schema !== "spectrumfog.web/1"`）直接显示错误，不要降级渲染。

## 本地跑

```bash
# 1) 生成数据（改了想定或物理就要重跑这一步）
spectrumfog export-web scenarios/02_standoff.yaml -o viz/web/public/data.json

# 2) 开发模式
cd viz/web
npm install
npm run dev          # http://localhost:5173

# 3) 生产构建（纯静态，dist/ 直接丢 GitHub Pages）
npm run build
npm run preview

# 4) 前端单测（查表逻辑 + 首屏渲染）
npm test
```

## 验证状态（诚实记录）

已验证：

* `npm run build` 通过（tsc 类型检查 + vite 打包，JS 230 KB / gzip 73 KB，无 WebGL 依赖）；
* 16 项前端单测通过：解码、log 遍历、双线性插值、环插值、越界夹取、
  用真实数据包做首屏 SSR 渲染（TRACKING / JAMMED 两种状态）；
* 静态产物经 `vite preview` 实测：`index.html`、`data.json`、JS/CSS 均 200；
* **headless Chrome 截图**（`docs/images/web_explorer.png`）：初始 60 km 处为 TRACKING，
  干扰机贴到 15 km 时切换为 JAMMED、边界从 20.5 km 缩到 10.4 km、Pd 从 0.92 掉到 0.31。

**未验证**（本机没有交互式浏览器，只能靠你点一下）：

* 拖动的手感与帧率。若明显卡顿，先把 `App.tsx` 里的 `HEAT` 从 220 降到 160；
* 地图与曲线在高 DPI 屏上的清晰度；
* 移动端触摸拖动。

截图命令（无浏览器环境也能复现）：

```bash
npm run build && npm run preview -- --port 4321
"C:\Program Files\Google\Chrome\Application\chrome.exe" \
  --headless=new --disable-gpu --window-size=1200,860 \
  --virtual-time-budget=8000 --screenshot=shot.png http://localhost:4321/
```

## 为什么没用 deck.gl

T2 的验收是"拖动 + 热力图 + 曲线 <100 ms"，二维、几百个图元，canvas 自绘足够。
deck.gl 会带来 ~1 MB 的 WebGL 依赖和一层 GPU 抽象，在只有一个热力图的页面上不划算。
等 T3b 要做多平台 / 大规模轨迹时再上 deck.gl，届时的数据包可以复用同一份表。

