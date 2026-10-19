# Bandori 国服分数预测 — 统一发行版（fresh / ridge / stack）

一个自包含的预测包：同一输入、同一输出格式，三个可互换的冻结模型。
推理只需 **Python 3.10+ 与 NumPy**（`requirements.txt` 只有 `numpy>=1.24`），不需要
PyTorch、不需要仓库其余部分，也不会在线重训。

- `fresh` — 规则变更后联合统计模型（120 系数，strength 0.01）
- `ridge` — 规则感知 Ridge + 24h 观测器（319 场，gain 0.5，`1+gain·delta`）
- `stack` — 规则+显式 600 列联合神经网络校正（JointNet，gain 1.0，`exp(delta)`）

**生产默认：stack + 简洁报文（T100/T500/T1000/T1500/T2000）。**
模型选择回车或脚本省略 `--model` 均尊重注册表的 `default: stack`；
`--model auto` 仍可显式启用原有的 240 小时切换规则。

## 安装与运行

```sh
python3 -m pip install -r requirements.txt
python3 predict.py
```

macOS 双击 `run.command`，Windows 双击 `run.bat`，任意平台 `./run.sh` —— 三者都会在
首次运行时创建 `.venv`、安装依赖并执行 `predict.py`（透传全部参数）。

常用示例：

```sh
python3 predict.py --event 326 --as-of 2026-10-17T20:00:00+08:00 \
    --snapshot examples/event_326_20261017_2000.json.gz
python3 predict.py --event latest --as-of now --data /path/to/bandori_cn_data
python3 predict.py --model stack --report full --no-save
```

当前默认 Stack 的含 σ 示例见 `examples/stack_event_326_171h.txt`（同名 JSON 保留
完整未舍入数值）。原 `event_326_171h`/`event_325_3h` 示例保留此前显式 auto 的结果。

不带参数运行时进入交互：输入活动编号（回车=按国服开始时间的最新活动），输入数据
截止时间（回车=活动内最近一次 08:00，now=当前时刻，或任意 `YYYY-MM-DD HH:MM`
中国时间），输入模型（回车=stack，或 auto/fresh/ridge/stack 指定）。未来时刻与活动
区间外的时刻会被拒绝。只使用数据截止时刻及之前已知的数据，预测目标始终是活动
结束时的最终分数。

## 命令行参数

| 参数 | 说明 |
|---|---|
| `--model auto\|fresh\|ridge\|stack` | 模型选择。省略或交互回车用注册表默认 `stack`。显式 `auto` 按 240 小时规则切换；其余强制指定。 |
| `--event N\|latest` | 活动编号，或 `latest`（按实际国服开始时间）。 |
| `--as-of TIME\|now` | 数据截止时间：中国时间 `2026-10-19 14:30` / ISO 带时区，或 `now`。不带时区按 UTC+8。省略且给了 `--event` 时取活动内最近一次 08:00。 |
| `--snapshot PATH` | 离线输入：`*.json.gz` 快照。与 `--data` 互斥。 |
| `--data DIR` | 本地已下载的数据目录（如 `../bandori_cn_data`）。与 `--snapshot` 互斥。两者都不给则走实时抓取（server=3，仅刷新所选活动的 24 档）。 |
| `--report focus\|full` | 文本报档次。`focus`（默认）只显示 T100/T500/T1000/T1500/T2000；`full` 显示全部分数线。JSON 始终保留全部档位。 |
| `--output-prefix PATH` | 保存的 JSON/TXT 路径前缀（写 `PREFIX.json` 与 `PREFIX.txt`）。默认 `output/event_N_时间戳`。已存在则拒绝覆盖。 |
| `--no-save` | 只打印，不写文件。 |
| `--max-stale-hours H` | 允许的最大分数观测年龄（小时）。默认 1 小时；放宽时（如 48）报文会标记旧观测。 |
| `--cache DIR` | 实时抓取缓存目录（默认包内 `cache/`）。 |
| `--timeout S` | 实时抓取超时（默认 10 秒）。 |
| `--save-snapshot PATH` | 把本次实际使用的输入另存为快照。 |
| `--registry PATH` | 模型注册表（含 auto 路由），默认 `models/registry.json`。 |
| `--fresh-model DIR` | 覆盖 fresh 包目录（默认 `models/fresh`）。 |
| `--ridge-model DIR` | 覆盖 ridge 包目录（默认 `models/ridge`）。 |
| `--stack-model DIR` | 覆盖 stack 包目录（默认 `models/stack`）。 |

## 统一输出格式

每次预测写出同名 `.json` 与 `.txt`（`--no-save` 时只打印 TXT）。JSON 结构
（`api_version = 1`）：

```jsonc
{
  "api_version": 1,
  "model": { "name": "fresh|ridge|stack", "kind": "包 manifest 的 kind",
             "selection": "default|auto|manual", "reason": "选择原因（含时长与阈值）" },
  "event_id": 326, "event_name": "…", "event_type": "…",
  "checkpoint_cn": "起报（数据截止）时刻 ISO8601，中国时区",
  "closes_cn":        "截活（活动结束）时刻 ISO8601，中国时区",
  "hours_remaining":  "距截活的剩余小时数",
  "total_duration_hours": "活动总时长（小时），auto 路由的依据",
  "routing": { "rule": "路由规则文本", "threshold": 240.0,
               "selected": "实际使用的模型" },
  "predictions": [
    { "tier": 500,                 // 分数线档位
      "status": "ok|unavailable",
      "final": 3966815,            // 预测最终分数（math.ceil，向上取整）
      "current": 531860,           // 截止时刻当前分数
      "observed_at": "当前分数的观测时刻 ISO8601",
      "score_age_hours": 0.25,     // 分数观测年龄（小时）
      "sigma": { "value": 326000,  // 近似 1σ 绝对值；不可用时整段为 null
                 // 支持的档位/时效给出近似误差尺度，无支持则 value 为 null
                 "relative": 0.082,// 相对 σ
                 "method": "fresh_chronological_rms|ridge_out_of_fold_rms|stack_event_loo_rms|uncalibrated" },
      "detail": { … }              // 模型专有字段，见下
    }, …
  ],
  "limitations": ["…"]             // 模型局限 + 路由来源 + 运行时警告
}
```

`detail` 字段（因模型而异）：

- fresh：`anchor`/`pace`/`anchor_method`/`trained`/`reference_events`/
  `role_transfer_references`，以及 σ 的 `sigma_scope`/`sigma_events`。
- ridge：`baseline`/`remaining_multiplier`（= `1+0.5·delta`）/`raw_correction`/
  `trained`/`supervised_training_targets`，σ 借档时 `sigma_borrowed_from_tier`，
  情景缺测时 `stale_calibration_unavailable`。
- stack：`baseline`/`remaining_multiplier`（= `exp(1.0·delta)`）/`raw_correction`/
  `supervised_training_targets`/`trained`；σ 方法为 `stack_event_loo_rms`，JSON 还记录
  校准活动数、有效活动数、相对偏差、去均值标准差、时效合并与外推标记。
  无本档标签（T1/T10）、无标定包、或观测超过 1h 时保留 `uncalibrated`，不借档。
- `status = "unavailable"` 的行：`final/current/observed_at/sigma = null`，
  `detail.reason` 说明原因（缺测、超过允许观测年龄等）。

TXT 报文要点：物理学括号记法 `1.23(45) * 10^6` 表示 `(1.23 ± 0.45) * 10^6`，
括号为近似 1σ 并与末位对齐；σ 未标定时不写括号，行尾出现 `[σ未标定]` 等标记
（`σ借用T20`、`旧观测x.xh`、`无本档监督，回退先验` 等）。起报时间即数据截止
时间；截活时间即活动结束时间。

## 自动路由（240 小时规则）

`--model auto` 读取 `models/registry.json`：

> **活动总时长 > 240 小时 → ridge；否则 → fresh**（恰好 240 小时归 fresh）。

- 用已知的完整活动日程判断总时长，与剩余小时数无关：325 场约 373 小时，
  即便只剩 3 小时也选 ridge；326 场约 202 小时，选 fresh。
- 切换只换推理模型，不重训、不混合两模型输出、不读取当前活动终值。
- **来源（provenance）**：该规则是用户在查看 325 场结果后指定的
  （"specified by the user after viewing 325 results"），不是与检验独立得出的
  路由规则；325 已被用于指定规则，因此不能把历史成绩当作路由系统的独立测试。
- JSON 的 `routing` 与 `limitations` 均记录规则、阈值与该来源说明。
- 三个模型的拟合都**排除 325/326**（fit_cutoff = 325 开始时刻 1790647200）。

### 换模型（swap procedure）

一切由注册表驱动：`default` 改生产默认模型，`auto` 只影响显式的 `--model auto`。
改一行即可：

1. 编辑 `models/registry.json` 的 `default` 改生产默认；若要修改显式 auto 的路由，把 `auto.at_or_below_threshold` 的值改为目标
   模型（`"fresh"` → `"stack"`，一行）；或者引入新模型时，在 `models.<name>`
   下加一个 `{"package": "<目录名>"}` 条目并放入 `models/<目录名>/` 包。
2. 运行一次 `python3 predict.py --model auto …` 确认 JSON 中
   `model.name` / `routing.selected` 已切换（`--registry` / `--*-model` 可做
   临时覆盖，不必改文件）。
3. 改回原值即还原。

已执行的换栈测试（验收记录）：把 `at_or_below_threshold` 临时改为 `stack` 后，
326 例 `--model auto` 选中 `stack`（`reason: …202h <= 240h -> stack`），
焦点档 T500/T1000/T1500/T2000 = 3966815 / 1945733 / 1945733 / 1580956，与手动
`--model stack` 完全一致；改回 `fresh` 后同一命令重新选中 `fresh`，结果与
`fresh_326` 基准一致。

## 三个模型：出处与局限

三者都以 `models/<name>/manifest.json`（`name, kind, api_version, fit_cutoff,
supervised_events, assets{文件:sha256}, training_provenance, limitations`）+
`models/registry.json` 描述；推断前会校验包内资产的 sha256。

### fresh — `fresh_post_rule_joint_v1`

- **拟合**：12 场严格规则变更后活动（311、314–324；312/313 起始早于规则变更），
  120 个联合部分共享系数，strength 0.01，730 天特征时间衰减 + 变更后 3 倍权重。
  319 场曲线库只作完成率/临近截活增速等物理先验，**不作监督标签**。
  fit_cutoff = 325 开始；**325/326 不参与拟合**。
- **公式**：模型内部给出修正（先验锚点 + 配速），随后按分数线单调性调和、向上取整。
- **σ**：`fresh_chronological_rms` —— 来自 317–320 的逐场时间自适应预报残差，
  每档最多 4 场，**不跨档借用**；无本档样本时 σ 为空。
- **验证（held-out 325，描述性单次留出）**：external-24 **7.99%** /
  standing-164 **10.63%** MAPE。
- **局限**：只拟合 12 场规则变更后活动；T1/T10 无监督标签回退先验；σ 样本极小、
  含系统偏差、无 68% 覆盖保证；参考池止于 324。

### ridge — `bandori_cn_ridge_deployment_v1`

- **拟合**：319 场至 324 全量（**325/326 不参与拟合**），PCA 输入（rich 压缩 +
  24h 观测器摘要 + 规则元数据），JointRidge 收缩强度 0.01（唯一搜索参数，仅在
  317–320 按时间前推选择，不读 321–324 结果），730 天衰减 + 规则变更后 3 倍权重。
- **公式**：`1 + gain·delta`，**gain = 0.5**；随后单调性调和、向上取整。
- **出处**：打包自旧发行版 `old/dist_fresh/long_event_model`（同族于
  `runs/rule_aware_3x_20261018/full_refit`），`ridge_source_model_sha256`
  记录在 `ROUTING.json`；本发行版只打包与切换，不重拟合、不调参；未采用
  实验性 clock_v2 变体。
- **σ**：`ridge_out_of_fold_rms` —— 12 场规则变更后按时间留出的自身历史残差
  （多为收尾代理标签），T1/T10 借 T20 时标注 `σ借用T20`；未专门标定超长活动。
- **验证（held-out 325）**：external-24 **4.31%** / standing-164 **8.79%** MAPE。
- **局限**：σ 含系统偏差、无覆盖保证；强度选择与设计迭代使历史检查并非全新
  独立检验；超长活动（如 325 的 373h）超出训练分布时 JSON/报文会显式告警。

### stack — `bandori_cn_joint_deployment_v1`

- **拟合（post12_30 配方）**：**正好 30 场** = 12 场严格规则变更后（311、314–324）
  + 18 场最近的更早活动；加权单元格 15,132；**325/326 不参与拟合**
  （fit_cutoff = 325 开始 1790647200）。
- **特征**：完成态 pca 布局（1168 列）中选择 **600 列** = 显式块 16–111 +
  规则块 664–1167（`block_cols({'rule','expl'})`），与 `pca_layout` 一致。
- **结构与公式**：JointNet（hidden 8、seed 1729、240 epochs、lr 0.01、
  weight_decay 0.003、bias_penalty 3），**exp(gain·delta)，gain = 1.0**；
  随后单调性调和、向上取整。推理纯 NumPy（`bandori/stack_runtime.py`，
  语义对齐 `bandori/joint_runtime.JointPredictor`）。
- **配方与导出**：`runs/sweep_stack_20261019/train_post12_30_test325.py`
  （已验证；**不是** train_postrule30_*，后者误拟合 32 场）；包由
  `tools/export_stack_package.py` 导出并复验。
- **σ**：`stack_event_loo_rms`，只用 311、314–320 共 8 场严格规则变更后活动的
  留一预报残差。每折完整移除该场曲线/标签/元数据，重建受影响的训练先验，
  重拟合训练标准化与固定的 h8/seed1729 配方；不是训练集拟合误差。
  321–326 全部排除 σ 标定，321–324 仍在生产网络训练中。
- **σ 计算**：每档按剩余小时分为 `[0,24), [24,48), [48,72), [72,96),
  [96,144), [144,192), [192,∞)`；事件内部先平均平方误差，再以 730 天衰减
  对事件加权。`r=(预测−终值)/预测`，`σ=预测×sqrt(加权事件均值(r²))`。
  单元不足 4 场则合并邻近时效；不借 T1000 的 σ 给 T1500。
  4385 行残差只代表 8 个活动簇；重复时刻不是独立样本。
- **σ 支持**：标定活动总时长约 178–226h；超长场次会标记 `σ超出标定范围`。
  无本档残差或观测超过 1h 不给 σ；未单独标定停服/陈旧输入/活动类型。
  σ 含偏差，没有正态/68%覆盖保证；全量重训不自动缩小 σ。
- **验证（held-out 325）**：external-24 **4.50%**（偏差 +1.54%）/
  standing-164 **7.55%**（偏差 −2.84%）MAPE —— 描述性单次留出，非独立测试。
  （早期 32 场口径的 4.53%/7.74% 已作废。）
- **局限**：σ 只有 8 场且 LOO 允许拟合更晚的 pre-321 活动，并非按时间前推的独立覆盖检验；超参数继承自冻结配方、未针对该 30 场口径重选；标签含
  收尾代理；参考池止于 324；单次留出验证。

### 精度对照（held-out 325，描述性）

| 模型 | external-24 MAPE | standing-164 MAPE |
|---|---|---|
| stack | **4.50%** | **7.55%** |
| ridge | **4.31%** | **8.79%** |
| fresh | **7.99%** | **10.63%** |

（同表基线：8.65% / 14.73%。均为此前检验过的单次留出，不是全新独立测试。）

## 模型包与校验

```
models/
  registry.json        # 模型清单 + auto 路由（换模型改这里）
  fresh/               # model.json, correction.npz, policy.json,
                       # calibration_forecasts.csv, reference_bank.{json,npz}, manifest.json
  ridge/               # deployment.json, ridge.npz, compression.npz, input_masks.npz,
                       # sigma.json, config.json, reference_bank.{json,npz}, manifest.json
  stack/               # deployment.json, joint_model.npz, compression.npz, input_masks.npz,
                       # sigma.json, config.json, reference_bank.{json,npz}, manifest.json
```

- `ROUTING.json` —— 路由阈值与三个模型资产 sha256 的速查（含来源说明）。
- `DISTRIBUTION` —— 发行版标识：`unified-fresh-ridge-stack-v1`。
- `PACKAGE_SHA256.json` —— 整包文件清单（sha256）。校验/重建：

```sh
python3 verify_package.py            # 校验每个文件（不匹配或缺漏即失败）
python3 verify_package.py --regen    # 重建清单后再校验
```

构建期工具（推理不需要，需在仓库根以 `PYTHONPATH=. .venv/bin/python` 运行）：

- `tools/export_stack_package.py` —— 按 post12_30 配方重训并导出 stack 包，
  内置 held-out 325 分数断言与 runtime/harness 特征一致性断言。
  重导出后，需在仓库根运行 `python -m bandori.stack_sigma_calibration` 重新绑定 σ，
  然后运行 `python dist/verify_package.py --regen`；σ 绑定具体权重 SHA256，不能直接复制到新模型。
- `tools/build_manifests.py` —— 重建 fresh / ridge 的 `manifest.json`。
- `tools/verify_stack_parity.py` —— stack 自检：对两个示例快照，同一截止时刻下
  比较发行版输出与 harness 路径（`ridge_inputs ≡ finish_context`，atol 1e-10）及
  `bandori.joint_runtime.JointPredictor`（零填充到 1168 宽的等价权重）逐档整数
  finals。

回归基线：fresh/ridge 在两个示例上与旧发行版（`old/dist_fresh`，只读）逐档
finals/current/σ 完全一致；实时路径（`--data`）与快照路径在同一起报时间结果一致。

## 已知局限（汇总）

- 三个模型都排除 325/326 拟合；held-out 325 分数为描述性单次留出。
- 240 小时路由由用户在查看 325 结果后指定，非检验独立规则。
- 三个模型的 σ 样本都少、含系统偏差，无正态或 68% 覆盖保证；Stack 的 σ 只用 8 场 pre-321 LOO 残差。
- 参考/模拟池止于 324；训练分布内最长约 346 小时，超长活动会显式告警。
- 实时输入只刷新所选活动的 24 档历史；不修改本地数据目录。
