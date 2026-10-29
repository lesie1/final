# 复现指南

本文说明从原始数据到最终结果的完整流程：需要准备什么数据、按什么顺序执行哪些命令、Agent 怎么接入、结果输出到哪里。命令行参数和 Python 接口的完整说明见[接口参考](interfaces.md)。

## 1. 环境

| 项目 | 要求 |
|---|---|
| Python | 3.10～3.13（已验证 3.10.8、3.12.3、3.13.9） |
| 依赖 | `pip install -r requirements.txt`（numpy、pandas、pyarrow、xgboost ≥ 3.2、psutil、openpyxl） |
| GPU | 可选。配置里 `device="cuda:0"` 用 GPU，CPU 环境改成 `"cpu"` |
| 内存 | TypeA 全量实验使用 `memory_limit_gb=80`。内存不够时框架会主动报错，不会悄悄降采样 |
| 磁盘 | TypeA 的特征缓存加上会话约 10 GB |

先跑一遍回归测试，确认环境正常：

```bash
python -m unittest discover -s tests -v
```

测试只使用合成数据，Agent 的回复用替身代替，不会调用付费 API。

## 2. 准备数据

框架通过一份**字段映射**（mapping JSON）读取任意"设备 × 时间"的数值数据。映射的格式见 `examples/mapping_example.json`，规则见 `data_ingest.validate_mapping`。

### 2.1 内存 TypeA（本次结果所用数据）

TypeA 的模型输入 = **官方给出的 99 个特征 + 官方原始日志**，再经过我们的数据处理脚本得到。

**① 官方输入**

来源：全部数据（99 个预处理特征、原始 CE 日志、故障工单）均从 [SmartMem 内存故障预测挑战赛](https://hyxie2023.github.io/SmartMem.github.io)（WWW 2025，华为云举办，[Codabench 3586](https://www.codabench.org/competitions/3586/)）下载，使用第一阶段的 TypeA 数据（原始日志包为 `stage1_csv.zip`）。这也是本赛事通知为内存器件指定的数据来源；本赛事的数据说明见[赛事 GitCode 数据仓库](https://gitcode.com/dfx-contest-2026/data)（报名后授权访问）。

| 文件 | 内容 |
|---|---|
| `type_A/<设备ID>.feather` | 官方预处理好的 99 个时间窗统计特征（15 分钟、1 小时、6 小时等窗口的 CE 计数等）；文件名就是设备 ID；时间列 `LogTime`（Unix 秒） |
| 原始 CE 日志：`stage1_csv.zip`（或解压后的 `type_A` 原始 CSV 目录） | 每条可纠正错误（CE）的原始记录，包含地址、比特等信息 |
| `failure_ticket.csv` | 故障工单：`sn_name`、`alarm_time`（Unix 秒）、`sn_type`（取 `'A'`） |

文件名以官方数据包为准；上表是代码里使用的名称（映射中的 `type_A/*.feather`，`raw_cache.py --raw` 接受的 zip 或目录）。

**② 我们的数据处理**（按顺序）

| 步骤 | 脚本 | 做什么 | 产出 |
|---|---|---|---|
| 1. 整理输入目录 | 早期版本的整理脚本（**不在本仓库**，见 §3.4） | 剔除 2 个表结构不完整、且没有故障工单的特征文件；按下面的规则生成每台设备的观测区间 | 输入目录：`type_A/*.feather`、`failure_ticket.csv`、`followup.csv` |
| 2. 规范化 | `agent.py init` + `prepare`（`data_ingest.normalize`） | 按映射读取 99 个特征和工单；统一时间和设备 ID；删除故障之后的记录；按时间 6:2:2 划分 TRAIN/VAL/TEST | 99 特征的 TRAIN/VAL 缓存 |
| 3. 原始日志特征 | `raw_cache.py`（`raw_events.py`） | 解析原始 CE 日志，在**相同的设备和时间点**上补充 65 个比特、DQ、突发、行/列/单元分布特征；没有原始日志的设备保持缺失，不填 0 | 164 特征的缓存 |
| 4. 冻结负采样 | 早期版本的采样脚本（**不在本仓库**，见 §3.4） | 只用 TRAIN；保留全部正样本，按设备和周抽取负样本，并记录补偿权重 | `sampling_reference.json` 及行号、权重数组 |

`followup.csv` 的规则：观测起点是该设备第一条 CE 记录，终点是故障工单时间；没有工单的设备取最后一条 CE 记录。没有工单**不等于**健康，不当作完整随访。

**③ 模型输入**：164 个基础特征（99 个官方 + 65 个原始日志特征）。搜索过程中，特征 Agent 还会在这些特征上生成组合特征，例如最终采纳的 trial_008 多了 4 个，一共 168 个。

本次使用的完整映射在 [`results/typea_v3.2/session/mapping.json`](../results/typea_v3.2/session/mapping.json)，可以直接复用。故障定义是每台设备第一次出现 TypeA 内存故障工单的时间。

### 2.2 内存 TypeB

官方输入是 `type_B/<设备ID>.feather`（官方预处理好的 **66 个**统计特征）+ TypeB 的原始 CE 日志 + 故障工单（`sn_type` 取 `'B'`），然后按 §2.1 的四步处理。**原始日志一定要提供并经过第 3 步**，否则模型只有 66 个统计特征，缺少原始比特特征，效果会明显变差。原始日志的格式如果和 TypeA 不同，要先用 `raw_cache.py --smoke-parts 1` 确认解析正确。

### 2.3 硬盘（Backblaze）

来源：[Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)。每天一个 CSV，字段包括 `date`、`serial_number`、`model`、`failure` 和 `smart_*`。映射可以参考 `examples/mapping_example.json`：
- `failure.mode = "indicator"`，`column = "failure"`；
- `followup.mode = "last_observation"`：设备消失不等于健康，随访以最后一次观测为准；
- 评价口径会自动使用 `period_device_f1_v2`，不设最长提前量。

如果不提供 `--mapping`，`agent.py init` 会让特征 Agent 根据数据的表头和样例自动生成映射。生成的映射必须人工核对故障定义和随访规则后再继续。

## 3. 运行流程

### 3.1 新数据集从头开始

```bash
export DEEPSEEK_API_KEY=...        # 只放在环境变量里

# ① 建立会话：扫描数据、固定配置、记录代码封印、确定字段映射
#    --data 指向 §2.1 第 1 步整理好的输入目录（type_A/*.feather、failure_ticket.csv、followup.csv）
python agent.py init --data /data/typeA --workspace /ws/typea --domain memory \
    --mapping mapping.json --config examples/runtime.json
#   → 输出 SESSION: /ws/typea/sessions/<时间>_memory_<id>

# ② 构建 TRAIN/VAL 缓存，按时间 6:2:2 划分；VAL 再分成前半段"选择窗口"和后半段"确认窗口"
python agent.py prepare --session /ws/typea/sessions/<会话名>

# ②′ 训练开始之前：特征 Agent 根据领域知识和原始数据字段提出特征建议，人逐条决定是否采纳（见 3.3）
#    --before 填 TRAIN 的结束时间（在 <会话>/cache/manifest.json 的 windows.train 里），保证 Agent 只看到训练期的样例
python tools/propose_features.py propose --raw /data/stage1_csv.zip --time-column LogTime --before <TRAIN结束时间> \
    --current-features <会话>/mapping.json --task-file task.txt --out /ws/typea/feature_proposals
#    人工在 adoption.json 里填写决定和理由，然后生成审查记录
python tools/propose_features.py review --out /ws/typea/feature_proposals

# ③（仅内存）实现被采纳的原始日志特征：先用一个分片试跑，再全量构建
python raw_cache.py --source-cache <会话>/cache --raw /data/stage1_csv.zip --out /ws/typea/raw_cache --smoke-parts 1
python raw_cache.py --source-cache <会话>/cache --raw /data/stage1_csv.zip --out /ws/typea/raw_cache --workers 4

# ④ 用带原始特征的缓存新建搜索会话（复用 ① 的映射和数据划分）
python search_step.py --initialize-from <会话> --session /ws/typea/search --active-cache /ws/typea/raw_cache --arm arm_policy.json
python agent.py limits --session /ws/typea/search --memory-limit-gb 80 --max-trials 13

# ⑤ 运行 Agent 闭环（建议放在后台）
nohup python agent.py run --session /ws/typea/search > typea_run.log 2>&1 &
python agent.py status --session /ws/typea/search        # 查看进度

# ⑥ 搜索结束后，在确认窗口上评估基线和最优模型
python confirm_search.py --session /ws/typea/search --out /ws/typea/confirmation

# ⑦ 汇总比赛指标
python tools/session_report.py --session /ws/typea/search --json summary.json --md summary.md
```

硬盘等没有原始日志的数据跳过 ②′、③ 和 ④，在 ② 之后直接执行 `agent.py run`。

#### 运行前请根据需要调整初筛门槛 `screen_margin`

每组新实验训练完后，控制器会先用选择窗口的 F1 做一次初筛：比当前最优低 `screen_margin`（默认 0.02，即 2 个百分点）以上，就直接淘汰，**不再跑 CV**。这一步只是为了省时间，但选择窗口的故障往往很少，初筛结果有很大的偶然性。例如 TypeB 的选择窗口只有 10 台故障，差 1 台 F1 就相差约 4 个百分点；检验"累积 CE 负担 → 故障"假设的实验（trial_003）就是以 4.2 个百分点的差距被初筛淘汰的，没有得到 CV 评估。

| 情况 | 建议 |
|---|---|
| **时间充裕** | **设为 `1.0`，让所有实验都跑交叉验证**，是否采纳完全由 CV 证据决定 |
| 时间紧张，并且选择窗口里故障较多（几十台以上） | 可以保留默认值 0.02 |
| 选择窗口里故障很少（例如只有十几台） | 即使时间紧，也建议设为 `1.0`，或者明显放宽 |

设置方法：在传给 `agent.py init --config` 的配置文件里加上 `"screen_margin": 1.0`。会话一旦建立，就不能再改这个值（配置变化需要新建会话）。

**注意：** 用 `search_step.py --initialize-from` 复用旧缓存建会话时，`--arm` 配置目前**不接受** `screen_margin`，只能使用默认值 0.02。需要关闭初筛时，请用 `agent.py init --config` 建会话。

### 3.2 本次 TypeA 结果的实际执行链

```
官方 99 特征（type_A/*.feather）+ 原始 CE 日志 + 故障工单
  → 整理输入目录（剔除 2 个不完整文件，生成 followup.csv）
  → agent.py init / prepare                      99 特征的 TRAIN/VAL 缓存
  → raw_cache.py（v1 表示）                       + 65 个原始日志特征 = 164 特征缓存
  → 冻结负采样文件 sampling_reference.json        只用 TRAIN；所有正样本保留
  → v3.1 会话（search_step.py --initialize-from）
  → v3.2 会话（results/typea_v3.2/run_scripts/setup_run.py：复用 v3.1 的缓存、采样和配置，只换采纳规则和工作点）
  → agent.py run（13 组）→ confirm_search → 历史 TEST 评估（run_scripts/evaluate_search.py）
```

`run_scripts/` 下的脚本是在服务器上实际执行过的版本，里面写的是服务器的绝对路径，换机器使用时需要相应修改。

### 3.3 训练前的特征建议：Agent 提出，人采纳和实现

官方给出的预处理特征往往不够用。在**任何训练开始之前**，由特征 Agent 根据已有的领域知识和研究，看原始数据的字段和少量训练期样例，提出还需要补充哪些特征：

| 步骤 | 谁来做 | 产出 |
|---|---|---|
| 1. 提出建议 | 特征 Agent（`propose_features.py propose`） | `proposals.json` / `proposals.md`：每条建议的计算方法、时间窗口、因果性说明、故障机理、依据及其可信度；数据中缺少的信息另列在 `data_requests` |
| 2. 审查和决定 | 人 | 在 `adoption.json` 中给每条建议填写 `adopted` / `rejected` / `deferred`、审查人和理由 |
| 3. 实现 | 人 | 把采纳的建议实现为特征（内存数据实现在 `raw_cache.py` 里），在 `implemented_as` 里写上对应的特征名 |
| 4. 记录 | `propose_features.py review` | `review.md` / `review_summary.json`：建议、决定、实现情况的完整记录，可以直接放进报告 |

这一步 Agent **看不到**故障标签、故障时间和任何实验结果；样例只取 `--before` 之前（训练期）的数据；引用了原始数据中不存在的列会被要求修正；没把握的论文依据必须标为低可信度。之后的搜索阶段，特征 Agent 还会在这些特征的基础上继续根据诊断生成组合特征。

**关于本次 TypeA 结果：** 这个工具是在 TypeA 实验之后才加入的。TypeA 中的 65 个原始日志特征是项目成员参考相关研究后人工设计和实现的，不是由 Agent 提出的。之后的新数据都按上面的流程执行。

### 3.4 复现时的已知缺口

要**逐字节**复现 `results/typea_v3.2` 的数字，除了代码和原始数据，还需要两份不在仓库里的文件：

1. **输入目录整理脚本**（§2.1 第 1 步：剔除不完整文件、生成 `followup.csv`）：规则已经写在上面，但脚本是早期版本留下的，不在本仓库中；
2. **冻结负采样文件** `sampling_reference.json` 及其行号、权重数组（§2.1 第 4 步，策略：`frozen_selective_device_week_10pct_20pct_hard`）。生成脚本同样来自早期版本。

缺少采样文件时，框架会使用全部训练行（不采样）。流程完全可以运行，但具体数字会和本次结果不同。所有输入文件的 SHA-256 都记录在会话的 `config.json`、`bootstrap.json` 和缓存 `manifest.json` 里，可以用来核对。

## 4. 接入大模型 Agent

- 默认使用 DeepSeek 的 OpenAI 兼容接口：`deepseek_base_url="https://api.deepseek.com"`，三个角色的模型由 `analysis_model`、`feature_model`、`code_model` 指定（本次都是 `deepseek-v4-pro`）；
- 密钥**只**从环境变量 `DEEPSEEK_API_KEY` 读取。训练子进程会被剥离所有名称里含 KEY、TOKEN、SECRET、PASSWORD 的环境变量，也不会把密钥写入任何文件；
- 换成其他 OpenAI 兼容服务：在配置文件里改 `deepseek_base_url` 和三个 `*_model`，密钥仍然放在 `DEEPSEEK_API_KEY`；
- 换成完全不同的 LLM：实现一个带 `call(role, payload, folder, max_tokens=None) -> dict` 方法的类，传给 `Loop(session, provider)`。见[接口参考 §3](interfaces.md#3-llm-接入接口)；
- 正式运行前先检查连接：`python agent.py doctor --session <会话> --api-check`。三个角色各调用一次，会计入调用预算。

每次调用都会把同一份 [Skill](../skills/device-failure-experiment-loop/SKILL.md) 共享方法和角色提示词（`prompts/`）作为 system prompt 注入。

## 5. 输出在哪里

一个会话目录包含该次实验的全部记录：

| 路径 | 内容 |
|---|---|
| `config.json`、`state.json` | 固定配置；当前状态、所有 trial、最优模型 `best`、调用次数和 token 数 |
| `events.jsonl` | 按时间记录的全部事件（Agent 调用、决策、训练、CV、采纳） |
| `calls/<编号>_<角色>/` | 每次 LLM 调用的请求（不含密钥）、回复、用量 |
| `hypotheses.json`、`diagnostics/`、`feature_jobs/`、`feature_programs/` | 假设、误报/漏报诊断、特征 Agent 生成的特征程序和预筛结果 |
| `trials/trial_XXX/attempt_01/` | 每组实验：计划、训练代码、模型、验证报告、CV 结果、采纳判断 |
| `cache/` | 规范化的 TRAIN/VAL 数据（体积大，不进仓库） |

每个 trial 目录里的主要文件：

| 文件 | 内容 |
|---|---|
| `plan.json`、`candidate.py` | 被接受的实验计划；代码 Agent 写的训练配置 |
| `model.ubj` | 选择窗口上最优的那一步模型 |
| `model_full.ubj` | 训练到 `cv_rounds` 的完整模型；部署时截断到 CV 选出的步数 |
| `result.json`、`report_val.json`、`metrics_val.json` | 选择窗口指标、提前量、报警预算曲线、耗时 |
| `devices_val.csv` | 逐设备的分数、是否报警、命中、误报、漏报 |
| `cv/oof_complete.json` | 5 折 CV 的最优步数、阈值和合并指标 |
| `cv_comparison_best.json` | 和当前最优模型的配对比较：P(更好)、ΔF1、新增/丢失命中 |
| `feature_importance.csv`、`loss_history.json`、`threshold_search_val.csv` | 特征重要性、训练曲线、阈值网格 |
| `complete.json` | 所有输出文件的 SHA-256 回执 |

`confirm_search.py --out <目录>` 会写出 `frozen.json`（被评估的模型和工作点）、`<trial>/report.json`、`<trial>/devices_<工作点>.csv` 和 `summary.json`。

`agent.py export --session <会话>` 会把会话打包成 `<会话名>_report.zip`（不含缓存和大数组），方便提交或归档。

## 6. 数据隔离与评估纪律

- 搜索过程只使用 TRAIN 和 VAL 的前半段（选择窗口）。确认窗口和 TEST 都不参与搜索，`agent.py test` 在代码中被禁用；
- TEST 评估只能在搜索结束、模型和阈值冻结之后进行，并且不能根据 TEST 结果重新选阈值或换模型；
- 换数据、换特征表示或改代码，都要新建会话。旧会话不能"迁移"后继续使用（代码封印会拒绝）。
