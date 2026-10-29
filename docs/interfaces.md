# 接口参考

对外接口分三层：**命令行**（最常用）、**Python API**（嵌入其他程序时使用）、**LLM 接入接口**（更换大模型时使用）。完整运行流程见[复现指南](reproduce.md)。

## 1. 命令行

所有命令都在仓库根目录执行。`<会话>` 指 `workspace/sessions/` 下的一个会话目录。

### `agent.py`：会话与 Agent 闭环

| 命令 | 参数 | 作用 |
|---|---|---|
| `init` | `--data` `--workspace` `--domain {memory,hdd,generic}` 必填；`--mapping` `--config` `--instructions` `--description-file` `--complete-failure-coverage` 可选 | 新建会话：扫描数据、固定配置和评价口径、写入代码封印、确定字段映射。没有 `--mapping` 时由特征 Agent 生成映射 |
| `adapt` | `--session` | 只生成或校验字段映射 |
| `prepare` | `--session` `[--mapping]` | 构建 TRAIN/VAL 缓存，划分选择窗口和确认窗口 |
| `run` | `--session` | 运行 Agent 闭环，直到 Agent 主动停止或实验预算用完。中断后再执行一次会从断点继续 |
| `status` | `--session` | 打印 `state.json`（trial 列表、最优模型、调用次数和 token 数） |
| `doctor` | `--session` `[--api-check]` | 检查 XGBoost、设备、磁盘和缓存哈希；`--api-check` 会对三个角色各做一次连接测试 |
| `limits` | `--session` 加上 `--max-trials` `--max-agent-calls` `--max-reported-tokens` `--memory-limit-gb` `--trial-timeout-seconds` `--feature-timeout-seconds` 中的一个或多个 | 调整资源和预算。只能改这几项，修改会记录为 `limits_changed` 事件 |
| `freeze` | `--session` | 冻结基线和最优模型，记录模型哈希、阈值和工作点 |
| `export` | `--session` | 打包成 `<会话名>_report.zip`（不含缓存和大数组） |
| `test` | `--session` | **已禁用**。TEST 评估要在搜索之外单独执行 |

### 其他命令

| 命令 | 参数 | 作用 |
|---|---|---|
| `search_step.py --initialize-from <旧会话> --session <新会话>` | `[--active-cache <缓存>] [--arm arm_policy.json] [--data-dir]` | 复用旧会话的映射、数据划分、缓存和冻结采样，新建会话。不会导入旧模型 |
| `search_step.py --session <会话>` | – | 只执行一个 Agent 动作（供外部调度器使用） |
| `confirm_search.py --session <会话> --out <目录>` | – | 搜索结束后，在确认窗口上评估基线和最优模型，同时给出 CV OOF 和选择窗口两种工作点的结果 |
| `operating_point.py --trial <会话>/trials/<id>/attempt_01` | `[--mode cv_oof\|selection_window]` | 打印某个 trial 部署时使用的步数、阈值和模型文件 |
| `raw_cache.py --source-cache <缓存> --raw <zip 或目录> --out <目录>` | `[--workers N] [--smoke-parts N] [--representation v1\|spatial_v3] [--final-test]` | 从内存原始 CE 日志构建原始特征缓存 |
| `tools/propose_features.py propose --raw <原始数据> --task-file <任务说明> --out <新目录>` | `[--current-features mapping.json] [--time-column 列 --before 时间] [--rows 20] [--max-files 5] [--model] [--base-url] [--dry-run]` | **训练前**由特征 Agent 根据领域知识和原始字段提出特征建议。`--before` 限制只展示训练期的样例；`--dry-run` 只生成请求、不调用 API |
| `tools/propose_features.py review --out <目录>` | – | 校验人工填写的 `adoption.json`（每条建议的决定、审查人、理由、实现的特征名），生成 `review.md` |
| `tools/session_report.py --session <会话>` | `[--target-f1 F] [--json 文件] [--md 文件]` | 汇总比赛指标：搜索过程、token、自动化程度、模型复杂度、推理时延、预设目标达成情况 |

`xgb_runner.py`、`crossval.py`、`feature_worker.py`、`worker_entry.py` 由控制器在子进程里调用，不需要手动执行。

## 2. Python API

```python
from pathlib import Path
from agent import create_session, Loop, freeze, export_report, update_limits
from search_step import initialize, step
from confirm_search import confirm
from operating_point import choose, load

session = create_session(data='/data/typeA', workspace='/ws', domain='memory', instructions='',
                         config='examples/runtime.json', mapping='mapping.json')   # 返回会话路径
loop = Loop(session)            # 可以传入 provider=... 替换 LLM（见第 3 节）
loop.run()                      # 运行闭环；loop.s 是状态，loop.c 是配置
confirm(session, Path(session).parent/'confirmation')

folder = Path(session)/'trials'/loop.s['best']/'attempt_01'
op = choose(folder, loop.c['operating_point'])      # {'source','step','threshold','model',...}
model = load(folder, op, threads=4)                 # 截断到部署步数的 xgboost.Booster
scores = model.inplace_predict(x)                   # x: float32 矩阵，列顺序 = result.json 的 features
```

| 函数 | 说明 |
|---|---|
| `create_session(data, workspace, domain, instructions, config=None, mapping=None, complete_coverage=False, provider=None)` | 等同于 `agent.py init`，返回会话路径 |
| `Loop(session, provider=None)` | 打开会话。如果配置或代码和封印不一致会直接报错 |
| `Loop.prepare()` / `Loop.run()` | 等同于 `prepare` / `run` |
| `freeze(loop)`、`export_report(loop)`、`update_limits(loop, changes)` | 等同于对应的命令 |
| `initialize(source, session, active_cache=None, arm_path=None, data_dir=None)`、`step(session)` | 等同于 `search_step.py` |
| `confirm(session, out)` | 等同于 `confirm_search.py` |
| `choose(trial_folder, mode='cv_oof')`、`load(trial_folder, op, threads=1)` | 取得部署工作点和对应的模型 |

**直接拿模型做预测时要注意：** 输入特征必须和训练时的计算方式、列顺序完全一致（见 trial 的 `result.json` 中的 `features`）。如果用了特征 Agent 生成的特征，还要先执行 `feature_programs.materialize`。设备级的报警判定（按时间表打分、聚合、提前量窗口）在 `prediction.predict` 中完成。

## 3. LLM 接入接口

控制器只通过下面这一个方法和大模型交互：

```python
class MyProvider:
    def call(self, role, payload, folder, *, max_tokens=None) -> dict:
        """role: 'analysis' | 'feature' | 'code'
        payload: 发给该角色的 JSON 任务（payload['task'] 说明任务类型）
        folder:  本次调用的记录目录（calls/<编号>_<角色>/）
        返回：解析好的 JSON 对象（dict）"""
        ...
        (Path(folder)/'usage.json').write_text(json.dumps({'total_tokens': n}))   # 用于 token 计费和预算
        return reply

Loop(session, provider=MyProvider()).run()
```

- system prompt 由 `skills/.../experiment-method.md`（共享方法）和 `prompts/<角色>.txt` 拼接而成，参考 `providers.DeepSeek.call`；
- 回复格式不合法时，控制器会把错误信息附在下一次请求里让模型修正，最多 `max_repairs` 次；还不行就退回到预先定义好的单因素实验（记录为 `controller_fallback` 事件），**不会**假装是模型选的；
- 回复因长度被截断时，抛出 `providers.ReplyTruncated`，控制器会加大 `max_tokens` 重试；
- 只要是 OpenAI 兼容的服务，不用写代码，改配置里的 `deepseek_base_url` 和 `*_model` 就可以。

### 三个角色做什么

| 角色 | 任务（`payload['task']`） | 必须返回 |
|---|---|---|
| 特征 Agent | `adaptation`：根据数据表头生成字段映射 | `{"mapping": {...}}` |
| 特征 Agent | `hypotheses`：根据字段说明提出故障机理假设（看不到标签） | `{"hypotheses": [{id, mechanism, basis, fields, expected_direction, falsified_if}]}` |
| 特征 Agent | `features`：为指定假设写因果特征程序 | `{"source": "def transform(frame): ...", "features": [...], "reason": ...}` |
| 分析 Agent | `analysis`：读取实验历史和诊断证据，选择下一步动作 | 见下表 |
| 代码 Agent | `training`：把已接受的计划写成训练配置代码 | `{"source": "..."}` |

分析 Agent 可以选择的动作：

| `action` | 必需字段 | 说明 |
|---|---|---|
| `experiment` | `parent` `changes` `reason` `hypothesis` `expected` `falsification` | 一次只改一个因素（或一组耦合因素）：超参数（范围见 `experiment_contract.BOUNDS`）、`seed`、`base_weight`、`rerank` |
| `features` | `parent` `hypothesis_id` `question` `reason` `expected` `falsification`，可选 `target` `exploration_reason` | 请特征 Agent 为某个假设生成特征 |
| `diagnose` | `parent` `kind` `question` `reason` | 取得更多证据：`errors`、`coverage`、`score_distribution`、`feature_profiles`、`loss_curve`、`paired_devices`、`high_score_contrast`、`missed_fault_contrast`、`error_atlas` |
| `stop` | `reason` | 至少完成 3 组实验并做过一次特征实验后才允许停止 |

## 4. 主要配置项

默认值在 `settings.DEFAULTS`。`init --config` 只能覆盖其中已有的键。

| 键 | 默认值 | 说明 |
|---|---|---|
| `device` / `num_threads` / `memory_limit_gb` | `cuda:0` / 8 / 32 | 训练设备与资源上限 |
| `max_trials` / `max_agent_calls` / `max_reported_tokens` | 24 / 160 / 1,500,000 | 实验组数、LLM 调用次数和 token 的总预算 |
| `rounds` / `eval_every` / `patience` | 600 / 50 / 4 | XGBoost 最大轮数、验证间隔、早停耐心 |
| `cv_folds` / `cv_rounds` / `cv_bootstraps` | 5 / 400 / 1000 | 设备分组 CV 与配对 bootstrap |
| `acceptance` | `cv_paired` | 采纳依据（`selection_f1` 为旧版规则） |
| `accept_test` / `accept_probability` / `accept_lower_bound` | `probability` / 0.9 / 0.0 | 采纳条件：P(更好) ≥ 0.9 并且 CV ΔF1 > 0 |
| `accept_require_net_hits` | `false` | 设为 `true` 时，丢失的命中多于新增的命中就不采纳 |
| `operating_point` | `cv_oof` | 部署用的步数和阈值来源（或 `selection_window`） |
| `screen_margin` | 0.02 | 选择窗口 F1 比当前最优低这么多就不跑 CV |
| `analysis_model` / `feature_model` / `code_model` | `deepseek-v4-pro` | 三个角色使用的模型 |
| `deepseek_base_url` | `https://api.deepseek.com` | OpenAI 兼容服务的地址 |

评价口径由 `--domain` 决定，会话建立后就不能再改：内存是 `memory_15m_7d_device_f1_v2`（提前 15 分钟到 7 天内的设备级 F1），其他领域是 `period_device_f1_v2`。

## 5. 环境变量

| 变量 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` | LLM 密钥。必须设置，只在控制器进程中使用 |
| `TYPEA_SEARCH_ISOLATION` | 可选。指向一个 JSON 文件，`forbidden_reads` 列出搜索时禁止读取的路径（如原始数据、TEST），`readonly_roots` 列出禁止写入的旧会话目录 |
