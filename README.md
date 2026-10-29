# final · 大模型 Agent 驱动的器件故障预测与自动调优框架

「大模型驱动的复杂系统自检自愈与自优化挑战赛」A 题作品。分析、特征、代码三个 LLM Agent 围绕 XGBoost 组成实验闭环：诊断误报和漏报、提出可证伪的假设、生成特征或参数实验，再由受控执行器训练和交叉验证，按预先确定的统计规则决定是否采纳。整个流程有记录、可追溯、可复现。

当前版本 **v3.2**（代码提交 `8ebafc8`）。

## 结果一览

| 数据 | 状态 | 结果 |
|---|---|---|
| 内存 TypeA（SmartMem） | ✅ 完成 | 13 组全自动搜索，无人工干预。采纳 trial_008；历史 TEST 设备级 F1 由基线 13.16% 提高到 **20.22%**（TP 20→27，FP 183→139）。详见 [results/typea_v3.2](results/typea_v3.2/README.md) |
| 内存 TypeB | ✅ 完成 | 135 特征（含原始日志特征），`screen_margin=1.0`，13 组全部经过 CV，最终保留基线。确认窗口 F1 **54.55%**，历史 TEST F1 0（TP=0、FP=2、FN=12）。测试期多数故障在训练数据中缺乏可学习的前兆，详见 [results/typeb_v3.2 的原因分析](results/typeb_v3.2/README.md) |
| 硬盘（Backblaze） | ⏳ 运行中 | – |

历史 TEST 时间段在项目早期被查看过，因此以上测试成绩属于回顾性评估，不是全新的独立盲测。确认窗口的结果（搜索阶段没有用过的日期）一并保存在结果目录中。Accuracy、模型参数量、自动化覆盖率等全部指标的对比见 [results/README.md](results/README.md)。

## 目录结构

```
.
├── agent.py               # 主入口：会话生命周期、三角色调用、实验调度、采纳规则（CLI）
├── search_step.py         # 从已有会话复用缓存新建会话；单步调度（CLI）
├── confirm_search.py      # 搜索结束后在确认窗口上评估基线与最优模型（CLI）
├── operating_point.py     # 部署用的训练步数和阈值（CV OOF / 选择窗口）（CLI）
├── raw_cache.py           # 内存原始 CE 日志 → 原始比特/空间特征缓存（CLI）
├── settings.py            # 默认配置与校验
├── providers.py           # LLM 接口（DeepSeek / OpenAI 兼容）
├── prompts/               # 三个角色的提示词
├── skills/                # 标准化 Skill 与共享实验方法（每次调用都会注入）
├── data_ingest.py, history_store.py, sample_builder.py, training_selector.py   # 数据接入与训练样本
├── feature_programs.py, feature_worker.py, feature_contract.py                # Agent 生成特征的检查与执行
├── xgb_runner.py, prediction.py, evaluation.py                                # 训练、预测、设备级评分
├── crossval.py, comparison.py, contrast.py                                    # 设备分组 CV、配对比较、误报/漏报诊断
├── rerank.py, parallel_branch.py                                              # 可选的两阶段/并行分支
├── raw_events.py, raw_spatial_v3.py                                           # 内存原始日志解析
├── common.py, worker_entry.py, worker_errors.py, scratch.py, ...              # 子进程、文件锁、哈希、错误分类
├── tests/                 # 回归测试（合成数据，不调用付费 API）
├── tools/
│   ├── propose_features.py # 训练前：特征 Agent 根据领域知识提出特征建议，人来采纳和实现
│   └── session_report.py   # 比赛指标汇总：轮数、token、自动化程度、模型复杂度、时延
├── examples/              # 映射与运行配置示例
├── results/               # 最终实验结果（session、模型、确认与测试评估）
└── docs/                  # 复现指南、接口参考、三个 Agent 的说明、架构
```

根目录的 `.py`、`prompts/` 和 `skills/.../experiment-method.md` 都受**代码封印**（`code_seal.json`）保护：改动其中任何一个文件，已有会话都会拒绝继续运行。新增的工具放在 `tools/` 这样的子目录中，不会影响封印。

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests              # 回归测试，约 3～4 分钟
export DEEPSEEK_API_KEY=...                       # 不要写进任何文件
python agent.py init --data /path/to/data --workspace /path/to/ws --domain memory \
    --mapping mapping.json --config examples/runtime_small.json
python agent.py run --session /path/to/ws/sessions/<会话名>
python confirm_search.py --session <会话> --out <会话>/../confirmation
python tools/session_report.py --session <会话> --md summary.md
```

对于有原始日志的数据，训练开始前先运行 `tools/propose_features.py`，让特征 Agent 提出需要补充的特征，再由人决定是否采纳并实现（[复现指南 §3.3](docs/reproduce.md#33-训练前的特征建议agent-提出人采纳和实现)）。

完整步骤（数据准备、Agent 接入、输出位置）见 **[复现指南](docs/reproduce.md)**；命令行与 Python 接口见 **[接口参考](docs/interfaces.md)**。

## 文档

- [实验结果与指标汇总](results/README.md)：TypeA、TypeB 的结果，以及 Accuracy、参数量、自动化覆盖率等指标对比
- [TypeB 结果与原因分析](results/typeb_v3.2/README.md)：为什么确认窗口有效、测试期失败
- [复现指南](docs/reproduce.md)：要准备哪些数据、按什么顺序执行、输出在哪里
- [接口参考](docs/interfaces.md)：命令行、Python API、LLM 接入、配置项、会话目录结构
- [三个 Agent 的作用与整体流程图](docs/agents.md)
- [框架与模块说明](docs/architecture.md)
- [采纳规则与部署工作点](docs/acceptance_rule.md)：新模型凭什么取代最优模型、部署时用哪一轮和什么阈值，以及为什么这样设计
- [Agent Skill](skills/device-failure-experiment-loop/SKILL.md) 与[共享实验方法](skills/device-failure-experiment-loop/references/experiment-method.md)

v3.1 及更早的实验快照可在 git 提交 `4c27cf1` 中查看；v3 交接说明、v3.1 修复说明和 v3.1 验证报告可在提交 `9262e8f` 的 `docs/history/` 中查看。

仓库不包含原始数据、大型特征缓存、API 密钥和服务器上的完整续跑环境。
