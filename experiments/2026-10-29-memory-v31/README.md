# TypeA / TypeB v3.1 实验 session 与补充测试

这里保存 2026-10-28 开始的 TypeA、TypeB 两次 13 组实验的审阅快照，以及 2026-10-29 完成的冻结模型测试和未采纳候选比较。框架源码版本为 `e35150a021c52ed5605c4bb3a2c436b59113e8c0`。

## 从这里阅读

| 内容 | 入口 |
|---|---|
| 完整比较报告，可直接在 GitHub 阅读 | [report.md](report.md) |
| TypeA 原始事件日志 | [typea/session/events.jsonl](typea/session/events.jsonl) |
| TypeB 原始事件日志 | [typeb/session/events.jsonl](typeb/session/events.jsonl) |
| TypeA 搜索状态与采纳理由 | [state.json](typea/session/state.json) |
| TypeB 搜索状态与采纳理由 | [state.json](typeb/session/state.json) |
| TypeA Agent 请求、回复和调用用量 | [calls/](typea/session/calls/) |
| TypeB Agent 请求、回复和调用用量 | [calls/](typeb/session/calls/) |
| TypeA 各组实验计划、训练日志、模型和结果 | [trials/](typea/session/trials/) |
| TypeB 各组实验计划、训练日志、模型和结果 | [trials/](typeb/session/trials/) |
| 19 个未采纳候选及基线的比较表 | [comparison.csv](comparison/comparison.csv) |
| 完整 HTML 报告，下载后可在浏览器打开 | [report.html](comparison/report.html) |
| TypeB 修改前模型的同口径复测 | [historical_typeb/](comparison/historical_typeb/) |
| TypeB 逐台故障诊断 | [typeb_fault_review.csv](comparison/typeb_fault_review.csv) |
| 含逐设备 TEST 预测及模型的完整比较证据包 | [下载 ZIP](artifacts/memory_rejected_comparison_20261029.zip) |
| 所有导出文件的来源哈希及排除清单 | [export_manifest.json](export_manifest.json) |

## session 与测试记录的关系

- 两份 `session/events.jsonl` 是服务器原始训练搜索日志，字节未修改。TypeA 179 条事件、46 次 Agent 调用；TypeB 177 条事件、44 次调用；各完成 13 组训练实验。
- `session/state.json` 保留当时的 `trial_budget_complete` 和 `best=trial_001`。它记录自动采纳规则当时的决定，不会被补充测试排名反向改写。
- 原训练事件中的 `test_opened=false` 指搜索阶段未打开 TEST。此次用户授权的测试在独立目录执行，记录在 [comparison/typea/](comparison/typea/) 和 [comparison/typeb/](comparison/typeb/) 下的 `worker*.log`、`group*_status.json`、逐组 `report_test.json` 中；这不代表历史上从未查看过该测试时间段。
- 补充比较在预测前固定候选名单，重用已训练模型、原验证阈值及保存的特征程序。没有新训练、没有重新选择阈值、没有调用模型 API。名单见 [frozen.json](comparison/frozen.json)，执行程序见 [scripts/](comparison/scripts/)。
- 每个候选是否曾有验证或 CV 提升，以及原采纳判断，都保留在冻结清单和原 session 中。自动未采纳不等于候选在任何后续数据上都更差。

## 本次结果

- **TypeA：**10 个未采纳候选中 6 个测试 F1 高于基线。第 11 组最高，16.27% → 24.47%，命中故障 24 → 29，误报 170 → 107。
- **TypeB：**9 个候选只有第 13 组略高于基线，5.41% → 5.56%。它和旧版第 15 组都只命中 12 台故障中的 1 台，召回率均为 8.33%；旧版在相同范围的 F1 为 7.41%。主要问题是漏报，不能把少量误报变化当成故障识别能力的实质改善。
- 这一测试时间段已经在历史项目中使用过。本次是补充比较，最高测试分数是观察结果，不是新的独立盲测或自动更换最终模型的依据。

## 保存范围与复现边界

本目录保留两次搜索的事件、状态、配置、代码指纹、Agent 调用记录、假设、诊断、生成特征程序、实验计划、执行日志、各组主要训练模型及验证/CV 结果。原始日志与指标没有为了匹配本次结论而重写。

约 11GB 的大型特征缓存、数据数组、逐行分数、部分逐设备验证表和 CV 各折模型不放入 Git 仓库。每项排除的路径、大小和原因见导出清单；完整原始 session 仍在实验服务器。ZIP 证据包包含本次 TEST 的逐设备预测表。

**这是供审查的 session 快照，不是可以脱离原数据直接续跑的完整 checkpoint。** 原 `complete.json`、`code_seal.json`、缓存路径和哈希原样保留，其中可能引用未导出的文件。不能通过修改这些回执来绕过缺失文件或身份校验。

公开副本中的训练事件与原文件校验一致。上传前扫描了 API Key、GitHub Token、私钥及 URL 凭据等常见凭据形式，未发现匹配；记录见 [publication_checks.json](publication_checks.json)。
