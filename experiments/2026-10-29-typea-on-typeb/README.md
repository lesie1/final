# 用 TypeA 训练的模型在 TypeB 上测试

日期：2026-10-29。两者处理的是同一个内存故障提前预测任务；本实验检验已有 TypeA 模型能否直接用于 TypeB 数据。旧工件中的 `transfer` / “迁移”名称保留作历史标识，不表示改换预测任务。

## 阅读入口

| 内容 | 链接 |
|---|---|
| GitHub 可直接阅读的完整报告 | [report.md](report.md) |
| 实验原始事件日志，共 18 条 | [events.jsonl](events.jsonl) |
| 预测前冻结的模型、阈值及特征对应关系 | [frozen.json](frozen.json) |
| 结果比较 CSV | [comparison.csv](comparison.csv) |
| 机器可读结果及核验 | [summary.json](summary.json) / [verified_comparison.json](verified_comparison.json) |
| 各组逐设备预测、故障表和结果回执 | [trials/](trials/) |
| 原 TypeA 模型及训练结果 | [source_models/](source_models/) |
| 实际运行日志 | [run.log](run.log) |
| 执行和报告打包程序 | [scripts/](scripts/) |
| 原 HTML 报告，下载后用浏览器查看 | [report.html](report.html) |
| 原始完整证据包 | [下载 ZIP](artifacts/typea_to_typeb_transfer_20261029.zip) |
| 上一轮 TypeA / TypeB 训练 session 和候选比较 | [上一轮目录](../2026-10-29-memory-v31/README.md) |

## 核心结果

- TypeA 第 2、9 组在 TypeB 上各命中 10 / 12 台故障，召回率 83.33%，但各产生 542 次误报，精确率 1.81%，F1 3.55%。
- 第 2、9 组逐时评分哈希一致，不能当作两次独立验证。
- TypeA 第 10、11、12 组，以及原基线第 1 组，均未触发报警，漏掉 12 台故障。
- TypeB 本轮原有最好模型 F1 为 5.56%，命中 1 台、误报 23 台。本次现成 TypeA 模型没有在 F1 上超过它。

## 输入限制与结论边界

TypeA 模型使用 164 个基础特征；TypeB 只有其中同名、同时间窗的 66 个。缺少的 98 列在独立对齐缓存中保留为 NaN，没有填成 0，也没有伪造缺失的 6 小时窗口和原始日志特征。第 12 组另外 4 个生成特征也依赖缺失列。

本轮保留全部原模型和各自阈值，没有训练、没有调阈值、没有模型 API 调用。因此结果同时包含特征缺失、数据分布和阈值校准的影响，不能单独证明“B 因为样本少而学不会”，也不是完整同特征条件下的 A/B 对照。

后续讨论的“统一 66 个共有特征，分别用 A、B 或 A+B 训练，再在 B 留出数据上比较”尚未在本轮执行。

TypeA 候选按此前已查看的 TypeA TEST 表现挑选，TypeB TEST 也曾使用过。本结果属于探索性补充比较，不是新的独立盲测。原训练 session、模型、代码及最终选模状态保持不变。

## 文件核验与保存范围

原证据包的 60 个工件已逐项校验 SHA-256；642,104 行共享特征在进入模型时逐字节一致，预测后的计数和指标也从逐设备表重算核对。原件哈希见 [SHA256_MANIFEST.json](SHA256_MANIFEST.json)。

ZIP SHA-256：`8b4ecfac597951cf34f2a578408134fda5a1a2576ff97968144cd7aa4d423468`。

此目录保存报告、事件、执行程序、模型和逐设备 TEST 证据；对齐后的大型特征缓存及原始数据仍在实验服务器。源路径与原回执保留用于追溯，本目录不是脱离原数据即可续跑的完整环境。上传前扫描结果见 [publication_checks.json](publication_checks.json)。
