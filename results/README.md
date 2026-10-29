# 实验结果

| 目录 | 数据 | 状态 |
|---|---|---|
| [typea_v3.2/](typea_v3.2/README.md) | 内存 TypeA（SmartMem） | ✅ 完成 |
| `typeb/` | 内存 TypeB（含原始日志特征） | ⏳ 重跑中 |
| `hdd/` | 硬盘（Backblaze） | ⏳ 运行中 |

每个结果目录的结构相同：`session/`（完整搜索记录）、`confirmation/`（确认窗口评估）、`historical_test/`（TEST 评估，如有）、`summary.md`（用 `tools/session_report.py` 生成的比赛指标）。

新结果放进来的步骤：
1. 把服务器上的会话目录复制到 `results/<名称>/session/`，排除 `cache/`、`*.npy`、`*.parquet` 和 CV 各折模型这些大文件；
2. 把 `confirm_search.py` 的输出复制到 `results/<名称>/confirmation/`；
3. 运行 `python tools/session_report.py --session results/<名称>/session --json results/<名称>/summary.json --md results/<名称>/summary.md`；
4. 在本表和根目录 README 的"结果一览"里补上一行。

v3.1 及更早的实验快照（旧 TypeA/TypeB 会话、TypeA 模型迁移到 TypeB 的测试、共同 66 特征对照）已从当前目录移除，可以在 git 提交 `4c27cf1` 中查看。
