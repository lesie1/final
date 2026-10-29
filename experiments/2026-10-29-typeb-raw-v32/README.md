# TypeB 原始日志接入实验

[结果报告](report.md) · [网页报告](report.html) · [搜索各组结果](search_trials.csv)

本轮使用 v3.2 控制器与 135 个基础输入（66 个原特征、65 个原始日志衍生特征、4 个位图可观测性计数）。预算最多 13 组，实际完成 12 组后由 Agent 提前停止，最终保留 trial_001。

历史 TEST：TP=0、FP=2、FN=12，F1=0；确认窗口 F1=54.55%，不能代替 TEST 结论。

- `typeb/session/`：新会话、参数、主模型、Agent 调用、诊断和搜索事件。
- `typeb/historical_test/`：测试前冻结记录、结果和事件。
- `data_quality/`：原样本/标签不变、因果前缀检查及两阶段缓存清单。
- `evaluation_startup_repair.json`、`report_repair.json`：仅基础设施和报告修复，无重训或阈值重选。
- `export_manifest.json`：导出文件哈希与明确的排除清单。

此目录是历史运行证据，内部 runtime 固定为 8ebafc8d311e734d2089df9d1106c962e2026b66，不替换仓库根目录的当前实现。原始数据与完整缓存仍在服务器；此快照不保证可独立续跑。历史 TEST 先前已见过，不是新的独立盲测。
