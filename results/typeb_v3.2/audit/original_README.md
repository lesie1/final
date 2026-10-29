# TypeB screen_margin=1.0：审阅入口

本轮已完成 13 组，全部执行交叉验证。最终仍采纳 `trial_001` 基线。
冻结后的历史 TEST：TP=0，FP=2，FN=12，F1=0.00%。
CV F1 最高的候选为 `trial_004`：35.82%；是否采纳以预先固定的配对改善概率标准为准，不能只看最高点估计。

唯一实验设置变化是 screen_margin 从 0.02 到 1.0；输入仍为已验证的135个特征，采纳概率要求仍为0.9。这里的margin不是最终报警阈值。
这是新的自适应搜索；候选提案可以随反馈变化，不是对完全相同候选的配对重放。历史TEST已经使用过，不是独立盲测。

本轮在零分条件下曾按用户要求暂停发布，随后用户明确要求上传供审阅。相关暂停与解禁记录均保留，结果没有重训、换模型或改阈值。

- [完整报告](report.md)
- [各组结果](search_trials.csv)
- [搜索 events](typeb/session/events.jsonl)
- [测试结果](typeb/historical_test/summary.json)
- [配置差异](config_diff.json)
- [导出清单](export_manifest.json)

数据缓存、原始日志、大型数组和CV折模型留在服务器；本目录是审阅快照，不是可直接续跑的完整备份。
